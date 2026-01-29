from process import calculations
import numpy as np
import math
import copy
from scipy.stats import ks_2samp as _ks2
from model.settings.calculation import ConcordiaMode

# ---------------------------
# Per-node statistics record
# ---------------------------

def _find_optimal_index(values_to_compare) -> int:
    # values_to_compare is a 1D array-like of floats (lower is better)
    minIndex, minValue = min(enumerate(values_to_compare), key=lambda v: v[1])
    n = len(values_to_compare)

    startMinIndex = minIndex
    while startMinIndex > 0 and values_to_compare[startMinIndex - 1] == minValue:
        startMinIndex -= 1

    endMinIndex = minIndex
    while endMinIndex < n - 1 and values_to_compare[endMinIndex + 1] == minValue:
        endMinIndex += 1

    if (endMinIndex != n - 1 and startMinIndex != 0) or (endMinIndex == n - 1 and startMinIndex == 0):
        return (endMinIndex + startMinIndex) // 2
    if startMinIndex == 0:
        return 0
    return n - 1

class _KSSurface:
    def __init__(self, ages_ma, dvals):
        import numpy as _np
        self.ages  = _np.asarray(ages_ma, float)
        self.dvals = _np.asarray(dvals,  float)

    def goodness(self, mode="1-D", sigma=0.02):
        import numpy as _np
        if mode == "1-D":
            return 1.0 - self.dvals
        elif mode == "exp":
            return _np.exp(-self.dvals / float(sigma))
        raise ValueError(mode)


class MonteCarloRunPbLossAgeStatistics:
    """
    Holds raw K–S result (D, p), invalid count, and the penalised dissimilarity:
        score = base + (1 - base) * inv_frac
    where base = KS-D, inv_frac = (#invalid) / (#total discordant).
    (Algebraically identical to: 1 - (1 - D)*(1 - inv_frac))
    """
    def __init__(self, concordant_ages, discordant_ages, dissimilarity_test, penalise_invalid_ages):
        ca = np.asarray(concordant_ages or [], dtype=float)
        da = np.asarray(discordant_ages or [], dtype=float)

        self.valid_concordant_ages = ca[np.isfinite(ca)].tolist()
        self.valid_discordant_ages = da[np.isfinite(da)].tolist()

        self.number_of_ages         = int(da.size)
        self.number_of_invalid_ages = int(da.size - len(self.valid_discordant_ages))

        # Old behavior: if no valid discordants, force KS=(1.0, 1.0)
        if len(self.valid_discordant_ages) == 0:
            ks_d, ks_p = 1.0, 1.0
        else:
            # Try the provided test; if that fails, use scipy’s ks_2samp for parity
            try:
                ks_d, ks_p = dissimilarity_test.perform(
                    self.valid_concordant_ages, self.valid_discordant_ages
                )
            except Exception:
                ks_d, ks_p = _ks2(
                    self.valid_concordant_ages, self.valid_discordant_ages, alternative="two-sided"
                )
        self.test_statistics = (float(ks_d), float(ks_p))
        base = float(dissimilarity_test.getComparisonValue(self.test_statistics))

        if penalise_invalid_ages:
            inv_frac = 1.0 if self.number_of_ages == 0 else (
                1.0 - len(self.valid_discordant_ages) / float(self.number_of_ages)
            )
            self.score = base + (1.0 - base) * float(inv_frac)
        else:
            self.score = base


class MonteCarloRun:
    """
    One Monte Carlo realisation. Supports optional clustering:
    pass either `discordant_labels` (new) or `discord_clusters` (old). At each grid
    age we evaluate clusters separately and keep:
        • RAW (KS–D) from the RAW-minimising cluster
        • PENALISED score from the PEN-minimising cluster
    """

    def __init__(self,
                 run_number,
                 sample_name,
                 concordant_uPb,
                 concordant_pbPb,
                 discordant_uPb,
                 discordant_pbPb,
                 discordant_labels=None,     # new name
                 discord_clusters=None,      # old alias (still accepted)
                 settings=None):

        self.run_number   = run_number
        self.sample_name  = sample_name
        self.settings     = settings

        self.concordiaMode = ConcordiaMode.coerce(
            getattr(settings, "concordiaMode", ConcordiaMode.TW)
        ) if settings is not None else ConcordiaMode.TW

        # --- keep GUI-facing names ---
        self.concordant_uPb  = np.asarray(concordant_uPb, float)
        self.concordant_pbPb = np.asarray(concordant_pbPb, float)
        self.discordant_uPb  = np.asarray(discordant_uPb, float)
        self.discordant_pbPb = np.asarray(discordant_pbPb, float)

        # (optional short aliases)
        self.con_u = self.concordant_uPb
        self.con_p = self.concordant_pbPb
        self.dis_u = self.discordant_uPb
        self.dis_p = self.discordant_pbPb

        # Accept both label argument names
        if discordant_labels is None and discord_clusters is not None:
            discordant_labels = discord_clusters

        self.labels = None
        if discordant_labels is not None:
            lab = np.asarray(discordant_labels)
            if lab.shape[0] == self.dis_u.shape[0]:
                self.labels = lab.astype(int)

        # Cache concordant ages (YEARS) for this run
        self.concordant_ages = []

        if self.concordiaMode == ConcordiaMode.WETHERILL:
            # Inputs are Wetherill coords in this mode:
            #   concordant_uPb  -> x = 207/235
            #   concordant_pbPb -> y = 206/238
            for x, y in zip(self.concordant_uPb, self.concordant_pbPb):
                try:
                    x = float(x); y = float(y)
                except Exception:
                    continue
                if not (math.isfinite(x) and math.isfinite(y)) or x <= 0.0 or y <= 0.0:
                    continue
                try:
                    t = calculations.concordant_age_wetherill(x, y)
                    if isinstance(t, (int, float)) and math.isfinite(t):
                        self.concordant_ages.append(float(t))
                except Exception:
                    pass
        else:
            # TW coords in this mode:
            #   concordant_uPb  -> u = 238/206
            #   concordant_pbPb -> v = 207/206
            for u, v in zip(self.concordant_uPb, self.concordant_pbPb):
                try:
                    t = calculations.concordant_age(float(u), float(v))
                    if isinstance(t, (int, float)) and math.isfinite(t):
                        self.concordant_ages.append(float(t))
                except Exception:
                    pass

        self.statistics_by_pb_loss_age = {}  # key: age (YEARS) -> MonteCarloRunPbLossAgeStatistics
        self.optimal_pb_loss_age = None
        self.optimal_uPb = None
        self.optimal_pbPb = None
        self.optimal_statistic = None

        self.heatmapColumnData = None
        self.lead_loss_ages = []

        # Per-cluster, per-age KS stats for later stacking (used in processing)
        self._stats_by_age_by_cluster = {}  # {cluster_id: {age_years: MonteCarloRunPbLossAgeStatistics}}

        # legacy surface shim (penalised dissimilarity)
        self.ks_surface = None

    # ---- main per-node evaluation -------------------------------------------

    def samplePbLossAge(self, leadLossAge, dissimilarity_test, penalise_invalid_ages):
        """Evaluate this run at a given lower intercept age (YEARS)."""

        all_ui = np.full_like(self.discordant_uPb, np.nan, dtype=float)

        if self.concordiaMode == ConcordiaMode.WETHERILL:
            # Anchor on Wetherill concordia at Pb-loss age
            xL = calculations.pb207u235_from_age(float(leadLossAge))  # 207/235
            yL = calculations.pb206u238_from_age(float(leadLossAge))  # 206/238

            # discordant_uPb  -> x2 (207/235)
            # discordant_pbPb -> y2 (206/238)
            for i, (x2, y2) in enumerate(zip(self.discordant_uPb, self.discordant_pbPb)):
                try:
                    x2 = float(x2); y2 = float(y2)
                except Exception:
                    continue

                if not (math.isfinite(x2) and math.isfinite(y2)) or (x2 <= 0.0) or (y2 <= 0.0):
                    continue

                ui = calculations.discordant_age_wetherill(xL, yL, x2, y2)
                if ui is not None and math.isfinite(ui):
                    all_ui[i] = float(ui)

        else:
            # TW mode (unchanged)
            xL = calculations.u238pb206_from_age(float(leadLossAge))
            yL = calculations.pb207pb206_from_age(float(leadLossAge))

            for i, (du, dv) in enumerate(zip(self.discordant_uPb, self.discordant_pbPb)):
                ui = calculations.discordant_age(xL, yL, float(du), float(dv))
                if ui is not None and math.isfinite(ui):
                    all_ui[i] = float(ui)


        # No clustering → evaluate all together
        if self.labels is None:
            st = MonteCarloRunPbLossAgeStatistics(
                self.concordant_ages, all_ui.tolist(), dissimilarity_test, penalise_invalid_ages
            )
            self.statistics_by_pb_loss_age[leadLossAge] = st
            self._stats_by_age_by_cluster.setdefault(0, {})[float(leadLossAge)] = st
            return

        # Clustered path: keep RAW-min and PEN-min (possibly different clusters)
        best_raw_val, best_raw_stat = float("inf"), None
        best_pen_val, best_pen_stat = float("inf"), None

        for lab in np.unique(self.labels):
            mask = (self.labels == lab)
            if not np.any(mask):
                continue
            ages_k = all_ui[mask].tolist()
            st_k = MonteCarloRunPbLossAgeStatistics(
                self.concordant_ages, ages_k, dissimilarity_test, penalise_invalid_ages
            )
            self._stats_by_age_by_cluster.setdefault(int(lab), {})[float(leadLossAge)] = st_k

            D_raw  = float(st_k.test_statistics[0])
            Sc_pen = float(st_k.score)

            if D_raw < best_raw_val:
                best_raw_val, best_raw_stat = D_raw, st_k
            if Sc_pen < best_pen_val:
                best_pen_val, best_pen_stat = Sc_pen, st_k

        out = copy.copy(best_raw_stat)
        out.score = best_pen_val
        if best_pen_stat is not None:
            out.number_of_invalid_ages = best_pen_stat.number_of_invalid_ages
            out.number_of_ages         = best_pen_stat.number_of_ages

        self.statistics_by_pb_loss_age[leadLossAge] = out

    def calculateOptimalAge(self):
        """
        Choose the node with the MINIMUM penalised dissimilarity (score = D*).
        Also compute per-run peaks on RAW and PEN goodness surfaces.
        Keep a small ks_surface shim for downstream code.
        """
        if not self.statistics_by_pb_loss_age:
            self.optimal_pb_loss_age = float("nan")
            self.optimal_statistic = None
            self.peaks_ma_raw = np.array([], float)
            self.peaks_ma_pen = np.array([], float)
            self.peaks_ma     = self.peaks_ma_raw
            return

        # Sort by age (YEARS) and rebuild arrays
        items = sorted(self.statistics_by_pb_loss_age.items(), key=lambda kv: kv[0])
        ages_year = np.asarray([a for a, _ in items], float)
        age_ma    = ages_year / 1e6
        D_pen     = np.asarray([st.score              for _, st in items], float)
        D_raw     = np.asarray([st.test_statistics[0] for _, st in items], float)

        # Run-level optimum from penalised curve
        j = _find_optimal_index(D_pen)
        best_age_y = float(ages_year[j])
        self.optimal_pb_loss_age = best_age_y

        # store optimum point
        if self.concordiaMode == ConcordiaMode.WETHERILL:
            self.optimal_uPb  = calculations.pb207u235_from_age(best_age_y)  # x
            self.optimal_pbPb = calculations.pb206u238_from_age(best_age_y)  # y
        else:
            self.optimal_uPb  = calculations.u238pb206_from_age(best_age_y)
            self.optimal_pbPb = calculations.pb207pb206_from_age(best_age_y)

        self.optimal_statistic = self.statistics_by_pb_loss_age[best_age_y]

        # Legacy surface (penalised dissimilarity)
        self.ks_surface = _KSSurface(age_ma, D_pen)

    def createHeatmapData(self, minAge, maxAge, resolution):
        """
        Build a per-run column vector over the grid (length <= resolution) with
        penalised dissimilarity (D*) values, linearly interpolated across gaps.
        """
        ageInc = (maxAge - minAge) / resolution
        runAges = sorted(list(self.statistics_by_pb_loss_age.keys()))
        if not runAges:
            self.heatmapColumnData = []
            return

        colAges = [[] for _ in range(resolution)]
        for age in runAges:
            col = (resolution - 1) if age == maxAge else int((age - minAge) // ageInc)
            colAges[col].append(age)

        colData = []
        for col in range(resolution):
            prevNonEmptyCol = col
            nextNonEmptyCol = col
            while prevNonEmptyCol > 0 and len(colAges[prevNonEmptyCol]) == 0:
                prevNonEmptyCol -= 1
            while nextNonEmptyCol < resolution - 1 and len(colAges[nextNonEmptyCol]) == 0:
                nextNonEmptyCol += 1
            if len(colAges[prevNonEmptyCol]) == 0 or len(colAges[nextNonEmptyCol]) == 0:
                continue

            if prevNonEmptyCol != nextNonEmptyCol:
                prevAge  = max(colAges[prevNonEmptyCol])
                nextAge  = min(colAges[nextNonEmptyCol])
                prevStat = self.statistics_by_pb_loss_age[prevAge].score
                nextStat = self.statistics_by_pb_loss_age[nextAge].score
                prevDiff = col - prevNonEmptyCol
                nextDiff = nextNonEmptyCol - col
                totalDiff = nextDiff + prevDiff
                value = (nextDiff * prevStat + prevDiff * nextStat) / totalDiff
            else:
                value = float(np.mean([self.statistics_by_pb_loss_age[a].score for a in colAges[col]]))
            colData.append(value)
        self.heatmapColumnData = colData

    def toList(self):
        mode = "WETHERILL" if self.concordiaMode == ConcordiaMode.WETHERILL else "TW"
        age_ma = (self.optimal_pb_loss_age / 1_000_000.0) if self.optimal_pb_loss_age is not None else float("nan")
        return [self.sample_name, self.run_number, mode, age_ma]

