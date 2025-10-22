from process import calculations, processing
from process.ensemble import per_run_peaks
import numpy as np
import math
import copy
from scipy.stats import ks_2samp as _ks2  # exact-parity fallback for KS

# ---------------------------
# Per-node statistics record
# ---------------------------

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
        for u, p in zip(self.concordant_uPb, self.concordant_pbPb):
            try:
                t = calculations.concordant_age(float(u), float(p))
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

        # --- per-run peak attributes (RAW & PEN) ---
        self.peaks_ma_raw = None
        self.peaks_ma_pen = None
        self.peaks_ma     = None   # legacy alias (RAW by default)

        # legacy surface shim (penalised dissimilarity)
        self.ks_surface = None

    # ---- main per-node evaluation -------------------------------------------

    def samplePbLossAge(self, leadLossAge, dissimilarity_test, penalise_invalid_ages):
        """Evaluate this run at a given lower intercept age (YEARS)."""
        xL = calculations.u238pb206_from_age(float(leadLossAge))
        yL = calculations.pb207pb206_from_age(float(leadLossAge))

        # Project all discordant points once (old execution order)
        all_ui = np.empty_like(self.discordant_uPb, dtype=float)
        for i, (du, dp) in enumerate(zip(self.discordant_uPb, self.discordant_pbPb)):
            ui = calculations.discordant_age(xL, yL, float(du), float(dp))
            all_ui[i] = np.nan if ui is None else float(ui)

        # No clustering → evaluate all together
        if self.labels is None:
            st = MonteCarloRunPbLossAgeStatistics(
                self.concordant_ages, all_ui.tolist(), dissimilarity_test, penalise_invalid_ages
            )
            self.statistics_by_pb_loss_age[leadLossAge] = st
            self._stats_by_age_by_cluster.setdefault(0, {})[float(leadLossAge)] = st
            return

        # Clustered path: keep RAW-min and PEN-min (possibly different clusters)
        best_raw_val, best_raw_stat = float('inf'), None
        best_pen_val, best_pen_stat = float('inf'), None

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

        # Compose: raw fields from RAW-minimising; penalised score from PEN-minimising
        out = copy.copy(best_raw_stat)
        out.score = best_pen_val
        if best_pen_stat is not None:
            out.number_of_invalid_ages = best_pen_stat.number_of_invalid_ages
            out.number_of_ages         = best_pen_stat.number_of_ages

        self.statistics_by_pb_loss_age[leadLossAge] = out

    def calculateOptimalAge(self):
        """
        Choose the node with the MINIMUM penalised dissimilarity (score = D*).
        Also compute per-run peaks on RAW and PEN goodness surfaces (old thresholds).
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
        j = processing._findOptimalIndex(D_pen)
        best_age_y = float(ages_year[j])

        self.optimal_pb_loss_age = best_age_y
        self.optimal_uPb  = calculations.u238pb206_from_age(best_age_y)
        self.optimal_pbPb = calculations.pb207pb206_from_age(best_age_y)
        self.optimal_statistic = self.statistics_by_pb_loss_age[best_age_y]

        # Legacy surface (penalised dissimilarity)
        self.ks_surface = _KSSurface(age_ma, D_pen)

        # Per-run peaks on BOTH surfaces (old semantics and thresholds)
        S_raw = 1.0 - D_raw
        S_pen = 1.0 - D_pen
        try:
            self.peaks_ma_raw = per_run_peaks(
                age_ma, S_raw,
                prom_frac=0.04, min_dist=3, min_width_nodes=3,
                require_full_prom=False, max_keep=None, fallback_global_max=False
            )
            self.peaks_ma_pen = per_run_peaks(
                age_ma, S_pen,
                prom_frac=0.03, min_dist=3, min_width_nodes=3,
                require_full_prom=False, max_keep=None, fallback_global_max=False
            )
        except TypeError:
            # Legacy signature fallback (match old thresholds)
            self.peaks_ma_raw = per_run_peaks(age_ma, S_raw, prom_frac=0.03, min_dist=3)
            self.peaks_ma_pen = per_run_peaks(age_ma, S_pen, prom_frac=0.03, min_dist=3)

        # Legacy alias (RAW by default)
        self.peaks_ma = self.peaks_ma_raw

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
        # Convert to Ma in the exported row
        return [self.sample_name, self.run_number,
                (self.optimal_pb_loss_age / 1_000_000.0) if self.optimal_pb_loss_age is not None else float("nan")]
