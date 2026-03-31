from process import calculations
from process.cdc_config import (
    PER_RUN_MIN_DIST,
    PER_RUN_MIN_WIDTH,
    PER_RUN_PROM_FRAC,
)
from process.ensemble import per_run_peaks
import numpy as np
import math
from scipy.stats import ks_2samp as _ks2  # exact-parity fallback for KS

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

        self.number_of_ages = int(da.size)
        self.number_of_invalid_ages = int(da.size - len(self.valid_discordant_ages))

        # Old behavior: if no valid discordants, force KS=(1.0, 0.0-equivalent)
        if len(self.valid_discordant_ages) == 0:
            ks_d, ks_p = 1.0, 0.0
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
    """One Monte Carlo realisation over the Pb-loss age grid."""

    def __init__(self,
                 run_number,
                 sample_name,
                 concordant_uPb,
                 concordant_pbPb,
                 discordant_uPb,
                 discordant_pbPb,
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
        self._raw_statistics_by_pb_loss_age = {}  # key: age (YEARS) -> MonteCarloRunPbLossAgeStatistics
        self.optimal_pb_loss_age = None
        self.optimal_uPb = None
        self.optimal_pbPb = None
        self.optimal_statistic = None

        self.heatmapColumnData = None
        self.lead_loss_ages = []

        self._all_statistics_by_pb_loss_age = {}
        self._heatmap_view_which = None

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

        # Project all discordant points once for the current lower-intercept age.
        all_ui = np.empty_like(self.discordant_uPb, dtype=float)
        for i, (du, dp) in enumerate(zip(self.discordant_uPb, self.discordant_pbPb)):
            ui = calculations.discordant_age(xL, yL, float(du), float(dp))
            all_ui[i] = np.nan if ui is None else float(ui)

        # Store one statistics object per age using all discordant analyses.
        st_all = MonteCarloRunPbLossAgeStatistics(
            self.concordant_ages, all_ui.tolist(), dissimilarity_test, penalise_invalid_ages
        )
        self._all_statistics_by_pb_loss_age[leadLossAge] = st_all
        self.statistics_by_pb_loss_age[leadLossAge] = st_all
        self._raw_statistics_by_pb_loss_age[leadLossAge] = st_all

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
        D_raw = np.asarray(
            [
                self._raw_statistics_by_pb_loss_age.get(a, self.statistics_by_pb_loss_age[a]).test_statistics[0]
                for a in ages_year
            ],
            float,
        )

        # Run-level optimum follows the active primary channel.
        prefer_pen = bool(getattr(self.settings, "penaliseInvalidAges", True))
        D_primary = D_pen if prefer_pen else D_raw
        j = _find_optimal_index(D_primary)
        best_age_y = float(ages_year[j])

        self.optimal_pb_loss_age = best_age_y
        self.optimal_uPb  = calculations.u238pb206_from_age(best_age_y)
        self.optimal_pbPb = calculations.pb207pb206_from_age(best_age_y)
        if prefer_pen:
            self.optimal_statistic = self.statistics_by_pb_loss_age[best_age_y]
        else:
            self.optimal_statistic = self._raw_statistics_by_pb_loss_age.get(
                best_age_y,
                self.statistics_by_pb_loss_age[best_age_y],
            )

        # Legacy surface shim now follows active primary channel.
        self.ks_surface = _KSSurface(age_ma, D_primary)

        # Keep legacy per-run peak arrays aligned with the configured ensemble gates.
        S_raw = 1.0 - D_raw
        S_pen = 1.0 - D_pen
        try:
            self.peaks_ma_raw = per_run_peaks(
                age_ma, S_raw,
                prom_frac=float(PER_RUN_PROM_FRAC),
                min_dist=int(PER_RUN_MIN_DIST),
                min_width_nodes=int(PER_RUN_MIN_WIDTH),
                require_full_prom=False, max_keep=None, fallback_global_max=False
            )
            self.peaks_ma_pen = per_run_peaks(
                age_ma, S_pen,
                prom_frac=float(PER_RUN_PROM_FRAC),
                min_dist=int(PER_RUN_MIN_DIST),
                min_width_nodes=int(PER_RUN_MIN_WIDTH),
                require_full_prom=False, max_keep=None, fallback_global_max=False
            )
        except TypeError:
            # Legacy signature fallback uses the same configured prominence and distance.
            self.peaks_ma_raw = per_run_peaks(
                age_ma, S_raw,
                prom_frac=float(PER_RUN_PROM_FRAC),
                min_dist=int(PER_RUN_MIN_DIST),
            )
            self.peaks_ma_pen = per_run_peaks(
                age_ma, S_pen,
                prom_frac=float(PER_RUN_PROM_FRAC),
                min_dist=int(PER_RUN_MIN_DIST),
            )

        # Legacy alias (RAW by default)
        self.peaks_ma = self.peaks_ma_raw

    def createHeatmapData(self, minAge, maxAge, resolution):
        """
        Build a per-run column vector over the grid (length <= resolution) with
        primary dissimilarity values, linearly interpolated across gaps.

        Primary channel follows GUI intent:
          - penaliseInvalidAges=True  -> use penalised score (D*)
          - penaliseInvalidAges=False -> use raw KS D
        """
        which = str(getattr(self, "_heatmap_view_which", "") or "").strip().lower()
        if which == "raw":
            prefer_pen = False
        elif which == "pen":
            prefer_pen = True
        else:
            prefer_pen = bool(getattr(self.settings, "penaliseInvalidAges", True))

        stats_map = self.statistics_by_pb_loss_age if prefer_pen else self._raw_statistics_by_pb_loss_age
        if not isinstance(stats_map, dict) or not stats_map:
            stats_map = getattr(self, "_all_statistics_by_pb_loss_age", None)

        if not isinstance(stats_map, dict) or not stats_map:
            self.heatmapColumnData = []
            return

        def _value_at(age_key: float) -> float:
            st = stats_map.get(float(age_key))
            if st is None:
                return float("nan")
            if prefer_pen:
                v = float(st.score)
            else:
                v = float(st.test_statistics[0])
            if not np.isfinite(v):
                return float("nan")
            return float(np.clip(v, 0.0, 1.0))

        runAges = sorted(list(stats_map.keys()))

        ageInc = (maxAge - minAge) / resolution
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
                prevStat = _value_at(prevAge)
                nextStat = _value_at(nextAge)
                prevDiff = col - prevNonEmptyCol
                nextDiff = nextNonEmptyCol - col
                totalDiff = nextDiff + prevDiff
                value = (nextDiff * prevStat + prevDiff * nextStat) / totalDiff
            else:
                vals = np.asarray([_value_at(a) for a in colAges[col]], float)
                vals = vals[np.isfinite(vals)]
                value = float(np.mean(vals)) if vals.size else float("nan")
            colData.append(value)
        self.heatmapColumnData = colData

    def toList(self):
        # Convert to Ma in the exported row
        return [self.sample_name, self.run_number,
                (self.optimal_pb_loss_age / 1_000_000.0) if self.optimal_pb_loss_age is not None else float("nan")]
