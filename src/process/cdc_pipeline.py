"""CDC processing pipeline.

A refactor of the original `process/processing.py`:
- configuration lives in `process.cdc_config`
- diagnostics/paper exports live in `process.cdc_diagnostics`
- Tera-W maths lives in `process.cdc_tw`
- population clustering lives in `process.cdc_population`

The GUI should continue to call `process.processing.processSamples(...)`.
"""

from __future__ import annotations

import platform
import time
from enum import Enum
from typing import Dict, List
from model.settings.calculation import ConcordiaMode

import numpy as np

from model.monteCarloRun import MonteCarloRun
from model.settings.calculation import DiscordanceClassificationMethod
from process import calculations
from process.discordantClustering import (
    find_discordant_clusters,
    _labels_from_this_run,
    _labels_from_this_run_wetherill,
    _stack_min_across_clusters,
    lower_intercept_proxy,
    lower_intercept_proxy_wetherill,
    _soft_accept_labels,
    _adaptive_gates,
    stack_goodness_by_cluster,
)

from process.ensemble import robust_ensemble_curve, build_ensemble_catalogue
from process.cdc_config import (
    CATALOGUE_CSV_PEN,
    CATALOGUE_CSV_RAW,
    CATALOGUE_SURFACE,
    CDC_WRITE_OUTPUTS,
    ENS_DELTA_MIN,
    FD_DIST_FRAC,
    FH_HEIGHT_FRAC,
    FP_PROM_FRAC,
    FR_RUN_REL,
    FS_SUPPORT,
    FW_WIN_FRAC,
    KS_EXPORT_ROOT,
    PER_RUN_MIN_DIST,
    PER_RUN_MIN_WIDTH,
    PER_RUN_PROM_FRAC,
    RMIN_RUNS,
    FV_VALLEY_FRAC,
    RUNLOG,
    SMOOTH_MA,
    SMOOTH_FRAC,
    TIMING_MODE,
    USE_CLUSTER_CATALOGUE,
)

from process.cdc_diagnostics import (
    append_catalogue_rows as _append_catalogue_rows,
    ensure_output_dirs as _ensure_output_dirs,
    export_legacy_ks as _export_legacy_ks,
    reset_csv as _reset_csv,
    rss_mb as _rss_mb,
    write_npz_diagnostics as _write_npz_diagnostics,
    write_runlog as _write_runlog,
)
from process.cdc_population import (
    assign_discordant_to_populations as _assign_discordant_to_populations,
    cluster_concordant_populations as _cluster_concordant_populations,
)
from process.cdc_tw import is_reverse_discordant as _is_reverse_discordant
from process.cdc_utils import infer_tier as _infer_tier, seed_from_name as _seed_from_name
from utils import config
from utils.peakHelpers import fmt_peak_stats

TIME_PER_TASK = 0.0

class ProgressType(Enum):
    CONCORDANCE = 0
    SAMPLING    = 1
    OPTIMAL     = 2

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _spot_xy_mode(spot, is_wetherill):
    """
    Return (x,y) in the *MODE* space.
      TW:        x=u=238/206,  y=v=207/206
      Wetherill: x=207/235,    y=206/238
    Includes fallback derivation if one pair is missing.
    """
    U = _safe_float(calculations.U238U235_RATIO)

    if is_wetherill:
        # Prefer Wetherill
        x = _safe_float(getattr(spot, "pb207U235Value", np.nan))
        y = _safe_float(getattr(spot, "pb206U238Value", np.nan))
        if np.isfinite(x) and np.isfinite(y) and (y > 0.0):
            return x, y

        # Fallback from TW
        u = _safe_float(getattr(spot, "uPbValue", np.nan))
        v = _safe_float(getattr(spot, "pbPbValue", np.nan))
        if np.isfinite(u) and np.isfinite(v) and (u > 0.0) and np.isfinite(U) and (U > 0.0):
            return (v * U / u), (1.0 / u)

        return np.nan, np.nan

    else:
        # Prefer TW
        u = _safe_float(getattr(spot, "uPbValue", np.nan))
        v = _safe_float(getattr(spot, "pbPbValue", np.nan))
        if np.isfinite(u) and np.isfinite(v):
            return u, v

        # Fallback from Wetherill
        x = _safe_float(getattr(spot, "pb207U235Value", np.nan))
        y = _safe_float(getattr(spot, "pb206U238Value", np.nan))
        if np.isfinite(x) and np.isfinite(y) and (y > 0.0) and np.isfinite(U) and (U > 0.0):
            return (1.0 / y), (x / (U * y))

        return np.nan, np.nan

def _spot_uv_tw(spot):
    """
    Return TW u=238/206 and v=207/206 for this spot.
    Prefer TW fields; otherwise derive from Wetherill ratios.
    """
    u = _safe_float(getattr(spot, "uPbValue", np.nan))
    v = _safe_float(getattr(spot, "pbPbValue", np.nan))
    if np.isfinite(u) and np.isfinite(v):
        return u, v

    x = _safe_float(getattr(spot, "pb207U235Value", np.nan))  # 207/235
    y = _safe_float(getattr(spot, "pb206U238Value", np.nan))  # 206/238
    U = _safe_float(calculations.U238U235_RATIO)

    if np.isfinite(x) and np.isfinite(y) and (y > 0.0) and np.isfinite(U) and (U > 0.0):
        u = 1.0 / y
        v = x / (U * y)
        return u, v

    return np.nan, np.nan

def _draw_in_mode(rng, spots, n, is_wetherill):
    """
    Return arrays shaped (n, nspots) in the MODE space.
    TW:        X=uPbValue(238/206), Y=pbPbValue(207/206)
    Wetherill: X=pb207U235Value(207/235), Y=pb206U238Value(206/238)
    """
    X, Y = [], []
    for s in spots:
        if is_wetherill:
            X.append(_normal_draws(rng, getattr(s, "pb207U235Value", np.nan), getattr(s, "pb207U235StDev", 0.0), n))
            Y.append(_normal_draws(rng, getattr(s, "pb206U238Value", np.nan), getattr(s, "pb206U238StDev", 0.0), n))
        else:
            X.append(_normal_draws(rng, getattr(s, "uPbValue", np.nan),  getattr(s, "uPbStDev", 0.0),  n))
            Y.append(_normal_draws(rng, getattr(s, "pbPbValue", np.nan), getattr(s, "pbPbStDev", 0.0), n))
    if not X:
        return np.empty((n, 0), float), np.empty((n, 0), float)
    return np.stack(X, axis=1), np.stack(Y, axis=1)


def _normal_draws(rng, mu, sd, n):
    """Return length-n draws; if mu invalid -> all NaN; if sd invalid -> treat as 0."""
    try:
        mu = float(mu)
    except Exception:
        return np.full(n, np.nan, float)
    if not np.isfinite(mu):
        return np.full(n, np.nan, float)

    try:
        sd = float(sd)
    except Exception:
        sd = 0.0
    if (not np.isfinite(sd)) or (sd < 0):
        sd = 0.0

    return rng.normal(mu, sd, n)


def _draw_tw_uv(rng, spots, n):
    """Draw TW u=238/206 and v=207/206 arrays with shape (n, nspots)."""
    U = []
    V = []
    for s in spots:
        U.append(_normal_draws(rng, getattr(s, "uPbValue", np.nan),  getattr(s, "uPbStDev", 0.0),  n))
        V.append(_normal_draws(rng, getattr(s, "pbPbValue", np.nan), getattr(s, "pbPbStDev", 0.0), n))
    if not U:
        return np.empty((n, 0), float), np.empty((n, 0), float)
    return np.stack(U, axis=1), np.stack(V, axis=1)

def processSamples(signals, samples):
    if CDC_WRITE_OUTPUTS:
        _reset_csv(CATALOGUE_CSV_PEN, "sample,peak_no,age_ma,ci_low,ci_high,support")
        _reset_csv(CATALOGUE_CSV_RAW, "sample,peak_no,age_ma,ci_low,ci_high,support")
        _reset_csv(RUNLOG,            "method,phase,sample,tier,R,n_grid,elapsed_s,per_run_median_s,per_run_p95_s,rss_peak_mb,python,numpy")

    for sample in samples:
        completed, skip_reason = _processSample(signals, sample)
        if not completed and skip_reason:
            signals.skipped(sample.name, skip_reason)

    signals.completed()

def _processSample(signals, sample):
    t0 = time.perf_counter()

    try:
        # 1) Classify concordant/discordant (incl. reverse flags)
        completed, skip_reason = _calculateConcordantAges(signals, sample)
        if not completed:
            return False, skip_reason

        # 2) First pass: MC sampling + ensemble
        completed, skip_reason = _performRimAgeSampling(signals, sample)
        if not completed:
            return False, skip_reason
        
        # >>> NEW: if we used population-split, skip edge-guard entirely <<<
        st = sample.calculationSettings
        if bool(getattr(st, "split_by_concordant_population", False)):
            return True, None

        # 3) Edge-guard: widen once if many per-run optima hug a boundary
        try:
            ages_ma = np.asarray(st.rimAges(), float) / 1e6
            if ages_ma.size >= 2 and sample.monteCarloRuns:
                # robust step estimate
                raw_step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 0.0
                if not np.isfinite(raw_step) or raw_step <= 0.0:
                    raw_step = 5.0
                step_ma = raw_step

                opt_ma  = np.array([r.optimal_pb_loss_age for r in sample.monteCarloRuns],
                                   dtype=float) / 1e6
                opt_ma  = opt_ma[np.isfinite(opt_ma)]
                if opt_ma.size:
                    hit_lo    = np.mean(opt_ma <= (ages_ma[0] + step_ma))      # young boundary
                    hit_hi    = np.mean(opt_ma >= (ages_ma[-1] - step_ma))     # old boundary
                    edge_hits = max(hit_lo, hit_hi)

                    if edge_hits > 0.20:
                        span_y        = float(st.maximumRimAge) - float(st.minimumRimAge)
                        expand_y      = 0.20 * span_y
                        widen_younger = (hit_lo >= hit_hi)

                        if widen_younger:
                            new_min = max(0.0, float(st.minimumRimAge) - expand_y)
                            new_max = float(st.maximumRimAge)
                        else:
                            new_min = float(st.minimumRimAge)
                            new_max = float(st.maximumRimAge) + expand_y

                        # keep grid step roughly constant
                        ages_ma_new = np.asarray(st.rimAges(), float) / 1e6
                        raw_step2 = float(np.median(np.diff(ages_ma_new))) if ages_ma_new.size >= 2 else 0.0
                        if not np.isfinite(raw_step2) or raw_step2 <= 0.0:
                            raw_step2 = step_ma or 5.0
                        step_ma2 = raw_step2

                        span_ma_new      = (new_max - new_min) / 1e6
                        safe_step_for_div = step_ma2 if step_ma2 > 1e-9 else max(span_ma_new, 1.0)
                        st.minimumRimAge  = new_min
                        st.maximumRimAge  = new_max
                        st.rimAgesSampled = int(max(3, round(span_ma_new / safe_step_for_div) + 1))

                        # re-run once with widened window
                        sample.monteCarloRuns = []
                        sample.peak_catalogue = []
                        completed, skip_reason = _performRimAgeSampling(signals, sample)
                        if not completed:
                            return False, skip_reason

        except Exception as _edge_guard_err:
            print(f"[CDC] Edge-guard diagnostic failed for {sample.name}: {_edge_guard_err}")

        return True, None

    finally:
        # runtime log (best effort)
        try:
            n_grid = len(sample.calculationSettings.rimAges())
            R_runs = sample.calculationSettings.monteCarloRuns
        except Exception:
            n_grid, R_runs = 0, 0
        _write_runlog(dict(
            method="CDC", phase="e2e_runtime",
            sample=sample.name, tier=_infer_tier(sample.name),
            R=R_runs, n_grid=n_grid,
            elapsed_s=round(time.perf_counter() - t0, 3),
            per_run_median_s="", per_run_p95_s="",
            rss_peak_mb=round(_rss_mb(), 1),
            python=platform.python_version(), numpy=np.__version__,
        ))

def _calculateConcordantAges(signals, sample):
    """
    Classify each valid spot as concordant/discordant and flag reverse discordance.

    - Concordance is determined either by percentage discordance or by error ellipse,
      according to sample.calculationSettings.discordanceClassificationMethod.
    - Reverse discordance is ONLY evaluated in TW space (geometry is TW-specific).

    Emits:
      - ProgressType.CONCORDANCE updates for UI.
      - sample.updateConcordance(concordancy, discordances, reverse_flags).
    """

    sampleNameText = f" for '{sample.name}'" if sample.name else ""
    signals.newTask("Classifying points" + sampleNameText + "...")

    settings   = sample.calculationSettings
    n_spots    = max(1, len(sample.validSpots))
    timePerRow = TIME_PER_TASK / n_spots

    # NEW: select concordia mode (robust to strings/old settings)
    mode = ConcordiaMode.coerce(getattr(settings, "concordiaMode", ConcordiaMode.TW))
    is_wetherill = (mode == ConcordiaMode.WETHERILL)

    concordancy   = []
    discordances  = []
    reverse_flags = []

    for i, spot in enumerate(sample.validSpots):
        signals.progress(ProgressType.CONCORDANCE, i / n_spots)
        time.sleep(timePerRow)
        if signals.halt():
            signals.cancelled()
            return False, "processing halted by user"

        # -----------------------------
        # Concordance test (TW vs Weth)
        # -----------------------------
        if settings.discordanceClassificationMethod == DiscordanceClassificationMethod.PERCENTAGE:
            if not is_wetherill:
                discordance = calculations.discordance(spot.uPbValue, spot.pbPbValue)
            else:
                discordance = calculations.discordance_wetherill(
                    spot.pb207U235Value,
                    spot.pb206U238Value
                )

            # Concordant if magnitude under threshold (and discordance computed)
            concordant = (discordance is not None) and (abs(discordance) < settings.discordancePercentageCutoff)

        else:
            discordance = None
            if not is_wetherill:
                concordant = calculations.isConcordantErrorEllipse(
                    spot.uPbValue,  spot.uPbStDev,
                    spot.pbPbValue, spot.pbPbStDev,
                    settings.discordanceEllipseSigmas
                )
            else:
                concordant = calculations.isConcordantErrorEllipseWetherill(
                    spot.pb207U235Value, spot.pb207U235StDev,
                    spot.pb206U238Value, spot.pb206U238StDev,
                    settings.discordanceEllipseSigmas
                )

        # -----------------------------------------
        # Reverse discordance (TW-only geometry)
        # -----------------------------------------
        if not is_wetherill:
            is_rev_geom = _is_reverse_discordant(spot.uPbValue, spot.pbPbValue)
        else:
            is_rev_geom = calculations.is_reverse_discordant_wetherill(
                spot.pb207U235Value,
                spot.pb206U238Value,
            )

        # Only mark reverse if it's discordant AND reverse geometry (same as before)
        spot.reverseDiscordant = bool(is_rev_geom and not concordant)

        discordances.append(discordance)
        concordancy.append(concordant)

    reverse_flags = [bool(s.reverseDiscordant) for s in sample.validSpots]
    sample.updateConcordance(concordancy, discordances, reverse_flags)

    signals.progress(ProgressType.CONCORDANCE, 1.0, sample.name, concordancy, discordances, reverse_flags)
    return True, None

# ======================  MC Sampling  ======================

def _performSingleRun(settings, run):
    for age in settings.rimAges():
        run.samplePbLossAge(age, settings.dissimilarityTest, settings.penaliseInvalidAges)
    run.calculateOptimalAge()
    run.createHeatmapData(settings.minimumRimAge, settings.maximumRimAge, config.HEATMAP_RESOLUTION)

def _performRimAgeSampling(signals, sample):
    sample.monteCarloRuns = []
    sample.peak_catalogue = []
    sampleNameText = f" for '{sample.name}'" if sample.name else ""
    signals.newTask("Sampling Pb-loss age distributions" + sampleNameText + "...")

    settings = sample.calculationSettings
    setattr(settings, "timing_mode", TIMING_MODE)
    setattr(settings, "write_outputs", CDC_WRITE_OUTPUTS)

    if bool(getattr(settings, "split_by_concordant_population", False)):
        return _performRimAgeSampling_split_by_population(signals, sample)

    eligibleSpots   = [s for s in sample.validSpots if not getattr(s, "reverseDiscordant", False)]
    concordantSpots = [s for s in eligibleSpots if s.concordant]
    discordantSpots = [s for s in eligibleSpots if not s.concordant]

    sample._ks_concordantSpots = list(concordantSpots)
    sample._ks_discordantSpots = list(discordantSpots)

    if not concordantSpots:
        return False, "no concordant spots"
    if not discordantSpots:
        return False, "no discordant spots"
    if len(discordantSpots) <= 2:
        return False, "fewer than 3 discordant spots"

    stabilitySamples = int(settings.monteCarloRuns)
    rng = np.random.default_rng(_seed_from_name(sample.name))

    mode = ConcordiaMode.coerce(getattr(settings, "concordiaMode", ConcordiaMode.TW))
    is_wetherill = (mode == ConcordiaMode.WETHERILL)

    concordantUPbValues, concordantPbPbValues = _draw_in_mode(rng, concordantSpots, stabilitySamples, is_wetherill)
    discordantUPbValues,  discordantPbPbValues  = _draw_in_mode(rng, discordantSpots,  stabilitySamples, is_wetherill)

    use_dc = bool(getattr(settings, "use_discordant_clustering", False))
    relabel_per_run = use_dc and bool(getattr(settings, "relabel_clusters_per_run", False))

    # ---- optional clustering of discordant ages ----
    full_labels = np.zeros(len(discordantSpots), dtype=int)

    if use_dc:
        ages_y = np.asarray(settings.rimAges(), float)  # YEARS grid

        proxy_ma, keep_idx = [], []
        for idx, spot in enumerate(discordantSpots):
            x, y = _spot_xy_mode(spot, is_wetherill)
            proxy_y = (
                lower_intercept_proxy_wetherill(x, y, ages_y)
                if is_wetherill else
                lower_intercept_proxy(x, y, ages_y)
            )

            if proxy_y is not None and np.isfinite(proxy_y):
                proxy_ma.append(proxy_y / 1e6)
                keep_idx.append(idx)

        proxy_ma = np.asarray(proxy_ma, float)
        keep_idx = np.asarray(keep_idx, int)

        if proxy_ma.size >= 3:
            core_labels, *_ = find_discordant_clusters(proxy_ma)
            min_pts, min_frac, sep_sig = _adaptive_gates(proxy_ma.size)

            if proxy_ma.size >= min_pts:
                soft = _soft_accept_labels(
                    core_labels, proxy_ma,
                    min_points=min_pts,
                    min_frac=min_frac,
                    sep_sig_thr=sep_sig,
                )
                if soft.max() >= 1:
                    full_labels[keep_idx] = soft

    for spot, lab in zip(discordantSpots, full_labels):
        spot.cluster_id = int(lab)

    # --------- sampling loop ---------
    per_run_times = []
    t0 = time.perf_counter()

    ages_y = np.asarray(settings.rimAges(), float)  # ensure defined if relabel_per_run

    for j in range(stabilitySamples):
        if signals.halt():
            signals.cancelled()
            return False, "processing halted by user"

        t_run = time.perf_counter()

        labels_for_run = full_labels
        if relabel_per_run:
            if is_wetherill:
                labels_for_run = _labels_from_this_run_wetherill(
                    discordantUPbValues[j], discordantPbPbValues[j], ages_y
                )
            else:
                labels_for_run = _labels_from_this_run(
                    discordantUPbValues[j], discordantPbPbValues[j], ages_y
                )

        run = MonteCarloRun(
            j, sample.name,
            concordantUPbValues[j],  concordantPbPbValues[j],
            discordantUPbValues[j],  discordantPbPbValues[j],
            discordant_labels=labels_for_run,
            settings=settings
        )
        _performSingleRun(settings, run)
        per_run_times.append(time.perf_counter() - t_run)

        progress = (j + 1) / stabilitySamples
        sample.addMonteCarloRun(run)
        signals.progress(ProgressType.SAMPLING, progress, sample.name, run)

    mc_elapsed = time.perf_counter() - t0
    grid_len = len(settings.rimAges()) if hasattr(settings, "rimAges") else 0

    _write_runlog(dict(
        method="CDC", phase="MC",
        sample=sample.name, tier=_infer_tier(sample.name),
        R=stabilitySamples, n_grid=grid_len,
        elapsed_s=round(mc_elapsed, 3),
        per_run_median_s=round(float(np.median(per_run_times)), 4) if per_run_times else 0.0,
        per_run_p95_s=round(float(np.percentile(per_run_times, 95)), 4) if per_run_times else 0.0,
        rss_peak_mb=round(_rss_mb(), 1),
        python=platform.python_version(), numpy=np.__version__,
    ))

    _calculateOptimalAge(signals, sample, 1.0)
    return True, None

def _performRimAgeSampling_split_by_population(signals, sample):
    """
    Run CDC+ensemble separately for each concordant age population and
    merge the resulting peaks into a single catalogue.
    """
    settings = sample.calculationSettings
    setattr(settings, "timing_mode", TIMING_MODE)
    setattr(settings, "write_outputs", CDC_WRITE_OUTPUTS)

    eligibleSpots   = [s for s in sample.validSpots if not getattr(s, "reverseDiscordant", False)]
    concordantSpots = [s for s in eligibleSpots if s.concordant]
    discordantSpots = [s for s in eligibleSpots if not s.concordant]

    if not concordantSpots or not discordantSpots or len(discordantSpots) <= 2:
        return False, "insufficient spots for population split"

    # Cluster concordant ages into populations
    pop_labels_conc, n_pops, pop_means = _cluster_concordant_populations(concordantSpots, max_pops=3)
    pop_labels_disc = _assign_discordant_to_populations(discordantSpots, pop_means)

    all_runs = []
    all_peaks = []

    for pop_id in range(n_pops):
        sub_conc = [s for s, lab in zip(concordantSpots, pop_labels_conc) if lab == pop_id]
        sub_disc = [s for s, lab in zip(discordantSpots, pop_labels_disc) if lab == pop_id]
        if len(sub_conc) == 0 or len(sub_disc) < 3:
            continue

        # --- run the existing MC+ensemble logic on this subset ONLY ---
        ok, reason, runs_pop, peaks_pop = _run_cdc_on_subset(
            signals, f"{sample.name}_pop{pop_id}", settings, sub_conc, sub_disc
        )

        # tag peaks with population id
        for p in peaks_pop:
            p["population_id"] = pop_id
        all_runs.extend(runs_pop)
        all_peaks.extend(peaks_pop)

    # aggregate results back into the sample
    sample.monteCarloRuns = all_runs

    # Emit SAMPLING progress events for all runs so the UI sees them
    total_runs = len(all_runs)
    if total_runs:
        for j, run in enumerate(all_runs):
            progress = float(j + 1) / float(total_runs)
            signals.progress(ProgressType.SAMPLING, progress, sample.name, run)

    # Use the usual pipeline to build Goodness surfaces etc.
    _calculateOptimalAge(signals, sample, 1.0)

    # Overwrite catalogue with population-aware peaks (all_peaks)
    if all_peaks:
        sample.peak_catalogue = [
            dict(
                sample=sample.name,
                peak_no=i + 1,
                ci_low=p["ci_low"],
                age_ma=p["age_ma"],
                ci_high=p["ci_high"],
                support=p.get("support", float("nan")),
                population_id=p.get("population_id", None),
            )
            for i, p in enumerate(all_peaks)
        ]
        sample.summedKS_peaks_Ma = np.asarray([p["age_ma"] for p in all_peaks], float)
        sample.peak_uncertainty_str = fmt_peak_stats(
            [
                (p["age_ma"], p["ci_low"], p["ci_high"], p.get("support", float("nan")))
                for p in all_peaks
            ]
        )

    return True, None

def _run_cdc_on_subset(signals, sample_name: str, settings, concordantSpots, discordantSpots):
    if not concordantSpots:
        return False, "no concordant spots in subset", [], []
    if len(discordantSpots) <= 2:
        return False, "fewer than 3 discordant spots in subset", [], []

    stabilitySamples = int(settings.monteCarloRuns)
    rng = np.random.default_rng(_seed_from_name(sample_name + "_pop"))

    mode = ConcordiaMode.coerce(getattr(settings, "concordiaMode", ConcordiaMode.TW))
    is_wetherill = (mode == ConcordiaMode.WETHERILL)

    concU, concPb = _draw_in_mode(rng, concordantSpots, stabilitySamples, is_wetherill)
    discU, discPb = _draw_in_mode(rng, discordantSpots,  stabilitySamples, is_wetherill)

    use_dc = bool(getattr(settings, "use_discordant_clustering", False))
    relabel_per_run = use_dc and bool(getattr(settings, "relabel_clusters_per_run", False))

    ages_y = np.asarray(settings.rimAges(), float)  # YEARS grid

    full_labels = np.zeros(len(discordantSpots), dtype=int)

    if use_dc:
        proxy_ma, keep_idx = [], []
        for idx, spot in enumerate(discordantSpots):
            x, y = _spot_xy_mode(spot, is_wetherill)
            proxy_y = (
                lower_intercept_proxy_wetherill(x, y, ages_y)
                if is_wetherill else
                lower_intercept_proxy(x, y, ages_y)
            )
            if proxy_y is not None and np.isfinite(proxy_y):
                proxy_ma.append(proxy_y / 1e6)
                keep_idx.append(idx)

        proxy_ma = np.asarray(proxy_ma, float)
        keep_idx = np.asarray(keep_idx, int)

        if proxy_ma.size >= 3:
            core_labels, *_ = find_discordant_clusters(proxy_ma)
            min_pts, min_frac, sep_sig = _adaptive_gates(proxy_ma.size)
            if proxy_ma.size >= min_pts:
                soft = _soft_accept_labels(
                    core_labels, proxy_ma,
                    min_points=min_pts,
                    min_frac=min_frac,
                    sep_sig_thr=sep_sig,
                )
                if soft.max() >= 1:
                    full_labels[keep_idx] = soft

    # IMPORTANT: do this OUTSIDE the if-block
    for spot, lab in zip(discordantSpots, full_labels):
        spot.cluster_id = int(lab)

    runs_pop: List[MonteCarloRun] = []

    for j in range(stabilitySamples):
        if signals.halt():
            return False, "processing halted by user", [], []

        labels_for_run = full_labels
        if relabel_per_run:
            if is_wetherill:
                labels_for_run = _labels_from_this_run_wetherill(discU[j], discPb[j], ages_y)
            else:
                labels_for_run = _labels_from_this_run(discU[j], discPb[j], ages_y)

        run = MonteCarloRun(
            j, sample_name,
            concU[j], concPb[j],
            discU[j], discPb[j],
            discordant_labels=labels_for_run,
            settings=settings
        )
        _performSingleRun(settings, run)
        runs_pop.append(run)

    if not runs_pop:
        return False, "no runs produced", [], []

    ages_ma = ages_y / 1e6
    S_runs_pen = _stack_min_across_clusters(runs_pop, ages_y, which="pen")
    smf = _smooth_frac_for_grid(ages_ma)
    Smed_pen, _, _ = robust_ensemble_curve(S_runs_pen, smooth_frac=smf)

    optima_ma_pop = np.array([r.optimal_pb_loss_age for r in runs_pop], float) / 1e6

    peaks_pop = build_ensemble_catalogue(
        sample_name, _infer_tier(sample_name), ages_ma, S_runs_pen,
        orientation='max', smooth_frac=smf,
        f_d=FD_DIST_FRAC, f_p=FP_PROM_FRAC, f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
        w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
        per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
        per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
        pen_ok_mask=None, cand_curve=Smed_pen,
        optima_ma=optima_ma_pop,
    ) or []

    if not peaks_pop:
        opt_all = np.sort([r.optimal_pb_loss_age for r in runs_pop])
        if opt_all.size:
            optimalAge = float(np.median(opt_all))
            lower95    = float(opt_all[int(0.025 * len(opt_all))])
            upper95    = float(opt_all[int(np.ceil(0.975 * len(opt_all)) - 1)])
            peaks_pop = [dict(
                age_ma=optimalAge / 1e6,
                ci_low=lower95   / 1e6,
                ci_high=upper95  / 1e6,
                support=1.0,
                source="fallback_optimal",
            )]

    return True, None, runs_pop, peaks_pop

# ======================  Ensemble, KS, catalogue ======================

def _findOptimalIndex(valuesToCompare):
    minIndex, minValue = min(enumerate(valuesToCompare), key=lambda v: v[1])
    n = len(valuesToCompare)

    startMinIndex = minIndex
    while startMinIndex > 0 and valuesToCompare[startMinIndex - 1] == minValue:
        startMinIndex -= 1

    endMinIndex = minIndex
    while endMinIndex < n - 1 and valuesToCompare[endMinIndex + 1] == minValue:
        endMinIndex += 1

    if (endMinIndex != n - 1 and startMinIndex != 0) or (endMinIndex == n - 1 and startMinIndex == 0):
        return (endMinIndex + startMinIndex) // 2
    if startMinIndex == 0:
        return 0
    return n - 1

def _emit_summedKS(signals, sample, progress, ages_ma, y_curve, rows_for_ui):
    """
    Send the plotted curve together with peak positions and CI arrays, to BOTH:
      • sample.signals.summedKS (figure listens to this)
      • global signals.progress("summedKS", ...) (legacy bus)
    """
    ui_peaks_age = [float(r["age_ma"]) for r in rows_for_ui]
    ui_peaks_ci  = [[float(r["ci_low"]), float(r["ci_high"])] for r in rows_for_ui]
    ui_support   = [float(r.get("support", float("nan"))) for r in rows_for_ui]

    sample.summedKS_peaks_Ma   = np.asarray(ui_peaks_age, float)
    sample.summedKS_ci_low_Ma  = np.asarray([lo for lo, _ in ui_peaks_ci], float)
    sample.summedKS_ci_high_Ma = np.asarray([hi for _, hi in ui_peaks_ci], float)

    payload = (ages_ma.tolist(), y_curve.tolist(), ui_peaks_age, ui_peaks_ci, ui_support)
    try:
        if hasattr(sample.signals, "summedKS"):
            sample.signals.summedKS.emit(payload)
    except Exception:
        pass
    try:
        signals.progress("summedKS", progress, sample.name, payload)
    except TypeError:
        try:
            signals.progress("summedKS", progress, sample.name, (payload[0], payload[1], payload[2]))
        except Exception:
            pass

def _smooth_frac_for_grid(ages_ma):
    """Convert SMOOTH_MA (if >0) into a node fraction; else use SMOOTH_FRAC."""
    n = len(ages_ma)
    if n <= 1:
        return SMOOTH_FRAC
    if SMOOTH_MA > 0:
        step_ma = float(np.median(np.diff(ages_ma))) or 1e-9
        sigma_nodes = SMOOTH_MA / step_ma
        # robust_ensemble_curve expects sigma as a fraction of N
        return min(0.25, sigma_nodes / n)  # cap to avoid over-smoothing
    return SMOOTH_FRAC

def _collapse_ci_clusters(rows, width_mult: float = 1.0):
    """
    Collapse chains of peaks that are either:
      • CI-overlapping, OR
      • close in age compared to their widths.

    For each cluster, keep only the best-supported, narrowest peak
    and keep *its own* CI (no union widening).
    """
    if not rows or len(rows) <= 1:
        return rows

    # sort by age so we can scan left→right
    rows = sorted(rows, key=lambda r: float(r["age_ma"]))
    clusters = []
    current_cluster = [dict(rows[0])]

    def _same_cluster(a, b) -> bool:
        lo1, hi1 = float(a["ci_low"]),  float(a["ci_high"])
        lo2, hi2 = float(b["ci_low"]),  float(b["ci_high"])
        a1, a2   = float(a["age_ma"]),  float(b["age_ma"])

        # CI overlap?
        overlap = (lo2 <= hi1) and (hi2 >= lo1)

        # Age separation vs widths
        w1 = max(hi1 - lo1, 0.0)
        w2 = max(hi2 - lo2, 0.0)
        sep = abs(a2 - a1)

        # Treat as the same cluster if they overlap OR
        # separation is smaller than some multiple of the larger width.
        close = (w1 > 0.0 or w2 > 0.0) and sep <= width_mult * max(w1, w2)

        return overlap or close

    for r in rows[1:]:
        last = current_cluster[-1]
        if _same_cluster(last, r):
            current_cluster.append(dict(r))
        else:
            clusters.append(current_cluster)
            current_cluster = [dict(r)]
    clusters.append(current_cluster)

    collapsed = []
    for cl in clusters:
        # choose best-supported peak in the cluster; break ties with narrowest CI
        def _score(rr):
            sup   = float(rr.get("support", 0.0))
            width = float(rr["ci_high"]) - float(rr["ci_low"])
            return (sup, -width)  # higher support, then narrower

        best = max(cl, key=_score)
        collapsed.append(dict(best))  # copy

    # renumber
    for i, rr in enumerate(collapsed, 1):
        rr["peak_no"] = i

    return collapsed

def _calculateOptimalAge(signals, sample, progress):
    """
      • UI shows median-of-run-optima with consistent CI.
      • Exports and figures use legacy surface optimum and curve.
    """
    settings, runs = sample.calculationSettings, sample.monteCarloRuns
    if not runs:
        return

    # Grid
    ages_y = np.asarray(settings.rimAges(), float)   # years
    ages_ma = ages_y / 1e6
    optima_ma = np.array([r.optimal_pb_loss_age for r in runs], float) / 1e6

    # Per-run goodness matrices
    S_runs_raw = _stack_min_across_clusters(runs, ages_y, which="raw")  # 1 - KS D
    S_runs_pen = _stack_min_across_clusters(runs, ages_y, which="pen")  # 1 - score

    # Smoothing + ensemble curves (only relevant if ensemble mode ON)
    smf = _smooth_frac_for_grid(ages_ma)
    Smed_raw, Delta_raw, _ = robust_ensemble_curve(S_runs_raw, smooth_frac=smf)
    Smed_pen, Delta_pen, _ = robust_ensemble_curve(S_runs_pen, smooth_frac=smf)
    S_view = Smed_raw if (CATALOGUE_SURFACE == "RAW") else Smed_pen

    # ---------- (A) Run-optima median & CI (for UI) ----------
    opt_all = np.sort(np.asarray([r.optimal_pb_loss_age for r in runs], float))
    n = opt_all.size
    if n:
        optimalAge_ui = float(np.median(opt_all))  # years
        lower95 = float(opt_all[int(np.floor(0.025 * n))])
        upper95 = float(opt_all[int(np.ceil(0.975 * n)) - 1])
    else:
        optimalAge_ui = lower95 = upper95 = float("nan")
    optimalAge = optimalAge_ui 
    # ---------- (B) Legacy surface optimum (for export/figure) ----------
    mean_score = np.array(
        [np.mean([r.statistics_by_pb_loss_age[a].score for r in runs]) for a in ages_y],
        dtype=float,
    )
    mean_score = np.where(np.isfinite(mean_score), mean_score, np.inf)
    legacy_idx = _findOptimalIndex(mean_score.tolist())
    optimalAge_legacy = float(ages_y[legacy_idx])  # years (grid node)
    S_legacy_curve = 1.0 - mean_score  # legacy S(t)
    sample.legacy_surface_optimal_age = optimalAge_legacy

    # ---------- (C) Mean stats at each run’s own optimum ----------
    stats = [getattr(r, "optimal_statistic", None) for r in runs]
    stats = [s for s in stats if s is not None]
    if stats:
        meanD = float(np.mean([s.test_statistics[0] for s in stats]))
        meanP = float(np.mean([s.test_statistics[1] for s in stats]))
        meanInv = float(np.mean([s.number_of_invalid_ages for s in stats]))
        meanSc = float(np.mean([s.score for s in stats]))
    else:
        meanD = meanP = meanInv = meanSc = float("nan")

    # ---------- (D) Ensemble OFF branch ----------
    enabled = bool(getattr(settings, "enable_ensemble_peak_picking", False))
    if not enabled:
        # UI point estimate = median of run optima
        optimalAge = optimalAge_ui
        sample.legacy_surface_optimal_age = optimalAge_legacy

        sample.peak_catalogue = []
        try:
            sample.signals.optimalAgeCalculated.emit()
        except Exception:
            pass

        # Plot legacy curve for diagnostics (optional)
        _emit_summedKS(signals, sample, progress, ages_ma, S_legacy_curve, rows_for_ui=[])

        # Export legacy surface optimum + curve (for paper figure)
        if KS_EXPORT_ROOT is not None:
            _export_legacy_ks(
                sample, settings, runs, ages_y,
                D_pen=mean_score,  # keep legacy curve export if you want
                ui_opt_years=optimalAge_ui,
                ui_low95_years=lower95,
                ui_high95_years=upper95,
                run_optima_years=opt_all,
                legacy_opt_years=optimalAge_legacy,
            )


        payload = (optimalAge, lower95, upper95, meanD, meanP, meanInv, meanSc, "—", [])
        try:
            signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, payload)
        except TypeError:
            signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, payload[:7])
        return
    # ------------------------------------------------------------------
    # Build catalogues strictly from the ensemble (no winners override).
    # If an ensemble surface is too flat (Δ < ENS_DELTA_MIN), we do not
    # attempt to pick peaks from it and rely on the optima fallback instead.
    # ------------------------------------------------------------------
    use_dc = bool(getattr(settings, "use_discordant_clustering", False))

    rows_raw: List[Dict] = []
    rows_pen: List[Dict] = []

    # Per-cluster ensembles (only if clustering enabled)
    if use_dc and USE_CLUSTER_CATALOGUE:
        S_by_cluster_raw = stack_goodness_by_cluster(runs, ages_y, which="raw")
        S_by_cluster_pen = stack_goodness_by_cluster(runs, ages_y, which="pen")

        for cid in sorted(S_by_cluster_raw.keys()):
            S_raw_k = np.asarray(S_by_cluster_raw[cid], float)
            S_pen_k = np.asarray(S_by_cluster_pen.get(cid, S_raw_k * np.nan), float)

            # Keep only runs that actually have this cluster (any finite entries)
            mask_runs = np.isfinite(S_raw_k).any(axis=1)
            if mask_runs.sum() < max(RMIN_RUNS, 2):
                continue

            S_raw_k = S_raw_k[mask_runs]
            S_pen_k = S_pen_k[mask_runs]
            optima_ma_k = optima_ma[mask_runs] 

            # Ensemble for this cluster
            Smed_raw_k, Delta_raw_k, _ = robust_ensemble_curve(S_raw_k, smooth_frac=smf)
            Smed_pen_k, Delta_pen_k, _ = robust_ensemble_curve(S_pen_k, smooth_frac=smf)

            # If both surfaces for this cluster are essentially flat or empty, skip it
            if (Smed_raw_k.size == 0 and Smed_pen_k.size == 0) or \
               (Delta_raw_k < ENS_DELTA_MIN and Delta_pen_k < ENS_DELTA_MIN):
                continue

            # RAW catalogue for this cluster (only if not too flat)
            if Smed_raw_k.size and Delta_raw_k >= ENS_DELTA_MIN:
                rows_raw_k = build_ensemble_catalogue(
                    sample.name, _infer_tier(sample.name), ages_ma, S_raw_k,
                    orientation="max", smooth_frac=smf,
                    f_d=FD_DIST_FRAC, f_p=FP_PROM_FRAC, f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
                    w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
                    per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
                    per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
                    pen_ok_mask=None, cand_curve=Smed_raw_k, height_frac=FH_HEIGHT_FRAC, optima_ma=optima_ma_k,
                ) or []
                for r in rows_raw_k:
                    r["cluster_id"] = int(cid)
                rows_raw.extend(rows_raw_k)

            # PEN catalogue for this cluster (only if not too flat)
            if Smed_pen_k.size and Delta_pen_k >= ENS_DELTA_MIN:
                rows_pen_k = build_ensemble_catalogue(
                    sample.name, _infer_tier(sample.name), ages_ma, S_pen_k,
                    orientation="max", smooth_frac=smf,
                    f_d=FD_DIST_FRAC, f_p=FP_PROM_FRAC, f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
                    w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
                    per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
                    per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
                    pen_ok_mask=None, cand_curve=Smed_pen_k, height_frac=FH_HEIGHT_FRAC, optima_ma=optima_ma_k,
                ) or []
                for r in rows_pen_k:
                    r["cluster_id"] = int(cid)
                rows_pen.extend(rows_pen_k)

    # Global RAW ensemble (only if not flat and nothing robust from clusters)
    if not rows_raw and Delta_raw >= ENS_DELTA_MIN:
        rows_raw = build_ensemble_catalogue(
            sample.name, _infer_tier(sample.name), ages_ma, S_runs_raw,
            orientation="max", smooth_frac=smf,
            f_d=FD_DIST_FRAC, f_p=FP_PROM_FRAC, f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
            w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
            per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
            per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
            pen_ok_mask=None, cand_curve=Smed_raw, height_frac=FH_HEIGHT_FRAC, optima_ma=optima_ma,
        ) or []

        if not rows_raw and Delta_raw >= ENS_DELTA_MIN:
            rows_raw = build_ensemble_catalogue(
                sample.name, _infer_tier(sample.name), ages_ma, S_runs_raw,
                orientation="max", smooth_frac=smf,
                f_d=FD_DIST_FRAC, f_p=max(0.5 * FP_PROM_FRAC, 0.01),
                f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
                w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
                per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
                per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
                pen_ok_mask=None, cand_curve=Smed_raw, height_frac=FH_HEIGHT_FRAC, optima_ma=optima_ma,
            ) or []

    # Global PEN ensemble (only if not flat)
    if not rows_pen and Delta_pen >= ENS_DELTA_MIN:
        rows_pen = build_ensemble_catalogue(
            sample.name, _infer_tier(sample.name), ages_ma, S_runs_pen,
            orientation="max", smooth_frac=smf,
            f_d=FD_DIST_FRAC, f_p=FP_PROM_FRAC, f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
            w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
            per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
            per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
            pen_ok_mask=None, cand_curve=Smed_pen, height_frac=FH_HEIGHT_FRAC, optima_ma=optima_ma,
        ) or []

    # Fallback:
    # Only promote the median of the per-run optima to a pseudo-peak if the
    # ensemble surface has some structure (Δ >= ENS_DELTA_MIN). If both RAW
    # and PEN surfaces are essentially flat, leave the catalogue empty.
    if (not rows_raw) and (not rows_pen) and len(opt_all) > 0:

        # Fallback: if no ensemble peaks survive, **do not** invent one.
        # Leave the catalogue empty – the single CDC optimal age is still
        # reported in the summary, but there is no robust ensemble peak.        
        rows_raw = []
        rows_pen = []

    # Choose which to DISPLAY in the UI strictly from the ensemble surface.
    ui_surface = CATALOGUE_SURFACE  # "RAW" or "PEN"
    if (ui_surface == "RAW") and rows_raw:
        rows_for_ui = rows_raw
        S_view = Smed_raw
    elif rows_pen:
        rows_for_ui = rows_pen
        S_view = Smed_pen
    else:
        rows_for_ui = []
        S_view = Smed_pen if ui_surface != "RAW" else Smed_raw

    _ensure_output_dirs()

    rows_for_ui = _collapse_ci_clusters(rows_for_ui)
    rows_raw    = _collapse_ci_clusters(rows_raw)
    rows_pen    = _collapse_ci_clusters(rows_pen)

    # Ensure snapped age lies inside its CI (preserve width >= 1 step)
    if rows_for_ui:
        step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 5.0
        min_age, max_age = float(ages_ma[0]), float(ages_ma[-1])
        fixed = []
        for r in rows_for_ui:
            a  = float(r["age_ma"])
            lo = float(r["ci_low"])
            hi = float(r["ci_high"])
            w  = max(hi - lo, step)

            if (a < lo) or (a > hi):
                lo, hi = a - 0.5 * w, a + 0.5 * w
                lo, hi = max(lo, min_age), min(hi, max_age)
                if (hi - lo) < step:
                    lo, hi = max(a - step, min_age), min(a + step, max_age)

            fixed.append(dict(r, ci_low=lo, ci_high=hi))
        rows_for_ui = fixed

    # Enforce a minimum CI width and drop boundary‑degenerate peaks
    if rows_for_ui:
        step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 5.0
        min_age, max_age = float(ages_ma[0]), float(ages_ma[-1])
        cleaned = []
        for r in rows_for_ui:
            a  = float(r["age_ma"])
            lo = float(r["ci_low"])
            hi = float(r["ci_high"])
            if (hi - lo) < step:
                lo, hi = a - step, a + step
            near_edge  = (a - min_age) <= step or (max_age - a) <= step
            degenerate = (hi - lo) <= 0.75 * step
            if near_edge and degenerate:
                if float(r.get("support", 0.0)) >= max(FS_SUPPORT, 0.12):
                    lo, hi = a - step, a + step
                else:
                    continue
            cleaned.append(dict(r, ci_low=lo, ci_high=hi))
        rows_for_ui = cleaned

    for r in rows_for_ui:
        a = float(r["age_ma"])
        if not (float(r["ci_low"]) <= a <= float(r["ci_high"])):
            r["ci_low"]  = min(float(r["ci_low"]),  a)
            r["ci_high"] = max(float(r["ci_high"]), a)

    # ---- Drop peaks with absurdly wide CIs (essentially the entire grid) ----
    if rows_for_ui:
        total_span = float(ages_ma[-1] - ages_ma[0])
        MAX_CI_FRAC = 0.5  # drop peaks whose CI spans >50% of the modelling window

        filtered = []
        for r in rows_for_ui:
            width = float(r["ci_high"] - r["ci_low"])
            if width > MAX_CI_FRAC * total_span:
                continue
            filtered.append(r)
        rows_for_ui = filtered

        # Keep rows_raw/rows_pen in sync with what we actually show
        def _keep_same(rows, keep):
            if not rows or not keep:
                return []
            keep_ages = {float(r["age_ma"]) for r in keep}
            return [r for r in rows if float(r.get("age_ma", float("nan"))) in keep_ages]

        rows_raw = _keep_same(rows_raw, rows_for_ui)
        rows_pen = _keep_same(rows_pen, rows_for_ui)

    # Renumber peak_no after filtering so CSV doesn't have gaps
    for i, r in enumerate(rows_for_ui, 1):
        r["peak_no"] = i
    for i, r in enumerate(rows_raw, 1):
        r["peak_no"] = i
    for i, r in enumerate(rows_pen, 1):
        r["peak_no"] = i

    # Export rows_for_ui/rows_raw/rows_pen are final
    if CDC_WRITE_OUTPUTS:
        _append_catalogue_rows(sample.name, rows_pen, dest_path=CATALOGUE_CSV_PEN)
        _append_catalogue_rows(sample.name, rows_raw, dest_path=CATALOGUE_CSV_RAW)
        _write_npz_diagnostics(
            sample_name=sample.name,
            ages_ma=ages_ma,
            ages_y=ages_y,
            runs=runs,
            S_runs_raw=S_runs_raw,
            S_runs_pen=S_runs_pen,
            Smed_raw=Smed_raw,
            Smed_pen=Smed_pen,
            S_view=S_view,
            rows_for_ui=rows_for_ui,
        )

    # Publish to UI
    catalogue = [(r["age_ma"], r["ci_low"], r["ci_high"], r["support"]) for r in rows_for_ui]

    red_peaks = np.asarray([m for m, *_ in catalogue], float)
    peak_str  = fmt_peak_stats(catalogue) if catalogue else "—"

    sample.summedKS_peaks_Ma = red_peaks
    sample.peak_uncertainty_str = peak_str
    sample.peak_catalogue = [
        dict(sample=sample.name, peak_no=i + 1, ci_low=lo, age_ma=med, ci_high=hi, support=sup)
        for i, (med, lo, hi, sup) in enumerate(catalogue)
    ]

    try:
        sample.signals.optimalAgeCalculated.emit()
    except Exception:
        pass

    _emit_summedKS(signals, sample, progress, ages_ma, S_view, rows_for_ui)

    payload = (optimalAge, lower95, upper95, meanD, meanP, meanInv, meanSc, peak_str, catalogue)
    try:
        signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, payload)
    except TypeError:
        signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, payload[:7])

