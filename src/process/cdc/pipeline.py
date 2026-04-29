"""Core CDC pipeline orchestration."""

from __future__ import annotations

import platform
import time

import numpy as np

from model.monteCarloRun import MonteCarloRun
from model.settings.calculation import DiscordanceClassificationMethod
from process import calculations
from process.cdc.state import ProgressType
from process.cdc.surfaces import (
    _build_global_catalogue_rows,
    _build_surface_states,
    _compute_mean_stats,
    _compute_optimal_age,
    _compute_optimal_age_ci,
    _initialise_surface_view_state,
)
from process.cdc.filtering import (
    _dedupe_rejected_rows,
    _run_filter_pipeline,
)
from process.cdc.guards import _apply_guards_and_fallbacks
from process.cdc.publish import (
    _publish_legacy_only,
    _publish_results,
    reset_output_exports,
)
from process.cdcConfig import (
    CDC_WRITE_OUTPUTS,
    MERGE_NEARBY_PEAKS,
    FS_SUPPORT,
    TIMING_MODE,
)
from process.cdcDiagnostics import (
    ensure_output_dirs as _ensure_output_dirs,
    rss_mb as _rss_mb,
    write_runlog as _write_runlog,
)
from process.cdcTW import is_reverse_discordant as _is_reverse_discordant
from process.cdcUtils import infer_tier as _infer_tier, seed_from_name as _seed_from_name
from utils import config

TIME_PER_TASK = 0.0


def processSamples(signals, samples):
    if CDC_WRITE_OUTPUTS:
        reset_output_exports()

    for sample in samples:
        completed, skip_reason = _processSample(signals, sample)
        if not completed and skip_reason:
            signals.skipped(sample.name, skip_reason)

    signals.completed()


def _processSample(signals, sample):
    t0 = time.perf_counter()

    try:
        completed, skip_reason = _calculateConcordantAges(signals, sample)
        if not completed:
            return False, skip_reason

        completed, skip_reason = _performRimAgeSampling(signals, sample)
        if not completed:
            return False, skip_reason

        return True, None

    finally:
        n_grid = len(sample.calculationSettings.rimAges())
        R_runs = sample.calculationSettings.monteCarloRuns
        _write_runlog(
            dict(
                method="CDC",
                phase="e2e_runtime",
                sample=sample.name,
                tier=_infer_tier(sample.name),
                R=R_runs,
                n_grid=n_grid,
                elapsed_s=round(time.perf_counter() - t0, 3),
                per_run_median_s="",
                per_run_p95_s="",
                rss_peak_mb=round(_rss_mb(), 1),
                python=platform.python_version(),
                numpy=np.__version__,
            )
        )


def _calculateConcordantAges(signals, sample):
    """Classify each valid spot as concordant/discordant and flag reverse discordance."""
    sampleNameText = f" for '{sample.name}'" if sample.name else ""
    signals.newTask("Classifying points" + sampleNameText + "...")

    settings = sample.calculationSettings
    n_spots = max(1, len(sample.validSpots))
    timePerRow = TIME_PER_TASK / n_spots

    concordancy = []
    discordances = []

    for i, spot in enumerate(sample.validSpots):
        signals.progress(ProgressType.CONCORDANCE, i / n_spots)
        time.sleep(timePerRow)
        if signals.halt():
            signals.cancelled()
            return False, "processing halted by user"

        if settings.discordanceClassificationMethod == DiscordanceClassificationMethod.PERCENTAGE:
            discordance = calculations.discordance(spot.uPbValue, spot.pbPbValue)
            concordant = abs(discordance) < settings.discordancePercentageCutoff
        else:
            discordance = None
            concordant = calculations.isConcordantErrorEllipse(
                spot.uPbValue,
                spot.uPbStDev,
                spot.pbPbValue,
                spot.pbPbStDev,
                settings.discordanceEllipseSigmas,
            )

        is_rev_geom = _is_reverse_discordant(spot.uPbValue, spot.pbPbValue)
        spot.reverseDiscordant = bool(is_rev_geom and not concordant)

        discordances.append(discordance)
        concordancy.append(concordant)

    reverse_flags = [bool(s.reverseDiscordant) for s in sample.validSpots]
    sample.updateConcordance(concordancy, discordances, reverse_flags)
    signals.progress(ProgressType.CONCORDANCE, 1.0, sample.name, concordancy, discordances, reverse_flags)
    return True, None


def _performSingleRun(settings, run):
    for age in settings.rimAges():
        run.samplePbLossAge(age, settings.dissimilarityTest, settings.penaliseInvalidAges)
    run.calculateOptimalAge()
    run.createHeatmapData(settings.minimumRimAge, settings.maximumRimAge, config.HEATMAP_RESOLUTION)


def _performRimAgeSampling(signals, sample):
    """Run Monte Carlo sampling of Pb-loss ages for a single sample."""
    sample.monteCarloRuns = []
    sample.peak_catalogue = []
    sample.rejected_peak_candidates = []
    sampleNameText = f" for '{sample.name}'" if sample.name else ""
    signals.newTask("Sampling Pb-loss age distributions" + sampleNameText + "...")

    settings = sample.calculationSettings
    setattr(settings, "timing_mode", TIMING_MODE)
    setattr(settings, "write_outputs", CDC_WRITE_OUTPUTS)

    eligibleSpots = [s for s in sample.validSpots if not getattr(s, "reverseDiscordant", False)]
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

    concordantUPbValues = np.stack([rng.normal(s.uPbValue, s.uPbStDev, stabilitySamples) for s in concordantSpots], axis=1)
    concordantPbPbValues = np.stack([rng.normal(s.pbPbValue, s.pbPbStDev, stabilitySamples) for s in concordantSpots], axis=1)
    discordantUPbValues = np.stack([rng.normal(s.uPbValue, s.uPbStDev, stabilitySamples) for s in discordantSpots], axis=1)
    discordantPbPbValues = np.stack([rng.normal(s.pbPbValue, s.pbPbStDev, stabilitySamples) for s in discordantSpots], axis=1)

    per_run_times = []
    t0 = time.perf_counter()
    for j in range(stabilitySamples):
        if signals.halt():
            signals.cancelled()
            return False, "processing halted by user"

        t_run = time.perf_counter()
        run = MonteCarloRun(
            j,
            sample.name,
            concordantUPbValues[j],
            concordantPbPbValues[j],
            discordantUPbValues[j],
            discordantPbPbValues[j],
            settings=settings,
        )
        _performSingleRun(settings, run)
        per_run_times.append(time.perf_counter() - t_run)

        progress = (j + 1) / stabilitySamples
        sample.addMonteCarloRun(run)
        signals.progress(ProgressType.SAMPLING, progress, sample.name, run)

    mc_elapsed = time.perf_counter() - t0
    grid_len = len(settings.rimAges())
    _write_runlog(
        dict(
            method="CDC",
            phase="MC",
            sample=sample.name,
            tier=_infer_tier(sample.name),
            R=stabilitySamples,
            n_grid=grid_len,
            elapsed_s=round(mc_elapsed, 3),
            per_run_median_s=round(float(np.median(per_run_times)), 4) if per_run_times else 0.0,
            per_run_p95_s=round(float(np.percentile(per_run_times, 95)), 4) if per_run_times else 0.0,
            rss_peak_mb=round(_rss_mb(), 1),
            python=platform.python_version(),
            numpy=np.__version__,
        )
    )

    _calculateOptimalAge(signals, sample, 1.0)
    return True, None


def _calculateOptimalAge(signals, sample, progress):
    """Build the legacy single-age summary and the ensemble peak catalogue."""
    settings, runs = sample.calculationSettings, sample.monteCarloRuns
    if not runs:
        return
    sample.rejected_peak_candidates = []

    merge_nearby = bool(getattr(settings, "merge_nearby_peaks", MERGE_NEARBY_PEAKS))
    abstain_on_monotonic = bool(getattr(settings, "conservative_abstain_on_monotonic", True))
    support_filter_mode = "DIRECT"
    prefer_pen = bool(getattr(settings, "penaliseInvalidAges", False))
    primary_which = "pen" if prefer_pen else "raw"

    ages_y = np.asarray(settings.rimAges(), float)
    ages_ma = ages_y / 1e6
    smf, raw, pen = _build_surface_states(settings, runs, ages_y, ages_ma, abstain_on_monotonic)
    ui_surface, S_view = _initialise_surface_view_state(sample, settings, raw, pen, primary_which)

    lower95, upper95, opt_all = _compute_optimal_age_ci(raw, pen, prefer_pen, runs)
    optimalAge, S_optimal_curve, mean_primary = _compute_optimal_age(raw, pen, prefer_pen, ages_y)
    sample.legacy_surface_optimal_age = optimalAge
    meanD, meanP, meanInv, meanSc = _compute_mean_stats(runs, primary_which, prefer_pen)

    if not bool(getattr(settings, "enable_ensemble_peak_picking", False)):
        _publish_legacy_only(
            signals,
            sample,
            progress,
            settings,
            runs,
            ages_ma,
            ages_y,
            S_optimal_curve,
            optimalAge,
            lower95,
            upper95,
            opt_all,
            meanD,
            meanP,
            meanInv,
            meanSc,
            mean_primary,
        )
        return

    for surf in (raw, pen):
        surf.rows = _build_global_catalogue_rows(
            sample.name,
            _infer_tier(sample.name),
            ages_ma,
            surf.S_runs,
            surf.Smed,
            smf=smf,
            merge_nearby=merge_nearby,
            pickable=surf.pickable,
            optima_ma=surf.optima_ma,
            diagnostic_rows=surf.rejected,
        )

    if (not raw.rows) and (not pen.rows) and len(opt_all) > 0:
        raw.rows = []
        pen.rows = []
        if (not raw.pickable) and (not pen.pickable):
            sample.ensemble_abstain_reason = "flat_or_monotonic_surface"

    if ui_surface == "RAW" and raw.rows:
        rows_for_ui, S_runs_view, S_view, view_which, rejected_stage_ui = raw.rows, raw.S_runs, raw.Smed, "raw", raw.rejected
    elif pen.rows:
        rows_for_ui, S_runs_view, S_view, view_which, rejected_stage_ui = pen.rows, pen.S_runs, pen.Smed, "pen", pen.rejected
    else:
        chosen_fallback = raw if ui_surface == "RAW" else pen
        rows_for_ui, S_runs_view, S_view = [], chosen_fallback.S_runs, chosen_fallback.Smed
        view_which, rejected_stage_ui = ("raw" if chosen_fallback is raw else "pen"), chosen_fallback.rejected

    for row_list in (raw.rows, pen.rows, rows_for_ui):
        for row in row_list:
            row.setdefault("selection", "strict")

    _ensure_output_dirs()
    rejected_rows = _dedupe_rejected_rows([dict(r) for r in (rejected_stage_ui or [])], ages_ma)
    support_floor = max(float(FS_SUPPORT), 0.03)
    optima_ma_ui_vote = raw.optima_ma if ui_surface == "RAW" else pen.optima_ma
    rows_for_ui, rejected_rows = _run_filter_pipeline(
        raw,
        pen,
        rows_for_ui,
        rejected_rows,
        ages_ma,
        merge_nearby,
        support_floor,
        support_filter_mode,
        optima_ma_ui_vote,
    )

    rows_for_ui, rejected_rows = _apply_guards_and_fallbacks(
        sample,
        settings,
        runs,
        raw,
        pen,
        rows_for_ui,
        rejected_rows,
        ages_ma,
        S_view,
        S_runs_view,
        view_which,
        ui_surface,
        support_floor,
    )

    _publish_results(
        signals,
        sample,
        progress,
        settings,
        runs,
        raw,
        pen,
        rows_for_ui,
        rejected_rows,
        ages_ma,
        ages_y,
        S_view,
        optimalAge,
        lower95,
        upper95,
        opt_all,
        meanD,
        meanP,
        meanInv,
        meanSc,
    )
