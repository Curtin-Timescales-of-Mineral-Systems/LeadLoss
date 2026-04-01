"""CDC processing pipeline.

A refactor of the original `process/processing.py`:
- configuration lives in `process.cdc_config`
- diagnostics/paper exports live in `process.cdc_diagnostics`
- Tera-W maths lives in `process.cdc_tw`

The GUI should continue to call `process.processing.processSamples(...)`.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from model.monteCarloRun import MonteCarloRun
from model.settings.calculation import DiscordanceClassificationMethod
from process import calculations
from process.ensemble import (
    robust_ensemble_curve,
    build_ensemble_catalogue,
    widen_rows_to_curvature_floor,
)
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
    MERGE_NEARBY_PEAKS,
    MONO_DY_EPS_FRAC,
    MONO_MAX_TURNS,
    PLATEAU_DEDUPE,
    PLATEAU_DEDUPE_MIN_OVERLAP_FRAC,
    PLATEAU_DEDUPE_RADIUS_STEPS,
    PER_RUN_MIN_DIST,
    PER_RUN_MIN_WIDTH,
    PER_RUN_PROM_FRAC,
    RMIN_RUNS,
    FV_VALLEY_FRAC,
    RUNLOG,
    SMOOTH_MA,
    SMOOTH_FRAC,
    TIMING_MODE,
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
from process.cdc_tw import is_reverse_discordant as _is_reverse_discordant
from process.cdc_utils import infer_tier as _infer_tier, seed_from_name as _seed_from_name
from utils import config
from utils.peakHelpers import fmt_peak_stats

TIME_PER_TASK = 0.0
_BOUNDARY_NEAR_GRID_STEPS = 8.0
_BOUNDARY_FAR_GRID_STEPS = 5.0
_DEGENERATE_CI_GRID_FRAC = 0.75
_SINGLE_CREST_PROM_FRAC = 0.03

class ProgressType(Enum):
    CONCORDANCE = 0
    SAMPLING    = 1
    OPTIMAL     = 2

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
        
        return True, None

    finally:
        # runtime log (best effort)
        n_grid = len(sample.calculationSettings.rimAges())
        R_runs = sample.calculationSettings.monteCarloRuns
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
    - A spot is marked as reverseDiscordant if it is geometrically reverse in TW space
      and fails the concordance test.

    Emits:
      - ProgressType.CONCORDANCE updates for UI.
      - sample.updateConcordance(concordancy, discordances, reverse_flags).
    """

    sampleNameText = f" for '{sample.name}'" if sample.name else ""
    signals.newTask("Classifying points" + sampleNameText + "...")

    settings   = sample.calculationSettings
    n_spots    = max(1, len(sample.validSpots))
    timePerRow = TIME_PER_TASK / n_spots

    concordancy   = []
    discordances  = []
    reverse_flags = []   # << NEW

    for i, spot in enumerate(sample.validSpots):
        signals.progress(ProgressType.CONCORDANCE, i / n_spots)
        time.sleep(timePerRow)
        if signals.halt():
            signals.cancelled()
            return False, "processing halted by user"

        if settings.discordanceClassificationMethod == DiscordanceClassificationMethod.PERCENTAGE:
            discordance = calculations.discordance(spot.uPbValue, spot.pbPbValue)
            # Concordant if the *magnitude* is under the threshold
            concordant  = abs(discordance) < settings.discordancePercentageCutoff
        else:
            discordance = None
            concordant = calculations.isConcordantErrorEllipse(
                spot.uPbValue,  spot.uPbStDev,
                spot.pbPbValue, spot.pbPbStDev,
                settings.discordanceEllipseSigmas
            )

        is_rev_geom = _is_reverse_discordant(spot.uPbValue, spot.pbPbValue)

        # Only mark reverse if it’s *discordant* and geometrically reverse
        spot.reverseDiscordant = bool(is_rev_geom and not concordant)

        discordances.append(discordance)
        concordancy.append(concordant)

    reverse_flags = [bool(s.reverseDiscordant) for s in sample.validSpots]
    sample.updateConcordance(concordancy, discordances, reverse_flags)

    n_rev = sum(reverse_flags)
    n_fwd = sum(1 for c, r in zip(concordancy, reverse_flags) if (not c) and (not r))
    n_con = sum(1 for c in concordancy if c)

    signals.progress(ProgressType.CONCORDANCE, 1.0, sample.name, concordancy, discordances, reverse_flags)
    return True, None

# ======================  MC Sampling  ======================

def _performSingleRun(settings, run):
    for age in settings.rimAges():
        run.samplePbLossAge(age, settings.dissimilarityTest, settings.penaliseInvalidAges)
    run.calculateOptimalAge()
    run.createHeatmapData(settings.minimumRimAge, settings.maximumRimAge, config.HEATMAP_RESOLUTION)


def _performRimAgeSampling(signals, sample):
    """
    Run Monte Carlo sampling of Pb-loss ages for a single sample.

    - Filters out reverse-discordant spots.
    - Requires ≥1 concordant and ≥3 forward-discordant spots.
    - Draws MC replicates for U/Pb, Pb/Pb for each spot.
    - For each MC run, constructs a MonteCarloRun over the Pb-loss grid,
      finds the optimal Pb-loss age, and emits ProgressType.SAMPLING.
    """
    sample.monteCarloRuns = []
    sample.peak_catalogue = []
    sample.rejected_peak_candidates = []
    sampleNameText = f" for '{sample.name}'" if sample.name else ""
    signals.newTask("Sampling Pb-loss age distributions" + sampleNameText + "...")

    settings = sample.calculationSettings
    setattr(settings, "timing_mode", TIMING_MODE)
    setattr(settings, "write_outputs", CDC_WRITE_OUTPUTS)

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
    merge_nearby = bool(getattr(settings, "merge_nearby_peaks", MERGE_NEARBY_PEAKS))
    rng = np.random.default_rng(_seed_from_name(sample.name))

    concordantUPbValues  = np.stack([rng.normal(s.uPbValue,  s.uPbStDev,  stabilitySamples) for s in concordantSpots], axis=1)
    concordantPbPbValues = np.stack([rng.normal(s.pbPbValue, s.pbPbStDev, stabilitySamples) for s in concordantSpots], axis=1)
    discordantUPbValues  = np.stack([rng.normal(s.uPbValue,  s.uPbStDev,  stabilitySamples) for s in discordantSpots], axis=1)
    discordantPbPbValues = np.stack([rng.normal(s.pbPbValue, s.pbPbStDev, stabilitySamples) for s in discordantSpots], axis=1)

    # --------- sampling loop ---------
    per_run_times = []
    t0 = time.perf_counter()
    for j in range(stabilitySamples):
        if signals.halt():
            signals.cancelled()
            return False, "processing halted by user"

        t_run = time.perf_counter()

        run = MonteCarloRun(
            j, sample.name,
            concordantUPbValues[j],  concordantPbPbValues[j],
            discordantUPbValues[j],  discordantPbPbValues[j],
            settings=settings
        )
        _performSingleRun(settings, run)
        per_run_times.append(time.perf_counter() - t_run)

        progress = (j + 1) / stabilitySamples
        sample.addMonteCarloRun(run)
        signals.progress(ProgressType.SAMPLING, progress, sample.name, run)

    mc_elapsed = time.perf_counter() - t0
    grid_len = len(settings.rimAges())

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
    plot_rows = [dict(r) for r in rows_for_ui if str(r.get("mode", "")) != "recent_boundary"]
    ui_peaks_age = [float(r["age_ma"]) for r in plot_rows]
    ui_peaks_ci  = [[float(r["ci_low"]), float(r["ci_high"])] for r in plot_rows]
    ui_support   = [float(r.get("support", float("nan"))) for r in plot_rows]

    sample.summedKS_peaks_Ma   = np.asarray(ui_peaks_age, float)
    sample.summedKS_ci_low_Ma  = np.asarray([lo for lo, _ in ui_peaks_ci], float)
    sample.summedKS_ci_high_Ma = np.asarray([hi for _, hi in ui_peaks_ci], float)

    payload = (ages_ma.tolist(), y_curve.tolist(), ui_peaks_age, ui_peaks_ci, ui_support)
    if hasattr(sample.signals, "summedKS"):
        try:
            sample.signals.summedKS.emit(payload)
        except (AttributeError, RuntimeError, TypeError):
            import traceback
            traceback.print_exc()
    try:
        signals.progress("summedKS", progress, sample.name, payload)
    except TypeError:
        try:
            signals.progress("summedKS", progress, sample.name, (payload[0], payload[1], payload[2]))
        except (AttributeError, RuntimeError, TypeError):
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

def _is_effectively_monotonic(y_curve, delta):
    """
    Return True when the smoothed ensemble curve is effectively monotonic.

    Tiny wiggles are ignored using a derivative epsilon scaled by ensemble
    dynamic range (delta), so "boundary optima" are abstained rather than
    promoted to discrete peaks.
    """
    y = np.asarray(y_curve, float)
    if y.size < 4 or (not np.isfinite(y).any()):
        return True

    dy = np.diff(y)
    eps = max(1e-12, float(MONO_DY_EPS_FRAC) * max(float(delta), 1e-12))
    sgn = np.zeros_like(dy, dtype=int)
    sgn[dy > eps] = 1
    sgn[dy < -eps] = -1
    sgn = sgn[sgn != 0]
    if sgn.size == 0:
        return True

    turns = int(np.sum(sgn[1:] != sgn[:-1]))
    return turns <= int(MONO_MAX_TURNS)

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

def _recompute_winner_support(rows, optima_ma, ages_ma, min_support=None):
    """
    Recompute support as winner-vote fraction from per-run optima.

    Each run contributes to at most one peak:
      1) prefer peaks whose CI contains that run optimum,
      2) tie-break by nearest peak age,
      3) if none contain it, allow nearest peak within a small cap distance.
    """
    if not rows:
        return rows

    rows = sorted([dict(r) for r in rows], key=lambda rr: float(rr["age_ma"]))
    centers = np.array([float(r["age_ma"]) for r in rows], float)
    ci_lo = np.array([float(r["ci_low"]) for r in rows], float)
    ci_hi = np.array([float(r["ci_high"]) for r in rows], float)
    counts = np.zeros(len(rows), float)

    opts = np.asarray(optima_ma, float)
    opts = opts[np.isfinite(opts)]
    if opts.size == 0:
        return rows

    step = float(np.median(np.diff(ages_ma))) if np.asarray(ages_ma).size >= 2 else 5.0
    cap = max(3.0 * step, 30.0)

    for o in opts:
        in_ci = np.where((o >= ci_lo) & (o <= ci_hi))[0]
        if in_ci.size > 0:
            loc = np.argmin(np.abs(centers[in_ci] - o))
            counts[int(in_ci[loc])] += 1.0
            continue

        j = int(np.argmin(np.abs(centers - o)))
        if abs(float(centers[j]) - float(o)) <= cap:
            counts[j] += 1.0

    denom = float(max(opts.size, 1))
    out = []
    for i, r in enumerate(rows):
        winner_sup = float(counts[i] / denom)
        direct_sup = float(r.get("direct_support", r.get("support", float("nan"))))
        if not np.isfinite(direct_sup):
            direct_sup = winner_sup
        rr = dict(
            r,
            direct_support=direct_sup,
            winner_support=winner_sup,
            support=direct_sup,   # keep legacy field as reproducibility support
        )
        if (min_support is not None) and (direct_sup < float(min_support)):
            continue
        out.append(rr)

    out = sorted(out, key=lambda rr: float(rr["age_ma"]))
    for i, rr in enumerate(out, 1):
        rr["peak_no"] = i
    return out


def _support_score(row, mode):
    """Return support score used for inclusion filtering."""
    mode = str(mode).strip().upper()
    winner = float(row.get("winner_support", row.get("support", 0.0)))
    direct = float(row.get("direct_support", row.get("support", 0.0)))
    if mode == "DIRECT":
        return direct
    if mode == "MAX":
        return max(winner, direct)
    return winner  # legacy


def _apply_support_filter(rows, min_support, mode):
    """
    Apply configurable inclusion filtering on support:
      - WINNER: winner_support
      - DIRECT: direct_support
      - MAX: max(direct_support, winner_support)
    """
    if not rows:
        return rows
    out = []
    for r in rows:
        score = float(_support_score(r, mode))
        rr = dict(r, filter_support=score)
        if score >= float(min_support):
            out.append(rr)
    out = sorted(out, key=lambda rr: float(rr["age_ma"]))
    for i, rr in enumerate(out, 1):
        rr["peak_no"] = i
    return out

def _step_ma_from_grid(ages_ma):
    ages_ma = np.asarray(ages_ma, float)
    if ages_ma.size >= 2:
        step = float(np.median(np.diff(ages_ma)))
        if np.isfinite(step) and step > 0.0:
            return step
    return 5.0


def _row_match_index(target_row, candidates, used, tol_ma):
    """Return the unmatched candidate index closest in age to target_row within tol_ma."""
    tgt = float(target_row.get("age_ma", np.nan))
    if not np.isfinite(tgt):
        return None
    best_i = None
    best_d = None
    for i, row in enumerate(candidates):
        if used[i]:
            continue
        age = float(row.get("age_ma", np.nan))
        if not np.isfinite(age):
            continue
        d = abs(age - tgt)
        if d <= tol_ma and (best_d is None or d < best_d):
            best_i = i
            best_d = d
    return best_i


def _append_rejected_peak(rejected_rows, row, reason_code):
    """Append one rejected candidate row unless a near-identical age is already recorded."""
    age = float(row.get("age_ma", np.nan))
    if not np.isfinite(age):
        return
    for rr in rejected_rows:
        if abs(float(rr.get("age_ma", np.nan)) - age) <= 1e-6:
            return
    direct = float(row.get("direct_support", row.get("support", np.nan)))
    winner = float(row.get("winner_support", row.get("support", np.nan)))
    rejected_rows.append(
        dict(
            age_ma=age,
            ci_low=float(row.get("ci_low", np.nan)),
            ci_high=float(row.get("ci_high", np.nan)),
            direct_support=direct,
            winner_support=winner,
            reason=str(reason_code),
        )
    )


def _capture_rejected_step(before_rows, after_rows, rejected_rows, reason_code, ages_ma):
    """Record rows present in before_rows that disappear in after_rows as rejected with reason_code."""
    if not before_rows:
        return
    after_rows = [dict(r) for r in (after_rows or [])]
    used = [False] * len(after_rows)
    tol = max(0.51 * _step_ma_from_grid(ages_ma), 1e-6)
    for row in before_rows:
        j = _row_match_index(row, after_rows, used, tol)
        if j is None:
            _append_rejected_peak(rejected_rows, row, reason_code)
        else:
            used[j] = True

def _apply_boundary_dominance_guard(rows, optima_ma, ages_ma):
    """
    Suppress near-edge peaks when per-run optima are overwhelmingly boundary-dominated.

    This targets failure modes where a broad/monotonic surface produces a spurious
    near-edge local hump with high reproducibility.
    """
    if (not rows) or (len(rows) == 0):
        return rows, None

    ages_ma = np.asarray(ages_ma, float)
    opts = np.asarray(optima_ma, float)
    opts = opts[np.isfinite(opts)]
    if ages_ma.size < 2 or opts.size == 0:
        return rows, None

    lo = float(ages_ma[0])
    hi = float(ages_ma[-1])
    step = float(np.median(np.diff(ages_ma)))
    if (not np.isfinite(step)) or step <= 0.0:
        step = max((hi - lo) / max(len(ages_ma) - 1, 1), 1.0)

    # Boundary-dominance measured from run-level optima.
    edge_band = max(1.0 * step, 10.0)
    lo_frac = float(np.mean(opts <= (lo + edge_band)))
    hi_frac = float(np.mean(opts >= (hi - edge_band)))

    side = None
    edge_frac = 0.0
    if lo_frac >= hi_frac:
        side, edge_frac = "low", lo_frac
    else:
        side, edge_frac = "high", hi_frac

    # Require substantial boundary pile-up before suppressing.
    if edge_frac < 0.70:
        return rows, None

    opt_med = float(np.median(opts))
    span = max(hi - lo, step)
    # Keep edge zones scale-aware so genuinely young events are not suppressed
    # just because absolute fallback floors dominate small modelling windows.
    # Keep boundary heuristics tied to grid resolution so coarse meshes do not
    # label nearby interior peaks as edge modes.
    near_edge_zone = min(max(_BOUNDARY_NEAR_GRID_STEPS * step, 0.08 * span), 0.25 * span)
    far_from_edge_zone = min(max(_BOUNDARY_FAR_GRID_STEPS * step, 0.05 * span), 0.20 * span)
    strict_boundary_mode = edge_frac >= 0.85

    out = []
    for r in rows:
        if str(r.get("mode", "")) == "recent_boundary":
            out.append(dict(r))
            continue

        age = float(r.get("age_ma", np.nan))
        if not np.isfinite(age):
            continue

        support = float(r.get("support", 0.0))
        direct = float(r.get("direct_support", support))
        winner = float(r.get("winner_support", support))
        if not np.isfinite(direct):
            direct = support
        if not np.isfinite(winner):
            winner = support

        if side == "low" and age <= (lo + near_edge_zone):
            mismatch = (
                (opt_med <= (lo + edge_band))
                and ((age - lo) >= far_from_edge_zone)
                and (winner < max(0.45, 0.65 * direct))
            )
            if strict_boundary_mode or mismatch:
                continue
        if side == "high" and age >= (hi - near_edge_zone):
            mismatch = (
                (opt_med >= (hi - edge_band))
                and ((hi - age) >= far_from_edge_zone)
                and (winner < max(0.45, 0.65 * direct))
            )
            if strict_boundary_mode or mismatch:
                continue
        out.append(dict(r))

    # If everything was near-edge under boundary dominance, abstain.
    if len(out) == 0:
        return [], "boundary_dominated_surface"
    return out, None


def _recent_boundary_mode_row(
    optima_ma: np.ndarray,
    total_runs: int,
    ages_ma: np.ndarray,
) -> Optional[Dict]:
    vals = np.asarray(optima_ma, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    ages_ma = np.asarray(ages_ma, float)
    if ages_ma.size == 0:
        return None

    step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 5.0
    if (not np.isfinite(step)) or step <= 0.0:
        step = 5.0
    young_edge = float(ages_ma[0])
    edge_hits = vals <= (young_edge + step)
    n_hits = int(np.count_nonzero(edge_hits))
    min_support = max(float(FS_SUPPORT), float(RMIN_RUNS) / float(max(total_runs, 1)), 0.40)
    support = float(n_hits / float(max(total_runs, 1)))
    if n_hits == 0 or support < min_support:
        return None

    edge_vals = vals[edge_hits]
    upper = float(np.nanpercentile(edge_vals, 97.5)) if edge_vals.size >= 3 else float(young_edge + step)
    upper = max(upper, float(young_edge + step))
    return dict(
        sample="",
        peak_no=0,
        age_ma=float(young_edge),
        ci_low=float(young_edge),
        ci_high=float(upper),
        support=float(support),
        direct_support=float(support),
        winner_support=float(support),
        mode="recent_boundary",
        label="Recent boundary mode",
    )


def _inject_recent_boundary_mode(
    rows: List[Dict],
    optima_ma: np.ndarray,
    total_runs: int,
    ages_ma: np.ndarray,
) -> Tuple[List[Dict], Optional[Dict]]:
    if not rows:
        return [], None

    row_boundary = _recent_boundary_mode_row(optima_ma, total_runs, ages_ma)
    if row_boundary is None:
        return [dict(r) for r in rows], None

    ages_ma = np.asarray(ages_ma, float)
    step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 5.0
    if (not np.isfinite(step)) or step <= 0.0:
        step = 5.0
    young_edge = float(ages_ma[0]) if ages_ma.size else 0.0
    replace_limit = young_edge + (2.0 * step)

    interior_rows = [
        dict(r)
        for r in rows
        if str(r.get("mode", "")) != "recent_boundary"
        and np.isfinite(float(r.get("age_ma", np.nan)))
        and float(r.get("age_ma", np.nan)) > replace_limit
    ]
    if interior_rows:
        strongest_interior = max(
            float(r.get("direct_support", r.get("support", 0.0)))
            for r in interior_rows
        )
        boundary_support = float(
            row_boundary.get("direct_support", row_boundary.get("support", 0.0))
        )
        if boundary_support + 1e-12 < strongest_interior:
            return [dict(r) for r in rows], None

    out: List[Dict] = []
    inserted = False
    for rr in rows:
        mode = str(rr.get("mode", ""))
        if mode == "recent_boundary":
            if not inserted:
                out.append(dict(row_boundary))
                inserted = True
            continue

        age = float(rr.get("age_ma", np.nan))
        if np.isfinite(age) and age <= replace_limit:
            continue
        out.append(dict(rr))

    if not inserted:
        out.insert(0, dict(row_boundary))

    out.sort(key=lambda rr: (0 if str(rr.get("mode", "")) == "recent_boundary" else 1, float(rr.get("age_ma", np.nan))))
    return out, dict(row_boundary)


def _plateau_dedupe_rows(rows, ages_ma):
    """
    Collapse near-identical peaks that sit on the same broad/flat crest.
    This is lighter than full peak merging and mainly removes duplicate picks
    emitted from adjacent surface windows.
    """
    if (not rows) or len(rows) <= 1:
        return rows

    rows = sorted([dict(r) for r in rows], key=lambda rr: float(rr["age_ma"]))
    step = float(np.median(np.diff(ages_ma))) if np.asarray(ages_ma).size >= 2 else 5.0
    near_ma = max(float(PLATEAU_DEDUPE_RADIUS_STEPS) * step, 20.0)
    min_ov = float(PLATEAU_DEDUPE_MIN_OVERLAP_FRAC)

    def _width(rr):
        return max(0.0, float(rr["ci_high"]) - float(rr["ci_low"]))

    def _overlap_frac(a, b):
        lo = max(float(a["ci_low"]), float(b["ci_low"]))
        hi = min(float(a["ci_high"]), float(b["ci_high"]))
        ov = max(0.0, hi - lo)
        wa, wb = _width(a), _width(b)
        denom = max(min(wa, wb), 1e-9)
        return ov / denom

    def _score(rr):
        win = float(rr.get("winner_support", rr.get("support", 0.0)))
        direct = float(rr.get("direct_support", rr.get("support", 0.0)))
        width = _width(rr)
        return (win, direct, -width)

    deduped = [dict(rows[0])]
    for rr in rows[1:]:
        prev = deduped[-1]
        sep = abs(float(rr["age_ma"]) - float(prev["age_ma"]))
        same_crest = (sep <= near_ma) and (_overlap_frac(prev, rr) >= min_ov)
        if same_crest:
            if _score(rr) > _score(prev):
                deduped[-1] = dict(rr)
        else:
            deduped.append(dict(rr))

    for i, rr in enumerate(deduped, 1):
        rr["peak_no"] = i
    return deduped


def _single_crest_fallback_row(ages_ma, S_curve, optima_ma, min_support):
    """
    Conservative fallback when strict peak gating abstains:
      - require one strong interior crest on the displayed ensemble curve,
      - require non-trivial run-optima concentration in that crest window.
    Returns a single peak row dict or None.
    """
    x = np.asarray(ages_ma, float)
    y = np.asarray(S_curve, float)
    if x.size < 7 or y.size != x.size:
        return None

    finite = np.isfinite(y)
    if not np.any(finite):
        return None
    y = np.where(finite, y, np.nan)

    q5, q95 = np.nanpercentile(y, [5, 95])
    delta = float(max(q95 - q5, 0.0))
    if delta < max(float(ENS_DELTA_MIN), 1e-6):
        return None

    step = float(np.median(np.diff(x))) if x.size >= 2 else 1.0
    if (not np.isfinite(step)) or step <= 0.0:
        step = max((float(x[-1]) - float(x[0])) / max(int(x.size) - 1, 1), 1.0)
    lo_age, hi_age = float(x[0]), float(x[-1])
    span = max(hi_age - lo_age, step)
    edge_margin = min(max(6.0 * step, 0.07 * span), 0.30 * span)

    # Simple local maxima detector on the smoothed ensemble curve.
    loc = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    if loc.size == 0:
        finite_idx = np.flatnonzero(np.isfinite(y))
        if finite_idx.size == 0:
            return None
        best_rel = int(np.nanargmax(y[finite_idx]))
        loc = np.array([int(finite_idx[best_rel])], dtype=int)

    interior = [int(j) for j in loc if (x[j] > (lo_age + edge_margin)) and (x[j] < (hi_age - edge_margin))]
    if not interior:
        return None

    j = max(interior, key=lambda idx: float(y[idx]))
    left_min = float(np.nanmin(y[: j + 1]))
    right_min = float(np.nanmin(y[j:]))
    left_lift = float(y[j] - left_min)
    right_lift = float(y[j] - right_min)
    prom_balanced = float(y[j] - max(left_min, right_min))
    prom_one_sided = float(max(left_lift, right_lift))
    # Accept broad single-crest surfaces where absolute prominence is modest
    # compared to full-window delta, but still clearly above local roughness.
    rough = float(np.nanmedian(np.abs(np.diff(y)))) if y.size >= 3 else 0.0
    # Broad single-crest surfaces are accepted once they rise by roughly 3% of
    # the full window dynamic range and stand above local roughness.
    prom_min = max(_SINGLE_CREST_PROM_FRAC * delta, 3.0 * rough, 0.008)
    weak_side_min = max(rough, 0.002)
    if prom_one_sided < prom_min:
        return None
    if min(left_lift, right_lift) < weak_side_min:
        return None

    # Half-prominence window around the crest.
    prom_for_window = max(prom_balanced, prom_min)
    half_level = float(y[j] - 0.5 * prom_for_window)
    jl = int(j)
    while jl > 0 and float(y[jl]) >= half_level:
        jl -= 1
    jr = int(j)
    n = int(y.size)
    while jr < (n - 1) and float(y[jr]) >= half_level:
        jr += 1

    lo_win = float(x[max(jl, 0)])
    hi_win = float(x[min(jr, n - 1)])
    if hi_win <= lo_win:
        pad = max(2.0 * step, 0.04 * span)
        lo_win = max(lo_age, float(x[j]) - pad)
        hi_win = min(hi_age, float(x[j]) + pad)

    opts = np.asarray(optima_ma, float)
    opts = opts[np.isfinite(opts)]
    if opts.size == 0:
        return None

    # Do not synthesize a fallback peak when run-level optima are dominated by
    # a modelling-window boundary (classic fan-to-zero / floor artefact).
    edge_band = max(5.0 * step, 40.0)
    lo_frac = float(np.mean(opts <= (lo_age + edge_band)))
    hi_frac = float(np.mean(opts >= (hi_age - edge_band)))
    if max(lo_frac, hi_frac) >= 0.70:
        return None

    in_win = opts[(opts >= lo_win) & (opts <= hi_win)]
    winner_support = float(in_win.size / float(max(opts.size, 1)))
    if winner_support < float(min_support):
        return None

    if in_win.size >= 3:
        ci_low, ci_high = np.nanpercentile(in_win, [2.5, 97.5])
        ci_low = float(max(ci_low, lo_age))
        ci_high = float(min(ci_high, hi_age))
    else:
        ci_low, ci_high = lo_win, hi_win

    if (ci_high - ci_low) < step:
        ci_low = max(lo_age, float(x[j]) - step)
        ci_high = min(hi_age, float(x[j]) + step)
    age = float(x[j])
    if not (ci_low <= age <= ci_high):
        ci_low = min(ci_low, age)
        ci_high = max(ci_high, age)

    return dict(
        age_ma=age,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        support=winner_support,
        direct_support=winner_support,
        winner_support=winner_support,
        selection="fallback",
        peak_no=1,
    )


def _snap_rows_to_curve(rows, ages_ma, S_view):
    """
    Keep accepted catalogue rows unchanged for reporting. Only snap plotted
    marker ages when the nearest displayed-curve crest is genuinely close to
    the reported age; otherwise preserve the reported age/interval exactly.
    """
    if not rows:
        return []

    x = np.asarray(ages_ma, float)
    y = np.asarray(S_view, float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return [dict(r) for r in rows]

    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return [dict(r) for r in rows]

    # Candidate local maxima on the displayed curve.
    core = np.where(
        finite[1:-1] &
        (y[1:-1] >= y[:-2]) &
        (y[1:-1] >= y[2:])
    )[0] + 1
    if core.size == 0:
        idx_all = np.where(finite)[0]
        if idx_all.size:
            core = np.array([int(idx_all[np.nanargmax(y[idx_all])])], dtype=int)
        else:
            return [dict(r) for r in rows]

    # Deduplicate adjacent maxima (plateaus) by keeping the midpoint of the
    # flat crest when multiple adjacent nodes share the same local maximum.
    def _representative_max(run_idx):
        run_idx = np.asarray(run_idx, dtype=int)
        if run_idx.size == 0:
            raise ValueError("run_idx must not be empty")
        y_run = y[run_idx]
        y_max = float(np.nanmax(y_run))
        tied = run_idx[np.isclose(y_run, y_max, rtol=1e-12, atol=1e-15)]
        if tied.size:
            return int(tied[tied.size // 2])
        return int(run_idx[int(np.nanargmax(y_run))])

    maxima = []
    run = [int(core[0])]
    for idx in core[1:]:
        idx = int(idx)
        if idx == run[-1] + 1:
            run.append(idx)
        else:
            maxima.append(_representative_max(run))
            run = [idx]
    if run:
        maxima.append(_representative_max(run))
    maxima = np.asarray(maxima, dtype=int)

    out = []
    used = np.zeros(maxima.size, dtype=bool)
    step = float(np.median(np.diff(x))) if x.size >= 2 else 1.0
    lo_grid = float(np.nanmin(x[finite]))
    hi_grid = float(np.nanmax(x[finite]))

    # Assign each displayed row to the nearest available displayed-curve crest,
    # but do not drag the marker a long way from the reported age just to land
    # on a local maximum. That was producing misleading plots where the
    # catalogue age and the plotted marker diverged by hundreds of Ma.
    order = np.argsort([float(dict(r).get("age_ma", np.nan)) for r in rows])
    snapped = [None] * len(rows)
    for ii in order:
        rr = dict(rows[ii])
        a0 = float(rr.get("age_ma", np.nan))
        try:
            lo_old = float(rr.get("ci_low", np.nan))
            hi_old = float(rr.get("ci_high", np.nan))
        except (TypeError, ValueError):
            lo_old, hi_old = np.nan, np.nan

        if maxima.size > 0 and np.any(~used):
            avail = np.where(~used)[0]
            j_idx = int(avail[np.argmin(np.abs(x[maxima[avail]] - a0))])
            j = int(maxima[j_idx])
            used[j_idx] = True
        elif maxima.size > 0:
            j = int(maxima[np.argmin(np.abs(x[maxima] - a0))])
        else:
            idx_all = np.where(finite)[0]
            j = int(idx_all[np.argmin(np.abs(x[idx_all] - a0))])

        width = hi_old - lo_old if (np.isfinite(lo_old) and np.isfinite(hi_old)) else np.nan
        if (not np.isfinite(width)) or (width <= 0.0):
            width = 2.0 * step
        snap_tol = max(2.0 * step, 0.35 * width)

        a_new = float(x[j])
        if np.isfinite(a0) and abs(a_new - a0) <= snap_tol:
            rr["age_ma"] = a_new
        else:
            a_new = a0 if np.isfinite(a0) else a_new
            rr["age_ma"] = a_new

        lo_new = max(lo_grid, a_new - 0.5 * width)
        hi_new = min(hi_grid, a_new + 0.5 * width)
        if (hi_new - lo_new) < step:
            lo_new = max(lo_grid, a_new - step)
            hi_new = min(hi_grid, a_new + step)
        rr["ci_low"] = float(min(lo_new, a_new))
        rr["ci_high"] = float(max(hi_new, a_new))
        snapped[ii] = rr

    out = [r for r in snapped if isinstance(r, dict)]
    return out


def _raw_optimum_age_ma(run) -> float:
    """
    Return the per-run RAW optimum age (Ma) from `_raw_statistics_by_pb_loss_age`.

    Falls back to the penalised optimum if RAW data are unavailable.
    """
    raw_map = getattr(run, "_raw_statistics_by_pb_loss_age", None)
    if isinstance(raw_map, dict) and raw_map:
        ages = np.array(sorted(raw_map.keys()), float)
        dvals = np.array([raw_map[a].test_statistics[0] for a in ages], float)
        dvals = np.where(np.isfinite(dvals), dvals, np.inf)
        if ages.size and np.isfinite(dvals).any():
            idx = _findOptimalIndex(dvals.tolist())
            return float(ages[idx] / 1e6)

    pen = float(getattr(run, "optimal_pb_loss_age", np.nan))
    return float(pen / 1e6) if np.isfinite(pen) else float("nan")


def _stack_goodness_from_stats_attr(runs, ages_y, stats_attr: str, which: str = "pen") -> np.ndarray:
    """
    Build an R×G goodness matrix from a run-level stats map attribute.

    Parameters
    ----------
    runs : list[MonteCarloRun]
        Monte Carlo runs.
    ages_y : array_like
        Age grid in YEARS.
    stats_attr : str
        Attribute name on each run that maps age_years -> statistics object.
    which : {'raw', 'pen'}
        'raw' -> 1 - KS D ; 'pen' -> 1 - penalized score.
    """
    ages_y = np.asarray(ages_y, float)
    R = len(runs)
    G = len(ages_y)
    S = np.full((R, G), np.nan, float)

    for r_i, run in enumerate(runs):
        stats_map = getattr(run, stats_attr, None)
        if not isinstance(stats_map, dict) or not stats_map:
            stats_map = getattr(run, "statistics_by_pb_loss_age", None)
        if not isinstance(stats_map, dict) or not stats_map:
            continue

        arr = np.full(G, np.nan, float)
        for g_i, age in enumerate(ages_y):
            st = stats_map.get(float(age))
            if st is None:
                continue
            if which == "raw":
                arr[g_i] = 1.0 - float(st.test_statistics[0])
            else:
                ds = float(st.score)
                ds = min(1.0, max(0.0, ds))
                arr[g_i] = 1.0 - ds
        S[r_i] = arr

    return S


def _optimum_age_ma_from_stats_attr(run, stats_attr: str, which: str = "pen") -> float:
    """
    Return per-run optimum age (Ma) from a run-level stats map attribute.
    """
    stats_map = getattr(run, stats_attr, None)
    if not isinstance(stats_map, dict) or not stats_map:
        stats_map = getattr(run, "statistics_by_pb_loss_age", None)
    if not isinstance(stats_map, dict) or not stats_map:
        return float("nan")

    ages = np.array(sorted(stats_map.keys()), float)
    if ages.size == 0:
        return float("nan")

    if which == "raw":
        vals = np.array([stats_map[a].test_statistics[0] for a in ages], float)
    else:
        vals = np.array([stats_map[a].score for a in ages], float)
    vals = np.where(np.isfinite(vals), vals, np.inf)
    if not np.isfinite(vals).any():
        return float("nan")

    idx = _findOptimalIndex(vals.tolist())
    return float(ages[idx] / 1e6)


def _optimum_stat_from_stats_attr(run, stats_attr: str, which: str = "pen"):
    """
    Return the run-level optimum statistics object from a chosen stats map.
    """
    stats_map = getattr(run, stats_attr, None)
    if not isinstance(stats_map, dict) or not stats_map:
        stats_map = getattr(run, "statistics_by_pb_loss_age", None)
    if not isinstance(stats_map, dict) or not stats_map:
        return None

    ages = np.array(sorted(stats_map.keys()), float)
    if ages.size == 0:
        return None

    if which == "raw":
        vals = np.array([stats_map[a].test_statistics[0] for a in ages], float)
    else:
        vals = np.array([stats_map[a].score for a in ages], float)
    vals = np.where(np.isfinite(vals), vals, np.inf)
    if not np.isfinite(vals).any():
        return None

    idx = _findOptimalIndex(vals.tolist())
    return stats_map[float(ages[idx])]


def _median_best_from_runs(runs, stats_attr: str, which: str = "pen") -> float:
    """
    Median run-level best dissimilarity from a stats map attribute.
    Lower is better.
    """
    vals = []
    for run in runs:
        stats_map = getattr(run, stats_attr, None)
        if not isinstance(stats_map, dict) or not stats_map:
            continue
        ages = sorted(stats_map.keys())
        if not ages:
            continue
        if which == "raw":
            arr = np.array([stats_map[a].test_statistics[0] for a in ages], float)
        else:
            arr = np.array([stats_map[a].score for a in ages], float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(float(np.min(arr)))
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, float)))


def _build_global_catalogue_rows(sample_name: str,
                                 tier: str,
                                 ages_ma: np.ndarray,
                                 S_runs: np.ndarray,
                                 Smed: np.ndarray,
                                 *,
                                 smf: float,
                                 merge_nearby: bool,
                                 pickable: bool,
                                 optima_ma: np.ndarray,
                                 diagnostic_rows: Optional[List[Dict]] = None):
    """Build global ensemble rows for one surface."""
    if (not pickable) or (Smed.size == 0):
        return []

    diag_rows: List[Dict] = []

    rows = build_ensemble_catalogue(
        sample_name, tier, ages_ma, S_runs,
        orientation="max", smooth_frac=smf,
        f_d=FD_DIST_FRAC, f_p=FP_PROM_FRAC, f_v=FV_VALLEY_FRAC, f_w=FW_WIN_FRAC,
        w_min_nodes=3, support_min=FS_SUPPORT, r_min=RMIN_RUNS, f_r=FR_RUN_REL,
        per_run_prom_frac=PER_RUN_PROM_FRAC, per_run_min_dist=PER_RUN_MIN_DIST,
        per_run_min_width=PER_RUN_MIN_WIDTH, per_run_require_full_prom=False,
        pen_ok_mask=None, cand_curve=Smed, height_frac=FH_HEIGHT_FRAC, optima_ma=optima_ma,
        merge_per_hump=merge_nearby, merge_shoulders=merge_nearby,
        diagnostic_rows=diag_rows,
    ) or []

    if diagnostic_rows is not None:
        diagnostic_rows.extend(dict(r) for r in diag_rows)

    return rows

@dataclass
class SurfaceState:
    """Bundles the per-surface (RAW or PEN) state that flows through the pipeline."""
    S_runs: np.ndarray
    Smed: np.ndarray
    Delta: float
    mono: bool
    pickable: bool
    optima_ma: np.ndarray
    rows: List[Dict] = field(default_factory=list)
    rejected: List[Dict] = field(default_factory=list)


def _keep_same(rows, keep):
    if not rows or not keep:
        return []
    keep_ages = {float(r["age_ma"]) for r in keep}
    return [r for r in rows if float(r.get("age_ma", float("nan"))) in keep_ages]


def _compute_run_optima_ci(raw, pen, prefer_pen, runs):
    """Section (A): median-of-run-optima with 95% CI."""
    optima_ma_primary = pen.optima_ma if prefer_pen else raw.optima_ma
    opt_all = np.sort(np.asarray(optima_ma_primary[np.isfinite(optima_ma_primary)] * 1e6, float))
    if opt_all.size == 0:
        if prefer_pen:
            opt_all = np.sort(np.asarray([r.optimal_pb_loss_age for r in runs], float))
        else:
            opt_all = np.sort(np.asarray([_raw_optimum_age_ma(r) * 1e6 for r in runs], float))
    n = opt_all.size
    if n:
        optimalAge = float(np.median(opt_all))
        lower95 = float(opt_all[int(np.floor(0.025 * n))])
        upper95 = float(opt_all[int(np.ceil(0.975 * n)) - 1])
    else:
        optimalAge = lower95 = upper95 = float("nan")
    return optimalAge, lower95, upper95, opt_all


def _compute_legacy_surface(raw, pen, prefer_pen, ages_y):
    """Section (B): legacy surface optimum for export/figure."""
    S_runs_primary = pen.S_runs if prefer_pen else raw.S_runs
    sum_good = np.nansum(S_runs_primary, axis=0)
    cnt_good = np.sum(np.isfinite(S_runs_primary), axis=0)
    mean_good = np.divide(
        sum_good, cnt_good,
        out=np.full_like(sum_good, np.nan, dtype=float),
        where=cnt_good > 0,
    )
    mean_primary = 1.0 - mean_good
    mean_primary = np.where(np.isfinite(mean_primary), mean_primary, np.inf)
    legacy_idx = _findOptimalIndex(mean_primary.tolist())
    optimalAge_legacy = float(ages_y[legacy_idx])
    S_legacy_curve = 1.0 - mean_primary
    return optimalAge_legacy, S_legacy_curve, mean_primary


def _compute_mean_stats(runs, primary_which, prefer_pen):
    """Section (C): mean stats at each run's own optimum."""
    stats = [_optimum_stat_from_stats_attr(r, "_all_statistics_by_pb_loss_age", which=primary_which) for r in runs]
    stats = [s for s in stats if s is not None]
    if stats:
        meanD = float(np.mean([s.test_statistics[0] for s in stats]))
        pvals = np.asarray([s.test_statistics[1] for s in stats], float)
        p_ok = np.isfinite(pvals)
        meanP = float(np.mean(pvals[p_ok])) if np.any(p_ok) else float("nan")
        meanInv = float(np.mean([s.number_of_invalid_ages for s in stats]))
        if prefer_pen:
            meanSc = float(np.mean([s.score for s in stats]))
        else:
            meanSc = float(np.mean([s.test_statistics[0] for s in stats]))
    else:
        meanD = meanP = meanInv = meanSc = float("nan")
    return meanD, meanP, meanInv, meanSc


def _apply_guards_and_fallbacks(
    sample, settings, runs, raw, pen,
    rows_for_ui, rejected_rows,
    ages_ma, S_view, S_runs_view,
    view_which, ui_surface,
    support_floor,
):
    """Boundary guards, CI calibration, wide-CI filter, single-crest fallback."""
    optima_ma_display = raw.optima_ma if view_which == "raw" else pen.optima_ma

    # Boundary-dominance guard
    pre_boundary_ui = [dict(r) for r in rows_for_ui]
    rows_for_ui, boundary_reason = _apply_boundary_dominance_guard(rows_for_ui, optima_ma_display, ages_ma)
    if boundary_reason is not None:
        _capture_rejected_step(pre_boundary_ui, rows_for_ui, rejected_rows, boundary_reason, ages_ma)
        raw.rows = []
        pen.rows = []
        sample.ensemble_abstain_reason = boundary_reason

    # Recent boundary mode injection
    pre_boundary_mode_ui = [dict(r) for r in rows_for_ui]
    rows_for_ui, boundary_row_ui = _inject_recent_boundary_mode(
        rows_for_ui, optima_ma_display, len(runs), ages_ma,
    )
    if boundary_row_ui is not None:
        if view_which == "raw":
            raw.rows = [dict(r) for r in rows_for_ui]
        else:
            pen.rows = [dict(r) for r in rows_for_ui]
        sample.ensemble_abstain_reason = None
        _capture_rejected_step(
            pre_boundary_mode_ui, rows_for_ui, rejected_rows,
            "boundary_dominated_surface", ages_ma,
        )

    # CI calibration: widen using local curvature floor
    for surf in (raw, pen):
        surf.rows = widen_rows_to_curvature_floor(surf.rows, ages_ma, surf.Smed, surf.S_runs, orientation="max")
    rows_for_ui = widen_rows_to_curvature_floor(rows_for_ui, ages_ma, S_view, S_runs_view, orientation="max")

    # Ensure snapped age lies inside its CI
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

    # Enforce minimum CI width and drop boundary-degenerate peaks
    if rows_for_ui:
        step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 5.0
        min_age, max_age = float(ages_ma[0]), float(ages_ma[-1])
        pre_clean_ui = [dict(r) for r in rows_for_ui]
        cleaned = []
        for r in rows_for_ui:
            a  = float(r["age_ma"])
            lo = float(r["ci_low"])
            hi = float(r["ci_high"])
            if (hi - lo) < step:
                lo, hi = a - step, a + step
            near_edge  = (a - min_age) <= step or (max_age - a) <= step
            degenerate = (hi - lo) <= _DEGENERATE_CI_GRID_FRAC * step
            if near_edge and degenerate:
                if float(r.get("filter_support", r.get("support", 0.0))) >= max(support_floor, 0.12):
                    lo, hi = a - step, a + step
                else:
                    continue
            cleaned.append(dict(r, ci_low=lo, ci_high=hi))
        rows_for_ui = cleaned
        _capture_rejected_step(pre_clean_ui, rows_for_ui, rejected_rows, "edge_degenerate_ci", ages_ma)

    for r in rows_for_ui:
        a = float(r["age_ma"])
        if not (float(r["ci_low"]) <= a <= float(r["ci_high"])):
            r["ci_low"]  = min(float(r["ci_low"]),  a)
            r["ci_high"] = max(float(r["ci_high"]), a)

    # Drop peaks with absurdly wide CIs
    if rows_for_ui:
        total_span = float(ages_ma[-1] - ages_ma[0])
        MAX_CI_FRAC = 0.5
        pre_width_ui = [dict(r) for r in rows_for_ui]
        filtered = []
        for r in rows_for_ui:
            if str(r.get("mode", "")) == "recent_boundary":
                filtered.append(r)
                continue
            width = float(r["ci_high"] - r["ci_low"])
            score = float(r.get("filter_support", r.get("support", 0.0)))
            if width > MAX_CI_FRAC * total_span and score < max(support_floor, 0.25):
                continue
            filtered.append(r)
        rows_for_ui = filtered
        _capture_rejected_step(pre_width_ui, rows_for_ui, rejected_rows, "wide_ci", ages_ma)
        raw.rows = _keep_same(raw.rows, rows_for_ui)
        pen.rows = _keep_same(pen.rows, rows_for_ui)

    # Single-crest fallback
    if not rows_for_ui:
        fb = _single_crest_fallback_row(
            ages_ma, S_view, optima_ma_display,
            min_support=max(float(support_floor), 0.10),
        )
        if fb is not None:
            rows_for_ui = [dict(fb)]
            if ui_surface == "RAW":
                raw.rows = [dict(fb)]
                pen.rows = []
            else:
                pen.rows = [dict(fb)]
                raw.rows = []
            sample.ensemble_abstain_reason = None

    if not rows_for_ui:
        if sample.ensemble_abstain_reason is None:
            sample.ensemble_abstain_reason = "no_supported_peaks"

    if isinstance(getattr(sample, "ensemble_surface_flags", None), dict):
        sample.ensemble_surface_flags["view_surface_source"] = "global_all"
    sample.display_heatmap_ages_ma = np.asarray(ages_ma, float)
    sample.display_heatmap_runs_S = np.asarray(S_runs_view, float)

    # Heatmap setup
    for run in runs:
        run._heatmap_view_which = view_which
        run.createHeatmapData(settings.minimumRimAge, settings.maximumRimAge, config.HEATMAP_RESOLUTION)

    # Deduplicate rejected rows
    if rejected_rows:
        used = [False] * len(rows_for_ui)
        tol = max(0.51 * _step_ma_from_grid(ages_ma), 1e-6)
        kept_rejected: List[Dict] = []
        for rr in rejected_rows:
            j = _row_match_index(rr, rows_for_ui, used, tol)
            if j is None:
                kept_rejected.append(rr)
            else:
                used[j] = True
        rejected_rows = sorted(kept_rejected, key=lambda r: float(r.get("age_ma", np.nan)))
    sample.rejected_peak_candidates = rejected_rows

    return rows_for_ui, rejected_rows


def _publish_results(
    signals, sample, progress, settings, runs, raw, pen,
    rows_for_ui, rejected_rows, ages_ma, ages_y, S_view,
    optimalAge, optimalAge_ui, optimalAge_legacy, lower95, upper95, opt_all,
    meanD, meanP, meanInv, meanSc,
):
    """Renumber peaks, export CSVs/NPZ, publish UI payload."""
    for row_list in (rows_for_ui, raw.rows, pen.rows):
        for i, r in enumerate(row_list, 1):
            r["peak_no"] = i

    if CDC_WRITE_OUTPUTS:
        _append_catalogue_rows(sample.name, pen.rows, dest_path=CATALOGUE_CSV_PEN)
        _append_catalogue_rows(sample.name, raw.rows, dest_path=CATALOGUE_CSV_RAW)
        _write_npz_diagnostics(
            sample_name=sample.name,
            ages_ma=ages_ma, ages_y=ages_y, runs=runs,
            S_runs_raw=raw.S_runs, S_runs_pen=pen.S_runs,
            Smed_raw=raw.Smed, Smed_pen=pen.Smed,
            S_view=S_view, rows_for_ui=rows_for_ui,
        )

    rows_for_plot = _snap_rows_to_curve(rows_for_ui, ages_ma, S_view)

    catalogue_legacy = [(r["age_ma"], r["ci_low"], r["ci_high"], r["support"]) for r in rows_for_ui]
    peak_str = fmt_peak_stats(catalogue_legacy) if catalogue_legacy else "—"

    plot_peaks = np.asarray([float(r.get("age_ma", np.nan)) for r in rows_for_plot], float)
    plot_ci_low = np.asarray([float(r.get("ci_low", np.nan)) for r in rows_for_plot], float)
    plot_ci_high = np.asarray([float(r.get("ci_high", np.nan)) for r in rows_for_plot], float)
    sample.summedKS_peaks_Ma = plot_peaks
    sample.summedKS_ci_low_Ma = plot_ci_low
    sample.summedKS_ci_high_Ma = plot_ci_high
    sample.peak_uncertainty_str = peak_str
    detailed_catalogue = [
        dict(
            sample=sample.name, peak_no=i + 1,
            ci_low=lo, age_ma=med, ci_high=hi, support=sup,
            direct_support=float(r.get("direct_support", sup)),
            winner_support=float(r.get("winner_support", sup)),
            selection=r.get("selection", "strict"),
            mode=r.get("mode", ""), label=r.get("label", ""),
        )
        for i, (r, (med, lo, hi, sup)) in enumerate(zip(rows_for_ui, catalogue_legacy))
    ]
    sample.peak_catalogue = detailed_catalogue

    _emit_summedKS(signals, sample, progress, ages_ma, S_view, rows_for_plot)

    if KS_EXPORT_ROOT is not None:
        _export_legacy_ks(
            sample, settings, runs, ages_y,
            ui_opt_years=optimalAge_ui, ui_low95_years=lower95,
            ui_high95_years=upper95, run_optima_years=opt_all,
            legacy_opt_years=optimalAge_legacy,
        )

    payload = (
        optimalAge, lower95, upper95, meanD, meanP, meanInv, meanSc,
        peak_str, detailed_catalogue,
        {"rejected_peak_candidates": list(rejected_rows or [])},
    )
    try:
        signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, payload)
    except TypeError:
        signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, payload[:7])


def _calculateOptimalAge(signals, sample, progress):
    """
      • UI shows median-of-run-optima with consistent CI.
      • Exports and figures use legacy surface optimum and curve.
    """
    settings, runs = sample.calculationSettings, sample.monteCarloRuns
    if not runs:
        return
    sample.rejected_peak_candidates = []

    merge_nearby = bool(getattr(settings, "merge_nearby_peaks", MERGE_NEARBY_PEAKS))
    abstain_on_monotonic = bool(getattr(settings, "conservative_abstain_on_monotonic", True))
    support_filter_mode = "DIRECT"
    prefer_pen = bool(getattr(settings, "penaliseInvalidAges", False))
    primary_which = "pen" if prefer_pen else "raw"

    # Grid
    ages_y = np.asarray(settings.rimAges(), float)   # years
    ages_ma = ages_y / 1e6
    # Per-run optima and goodness matrices from the global all-discordants surface.
    smf = _smooth_frac_for_grid(ages_ma)

    _optima_raw = np.array(
        [_optimum_age_ma_from_stats_attr(r, "_all_statistics_by_pb_loss_age", which="raw") for r in runs], float,
    )
    _S_runs_raw = _stack_goodness_from_stats_attr(runs, ages_y, "_all_statistics_by_pb_loss_age", which="raw")
    _Smed_raw, _Delta_raw, _ = robust_ensemble_curve(_S_runs_raw, smooth_frac=smf)
    _mono_raw = _is_effectively_monotonic(_Smed_raw, _Delta_raw)

    _optima_pen = np.array(
        [_optimum_age_ma_from_stats_attr(r, "_all_statistics_by_pb_loss_age", which="pen") for r in runs], float,
    )
    _S_runs_pen = _stack_goodness_from_stats_attr(runs, ages_y, "_all_statistics_by_pb_loss_age", which="pen")
    _Smed_pen, _Delta_pen, _ = robust_ensemble_curve(_S_runs_pen, smooth_frac=smf)
    _mono_pen = _is_effectively_monotonic(_Smed_pen, _Delta_pen)

    raw = SurfaceState(
        S_runs=_S_runs_raw, Smed=_Smed_raw, Delta=_Delta_raw, mono=_mono_raw,
        pickable=(_Delta_raw >= ENS_DELTA_MIN) and ((not abstain_on_monotonic) or (not _mono_raw)),
        optima_ma=_optima_raw,
    )
    pen = SurfaceState(
        S_runs=_S_runs_pen, Smed=_Smed_pen, Delta=_Delta_pen, mono=_mono_pen,
        pickable=(_Delta_pen >= ENS_DELTA_MIN) and ((not abstain_on_monotonic) or (not _mono_pen)),
        optima_ma=_optima_pen,
    )

    ui_surface = str(getattr(settings, "catalogue_surface", CATALOGUE_SURFACE)).strip().upper()
    if ui_surface not in {"RAW", "PEN"}:
        ui_surface = "PEN"
    S_view = raw.Smed if (ui_surface == "RAW") else pen.Smed
    sample.ensemble_surface_flags = dict(
        raw_delta=float(raw.Delta),
        pen_delta=float(pen.Delta),
        raw_monotonic=bool(raw.mono),
        pen_monotonic=bool(pen.mono),
        primary_channel=str(primary_which),
        view_surface_source="global_all",
    )
    sample.ensemble_abstain_reason = None

    optimalAge, lower95, upper95, opt_all = _compute_run_optima_ci(raw, pen, prefer_pen, runs)
    optimalAge_ui = optimalAge

    optimalAge_legacy, S_legacy_curve, mean_primary = _compute_legacy_surface(raw, pen, prefer_pen, ages_y)
    sample.legacy_surface_optimal_age = optimalAge_legacy

    meanD, meanP, meanInv, meanSc = _compute_mean_stats(runs, primary_which, prefer_pen)

    # ---------- (D) Ensemble OFF branch ----------
    enabled = bool(getattr(settings, "enable_ensemble_peak_picking", False))
    if not enabled:
        # UI point estimate = median of run optima
        optimalAge = optimalAge_ui
        sample.legacy_surface_optimal_age = optimalAge_legacy

        sample.peak_catalogue = []

        # Plot legacy curve for diagnostics (optional)
        _emit_summedKS(signals, sample, progress, ages_ma, S_legacy_curve, rows_for_ui=[])

        # Export legacy surface optimum + curve (for paper figure)
        if KS_EXPORT_ROOT is not None:
            _export_legacy_ks(
                sample, settings, runs, ages_y,
                D_pen=mean_primary,
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
    # Build catalogues from the global all-discordants surface.
    # ------------------------------------------------------------------
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

    # Fallback: if no ensemble peaks survive, leave the catalogue empty.
    if (not raw.rows) and (not pen.rows) and len(opt_all) > 0:
        raw.rows = []
        pen.rows = []
        if (not raw.pickable) and (not pen.pickable):
            sample.ensemble_abstain_reason = "flat_or_monotonic_surface"

    # Choose which to DISPLAY in the UI strictly from the same ensemble surface
    # that produced the accepted rows.
    chosen = raw if (ui_surface == "RAW" and raw.rows) else pen if pen.rows else None
    if chosen is not None:
        view_which = "raw" if chosen is raw else "pen"
        rows_for_ui = chosen.rows
        S_runs_view = chosen.S_runs
        S_view = chosen.Smed
        rejected_stage_ui = chosen.rejected
    else:
        rows_for_ui = []
        chosen_fallback = raw if ui_surface == "RAW" else pen
        view_which = "raw" if chosen_fallback is raw else "pen"
        S_runs_view = chosen_fallback.S_runs
        S_view = chosen_fallback.Smed
        rejected_stage_ui = chosen_fallback.rejected

    for _rows in (raw.rows, pen.rows, rows_for_ui):
        for _r in _rows:
            _r.setdefault("selection", "strict")

    _ensure_output_dirs()

    rejected_rows: List[Dict] = [dict(r) for r in (rejected_stage_ui or [])]

    if rejected_rows:
        deduped_rejected: List[Dict] = []
        seen = set()
        step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 5.0
        tol = max(0.51 * step, 1e-6)
        for row in rejected_rows:
            age = float(row.get("age_ma", np.nan))
            reason = str(row.get("reason", ""))
            if np.isfinite(age):
                age_key = round(age / tol)
            else:
                age_key = None
            key = (reason, age_key)
            if key in seen:
                continue
            seen.add(key)
            deduped_rejected.append(row)
        rejected_rows = deduped_rejected

    if merge_nearby:
        pre_merge_ui = [dict(r) for r in rows_for_ui]
        rows_for_ui = _collapse_ci_clusters(rows_for_ui)
        for surf in (raw, pen):
            surf.rows = _collapse_ci_clusters(surf.rows)
        _capture_rejected_step(pre_merge_ui, rows_for_ui, rejected_rows, "merged_overlapping_candidates", ages_ma)

    # Recompute winner-vote support from per-run optima while preserving
    # direct per-run peak support as the primary "support" metric.
    support_floor = max(float(FS_SUPPORT), 0.03)
    optima_ma_ui_vote = raw.optima_ma if ui_surface == "RAW" else pen.optima_ma

    def _filter_pipeline(label=None):
        """Apply support recompute + filter to raw, pen, and rows_for_ui."""
        nonlocal rows_for_ui
        for surf in (raw, pen):
            surf.rows = _recompute_winner_support(surf.rows, surf.optima_ma, ages_ma, min_support=None)
            surf.rows = _apply_support_filter(surf.rows, support_floor, support_filter_mode)
        rows_for_ui = _recompute_winner_support(rows_for_ui, optima_ma_ui_vote, ages_ma, min_support=None)
        pre = [dict(r) for r in rows_for_ui]
        rows_for_ui = _apply_support_filter(rows_for_ui, support_floor, support_filter_mode)
        if label:
            _capture_rejected_step(pre, rows_for_ui, rejected_rows, label, ages_ma)

    _filter_pipeline("low_support")

    # In no-merge mode, remove near-identical plateau duplicates and re-vote so
    # winner support reflects the deduped set.
    if (not merge_nearby) and PLATEAU_DEDUPE:
        for surf in (raw, pen):
            surf.rows = _plateau_dedupe_rows(surf.rows, ages_ma)
        pre_dedupe_ui = [dict(r) for r in rows_for_ui]
        rows_for_ui = _plateau_dedupe_rows(rows_for_ui, ages_ma)
        _capture_rejected_step(pre_dedupe_ui, rows_for_ui, rejected_rows, "plateau_duplicate", ages_ma)
        _filter_pipeline("low_support")

    rows_for_ui, rejected_rows = _apply_guards_and_fallbacks(
        sample, settings, runs, raw, pen,
        rows_for_ui, rejected_rows,
        ages_ma, S_view, S_runs_view,
        view_which, ui_surface,
        support_floor,
    )

    _publish_results(
        signals, sample, progress, settings, runs, raw, pen,
        rows_for_ui, rejected_rows, ages_ma, ages_y, S_view,
        optimalAge, optimalAge_ui, optimalAge_legacy, lower95, upper95, opt_all,
        meanD, meanP, meanInv, meanSc,
    )
