from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from process.cdc_config import (
    CATALOGUE_SURFACE,
    ENS_DELTA_MIN,
    FD_DIST_FRAC,
    FH_HEIGHT_FRAC,
    FP_PROM_FRAC,
    FR_RUN_REL,
    FS_SUPPORT,
    FW_WIN_FRAC,
    MONO_DY_EPS_FRAC,
    MONO_MAX_TURNS,
    PER_RUN_MIN_DIST,
    PER_RUN_MIN_WIDTH,
    PER_RUN_PROM_FRAC,
    PLATEAU_ONSET_BLEND_FRAC,
    PLATEAU_ONSET_MIN_RIGHT_LEFT_RATIO,
    PLATEAU_ONSET_MIN_WIDTH_FRAC,
    PLATEAU_ONSET_MODE,
    RMIN_RUNS,
    FV_VALLEY_FRAC,
    SMOOTH_FRAC,
    SMOOTH_MA,
)
from process.cdc.state import SurfaceState
from process.ensemble import build_ensemble_catalogue, robust_ensemble_curve


def _findOptimalIndex(valuesToCompare):
    """
    Select the best index with the published plateau-aware tie handling.
    """
    vals = list(valuesToCompare)
    if len(vals) == 0:
        return 0

    minIndex, minValue = min(enumerate(vals), key=lambda v: v[1])
    n = len(vals)

    startMinIndex = minIndex
    while startMinIndex > 0 and vals[startMinIndex - 1] == minValue:
        startMinIndex -= 1

    endMinIndex = minIndex
    while endMinIndex < n - 1 and vals[endMinIndex + 1] == minValue:
        endMinIndex += 1

    if (endMinIndex != n - 1 and startMinIndex != 0) or (endMinIndex == n - 1 and startMinIndex == 0):
        return (endMinIndex + startMinIndex) // 2
    if startMinIndex == 0:
        return 0
    return n - 1


def _smooth_frac_for_grid(ages_ma):
    """Convert `SMOOTH_MA` into a node fraction when set, else use `SMOOTH_FRAC`."""
    n = len(ages_ma)
    if n <= 1:
        return SMOOTH_FRAC
    if SMOOTH_MA > 0:
        step_ma = float(np.median(np.diff(ages_ma))) or 1e-9
        sigma_nodes = SMOOTH_MA / step_ma
        return min(0.25, sigma_nodes / n)
    return SMOOTH_FRAC


def _is_effectively_monotonic(y_curve, delta):
    """
    Return True when the smoothed ensemble curve is effectively monotonic.

    Tiny wiggles are ignored using a derivative epsilon scaled by ensemble
    dynamic range, so boundary-driven surfaces abstain rather than becoming
    fake discrete peaks.
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


def _raw_optimum_age_ma(run) -> float:
    """
    Return the per-run RAW optimum age (Ma) from `_raw_statistics_by_pb_loss_age`.

    Falls back to the penalised optimum if RAW data are unavailable.
    """
    stats_map = getattr(run, "_raw_statistics_by_pb_loss_age", None)
    if isinstance(stats_map, dict) and stats_map:
        ages = np.array(sorted(stats_map.keys()), float)
        dvals = np.array([stats_map[a].test_statistics[0] for a in ages], float)
        finite = np.isfinite(dvals)
        if finite.any():
            vals = np.where(finite, dvals, np.inf)
            idx = _findOptimalIndex(vals.tolist())
            return float(ages[idx] / 1e6)
    return float(getattr(run, "optimal_pb_loss_age", np.nan) / 1e6)


def _stack_goodness_from_stats_attr(runs, ages_y, stats_attr: str, which: str = "pen") -> np.ndarray:
    """
    Stack run-wise CDC goodness curves onto a common age grid.

    `which="pen"` maps to the penalised score (`1 - score`).
    `which="raw"` maps to the unpenalised KS goodness (`1 - D`).
    """
    ages_y = np.asarray(ages_y, float)
    out = np.full((len(runs), ages_y.size), np.nan, float)
    for i, run in enumerate(runs):
        stats_map = getattr(run, stats_attr, None)
        if not isinstance(stats_map, dict) or not stats_map:
            stats_map = getattr(run, "statistics_by_pb_loss_age", None)
        if not isinstance(stats_map, dict) or not stats_map:
            continue
        if which == "raw":
            vals = np.array([1.0 - stats_map[float(a)].test_statistics[0] for a in ages_y], float)
        else:
            vals = np.array([1.0 - stats_map[float(a)].score for a in ages_y], float)
        out[i, :] = vals
    return out


def _optimum_age_ma_from_stats_attr(run, stats_attr: str, which: str = "pen") -> float:
    """
    Return one run's optimum age (Ma) from a stats map attribute.

    `which="pen"` uses the penalised score surface.
    `which="raw"` uses the unpenalised KS D surface.
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
    Return the stats object at a run's optimum age from a stats map attribute.
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


def _build_global_catalogue_rows(
    sample_name: str,
    tier: str,
    ages_ma: np.ndarray,
    S_runs: np.ndarray,
    Smed: np.ndarray,
    *,
    smf: float,
    merge_nearby: bool,
    pickable: bool,
    optima_ma: np.ndarray,
    diagnostic_rows: Optional[List[Dict]] = None,
):
    """Build ensemble rows for one global surface (raw or penalised)."""
    if (not pickable) or (Smed.size == 0):
        return []

    diag_rows: List[Dict] = []
    rows = build_ensemble_catalogue(
        sample_name,
        tier,
        ages_ma,
        S_runs,
        orientation="max",
        smooth_frac=smf,
        f_d=FD_DIST_FRAC,
        f_p=FP_PROM_FRAC,
        f_v=FV_VALLEY_FRAC,
        f_w=FW_WIN_FRAC,
        w_min_nodes=3,
        support_min=FS_SUPPORT,
        r_min=RMIN_RUNS,
        f_r=FR_RUN_REL,
        per_run_prom_frac=PER_RUN_PROM_FRAC,
        per_run_min_dist=PER_RUN_MIN_DIST,
        per_run_min_width=PER_RUN_MIN_WIDTH,
        per_run_require_full_prom=False,
        plateau_onset_mode=PLATEAU_ONSET_MODE,
        plateau_onset_min_width_frac=PLATEAU_ONSET_MIN_WIDTH_FRAC,
        plateau_onset_min_right_left_ratio=PLATEAU_ONSET_MIN_RIGHT_LEFT_RATIO,
        plateau_onset_blend_frac=PLATEAU_ONSET_BLEND_FRAC,
        height_frac=FH_HEIGHT_FRAC,
        optima_ma=optima_ma,
        merge_per_hump=merge_nearby,
        merge_shoulders=merge_nearby,
        diagnostic_rows=diag_rows,
    ) or []

    if diagnostic_rows is not None:
        diagnostic_rows.extend(dict(r) for r in diag_rows)
    return rows


def _compute_optimal_age_ci(raw, pen, prefer_pen, runs):
    """
    Empirical 2.5/97.5 percentile interval for the optimal Pb-loss age.
    """
    optima_ma_primary = pen.optima_ma if prefer_pen else raw.optima_ma
    opt_all = np.sort(np.asarray(optima_ma_primary[np.isfinite(optima_ma_primary)] * 1e6, float))
    if opt_all.size == 0:
        if prefer_pen:
            opt_all = np.sort(np.asarray([r.optimal_pb_loss_age for r in runs], float))
        else:
            opt_all = np.sort(np.asarray([_raw_optimum_age_ma(r) * 1e6 for r in runs], float))
    n = opt_all.size
    if n:
        lower95 = float(opt_all[int(np.floor(0.025 * n))])
        upper95 = float(opt_all[int(np.ceil(0.975 * n)) - 1])
    else:
        lower95 = upper95 = float("nan")
    return lower95, upper95, opt_all


def _compute_optimal_age(raw, pen, prefer_pen, ages_y):
    """
    Optimal Pb-loss age from the mean penalised goodness surface.
    """
    S_runs_primary = pen.S_runs if prefer_pen else raw.S_runs
    sum_good = np.nansum(S_runs_primary, axis=0)
    cnt_good = np.sum(np.isfinite(S_runs_primary), axis=0)
    mean_good = np.divide(
        sum_good,
        cnt_good,
        out=np.full_like(sum_good, np.nan, dtype=float),
        where=cnt_good > 0,
    )
    mean_primary = 1.0 - mean_good
    mean_primary = np.where(np.isfinite(mean_primary), mean_primary, np.inf)
    optimal_idx = _findOptimalIndex(mean_primary.tolist())
    optimalAge = float(ages_y[optimal_idx])
    S_optimal_curve = 1.0 - mean_primary
    return optimalAge, S_optimal_curve, mean_primary


def _compute_mean_stats(runs, primary_which, prefer_pen):
    """Mean D, p-value, invalid count and score at each run's own optimum."""
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


def _build_surface_states(settings, runs, ages_y, ages_ma, abstain_on_monotonic):
    """Build raw and penalised global surface states from the run stack."""
    smf = _smooth_frac_for_grid(ages_ma)

    optima_raw = np.array(
        [_optimum_age_ma_from_stats_attr(r, "_all_statistics_by_pb_loss_age", which="raw") for r in runs], float,
    )
    S_runs_raw = _stack_goodness_from_stats_attr(runs, ages_y, "_all_statistics_by_pb_loss_age", which="raw")
    Smed_raw, Delta_raw, _ = robust_ensemble_curve(S_runs_raw, smooth_frac=smf)
    mono_raw = _is_effectively_monotonic(Smed_raw, Delta_raw)

    optima_pen = np.array(
        [_optimum_age_ma_from_stats_attr(r, "_all_statistics_by_pb_loss_age", which="pen") for r in runs], float,
    )
    S_runs_pen = _stack_goodness_from_stats_attr(runs, ages_y, "_all_statistics_by_pb_loss_age", which="pen")
    Smed_pen, Delta_pen, _ = robust_ensemble_curve(S_runs_pen, smooth_frac=smf)
    mono_pen = _is_effectively_monotonic(Smed_pen, Delta_pen)

    raw = SurfaceState(
        S_runs=S_runs_raw,
        Smed=Smed_raw,
        Delta=Delta_raw,
        mono=mono_raw,
        pickable=(Delta_raw >= ENS_DELTA_MIN) and ((not abstain_on_monotonic) or (not mono_raw)),
        optima_ma=optima_raw,
    )
    pen = SurfaceState(
        S_runs=S_runs_pen,
        Smed=Smed_pen,
        Delta=Delta_pen,
        mono=mono_pen,
        pickable=(Delta_pen >= ENS_DELTA_MIN) and ((not abstain_on_monotonic) or (not mono_pen)),
        optima_ma=optima_pen,
    )
    return smf, raw, pen


def _initialise_surface_view_state(sample, settings, raw, pen, primary_which):
    """Record surface diagnostics and choose the default surface shown in the UI."""
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
    return ui_surface, S_view
