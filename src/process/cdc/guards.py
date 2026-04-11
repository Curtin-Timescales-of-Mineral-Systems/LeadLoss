from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from process.cdc_config import ENS_DELTA_MIN, FS_SUPPORT, RMIN_RUNS
from process.cdc.filtering import (
    _capture_rejected_step,
    _row_match_index,
    _step_ma_from_grid,
)
from utils import config

_BOUNDARY_NEAR_GRID_STEPS = 8.0
_BOUNDARY_FAR_GRID_STEPS = 5.0
_DEGENERATE_CI_GRID_FRAC = 0.75
_SINGLE_CREST_PROM_FRAC = 0.03


def _apply_boundary_dominance_guard(rows, optima_ma, ages_ma):
    """Suppress near-edge peaks when per-run optima are boundary-dominated."""
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

    edge_band = max(1.0 * step, 10.0)
    lo_frac = float(np.mean(opts <= (lo + edge_band)))
    hi_frac = float(np.mean(opts >= (hi - edge_band)))
    side = "low" if lo_frac >= hi_frac else "high"
    edge_frac = lo_frac if lo_frac >= hi_frac else hi_frac
    if edge_frac < 0.70:
        return rows, None

    opt_med = float(np.median(opts))
    span = max(hi - lo, step)
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
        boundary_support = float(row_boundary.get("direct_support", row_boundary.get("support", 0.0)))
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


def _single_crest_fallback_row(ages_ma, S_curve, optima_ma, min_support):
    """Conservative fallback when strict peak gating abstains."""
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
    rough = float(np.nanmedian(np.abs(np.diff(y)))) if y.size >= 3 else 0.0
    prom_min = max(_SINGLE_CREST_PROM_FRAC * delta, 3.0 * rough, 0.008)
    weak_side_min = max(rough, 0.002)
    if prom_one_sided < prom_min:
        return None
    if min(left_lift, right_lift) < weak_side_min:
        return None

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
    """Snap plotted marker ages only when the nearest crest is genuinely close."""
    if not rows:
        return []

    x = np.asarray(ages_ma, float)
    y = np.asarray(S_view, float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return [dict(r) for r in rows]

    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return [dict(r) for r in rows]

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

    snapped = [None] * len(rows)
    used = np.zeros(maxima.size, dtype=bool)
    step = float(np.median(np.diff(x))) if x.size >= 2 else 1.0
    lo_grid = float(np.nanmin(x[finite]))
    hi_grid = float(np.nanmax(x[finite]))
    order = np.argsort([float(dict(r).get("age_ma", np.nan)) for r in rows])

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

    return [r for r in snapped if isinstance(r, dict)]


def _keep_same(rows, keep):
    if not rows or not keep:
        return []
    keep_ages = {float(r["age_ma"]) for r in keep}
    return [r for r in rows if float(r.get("age_ma", float("nan"))) in keep_ages]


def _apply_guards_and_fallbacks(
    sample, settings, runs, raw, pen,
    rows_for_ui, rejected_rows,
    ages_ma, S_view, S_runs_view,
    view_which, ui_surface,
    support_floor,
):
    """Boundary guards, CI cleanup, wide-CI filter, and fallback handling."""
    optima_ma_display = raw.optima_ma if view_which == "raw" else pen.optima_ma

    pre_boundary_ui = [dict(r) for r in rows_for_ui]
    rows_for_ui, boundary_reason = _apply_boundary_dominance_guard(rows_for_ui, optima_ma_display, ages_ma)
    if boundary_reason is not None:
        _capture_rejected_step(pre_boundary_ui, rows_for_ui, rejected_rows, boundary_reason, ages_ma)
        raw.rows = []
        pen.rows = []
        sample.ensemble_abstain_reason = boundary_reason

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

    if rows_for_ui:
        step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 5.0
        min_age, max_age = float(ages_ma[0]), float(ages_ma[-1])
        fixed = []
        for r in rows_for_ui:
            a = float(r["age_ma"])
            lo = float(r["ci_low"])
            hi = float(r["ci_high"])
            w = max(hi - lo, step)
            if (a < lo) or (a > hi):
                lo, hi = a - 0.5 * w, a + 0.5 * w
                lo, hi = max(lo, min_age), min(hi, max_age)
                if (hi - lo) < step:
                    lo, hi = max(a - step, min_age), min(a + step, max_age)
            fixed.append(dict(r, ci_low=lo, ci_high=hi))
        rows_for_ui = fixed

    if rows_for_ui:
        step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 5.0
        min_age, max_age = float(ages_ma[0]), float(ages_ma[-1])
        pre_clean_ui = [dict(r) for r in rows_for_ui]
        cleaned = []
        for r in rows_for_ui:
            a = float(r["age_ma"])
            lo = float(r["ci_low"])
            hi = float(r["ci_high"])
            if (hi - lo) < step:
                lo, hi = a - step, a + step
            near_edge = (a - min_age) <= step or (max_age - a) <= step
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
            r["ci_low"] = min(float(r["ci_low"]), a)
            r["ci_high"] = max(float(r["ci_high"]), a)

    if rows_for_ui:
        total_span = float(ages_ma[-1] - ages_ma[0])
        max_ci_frac = 0.5
        pre_width_ui = [dict(r) for r in rows_for_ui]
        filtered = []
        for r in rows_for_ui:
            if str(r.get("mode", "")) == "recent_boundary":
                filtered.append(r)
                continue
            width = float(r["ci_high"] - r["ci_low"])
            score = float(r.get("filter_support", r.get("support", 0.0)))
            if width > max_ci_frac * total_span and score < max(support_floor, 0.25):
                continue
            filtered.append(r)
        rows_for_ui = filtered
        _capture_rejected_step(pre_width_ui, rows_for_ui, rejected_rows, "wide_ci", ages_ma)
        raw.rows = _keep_same(raw.rows, rows_for_ui)
        pen.rows = _keep_same(pen.rows, rows_for_ui)

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

    if not rows_for_ui and sample.ensemble_abstain_reason is None:
        sample.ensemble_abstain_reason = "no_supported_peaks"

    if isinstance(getattr(sample, "ensemble_surface_flags", None), dict):
        sample.ensemble_surface_flags["view_surface_source"] = "global_all"
    sample.display_heatmap_ages_ma = np.asarray(ages_ma, float)
    sample.display_heatmap_runs_S = np.asarray(S_runs_view, float)

    for run in runs:
        run._heatmap_view_which = view_which
        run.createHeatmapData(settings.minimumRimAge, settings.maximumRimAge, config.HEATMAP_RESOLUTION)

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

