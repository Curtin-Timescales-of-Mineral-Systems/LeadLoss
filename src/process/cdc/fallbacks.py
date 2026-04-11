"""Fallback and display-alignment helpers for CDC peak catalogues."""

from __future__ import annotations

from typing import Dict

import numpy as np

from process.cdc_config import ENS_DELTA_MIN

_DEGENERATE_CI_GRID_FRAC = 0.75
_SINGLE_CREST_PROM_FRAC = 0.03


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


def _normalise_ci_bounds(rows_for_ui, ages_ma):
    if not rows_for_ui:
        return rows_for_ui

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
    return fixed


def _remove_edge_degenerate_ci(rows_for_ui, rejected_rows, ages_ma, support_floor, capture_rejected_step):
    if not rows_for_ui:
        return rows_for_ui, rejected_rows

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
    capture_rejected_step(pre_clean_ui, rows_for_ui, rejected_rows, "edge_degenerate_ci", ages_ma)
    return rows_for_ui, rejected_rows


def _ensure_age_within_ci(rows_for_ui):
    for r in rows_for_ui:
        a = float(r["age_ma"])
        if not (float(r["ci_low"]) <= a <= float(r["ci_high"])):
            r["ci_low"] = min(float(r["ci_low"]), a)
            r["ci_high"] = max(float(r["ci_high"]), a)
    return rows_for_ui


def _filter_overwide_ci(rows_for_ui, raw, pen, rejected_rows, ages_ma, support_floor, capture_rejected_step):
    if not rows_for_ui:
        return rows_for_ui, rejected_rows

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
    capture_rejected_step(pre_width_ui, rows_for_ui, rejected_rows, "wide_ci", ages_ma)
    raw.rows = _keep_same(raw.rows, rows_for_ui)
    pen.rows = _keep_same(pen.rows, rows_for_ui)
    return rows_for_ui, rejected_rows
