"""Boundary-dominance handling for CDC peak catalogues."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from process.cdcConfig import FS_SUPPORT, RMIN_RUNS

_BOUNDARY_NEAR_GRID_STEPS = 8.0
_BOUNDARY_FAR_GRID_STEPS = 5.0


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
        support_low=float(young_edge),
        support_high=float(upper),
        stability_low=float(young_edge),
        stability_high=float(upper),
        support=float(support),
        direct_support=float(support),
        winner_support=float(support),
        ci_method="stability_bounds",
        ci_interpretation="bootstrap_percentile_stability_bounds_of_edge_optima",
        stability_method="edge_optima_percentile",
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
