"""Post-catalogue safety rules and fallback handling for CDC peak outputs.

This module does not build peaks. It applies defensive logic after the
candidate catalogue exists:
- suppress boundary-dominated artefacts
- inject explicit recent-boundary modes when appropriate
- snap rows back to the displayed curve
- create a single-crest fallback when peak detection yields nothing usable
"""

from __future__ import annotations

import numpy as np

from process.cdc.boundary import (
    _apply_boundary_dominance_guard,
    _inject_recent_boundary_mode,
    _recent_boundary_mode_row,
)
from process.cdc.fallbacks import (
    _ensure_age_within_ci,
    _filter_overwide_ci,
    _keep_same,
    _normalise_ci_bounds,
    _remove_edge_degenerate_ci,
    _single_crest_fallback_row,
    _snap_rows_to_curve,
)
from process.cdc.filtering import (
    _capture_rejected_step,
    _row_match_index,
    _step_ma_from_grid,
)
from utils import config


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

    rows_for_ui = _normalise_ci_bounds(rows_for_ui, ages_ma)
    rows_for_ui, rejected_rows = _remove_edge_degenerate_ci(
        rows_for_ui, rejected_rows, ages_ma, support_floor, _capture_rejected_step,
    )
    rows_for_ui = _ensure_age_within_ci(rows_for_ui)
    rows_for_ui, rejected_rows = _filter_overwide_ci(
        rows_for_ui, raw, pen, rejected_rows, ages_ma, support_floor, _capture_rejected_step,
    )

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
