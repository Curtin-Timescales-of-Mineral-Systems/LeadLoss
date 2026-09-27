"""Compatibility shim for historical snake-case CDC imports.

New code should import from ``process.cdc.pipeline`` and the specific
``process.cdc.*`` submodules instead of this re-export layer.
"""

from __future__ import annotations

from process.cdcPipeline import (
    ProgressType,
    _collapse_ci_clusters,
    _is_effectively_monotonic,
    _recompute_winner_support,
    _single_crest_fallback_row,
    _smooth_frac_for_grid,
    _snap_rows_to_curve,
    processSamples,
)

__all__ = [
    "ProgressType",
    "processSamples",
    "_collapse_ci_clusters",
    "_is_effectively_monotonic",
    "_recompute_winner_support",
    "_single_crest_fallback_row",
    "_smooth_frac_for_grid",
    "_snap_rows_to_curve",
]
