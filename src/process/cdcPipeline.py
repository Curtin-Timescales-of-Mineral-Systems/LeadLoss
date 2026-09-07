"""Compatibility shim for historical camel-case CDC imports.

New code should import from ``process.cdc.pipeline`` and the specific
``process.cdc.*`` submodules instead of this re-export layer.
"""

from __future__ import annotations

from process.cdc.pipeline import ProgressType, processSamples
from process.cdc.filtering import _collapse_ci_clusters, _recompute_winner_support
from process.cdc.fallbacks import _conditional_single_crest_row, _single_crest_fallback_row
from process.cdc.guards import _snap_rows_to_curve
from process.cdc.surfaces import _is_effectively_monotonic, _smooth_frac_for_grid

__all__ = [
    "ProgressType",
    "processSamples",
    "_collapse_ci_clusters",
    "_is_effectively_monotonic",
    "_recompute_winner_support",
    "_conditional_single_crest_row",
    "_single_crest_fallback_row",
    "_smooth_frac_for_grid",
    "_snap_rows_to_curve",
]
