"""Public CDC pipeline facade.

External callers should continue to import:
- `process.cdc_pipeline`
- `process.processing`

The implementation now lives in the internal `process.cdc` package. This file
exists as the stable public API boundary for the GUI, tests, and any external
scripts that import the pipeline directly.
"""

from __future__ import annotations

from process.cdc.pipeline import ProgressType, processSamples
from process.cdc.filtering import _collapse_ci_clusters, _recompute_winner_support
from process.cdc.guards import _single_crest_fallback_row, _snap_rows_to_curve
from process.cdc.surfaces import _is_effectively_monotonic, _smooth_frac_for_grid

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
