"""Backwards-compatible public entry points.

The original repository used `process.processing` as the main module for CDC.
To keep imports stable for the GUI/controller, we keep this file as a thin
re-export layer and move the implementation into smaller modules.

Public API
----------
- processSamples(signals, samples)
- calculateHeatmapData(signals, runs, settings)
- ProgressType (Enum used by the signals bus)
"""

from __future__ import annotations

from process.cdc_pipeline import ProgressType, processSamples
from process.cdc_heatmap import calculateHeatmapData

__all__ = ["ProgressType", "processSamples", "calculateHeatmapData"]
