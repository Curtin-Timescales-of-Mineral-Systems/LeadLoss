"""Re-exports CDC processing entry points for backwards compatibility."""

from __future__ import annotations

from process.cdcPipeline import ProgressType, processSamples
from process.cdcHeatmap import calculateHeatmapData

__all__ = ["ProgressType", "processSamples", "calculateHeatmapData"]
