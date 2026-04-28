"""Compatibility shim for legacy processing imports.

New code should import the CDC pipeline from ``process.cdc.pipeline``.
This module remains only to avoid breaking older app entry points.
"""

from __future__ import annotations

from process.cdcPipeline import ProgressType, processSamples
from process.cdcHeatmap import calculateHeatmapData

__all__ = ["ProgressType", "processSamples", "calculateHeatmapData"]
