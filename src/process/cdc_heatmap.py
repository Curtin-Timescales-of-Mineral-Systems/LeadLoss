"""Heatmap post-processing helper.

This is kept separate so the core CDC pipeline isn't cluttered with UI-specific
visualisation aggregation code.
"""

from __future__ import annotations

import numpy as np
import scipy as sp

from utils import config


def calculateHeatmapData(signals, runs, settings):
    """Aggregate per-run heatmap columns into a single probability heatmap."""
    resolution = config.HEATMAP_RESOLUTION

    colData = [[] for _ in range(resolution)]
    for run in runs:
        if run is None:
            continue
        for col in range(resolution):
            colData[col].append(run.heatmapColumnData[col])

    cache = {}
    data = [[0 for _ in range(resolution)] for _ in range(resolution)]
    for col in range(resolution):
        if len(colData[col]) == 0:
            continue

        mean = np.mean(colData[col])
        stdDev = np.std(colData[col])
        if stdDev < 10 ** -7:
            stdDev = 0

        if (mean, stdDev) not in cache:
            if stdDev == 0:
                meanRow = (resolution - 1) if mean == 1.0 else int(mean * resolution)
                result = [1 if i >= meanRow else 0 for i in range(resolution + 1)]
            else:
                rv = sp.stats.norm(mean, stdDev)
                result = rv.cdf(np.linspace(0, 1, resolution + 1))
            cache[(mean, stdDev)] = result

        cdfs = cache[(mean, stdDev)]
        for row in range(resolution):
            data[row][col] = cdfs[row + 1] - cdfs[row]

    signals.progress(data, settings)
