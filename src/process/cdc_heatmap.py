"""Heatmap post-processing helper.

This is kept separate so the core CDC pipeline isn't cluttered with UI-specific
visualisation aggregation code.
"""

from __future__ import annotations

import numpy as np
import scipy as sp

from utils import config


def calculateHeatmapData(signals, runs, settings, request_id=None):
    """Aggregate per-run heatmap columns into a single probability heatmap."""
    resolution = config.HEATMAP_RESOLUTION

    colData = [[] for _ in range(resolution)]
    for run in runs:
        if signals.halt():
            return
        if run is None:
            continue
        row = getattr(run, "heatmapColumnData", None)
        if row is None:
            continue
        for col in range(resolution):
            if col >= len(row):
                continue
            v = row[col]
            if v is None:
                continue
            vf = float(v)
            if not np.isfinite(vf):
                continue
            colData[col].append(vf)

    cache = {}
    data = [[0 for _ in range(resolution)] for _ in range(resolution)]
    prev_mean = None
    prev_std = None
    for col in range(resolution):
        if signals.halt():
            return
        vals = np.asarray(colData[col], float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            if prev_mean is None:
                mean, stdDev = 0.5, 0.0
            else:
                mean, stdDev = prev_mean, prev_std
        else:
            # Center each column on the run-median so the heatmap ridge aligns
            # with the ensemble median curve used in the lower panel.
            mean = float(np.median(vals))
            stdDev = float(np.std(vals))
        mean = float(np.clip(mean, 0.0, 1.0))
        if stdDev < 10 ** -7:
            stdDev = 0
        prev_mean, prev_std = mean, stdDev

        if (mean, stdDev) not in cache:
            if stdDev == 0:
                meanRow = (resolution - 1) if mean >= 1.0 else int(mean * resolution)
                result = [1 if i >= meanRow else 0 for i in range(resolution + 1)]
            else:
                rv = sp.stats.norm(mean, stdDev)
                result = rv.cdf(np.linspace(0, 1, resolution + 1))
            cache[(mean, stdDev)] = result

        cdfs = cache[(mean, stdDev)]
        for row in range(resolution):
            data[row][col] = cdfs[row + 1] - cdfs[row]

    # Optional request_id lets the UI ignore stale async heatmap frames.
    if request_id is None:
        signals.progress(data, settings)
    else:
        signals.progress(request_id, data, settings)
