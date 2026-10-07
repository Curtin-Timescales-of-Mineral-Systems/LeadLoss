from __future__ import annotations

import numpy as np
import scipy as sp

from utils import config


def build_density_heatmap_from_runs(ages_ma, S_runs, resolution=None):
    """Build the displayed heatmap density matrix from run-level goodness curves."""
    ages_ma = np.asarray(ages_ma, float)
    S_runs = np.asarray(S_runs, float)
    resolution = config.HEATMAP_RESOLUTION if resolution is None else int(resolution)

    if ages_ma.ndim != 1 or ages_ma.size == 0:
        raise ValueError("ages_ma must be a non-empty 1-D array")
    if S_runs.ndim != 2 or S_runs.shape[1] != ages_ma.size:
        raise ValueError("S_runs must be a 2-D array with one column per age")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    D_runs = 1.0 - S_runs
    D_runs = np.clip(D_runs, 0.0, 1.0)
    y_edges = np.linspace(0.0, 1.0, resolution + 1)
    data = np.zeros((resolution, ages_ma.size), float)
    prev_hist = None

    for col in range(ages_ma.size):
        vals = np.asarray(D_runs[:, col], float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            if prev_hist is None:
                hist = np.zeros(resolution, float)
                hist[resolution // 2] = 1.0
            else:
                hist = prev_hist.copy()
        else:
            hist, _ = np.histogram(vals, bins=y_edges)
            hist = hist.astype(float)
            total = float(np.sum(hist))
            if total > 0.0:
                hist /= total
            elif prev_hist is not None:
                hist = prev_hist.copy()
            else:
                hist[resolution // 2] = 1.0
        prev_hist = hist
        data[:, col] = hist

    if ages_ma.size == 1:
        step = 1.0
    else:
        step = float(np.median(np.diff(ages_ma)))
        if not np.isfinite(step) or step <= 0.0:
            step = 1.0
    x_edges = np.empty(ages_ma.size + 1, float)
    x_edges[1:-1] = 0.5 * (ages_ma[:-1] + ages_ma[1:])
    x_edges[0] = ages_ma[0] - 0.5 * step
    x_edges[-1] = ages_ma[-1] + 0.5 * step

    return x_edges, y_edges, data


def build_density_heatmap_from_run_columns(runs, settings, resolution=None):
    """Build the displayed heatmap density matrix from cached per-run columns."""
    resolution = config.HEATMAP_RESOLUTION if resolution is None else int(resolution)
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    min_age_ma = float(settings.minimumRimAge) / 1e6
    max_age_ma = float(settings.maximumRimAge) / 1e6
    if not np.isfinite(min_age_ma) or not np.isfinite(max_age_ma) or max_age_ma <= min_age_ma:
        raise ValueError("settings must define a valid age range")

    col_data = [[] for _ in range(resolution)]
    for run in runs or []:
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
            col_data[col].append(vf)

    cache = {}
    data = np.zeros((resolution, resolution), float)
    prev_mean = None
    prev_std = None
    y_edges = np.linspace(0.0, 1.0, resolution + 1)

    for col in range(resolution):
        vals = np.asarray(col_data[col], float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            if prev_mean is None:
                mean, std_dev = 0.5, 0.0
            else:
                mean, std_dev = prev_mean, prev_std
        else:
            mean = float(np.median(vals))
            std_dev = float(np.std(vals))
        mean = float(np.clip(mean, 0.0, 1.0))
        if std_dev < 1e-7:
            std_dev = 0.0
        prev_mean, prev_std = mean, std_dev

        key = (mean, std_dev)
        if key not in cache:
            if std_dev == 0.0:
                mean_row = (resolution - 1) if mean >= 1.0 else int(mean * resolution)
                result = np.array([1 if i >= mean_row else 0 for i in range(resolution + 1)], float)
            else:
                rv = sp.stats.norm(mean, std_dev)
                result = rv.cdf(y_edges)
            cache[key] = result

        cdfs = cache[key]
        data[:, col] = cdfs[1:] - cdfs[:-1]

    x_edges = np.linspace(min_age_ma, max_age_ma, resolution + 1)
    return x_edges, y_edges, data


def calculateHeatmapData(signals, runs, settings, request_id=None):
    """Aggregate per-run heatmap columns into a single probability heatmap."""
    _, _, data = build_density_heatmap_from_run_columns(runs, settings)

    # Optional request_id lets the UI ignore stale async heatmap frames.
    if request_id is None:
        signals.progress(data, settings)
    else:
        signals.progress(request_id, data, settings)
