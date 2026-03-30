# utils/peakHelpers.py
"""
Peak-finding and formatting utilities for the CDC pipeline.

Functions:
  • refine_peak(x, y, idx)         – parabolic sub-grid refinement
  • fmt_peak_stats(stats)          – "med ± half (support%)" formatter
  • find_peaks_1d_prom(...)        – adaptive prominence peak finder
  • summed_ks_surface(runs, ...)   – mean 'goodness' curve across runs
  • adaptive_peaks(x, y_raw, ...)  – tier-aware peak picking
  • keep_if_supported(...)         – bootstrap-style support gate
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "refine_peak",
    "fmt_peak_stats",
    "find_peaks_1d_prom",
    "summed_ks_surface",
    "adaptive_peaks",
    "keep_if_supported",
]

_EPS = 1e-12

# ---------------------------------------------------------------------------

def refine_peak(x: np.ndarray, y: np.ndarray, idx: int) -> float:
    """Parabolic vertex through (idx-1, idx, idx+1) -> refined x (identical math to old)."""
    x = np.asarray(x, float); y = np.asarray(y, float); idx = int(idx)
    if idx <= 0 or idx >= len(x) - 1:
        return float(x[idx])
    x0, x1, x2 = x[idx-1:idx+2]
    y0, y1, y2 = y[idx-1:idx+2]
    denom = (y0 - 2.0*y1 + y2)
    if denom == 0.0:
        return float(x1)
    delta = 0.5 * (y0 - y2) / denom
    return float(x1 + delta * (x2 - x1))

def fmt_peak_stats(stats) -> str:
    """Format (median, ci_low, ci_high, support) as 'med ± half (support%)' (old style)."""
    parts = []
    for med, lo, hi, sup in stats:
        med = float(med); lo = float(lo); hi = float(hi); sup = float(sup)
        half = max(med - lo, hi - med)
        parts.append(f"{med:.0f} ± {half:.0f} ({sup*100:.0f}%)")
    return "; ".join(parts)

def find_peaks_1d_prom(
    x, y, *,
    height_frac: float = 0.10,
    promin_frac: float = 0.05,
    min_distance: int = 3,
    local_win:   int = 15,
    use_local:   bool = False,
):
    """
    Return peak x-coordinates for a 1-D curve y(x) using absolute height and
    noise-adaptive prominence thresholds.

    use_local=False → global MAD (fast); True → rolling MAD over ±local_win.
    (Threshold algebra matches the old implementation.)
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if y.size < 3:
        return np.array([], float)

    # --- noise estimate σ̂
    if use_local:
        pad = int(local_win)
        y_pad = np.pad(y, pad, mode="edge")
        win = sliding_window_view(y_pad, 2*pad + 1)
        mad = np.median(np.abs(win - np.median(win, axis=1)[:, None]), axis=1)
        sigma_thr = 1.4826 * np.percentile(mad, 90)
    else:
        mad_all = np.median(np.abs(y - np.median(y)))
        sigma_thr = 1.4826 * mad_all

    # --- thresholds (scalar, old semantics)
    abs_height = float(height_frac) * float(np.nanmax(y))
    abs_prom   = max(float(promin_frac) * float(np.ptp(y)),
                     float(promin_frac) * float(sigma_thr))

    pk, _ = find_peaks(
        y,
        height=abs_height,
        prominence=abs_prom,
        distance=int(min_distance),
    )
    return x[pk]

def summed_ks_surface(runs, transform: str = "1-D", smooth_sigma: float = 1):
    """
    Average the 'goodness' curves over all Monte-Carlo runs.

    Returns (ages_ma, S) where:
      • ages_ma : 1-D array of rim-age grid nodes (Ma)
      • S       : averaged 'goodness' = 1 – KS-D (optionally Gaussian-smoothed)

    Old-first approach:
      1) Use run.ks_surface.{ages, goodness()} when present.
      2) Fallback: rebuild S from statistics_by_pb_loss_age (KS-D).
    """
    if not runs:
        return None, None

    # --- prefer ks_surface when present (old path)
    try:
        ages_ma = np.asarray(runs[0].ks_surface.ages, float)
        stack = np.vstack([
            np.asarray(r.ks_surface.goodness(transform), float)
            for r in runs if getattr(r, "ks_surface", None) is not None
        ])
        if stack.size == 0:
            raise ValueError("empty stack")
        S = np.nanmean(stack, axis=0)
        if smooth_sigma:
            S = gaussian_filter1d(S, float(smooth_sigma))
        return ages_ma, S
    except Exception:
        pass

    # --- fallback: rebuild from stored stats (same algebra as old)
    try:
        ages_y = np.array(sorted(runs[0].statistics_by_pb_loss_age.keys()), float)
        ages_ma = ages_y / 1e6
        stack = []
        for r in runs:
            row = [1.0 - float(r.statistics_by_pb_loss_age[a].test_statistics[0]) for a in ages_y]
            stack.append(row)
        S = np.nanmean(np.asarray(stack, float), axis=0)
        if smooth_sigma:
            S = gaussian_filter1d(S, float(smooth_sigma))
        return ages_ma, S
    except Exception:
        return None, None

def adaptive_peaks(x, y_raw, tier, smooth: float = 2):
    """
    Return sub-grid peak centres (Ma) from a CDC goodness curve.

    * Tier A/B – 4% of global P95–P05 amplitude as prominence.
    * Tier C   – 60% of local MAD (robust σ̂) as prominence.
    """
    x = np.asarray(x, float)
    y = gaussian_filter1d(np.asarray(y_raw, float), sigma=float(smooth))

    if str(tier).upper() == "C":
        win = 15  # ±15 nodes
        pad = np.pad(y, win, mode="edge")
        sw  = sliding_window_view(pad, 2*win + 1)
        mad = np.median(np.abs(sw - np.median(sw, axis=1)[:, None]), axis=1)
        amp = 1.4826 * np.percentile(mad, 90)  # robust σ̂ of ridge
        prom = 0.60 * amp
    else:
        amp  = np.percentile(y, 95) - np.percentile(y, 5)
        prom = 0.04 * amp

    pk, _ = find_peaks(y, prominence=float(prom), distance=3)

    return np.array([refine_peak(x, y, int(i)) for i in pk], float)

def keep_if_supported(
    x_grid,
    boot,
    peaks_ma,
    *,
    delta: float = 40.0,   # ±40 Ma voting window
    min_frac: float = 0.25
):
    """
    Accept a candidate peak only if ≥ min_frac of bootstrap curves exhibit
    *some* local maximum within ±delta Ma of that peak.
    """
    x_grid = np.asarray(x_grid, float)
    boot   = np.asarray(boot,   float)
    peaks_ma = np.asarray(peaks_ma, float).ravel()

    keep = []
    for pk in peaks_ma:
        lo, hi = float(pk) - float(delta), float(pk) + float(delta)
        votes = 0
        for curve in boot:
            loc, _ = find_peaks(curve)
            if loc.size and ((x_grid[loc] >= lo) & (x_grid[loc] <= hi)).any():
                votes += 1
        if boot.shape[0] > 0 and (votes / float(boot.shape[0])) >= float(min_frac):
            keep.append(pk)
    return np.array(keep, float)
