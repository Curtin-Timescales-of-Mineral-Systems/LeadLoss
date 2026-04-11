"""Low-level helpers for ensemble CDC peak picking."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import (
    find_peaks as _find_peaks,
    peak_prominences as _peak_prominences,
    peak_widths as _peak_widths,
)

find_peaks = _find_peaks
peak_prominences = _peak_prominences
peak_widths = _peak_widths

_EPS = 1e-12
_COARSE_SIGMA_GRID_FRAC = 0.03
_DEGENERATE_CI_GRID_FRAC = 0.75


def _step_from_grid(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if x.size >= 2:
        step = float(np.median(np.diff(x)))
        if np.isfinite(step) and step > 0.0:
            return step
    return 1.0


def _half_prominence_edges(x: np.ndarray, y: np.ndarray, pk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pk = np.asarray(pk, int)
    if x.size == 0 or y.size != x.size or pk.size == 0:
        return np.array([], float), np.array([], float)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="some peaks have a prominence of 0")
        prom, left_bases, right_bases = peak_prominences(y, pk)
        _, _, left_ips, right_ips = peak_widths(
            y,
            pk,
            rel_height=0.5,
            prominence_data=(prom, left_bases, right_bases),
        )

    idx = np.arange(x.size, dtype=float)
    return (
        np.asarray(np.interp(left_ips, idx, x), float),
        np.asarray(np.interp(right_ips, idx, x), float),
    )


def _append_diagnostic_peak(
    diagnostic_rows: Optional[List[Dict]],
    x: np.ndarray,
    y_ref: np.ndarray,
    idx: int,
    *,
    reason: str,
    direct_support: float = np.nan,
    winner_support: float = np.nan,
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
) -> None:
    if diagnostic_rows is None:
        return
    x = np.asarray(x, float)
    y_ref = np.asarray(y_ref, float)
    if x.size == 0 or y_ref.size != x.size:
        return

    j_ref = _crest_index(y_ref, int(idx), half_win=2)
    age = float(_parabolic_refine(x, y_ref, j_ref))
    step = _step_from_grid(x)
    lo = float(ci_low) if ci_low is not None else max(float(x[0]), age - step)
    hi = float(ci_high) if ci_high is not None else min(float(x[-1]), age + step)
    tol = max(0.51 * step, 1e-6)

    for row in diagnostic_rows:
        prev_age = float(row.get("age_ma", np.nan))
        if (str(row.get("reason", "")) == str(reason)) and np.isfinite(prev_age) and abs(prev_age - age) <= tol:
            return

    diagnostic_rows.append(
        dict(
            age_ma=age,
            ci_low=lo,
            ci_high=hi,
            direct_support=float(direct_support),
            winner_support=float(winner_support),
            reason=str(reason),
        )
    )


def _basin_bounds_from_peaks(y: np.ndarray, peak_idx: int, sorted_peaks: np.ndarray) -> Tuple[int, int]:
    y = np.asarray(y, float)
    peaks = np.asarray(sorted_peaks, int)
    n = y.size
    if n == 0:
        return 0, 0

    pos = int(np.searchsorted(peaks, int(peak_idx)))
    left_anchor = int(peaks[pos - 1]) if pos > 0 else 0
    right_anchor = int(peaks[pos + 1]) if pos < (peaks.size - 1) else (n - 1)

    if left_anchor >= int(peak_idx):
        left = max(0, int(peak_idx) - 1)
    else:
        left = int(left_anchor + np.argmin(y[left_anchor:int(peak_idx) + 1]))

    if right_anchor <= int(peak_idx):
        right = min(n - 1, int(peak_idx) + 1)
    else:
        right = int(int(peak_idx) + np.argmin(y[int(peak_idx):right_anchor + 1]))

    return max(0, left), min(n - 1, right)


def _estimate_window_support(
    lo_ma: float,
    hi_ma: float,
    per_run_peaks_list: List[np.ndarray],
    optima_ma: Optional[np.ndarray],
) -> Tuple[float, float]:
    R = max(len(per_run_peaks_list), 1)
    direct = 0
    for pr in per_run_peaks_list:
        arr = np.asarray(pr, float)
        if arr.size and np.any((arr >= lo_ma) & (arr <= hi_ma)):
            direct += 1

    winner = np.nan
    if optima_ma is not None:
        opts = np.asarray(optima_ma, float)
        opts = opts[np.isfinite(opts)]
        if opts.size:
            winner = float(np.mean((opts >= lo_ma) & (opts <= hi_ma)))

    return float(direct) / float(R), float(winner)


def _parabolic_refine(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Quadratic-vertex refinement around a local node, clamped to its bracket."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = y.size
    if k <= 0 or k >= n - 1:
        return float(x[k])
    x0, x1, x2 = float(x[k - 1]), float(x[k]), float(x[k + 1])
    y0, y1, y2 = float(y[k - 1]), float(y[k]), float(y[k + 1])
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) <= _EPS:
        return x1
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
    if abs(a) < 1e-12:
        return x1
    xv = -b / (2.0 * a)
    lo, hi = (x0, x2) if x0 <= x2 else (x2, x0)
    return float(min(max(xv, lo), hi))


def _crest_index(y: np.ndarray, k: int, half_win: int = 2) -> int:
    """Return a stable representative index for a local crest or flat top."""
    y = np.asarray(y, float)
    n = y.size
    a = max(1, int(k) - int(half_win))
    b = min(n - 2, int(k) + int(half_win))
    seg = y[a:b + 1]
    if seg.size == 0 or not np.isfinite(seg).any():
        return int(k)
    m = np.nanmax(seg)
    cand = np.where(np.isclose(seg, m, rtol=1e-12, atol=1e-15))[0]
    j_local = int(cand[len(cand) // 2])
    return int(a + j_local)


def _apply_plateau_onset_adjustment(
    age_out: float,
    peak_left_edge: float,
    peak_right_edge: float,
    *,
    base_age_mode: str,
    n_peaks: int,
    is_youngest: bool,
    total_span: float,
    mode: str,
    min_width_frac: float,
    min_right_left_ratio: float,
    blend_frac: float,
) -> Tuple[float, str, float, float]:
    """Shift broad older-tailed youngest peaks toward their onset."""
    age_mode = str(base_age_mode)
    peak_width_frac = (
        float(max(0.0, peak_right_edge - peak_left_edge)) / max(total_span, _EPS)
        if np.isfinite(peak_left_edge) and np.isfinite(peak_right_edge)
        else np.nan
    )
    right_left_ratio = np.nan

    if str(mode).lower() != "midpoint_left":
        return age_out, age_mode, peak_width_frac, right_left_ratio

    left_span = age_out - peak_left_edge if np.isfinite(peak_left_edge) else np.nan
    right_span = peak_right_edge - age_out if np.isfinite(peak_right_edge) else np.nan
    right_left_ratio = (
        float(right_span / max(left_span, _EPS))
        if np.isfinite(left_span) and np.isfinite(right_span)
        else np.nan
    )

    if (
        n_peaks > 1
        and is_youngest
        and np.isfinite(peak_left_edge)
        and np.isfinite(peak_width_frac)
        and peak_width_frac > float(min_width_frac)
        and np.isfinite(right_left_ratio)
        and right_left_ratio > float(min_right_left_ratio)
        and peak_left_edge < age_out
    ):
        blend = float(np.clip(blend_frac, 0.0, 1.0))
        age_out = float((1.0 - blend) * age_out + blend * peak_left_edge)
        age_mode = "plateau_onset_midpoint"

    return age_out, age_mode, peak_width_frac, right_left_ratio
