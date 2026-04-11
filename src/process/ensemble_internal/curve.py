"""Run-level and ensemble-curve helpers for CDC peak picking."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from process.ensemble_internal.primitives import (
    _COARSE_SIGMA_GRID_FRAC,
    _EPS,
    _crest_index,
    _parabolic_refine,
    find_peaks,
    peak_prominences,
    peak_widths,
)


def per_run_peaks(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prom_frac: float = 0.07,
    min_dist: int = 3,
    pad_left: bool = False,
    min_width_nodes: int = 3,
    require_full_prom: bool = True,
    max_keep: Optional[int] = None,
    fallback_global_max: bool = False,
    return_details: bool = False,
):
    """Find refined peak ages for one run trace."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size != y.size or x.size < 3:
        return (np.array([], float), []) if return_details else np.array([], float)

    rng = np.nanpercentile(y, 95) - np.nanpercentile(y, 5)
    prom_thr = prom_frac * max(rng, _EPS)

    if pad_left:
        yw = np.concatenate(([-np.inf], y))
        pk, _ = find_peaks(yw, distance=min_dist)
        pk = pk - 1
        pk = pk[pk >= 0]
    else:
        pk, _ = find_peaks(y, distance=min_dist)

    if pk.size == 0 and fallback_global_max:
        i = int(np.argmax(y))
        pk = np.array([i]) if 0 < i < (y.size - 1) else np.array([], int)

    if pk.size == 0:
        return (np.array([], float), []) if return_details else np.array([], float)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="some peaks have a width of 0")
        prom, _, _ = peak_prominences(y, pk)
        width, _, _, _ = peak_widths(y, pk, rel_height=0.5)

    keep = (prom >= prom_thr) & (width >= float(min_width_nodes))
    if require_full_prom:
        keep &= (pk > 0) & (pk < (y.size - 1))

    pk, prom, width = pk[keep], prom[keep], width[keep]
    if pk.size == 0:
        return (np.array([], float), []) if return_details else np.array([], float)

    pk_ref = np.array([_crest_index(y, int(i), half_win=2) for i in pk], int)
    refined = np.array([_parabolic_refine(x, y, int(j)) for j in pk_ref], float)

    order = np.argsort(refined)
    pk, refined = pk[order], refined[order]
    prom, width = prom[order], width[order]

    if max_keep is not None and max_keep > 0 and refined.size > max_keep:
        refined = refined[:max_keep]
        pk, prom, width = pk[:max_keep], prom[:max_keep], width[:max_keep]

    if not return_details:
        return refined
    det = [
        dict(
            idx=int(i),
            age_node=float(x[i]),
            age_refined=float(r),
            prom=float(p),
            width_nodes=float(w),
            height=float(y[i]),
        )
        for i, r, p, w in zip(pk, refined, prom, width)
    ]
    return refined, det


def robust_ensemble_curve(
    S_runs: np.ndarray,
    smooth_frac: float = 0.01,
    *_,
    **__,
) -> Tuple[np.ndarray, float, float]:
    """Return the lightly smoothed median ensemble curve and its dynamic range."""
    S_runs = np.asarray(S_runs, float)
    if S_runs.ndim != 2 or S_runs.shape[1] < 3:
        return np.array([]), 0.0, 0.0

    G = S_runs.shape[1]
    S_med = np.nanmedian(S_runs, axis=0)

    sigma_nodes = float(smooth_frac) * float(G)
    sigma_nodes = min(sigma_nodes, 2.0)
    S_med_s = gaussian_filter1d(S_med, sigma=sigma_nodes, mode="reflect")

    q5, q95 = np.nanpercentile(S_med_s, [5, 95])
    Delta = max(q95 - q5, _EPS)
    return S_med_s, float(Delta), float(sigma_nodes)
