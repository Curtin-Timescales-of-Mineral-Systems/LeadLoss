"""Ensemble peak picking for CDC goodness surfaces.

This module takes a stack of Monte Carlo goodness curves over a common age grid
and turns them into a small catalogue of supported Pb-loss peaks.

Core ideas:
- build a robust ensemble median surface from the run stack
- propose candidate humps on that surface
- keep only candidates that are reproducible across runs
- report one age and one empirical interval per surviving peak

The age reported for most peaks is the median of the run-level votes assigned to
that candidate. A narrow plateau-onset adjustment is available for broad,
older-tailed youngest peaks in multi-peak samples; that rule shifts the
reported age leftward toward the onset of the younger event while leaving the
rest of the catalogue unchanged.
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.ndimage import gaussian_filter1d
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

# -------------------------- utilities ----------------------------------------
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
    """
    Quadratic vertex refinement using 3 points around index k; clamps to bracket.
    If the local curvature is tiny (flat crest), fall back to the node.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = y.size
    if k <= 0 or k >= n - 1:
        return float(x[k])
    x0, x1, x2 = float(x[k-1]), float(x[k]), float(x[k+1])
    y0, y1, y2 = float(y[k-1]), float(y[k]), float(y[k+1])
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) <= _EPS:
        return x1
    a = (x2*(y1 - y0) + x1*(y0 - y2) + x0*(y2 - y1)) / denom
    b = (x2**2*(y0 - y1) + x1**2*(y2 - y0) + x0**2*(y1 - y2)) / denom
    # if crest is very flat, avoid nudging; keep the node
    if abs(a) < 1e-12:
        return x1
    xv = -b / (2.0*a)
    lo, hi = (x0, x2) if x0 <= x2 else (x2, x0)
    return float(min(max(xv, lo), hi))

def _crest_index(y: np.ndarray, k: int, half_win: int = 2) -> int:
    """
    Choose the index of the *local crest* near k:
      - search within [k-half_win, k+half_win],
      - take the midpoint of all nodes that attain the local maximum
        (stable on flat/plateau crests).
    """
    y = np.asarray(y, float); n = y.size
    a = max(1, int(k) - int(half_win))
    b = min(n - 2, int(k) + int(half_win))
    seg = y[a:b+1]
    if seg.size == 0 or not np.isfinite(seg).any():
        return int(k)
    m = np.nanmax(seg)
    cand = np.where(np.isclose(seg, m, rtol=1e-12, atol=1e-15))[0]
    # midpoint of the flat top if there are ties
    j_local = int(cand[len(cand)//2])
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
    """
    Shift broad older-tailed youngest peaks toward their onset.

    This rule is deliberately narrow: it applies only to the youngest member of
    a multi-peak catalogue, only when the half-prominence basin is broad, and
    only when the right span is longer than the left span. The goal is to avoid
    older-shifted vote-median ages on cancellation plateaus without changing
    normal sharp or single-peak behaviour.
    """
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

# -------------------------- per-run peaks --------------------
def per_run_peaks(
    x: np.ndarray, y: np.ndarray, *,
    prom_frac: float = 0.07,
    min_dist: int = 3,
    pad_left: bool = False,
    min_width_nodes: int = 3,
    require_full_prom: bool = True,
    max_keep: Optional[int] = None,
    fallback_global_max: bool = False,
    return_details: bool = False
):
    """
    Old per-run peak finder used by the previous ensemble picker.
    Returns refined ages (and optional details) for peaks in a single run trace.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size != y.size or x.size < 3:
        return (np.array([], float), []) if return_details else np.array([], float)

    rng = np.nanpercentile(y, 95) - np.nanpercentile(y, 5)
    prom_thr = prom_frac * max(rng, _EPS)

    if pad_left:
        yw = np.concatenate(([-np.inf], y))
        pk, _ = find_peaks(yw, distance=min_dist); pk = pk - 1; pk = pk[pk >= 0]
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

    # refine on the true local crest (stable on flat tops)
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
    det = [dict(idx=int(i), age_node=float(x[i]), age_refined=float(r),
                prom=float(p), width_nodes=float(w), height=float(y[i]))
           for i, r, p, w in zip(pk, refined, prom, width)]
    return refined, det

def robust_ensemble_curve(S_runs: np.ndarray, smooth_frac: float = 0.01,
                          *_, **__) -> Tuple[np.ndarray, float, float]:
    """
    Smoothed pointwise median across runs (larger = better).
    **Light** smoothing: grid-fraction with a small sigma cap to keep crests crisp.
    Returns (S_med_s, Delta, sigma_nodes).
    """
    S_runs = np.asarray(S_runs, float)
    if S_runs.ndim != 2 or S_runs.shape[1] < 3:
        return np.array([]), 0.0, 0.0

    G = S_runs.shape[1]
    S_med = np.nanmedian(S_runs, axis=0)

    # Light smoothing: sigma as a fraction of N, but cap at 2 nodes for crisp crests.
    sigma_nodes = float(smooth_frac) * float(G)
    sigma_nodes = min(sigma_nodes, 2.0) 
    S_med_s = gaussian_filter1d(S_med, sigma=sigma_nodes, mode="reflect")

    q5, q95 = np.nanpercentile(S_med_s, [5, 95])
    Delta = max(q95 - q5, _EPS)
    return S_med_s, float(Delta), float(sigma_nodes)

def _score_candidate_peaks(
    pk, x, S, S_med_s, sign, R,
    per_run_peaks_list, optima_ma, run_gate,
    f_r, support_min, r_min,
    left_bounds, right_bounds,
    j_ref_all, age_ref_all,
    left_edge_ma_all, right_edge_ma_all,
    plateau_onset_mode,
    plateau_onset_min_width_frac,
    plateau_onset_min_right_left_ratio,
    plateau_onset_blend_frac,
    diagnostic_rows,
):
    """Per-run voting, support computation, and CI estimation for candidate peaks."""
    out: List[Dict] = []

    for j_idx, j_c in enumerate(pk):
        j_ref = int(j_ref_all[j_idx])
        age_ref = float(age_ref_all[j_idx])

        a = max(1, int(left_bounds[j_idx]))
        b = min(S.shape[1] - 2, int(right_bounds[j_idx]))
        lo_ma_win, hi_ma_win = float(x[a]), float(x[b])

        f_r_eff = float(f_r)

        votes: List[float] = []
        direct_votes: List[float] = []
        optima_for_peak: List[float] = []

        for r in range(R):
            y_r = sign * S[r]
            pr_list = per_run_peaks_list[r] if r < len(per_run_peaks_list) else None
            pr_list = np.asarray(pr_list, float) if pr_list is not None else np.array([], float)
            cand = pr_list[(pr_list >= lo_ma_win) & (pr_list <= hi_ma_win)]

            if cand.size == 0:
                j_abs = int(a + np.argmax(y_r[a:b+1]))
                p5, p95 = run_gate[r, 0], run_gate[r, 1]
                thr = p5 + f_r_eff * max(p95 - p5, 0.0)
                if y_r[j_abs] < thr:
                    continue
                age_vote = float(_parabolic_refine(x, S[r], j_abs))
            else:
                j_choices = [int(np.argmin(np.abs(x - p))) for p in cand]
                heights_r = [float(y_r[jj]) for jj in j_choices]
                j_abs = int(j_choices[int(np.argmax(heights_r))])

                p5, p95 = run_gate[r, 0], run_gate[r, 1]
                thr = p5 + f_r_eff * max(p95 - p5, 0.0)
                if y_r[j_abs] < thr:
                    continue
                age_vote = float(_parabolic_refine(x, S[r], j_abs))
                direct_votes.append(age_vote)

            if age_ref_all.size > 1:
                nearest = int(np.argmin(np.abs(age_ref_all - age_vote)))
                if nearest != j_idx:
                    continue

            votes.append(age_vote)

            if optima_ma is not None and 0 <= r < optima_ma.size:
                opt_val = float(optima_ma[r])
                if np.isfinite(opt_val) and (lo_ma_win <= opt_val <= hi_ma_win):
                    optima_for_peak.append(opt_val)

        support = len(direct_votes) / float(R)

        if support < max(float(support_min), float(r_min) / float(max(R, 1))):
            _append_diagnostic_peak(
                diagnostic_rows, x, S_med_s, int(j_ref),
                reason="low_support",
                direct_support=float(support),
                winner_support=float(len(optima_for_peak) / float(max(R, 1))) if optima_ma is not None else np.nan,
                ci_low=lo_ma_win, ci_high=hi_ma_win,
            )
            continue

        step = float(x[1] - x[0])
        if len(direct_votes) >= 3:
            lo_ci, hi_ci = np.nanpercentile(direct_votes, [2.5, 97.5])
        elif optima_ma is not None and len(optima_for_peak) >= 3:
            lo_ci, hi_ci = np.nanpercentile(optima_for_peak, [2.5, 97.5])
        elif len(votes) >= 3:
            lo_ci, hi_ci = np.nanpercentile(votes, [2.5, 97.5])
        else:
            lo_ci, hi_ci = age_ref - step, age_ref + step

        if len(direct_votes) >= 1:
            med_votes = float(np.median(direct_votes))
            base_age_mode = "vote_median"
        elif len(optima_for_peak) >= 1:
            med_votes = float(np.median(optima_for_peak))
            base_age_mode = "vote_median"
        elif len(votes) >= 1:
            med_votes = float(np.median(votes))
            base_age_mode = "vote_median"
        else:
            med_votes = age_ref
            base_age_mode = "curve_crest"
        age_out = float(med_votes if np.isfinite(med_votes) else age_ref)

        peak_left_edge = float(left_edge_ma_all[j_idx]) if j_idx < len(left_edge_ma_all) else np.nan
        peak_right_edge = float(right_edge_ma_all[j_idx]) if j_idx < len(right_edge_ma_all) else np.nan
        total_span = max(float(x[-1]) - float(x[0]), _EPS)
        age_out, age_mode, peak_width_frac, right_left_ratio = _apply_plateau_onset_adjustment(
            age_out,
            peak_left_edge,
            peak_right_edge,
            base_age_mode=base_age_mode,
            n_peaks=len(pk),
            is_youngest=(j_idx == 0),
            total_span=total_span,
            mode=plateau_onset_mode,
            min_width_frac=plateau_onset_min_width_frac,
            min_right_left_ratio=plateau_onset_min_right_left_ratio,
            blend_frac=plateau_onset_blend_frac,
        )

        lo_ci = max(lo_ci, float(x[0]))
        hi_ci = min(hi_ci, float(x[-1]))
        if (not np.isfinite(lo_ci)) or (not np.isfinite(hi_ci)) or (hi_ci <= lo_ci):
            lo_ci, hi_ci = max(age_out - step, float(x[0])), min(age_out + step, float(x[-1]))
        if (hi_ci - lo_ci) < (_DEGENERATE_CI_GRID_FRAC * step):
            lo_ci, hi_ci = max(age_out - step, float(x[0])), min(age_out + step, float(x[-1]))
        if age_out < lo_ci:
            lo_ci = float(age_out)
        elif age_out > hi_ci:
            hi_ci = float(age_out)

        out.append(dict(
            peak_no=0,
            age_ma=age_out,
            ci_low=float(lo_ci),
            ci_high=float(hi_ci),
            support=float(support),
            age_mode=age_mode,
            peak_left_edge_ma=peak_left_edge,
            peak_right_edge_ma=peak_right_edge,
            peak_half_prom_width_frac=peak_width_frac,
            peak_right_left_ratio=right_left_ratio,
        ))

    return out


def build_ensemble_catalogue(
    sample_name: str,
    tier: str,
    age_grid: np.ndarray,
    goodness_runs: np.ndarray,
    *,
    orientation: str = "max",
    smooth_frac: float = 0.01,
    # range-agnostic knobs (fractions of N or of Δ)
    f_d: float = 0.05,
    f_p: float = 0.03,
    f_v: float = 0.10,
    f_w: float = 0.05,
    w_min_nodes: int = 3,
    # reproducibility
    support_min: float = 0.10,
    r_min: int = 3,
    f_r: float = 0.25,
    # per-run peak detection knobs
    per_run_prom_frac: float = 0.06,
    per_run_min_dist: int = 3,
    per_run_min_width: int = 3,
    per_run_require_full_prom: bool = False,
    plateau_onset_mode: str = "off",
    plateau_onset_min_width_frac: float = 0.20,
    plateau_onset_min_right_left_ratio: float = 1.30,
    plateau_onset_blend_frac: float = 0.50,
    # minimum ensemble dynamic range below which we do not pick peaks
    delta_min: float = 0.0,
    # only keep peaks whose crest is at least this fraction
    # of the highest crest on the ensemble curve.
    height_frac: float = 0.0,   # 0.0 = disabled
    # optional global optimum ages per run (Ma)
    optima_ma: Optional[np.ndarray] = None,
    # optional injection of per-run peak lists
    per_run_peaks_list: Optional[List[np.ndarray]] = None,
    # merge guards
    merge_per_hump: bool = True,
    merge_shoulders: bool = True,
    diagnostic_rows: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Ensemble picker:
      • candidate humps from a *coarse* ensemble median,
      • at most ONE fine-scale peak per coarse hump,
      • keep only peaks reproducible across runs by local votes,
      • CIs from per-peak optima distribution,
      • optional plateau-onset adjustment for broad older-tailed youngest peaks.

    Notes on support:
      • support is computed from runs with an explicit per-run peak inside the
        candidate window (fallback local-crest votes do not increase support).
    """
    x = np.asarray(age_grid, float)
    S = np.asarray(goodness_runs, float)
    if S.ndim != 2 or S.shape[0] == 0 or S.shape[1] < 3:
        return []

    R, G = S.shape
    sign = 1.0 if str(orientation).lower().startswith("max") else -1.0

    # Fine-scale ensemble curve (this is what you plot as “Goodness”)
    S_med_s, Delta, _ = robust_ensemble_curve(S, smooth_frac=smooth_frac)
    if S_med_s.size == 0:
        return []
    if delta_min > 0.0 and Delta < delta_min:
        return []

    y = sign * S_med_s

    prom_abs = max(float(f_p) * float(Delta), _EPS)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="some peaks have a prominence of 0")
        warnings.filterwarnings("ignore", message="some peaks have a width of 0")
        pk_visual, _ = find_peaks(y, distance=1)
        if pk_visual.size:
            prom_visual, _, _ = peak_prominences(y, pk_visual)
            width_visual, _, _, _ = peak_widths(y, pk_visual, rel_height=0.5)
        else:
            prom_visual = np.array([], float)
            width_visual = np.array([], float)
    pk_major = np.array([], dtype=int)
    major_rad = max(1, int(0.05 * G)) if merge_per_hump else 0
    if merge_per_hump:
        # A coarse surface blurred over ~3% of the grid helps merge shoulder
        # roughness while keeping distinct humps separate.
        sigma_coarse = max(5.0, _COARSE_SIGMA_GRID_FRAC * float(G))
        S_med_coarse = gaussian_filter1d(S_med_s, sigma=sigma_coarse, mode="reflect")
        y_coarse = sign * S_med_coarse
        pk_major, _ = find_peaks(
            y_coarse,
            distance=max(1, int(0.10 * G)),
            prominence=prom_abs,
        )
        if pk_major.size == 0:
            pk_diag, _ = find_peaks(
                y,
                distance=1,
                width=w_min_nodes,
                prominence=max(prom_abs, _EPS),
            )
            for idx in np.asarray(pk_diag, int):
                _append_diagnostic_peak(
                    diagnostic_rows,
                    x,
                    S_med_s,
                    int(idx),
                    reason="coarse_surface_no_separate_mode",
                )
            return []

    # ---------------- Fine-scale candidate peaks ----------------------------
    family_nodes = max(1, int(np.ceil(f_d * G)))       # peak-family window (nodes)
    edge_guard   = 0

    pk_fine, _ = find_peaks(y, distance=1, width=w_min_nodes, prominence=prom_abs)

    # If no interior fine peaks, fall back to using the coarse hump(s)
    if pk_fine.size == 0 and merge_per_hump:
        pk_fine = np.array(pk_major, dtype=int)

    if pk_fine.size == 0:
        # Still nothing at all → give up
        return []

    if merge_per_hump:
        # --------- FORCE: at most one fine peak per coarse hump -----------------
        # window size within which a fine peak can represent a coarse hump
        chosen_idx: List[int] = []
        for pM in pk_major:
            # indices in pk_fine that are close to this major crest
            idx_in_win = np.where(np.abs(pk_fine - int(pM)) <= major_rad)[0]
            if idx_in_win.size == 0:
                continue
            # pick the highest fine-scale crest in this window
            best_local = idx_in_win[np.argmax(y[pk_fine[idx_in_win]])]
            chosen_idx.append(int(best_local))

        if not chosen_idx:
            for idx in np.asarray(pk_fine, int):
                _append_diagnostic_peak(
                    diagnostic_rows,
                    x,
                    S_med_s,
                    int(idx),
                    reason="coarse_surface_no_separate_mode",
                )
            # coarse humps exist but none supported by a fine peak ⇒ no peaks
            return []

        chosen_idx = sorted(set(chosen_idx))
        chosen_set = set(chosen_idx)
        for local_i, idx in enumerate(np.asarray(pk_fine, int)):
            if local_i in chosen_set:
                continue
            close_to_major = np.any(np.abs(pk_major - int(idx)) <= major_rad)
            reason = "suppressed_nearby_weaker_peak" if close_to_major else "coarse_surface_no_separate_mode"
            _append_diagnostic_peak(
                diagnostic_rows,
                x,
                S_med_s,
                int(idx),
                reason=reason,
            )
        pk = pk_fine[chosen_idx]
    else:
        pk = np.asarray(np.unique(pk_fine), int)

    # guard against edges
    pk = pk[(pk >= edge_guard) & (pk <= G - 1 - edge_guard)]
    if pk.size == 0:
        return []

    # ---------------- Shoulder merge ---------------------
    if merge_shoulders:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="some peaks have a prominence of 0")
            prom, left_bases, right_bases = peak_prominences(y, pk)
        valid_prom = np.isfinite(prom) & (prom > 0.0)
        if not np.all(valid_prom):
            pk = pk[valid_prom]
            prom = prom[valid_prom]
            left_bases = left_bases[valid_prom]
            right_bases = right_bases[valid_prom]
            if pk.size == 0:
                return []
        order = np.argsort(pk)
        pk          = pk[order]
        prom        = prom[order]
        left_bases  = left_bases[order]
        right_bases = right_bases[order]
        pk_before_shoulder = pk.copy()

        kept_idx_local: List[int] = []
        i = 0
        while i < pk.size:
            winner = i
            j = i + 1
            while j < pk.size:
                sep = pk[j] - pk[winner]
                if sep >= family_nodes:
                    break

                lo, hi = int(min(pk[winner], pk[j])), int(max(pk[winner], pk[j]))
                valley = float(np.nanmin(y[lo:hi + 1]))
                lower_crest = min(y[pk[winner]], y[pk[j]])
                shallow = (lower_crest - valley) < f_v * min(prom[winner], prom[j])

                if shallow:
                    if prom[j] > prom[winner] * (1.0 + 0.05):
                        winner = j
                    elif (
                        abs(prom[j] - prom[winner])
                        <= 0.05 * max(prom[j], prom[winner])
                        and y[pk[j]] > y[pk[winner]]
                    ):
                        winner = j
                    j += 1
                else:
                    break

            kept_idx_local.append(winner)
            i = j

        kept_idx_local = sorted(set(kept_idx_local))
        removed_local = sorted(set(range(pk.size)) - set(kept_idx_local))
        for local_i in removed_local:
            _append_diagnostic_peak(
                diagnostic_rows,
                x,
                S_med_s,
                int(pk_before_shoulder[local_i]),
                reason="suppressed_nearby_weaker_peak",
            )
        pk          = pk[kept_idx_local]
        prom        = prom[kept_idx_local]
        left_bases  = left_bases[kept_idx_local]
        right_bases = right_bases[kept_idx_local]
        if pk.size == 0:
            return []

    # ---------------- Optional global height filter -------------------------
    if height_frac > 0.0:
        heights = y[pk]
        h_max = float(np.nanmax(heights))
        if np.isfinite(h_max) and h_max > 0.0:
            thr = h_max * float(height_frac)
            keep = heights >= thr
            for idx in np.asarray(pk[~keep], int):
                _append_diagnostic_peak(
                    diagnostic_rows,
                    x,
                    S_med_s,
                    int(idx),
                    reason="below_global_height_gate",
                )
            pk          = pk[keep]
            if pk.size == 0:
                return []

    pk = np.asarray(np.sort(pk), int)
    left_bounds = np.zeros(pk.size, dtype=int)
    right_bounds = np.full(pk.size, G - 1, dtype=int)
    for i in range(pk.size - 1):
        lo = int(pk[i])
        hi = int(pk[i + 1])
        if hi <= lo + 1:
            boundary = lo
        else:
            boundary = int(lo + np.argmin(y[lo:hi + 1]))
        right_bounds[i] = boundary
        left_bounds[i + 1] = boundary

    # ---------------- Per-run gates + voting ----------------------
    run_gate = np.empty((R, 2), float)
    for r in range(R):
        y_r = sign * S[r]
        p5, p95 = np.nanpercentile(y_r, [5, 95])
        run_gate[r, 0] = float(p5)
        run_gate[r, 1] = float(p95)

    if per_run_peaks_list is None:
        per_run_peaks_list = []
        for r in range(R):
            y_r = sign * S[r]
            pr = per_run_peaks(
                x, y_r,
                prom_frac=float(per_run_prom_frac),
                min_dist=int(per_run_min_dist),
                min_width_nodes=int(per_run_min_width),
                require_full_prom=bool(per_run_require_full_prom),
                fallback_global_max=False,
            )
            per_run_peaks_list.append(np.asarray(pr, float))

    optima_ma = np.asarray(optima_ma, float) if optima_ma is not None else None
    j_ref_all = np.array([_crest_index(S_med_s, int(jc), half_win=2) for jc in pk], int)
    age_ref_all = np.array([_parabolic_refine(x, S_med_s, int(jr)) for jr in j_ref_all], float)
    left_edge_ma_all, right_edge_ma_all = _half_prominence_edges(x, y, pk)

    out = _score_candidate_peaks(
        pk, x, S, S_med_s, sign, R,
        per_run_peaks_list, optima_ma, run_gate,
        f_r, support_min, r_min,
        left_bounds, right_bounds,
        j_ref_all, age_ref_all,
        left_edge_ma_all, right_edge_ma_all,
        plateau_onset_mode,
        plateau_onset_min_width_frac, plateau_onset_min_right_left_ratio,
        plateau_onset_blend_frac,
        diagnostic_rows,
    )
    for d in out:
        d["sample"] = sample_name

    if diagnostic_rows is not None and pk_visual.size:
        step = _step_from_grid(x)
        rough = float(np.nanmedian(np.abs(np.diff(y)))) if y.size >= 3 else 0.0
        diag_prom_thr = max(0.20 * prom_abs, 2.0 * rough, _EPS)
        diag_width_thr = max(1.0, 0.35 * float(w_min_nodes))
        accepted_ages = np.asarray([float(rr.get("age_ma", np.nan)) for rr in out], float)
        existing_diag_ages = np.asarray([float(rr.get("age_ma", np.nan)) for rr in diagnostic_rows], float)
        pk_fine_set = {int(v) for v in np.asarray(pk_fine, int)}
        pk_final_set = {int(v) for v in np.asarray(pk, int)}

        for idx, prom_v, width_v in zip(np.asarray(pk_visual, int), prom_visual, width_visual):
            if not np.isfinite(prom_v) or not np.isfinite(width_v):
                continue
            if (prom_v < diag_prom_thr) or (width_v < diag_width_thr):
                continue

            j_ref = int(_crest_index(S_med_s, int(idx), half_win=2))
            age = float(_parabolic_refine(x, S_med_s, j_ref))
            tol = max(0.51 * step, 1e-6)
            if accepted_ages.size and np.any(np.isfinite(accepted_ages) & (np.abs(accepted_ages - age) <= tol)):
                continue
            if existing_diag_ages.size and np.any(np.isfinite(existing_diag_ages) & (np.abs(existing_diag_ages - age) <= tol)):
                continue

            if int(idx) not in pk_fine_set:
                reason = "below_ensemble_prominence" if prom_v < prom_abs else "too_narrow_on_ensemble_curve"
            elif int(idx) not in pk_final_set:
                if merge_shoulders and np.any(np.abs(np.asarray(pk, int) - int(idx)) < family_nodes):
                    reason = "suppressed_nearby_weaker_peak"
                elif merge_per_hump and pk_major.size and not np.any(np.abs(pk_major - int(idx)) <= major_rad):
                    reason = "coarse_surface_no_separate_mode"
                else:
                    reason = "not_retained_as_formal_candidate"
            else:
                continue

            lo_idx, hi_idx = _basin_bounds_from_peaks(y, int(idx), np.asarray(pk_visual, int))
            lo_ma = float(x[lo_idx])
            hi_ma = float(x[hi_idx])
            direct_support, winner_support = _estimate_window_support(
                lo_ma,
                hi_ma,
                per_run_peaks_list,
                optima_ma,
            )
            _append_diagnostic_peak(
                diagnostic_rows,
                x,
                S_med_s,
                int(idx),
                reason=reason,
                direct_support=direct_support,
                winner_support=winner_support,
                ci_low=lo_ma,
                ci_high=hi_ma,
            )

    out.sort(key=lambda d: d["age_ma"])

    # In "no-merge" mode we still collapse obvious duplicate picks that sit on
    # the same flat crest (very close in age and strongly CI-overlapping).
    if (not merge_per_hump) and (not merge_shoulders) and len(out) > 1:
        step_ma = float(abs(x[1] - x[0])) if x.size >= 2 else 1.0
        near_ma = max(3.0 * step_ma, 20.0)
        deduped: List[Dict] = [dict(out[0])]

        def _score(rr: Dict) -> tuple:
            sup = float(rr.get("support", 0.0))
            width = float(rr["ci_high"]) - float(rr["ci_low"])
            return (sup, -width)

        for rr in out[1:]:
            prev = deduped[-1]
            sep = abs(float(rr["age_ma"]) - float(prev["age_ma"]))
            overlap = (float(rr["ci_low"]) <= float(prev["ci_high"])) and (
                float(rr["ci_high"]) >= float(prev["ci_low"])
            )
            if sep <= near_ma and overlap:
                if _score(rr) > _score(prev):
                    deduped[-1] = dict(rr)
            else:
                deduped.append(dict(rr))
        out = deduped

    for i, d in enumerate(out, 1):
        d["peak_no"] = i
    return out
