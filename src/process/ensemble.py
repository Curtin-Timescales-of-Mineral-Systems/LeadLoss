# # process/ensemble.py
# from __future__ import annotations

# import numpy as np
# from typing import List, Dict, Optional, Tuple, Any
# from scipy.ndimage import gaussian_filter1d
# from scipy.signal import (
#     find_peaks as _find_peaks,
#     peak_prominences as _peak_prominences,
#     peak_widths as _peak_widths,
# )

# # expose SciPy names we use everywhere under stable local aliases
# find_peaks = _find_peaks
# peak_prominences = _peak_prominences
# peak_widths = _peak_widths

# _EPS = 1e-12
# DEBUG_CAND = False

# # Set to 0.0 to keep all nearby apices. Set >0 (Ma) to auto-collapse close peaks later.
# _MERGE_WITHIN_MA = 50.0


# # -------------------------- utilities ----------------------------------------
# def _parabolic_refine(x: np.ndarray, y: np.ndarray, k: int) -> float:
#     """Quadratic vertex refinement using 3 points around index k; clamps to bracket."""
#     x = np.asarray(x, float); y = np.asarray(y, float)
#     n = y.size
#     if k <= 0 or k >= n - 1:
#         return float(x[k])
#     x0, x1, x2 = float(x[k-1]), float(x[k]), float(x[k+1])
#     y0, y1, y2 = float(y[k-1]), float(y[k]), float(y[k+1])
#     denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
#     if abs(denom) <= _EPS:
#         return x1
#     a = (x2*(y1 - y0) + x1*(y0 - y2) + x0*(y2 - y1)) / denom
#     b = (x2**2*(y0 - y1) + x1**2*(y2 - y0) + x0**2*(y1 - y2)) / denom
#     if a == 0.0:
#         return x1
#     xv = -b / (2.0*a)
#     lo, hi = (x0, x2) if x0 <= x2 else (x2, x0)
#     return float(min(max(xv, lo), hi))


# # -------------------------- per-run peaks (old API) --------------------------
# def per_run_peaks(x: np.ndarray, y: np.ndarray, *,
#                   prom_frac: float = 0.07,
#                   min_dist: int = 3,
#                   pad_left: bool = False,
#                   min_width_nodes: int = 3,
#                   require_full_prom: bool = True,
#                   max_keep: Optional[int] = None,
#                   fallback_global_max: bool = False,
#                   return_details: bool = False):
#     """
#     Old per-run peak finder used by the previous ensemble picker.
#     Returns refined ages (and optional details) for peaks in a single run trace.
#     """
#     x = np.asarray(x, float); y = np.asarray(y, float)
#     if x.size != y.size or x.size < 3:
#         return (np.array([], float), []) if return_details else np.array([], float)

#     rng = np.nanpercentile(y, 95) - np.nanpercentile(y, 5)
#     prom_thr = prom_frac * max(rng, _EPS)

#     if pad_left:
#         yw = np.concatenate(([-np.inf], y))
#         pk, _ = find_peaks(yw, distance=min_dist); pk = pk - 1; pk = pk[pk >= 0]
#     else:
#         pk, _ = find_peaks(y, distance=min_dist)

#     if pk.size == 0 and fallback_global_max:
#         i = int(np.argmax(y))
#         pk = np.array([i]) if 0 < i < (y.size - 1) else np.array([], int)
#     if pk.size == 0:
#         return (np.array([], float), []) if return_details else np.array([], float)

#     prom, lb, rb = peak_prominences(y, pk)
#     width, _, _, _ = peak_widths(y, pk, rel_height=0.5)

#     keep = (prom >= prom_thr) & (width >= float(min_width_nodes))
#     if require_full_prom:
#         keep &= (pk > 0) & (pk < (y.size - 1))

#     pk, prom, width = pk[keep], prom[keep], width[keep]
#     if pk.size == 0:
#         return (np.array([], float), []) if return_details else np.array([], float)

#     refined = np.array([_parabolic_refine(x, y, int(k)) for k in pk], float)
#     order = np.argsort(refined)
#     pk, refined = pk[order], refined[order]
#     prom, width = prom[order], width[order]

#     if max_keep is not None and max_keep > 0 and refined.size > max_keep:
#         refined = refined[:max_keep]
#         pk, prom, width = pk[:max_keep], prom[:max_keep], width[:max_keep]

#     if not return_details:
#         return refined
#     det = [dict(idx=int(i), age_node=float(x[i]), age_refined=float(r),
#                 prom=float(p), width_nodes=float(w), height=float(y[i]))
#            for i, r, p, w in zip(pk, refined, prom, width)]
#     return refined, det


# def robust_ensemble_curve(S_runs: np.ndarray, smooth_frac: float = 0.01,
#                           *_, **__) -> Tuple[np.ndarray, float, float]:
#     """
#     Smoothed pointwise median across runs (larger = better).
#     Returns (S_med_s, Delta, sigma_nodes), with sigma_nodes = 0.01 * N by default.
#     """
#     S_runs = np.asarray(S_runs, float)
#     if S_runs.ndim != 2 or S_runs.shape[1] < 3:
#         return np.array([]), 0.0, 0.0
#     G = S_runs.shape[1]
#     S_med = np.nanmedian(S_runs, axis=0)
#     sigma_nodes = max(0.5, float(smooth_frac) * float(G))
#     S_med_s = gaussian_filter1d(S_med, sigma=sigma_nodes, mode="reflect")
#     q5, q95 = np.nanpercentile(S_med_s, [5, 95])
#     Delta = max(q95 - q5, _EPS)
#     return S_med_s, float(Delta), float(sigma_nodes)

# # -------------------------- old candidate merging helpers --------------------
# def _merge_candidates(idx: np.ndarray, prom: np.ndarray,
#                       S_curve: np.ndarray, dist_nodes: int, f_v: float
#                       ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Merge shoulder/nearby candidates in index-space using:
#       - node distance (dist_nodes)
#       - shallow valley check relative to local prominences (fraction f_v)
#     Keep the stronger (by prominence; tie-break by higher crest).
#     """
#     idx = np.asarray(idx, int); prom = np.asarray(prom, float)
#     if idx.size == 0:
#         return idx, prom
#     order = np.argsort(idx); idx, prom = idx[order], prom[order]
#     keep_idx, keep_prom = [], []
#     i, n = 0, idx.size
#     while i < n:
#         winner = i; j = i + 1
#         while j < n:
#             sep = idx[j] - idx[winner]
#             lo = min(idx[winner], idx[j]); hi = max(idx[winner], idx[j])
#             valley = float(np.min(S_curve[lo:hi+1]))
#             hi_w = float(S_curve[idx[winner]]); hi_j = float(S_curve[idx[j]])
#             shallow = (min(hi_w, hi_j) - valley) < f_v * min(prom[winner], prom[j])
#             if not (sep < dist_nodes or shallow):
#                 break
#             eps = 0.05
#             if prom[j] > prom[winner] * (1.0 + eps):
#                 winner = j
#             elif abs(prom[j] - prom[winner]) <= eps * max(prom[j], prom[winner]) and \
#                  S_curve[idx[j]] > S_curve[idx[winner]]:
#                 winner = j
#             j += 1
#         keep_idx.append(int(idx[winner])); keep_prom.append(float(prom[winner]))
#         i = j
#     return np.asarray(keep_idx, int), np.asarray(keep_prom, float)


# def _suppress_close_by_height_ma(c_idx: np.ndarray, c_prom: np.ndarray,
#                                  S_curve: np.ndarray,
#                                  age_grid: np.ndarray,
#                                  min_sep_ma: float) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     After merging in index-space, optionally collapse peaks closer than `min_sep_ma`
#     (measured in Ma) keeping higher crest. Set min_sep_ma <= 0 to disable.
#     """
#     c_idx = np.asarray(c_idx, int); c_prom = np.asarray(c_prom, float)
#     if c_idx.size <= 1 or min_sep_ma <= 0.0:
#         return c_idx, c_prom
#     order = np.argsort(S_curve[c_idx])[::-1]
#     keep_mask = np.ones(c_idx.size, dtype=bool)
#     for ii in order:
#         if not keep_mask[ii]:
#             continue
#         ai = float(age_grid[c_idx[ii]])
#         for jj in order:
#             if jj == ii or not keep_mask[jj]:
#                 continue
#             aj = float(age_grid[c_idx[jj]])
#             if abs(aj - ai) <= min_sep_ma:
#                 keep_mask[jj] = False
#     kept = np.where(keep_mask)[0]
#     return c_idx[kept], c_prom[kept]

# def build_ensemble_catalogue(
#     sample_name: str,
#     tier: str,
#     age_grid: np.ndarray,
#     goodness_runs: np.ndarray,
#     *,
#     orientation: str = "max",
#     smooth_frac: float = 0.01,
#     # range-agnostic knobs (fractions of N or of Δ)
#     f_d: float = 0.05,            # min separation (nodes fraction)
#     f_p: float = 0.03,            # min prominence (of Δ)
#     f_v: float = 0.10,            # shallow valley merge (of smaller prominence)
#     f_w: float = 0.08,            # vote window half-width (nodes fraction)
#     w_min_nodes: int = 3,         # min FWHM in nodes
#     # reproducibility
#     support_min: float = 0.10,    # minimal support fraction
#     r_min: int = 3,               # minimal number of runs
#     f_r: float = 0.25,            # run vote gate (fraction of that run’s 5–95% range)
#     # ignored legacy/compat kwargs:
#     pen_ok_mask=None,             # kept for signature compatibility
#     **_ignored: Any,
# ) -> List[Dict]:
#     """
#     Manuscript picker:
#       • candidates from smoothed ensemble median only,
#       • explicit shoulder merge & endpoint handling,
#       • keep only peaks reproducible across runs by local votes.
#     """
#     x = np.asarray(age_grid, float)
#     S = np.asarray(goodness_runs, float)  # R × G
#     R, G = S.shape
#     if R == 0 or G < 3:
#         return []

#     # Oriented candidate curve (penalised or raw depending on caller)
#     sign = 1.0 if str(orientation).lower().startswith("max") else -1.0
#     S_med_s, Delta, _ = robust_ensemble_curve(S, smooth_frac=smooth_frac)
#     if S_med_s.size == 0:
#         return []
#     y = sign * S_med_s

#     # ----- candidate apices (interior only) -----
#     dist_nodes  = max(1, int(np.ceil(f_d * G)))
#     prom_abs    = max(float(f_p) * float(Delta), _EPS)

#     pk, props = find_peaks(y, distance=dist_nodes, width=w_min_nodes, prominence=prom_abs)
#     # Exclude exact endpoints (0 and G-1)
#     interior = (pk >= 1) & (pk <= G - 2)
#     pk = pk[interior]

#     if pk.size == 0:
#         return []

#     prom = peak_prominences(y, pk)[0]

#     # ----- shoulder merge in index space + valley check -----
#     order = np.argsort(pk)
#     pk, prom = pk[order], prom[order]

#     kept_idx = []
#     i = 0
#     while i < pk.size:
#         winner = i
#         j = i + 1
#         while j < pk.size:
#             sep = pk[j] - pk[winner]
#             if sep >= dist_nodes:
#                 # far enough, stop inner loop
#                 break
#             # valley depth between pk[winner] and pk[j]
#             lo, hi = int(min(pk[winner], pk[j])), int(max(pk[winner], pk[j]))
#             valley = float(np.min(y[lo:hi+1]))
#             lower_crest = min(y[pk[winner]], y[pk[j]])
#             shallow = (lower_crest - valley) < f_v * min(prom[winner], prom[j])
#             if shallow or sep < dist_nodes:
#                 # keep the stronger (by prominence; tie by higher crest)
#                 if prom[j] > prom[winner] * (1.0 + 0.05):
#                     winner = j
#                 elif abs(prom[j] - prom[winner]) <= 0.05 * max(prom[j], prom[winner]) and y[pk[j]] > y[pk[winner]]:
#                     winner = j
#                 j += 1
#             else:
#                 break
#         kept_idx.append(int(pk[winner]))
#         i = j

#     pk = np.asarray(sorted(set(kept_idx)), int)
#     if pk.size == 0:
#         return []

#     # ----- reproducibility by local voting -----
#     vote_nodes = max(1, int(np.ceil(f_w * G)))
#     out: List[Dict] = []

#     # precompute per-run dynamic ranges for the f_r gate
#     run_gate = []
#     for r in range(R):
#         y_r = sign * S[r]
#         p5, p95 = np.nanpercentile(y_r, [5, 95])
#         run_gate.append((float(p5), float(p95)))
#     run_gate = np.asarray(run_gate, float)  # R × 2

#     for j_c in pk:
#         # refine apex on *unoriented* ensemble curve for reporting
#         age_ref = float(_parabolic_refine(x, S_med_s, int(j_c)))

#         # voting window on oriented per-run traces
#         a = max(1, int(j_c) - vote_nodes)
#         b = min(G - 2, int(j_c) + vote_nodes)

#         votes = []
#         for r in range(R):
#             y_r = sign * S[r]
#             win = y_r[a:b+1]
#             if not np.isfinite(win).any():
#                 continue
#             # pick the tallest node in the window
#             j_local = int(np.nanargmax(win))
#             j_abs   = a + j_local

#             # relative-height gate using that run’s 5–95% range
#             p5, p95 = run_gate[r, 0], run_gate[r, 1]
#             thr = p5 + float(f_r) * max(p95 - p5, 0.0)
#             if y_r[j_abs] < thr:
#                 continue

#             votes.append(float(_parabolic_refine(x, S[r], j_abs)))

#         support = len(votes) / float(R)
#         if support < max(float(support_min), float(r_min) / float(max(R, 1))):
#             continue

#         if len(votes) >= 3:
#             lo_ci, hi_ci = np.nanpercentile(votes, [2.5, 97.5])
#         else:
#             lo_ci = age_ref - float(x[1] - x[0])
#             hi_ci = age_ref + float(x[1] - x[0])

#         # recenter CI on refined apex and enforce ≥ one grid step width
#         med_votes = float(np.median(votes)) if votes else age_ref
#         delta = age_ref - med_votes
#         lo_ci = min(lo_ci + delta, age_ref - float(x[1] - x[0]))
#         hi_ci = max(hi_ci + delta, age_ref + float(x[1] - x[0]))

#         out.append(dict(
#             sample=sample_name,
#             peak_no=0,  # set later
#             age_ma=age_ref,
#             ci_low=float(lo_ci),
#             ci_high=float(hi_ci),
#             support=float(support)
#         ))

#     out.sort(key=lambda d: d["age_ma"])
#     for i, d in enumerate(out, 1):
#         d["peak_no"] = i
#     return out


from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from scipy.ndimage import gaussian_filter1d
from scipy.signal import (
    find_peaks as _find_peaks,
    peak_prominences as _peak_prominences,
    peak_widths as _peak_widths,
)

# expose SciPy names we use everywhere under stable local aliases
find_peaks = _find_peaks
peak_prominences = _peak_prominences
peak_widths = _peak_widths

_EPS = 1e-12
DEBUG_CAND = False

# keep; optional post-merge MA collapse (unused by default here)
_MERGE_WITHIN_MA = 50.0


# -------------------------- utilities ----------------------------------------
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


# -------------------------- per-run peaks (legacy helper) --------------------
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

    # This is the behaviour you had: very light smoothing, capped.
    sigma_nodes = min(1.5, max(0.5, float(smooth_frac) * float(G)))
    S_med_s = gaussian_filter1d(S_med, sigma=sigma_nodes, mode="reflect")

    q5, q95 = np.nanpercentile(S_med_s, [5, 95])
    Delta = max(q95 - q5, _EPS)
    return S_med_s, float(Delta), float(sigma_nodes)


# -------------------------- ensemble catalogue -------------------------------
def build_ensemble_catalogue(
    sample_name: str,
    tier: str,
    age_grid: np.ndarray,
    goodness_runs: np.ndarray,
    *,
    orientation: str = "max",
    smooth_frac: float = 0.01,
    # range-agnostic knobs (fractions of N or of Δ)
    f_d: float = 0.05,            # min separation (nodes fraction)
    f_p: float = 0.03,            # min prominence (of Δ)
    f_v: float = 0.10,            # shallow valley merge (of smaller prominence)
    f_w: float = 0.08,            # vote window half-width (nodes fraction)
    w_min_nodes: int = 3,         # min FWHM in nodes
    # reproducibility
    support_min: float = 0.10,    # minimal support fraction
    r_min: int = 3,               # minimal number of runs
    f_r: float = 0.25,            # run vote gate (fraction of that run’s 5–95% range)
    # legacy signature compat:
    pen_ok_mask=None,
    **_ignored: Any,
) -> List[Dict]:
    """
    Manuscript picker:
      • candidates from smoothed ensemble median only,
      • explicit shoulder merge & endpoint handling,
      • keep only peaks reproducible across runs by local votes.
    """
    x = np.asarray(age_grid, float)
    S = np.asarray(goodness_runs, float)  # R × G
    R, G = S.shape
    if R == 0 or G < 3:
        return []

    # Oriented candidate curve (penalised or raw depending on caller)
    sign = 1.0 if str(orientation).lower().startswith("max") else -1.0
    S_med_s, Delta, _ = robust_ensemble_curve(S, smooth_frac=smooth_frac)
    if S_med_s.size == 0:
        return []
    y = sign * S_med_s

    # ----- candidate apices (interior only) -----
    dist_nodes  = max(1, int(np.ceil(f_d * G)))
    prom_abs    = max(float(f_p) * float(Delta), _EPS)

    pk, _ = find_peaks(y, distance=dist_nodes, width=w_min_nodes, prominence=prom_abs)
    # exclude exact endpoints
    pk = pk[(pk >= 1) & (pk <= G - 2)]
    if pk.size == 0:
        return []

    prom = peak_prominences(y, pk)[0]

    # ----- shoulder merge in index space + valley check -----
    order = np.argsort(pk)
    pk, prom = pk[order], prom[order]

    kept_idx = []
    i = 0
    while i < pk.size:
        winner = i
        j = i + 1
        while j < pk.size:
            sep = pk[j] - pk[winner]
            if sep >= dist_nodes:
                break
            lo, hi = int(min(pk[winner], pk[j])), int(max(pk[winner], pk[j]))
            valley = float(np.min(y[lo:hi+1]))
            lower_crest = min(y[pk[winner]], y[pk[j]])
            shallow = (lower_crest - valley) < f_v * min(prom[winner], prom[j])
            if shallow or sep < dist_nodes:
                # keep stronger by prominence; tie by higher crest
                if prom[j] > prom[winner] * (1.0 + 0.05):
                    winner = j
                elif abs(prom[j] - prom[winner]) <= 0.05 * max(prom[j], prom[winner]) and y[pk[j]] > y[pk[winner]]:
                    winner = j
                j += 1
            else:
                break
        kept_idx.append(int(pk[winner]))
        i = j

    pk = np.asarray(sorted(set(kept_idx)), int)
    if pk.size == 0:
        return []

    # ----- reproducibility by local voting -----
    vote_nodes = max(1, int(np.ceil(f_w * G)))
    out: List[Dict] = []

    # precompute per-run dynamic ranges for the f_r gate
    run_gate = []
    for r in range(R):
        y_r = sign * S[r]
        p5, p95 = np.nanpercentile(y_r, [5, 95])
        run_gate.append((float(p5), float(p95)))
    run_gate = np.asarray(run_gate, float)  # R × 2

    for j_c in pk:
        # refine on the *local crest* of the unoriented ensemble curve
        j_ref = _crest_index(S_med_s, int(j_c), half_win=2)
        age_ref = float(_parabolic_refine(x, S_med_s, int(j_ref)))

        # voting window on oriented per-run traces
        a = max(1, int(j_c) - vote_nodes)
        b = min(G - 2, int(j_c) + vote_nodes)

        votes = []
        for r in range(R):
            y_r = sign * S[r]
            win = y_r[a:b+1]
            if not np.isfinite(win).any():
                continue
            # choose midpoint of any flat local max, then refine there
            j_loc = _crest_index(win, int(np.argmax(win)), half_win=2)
            j_abs = int(a + (j_loc - 0))  # convert back to absolute index

            # relative-height gate using that run’s 5–95% range
            p5, p95 = run_gate[r, 0], run_gate[r, 1]
            thr = p5 + float(f_r) * max(p95 - p5, 0.0)
            if y_r[j_abs] < thr:
                continue

            votes.append(float(_parabolic_refine(x, S[r], j_abs)))

        support = len(votes) / float(R)
        if support < max(float(support_min), float(r_min) / float(max(R, 1))):
            continue

        if len(votes) >= 3:
            lo_ci, hi_ci = np.nanpercentile(votes, [2.5, 97.5])
        else:
            lo_ci = age_ref - float(x[1] - x[0])
            hi_ci = age_ref + float(x[1] - x[0])

        # recenter CI on refined apex and enforce ≥ one grid step width
        med_votes = float(np.median(votes)) if votes else age_ref
        delta = age_ref - med_votes
        lo_ci = min(lo_ci + delta, age_ref - float(x[1] - x[0]))
        hi_ci = max(hi_ci + delta, age_ref + float(x[1] - x[0]))

        out.append(dict(
            sample=sample_name,
            peak_no=0,  # set below
            age_ma=age_ref,
            ci_low=float(lo_ci),
            ci_high=float(hi_ci),
            support=float(support)
        ))

    out.sort(key=lambda d: d["age_ma"])
    for i, d in enumerate(out, 1):
        d["peak_no"] = i
    return out