from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
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

ADAPT_FR = True

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

    # Light smoothing: sigma as a fraction of N, but cap at 2 nodes for crisp crests.
    sigma_nodes = float(smooth_frac) * float(G)
    sigma_nodes = min(sigma_nodes, 2.0) 
    S_med_s = gaussian_filter1d(S_med, sigma=sigma_nodes, mode="reflect")

    q5, q95 = np.nanpercentile(S_med_s, [5, 95])
    Delta = max(q95 - q5, _EPS)
    return S_med_s, float(Delta), float(sigma_nodes)

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
    # minimum ensemble dynamic range below which we do not pick peaks
    delta_min: float = 0.0,
    # only keep peaks whose crest is at least this fraction
    # of the highest crest on the ensemble curve.
    height_frac: float = 0.0,   # 0.0 = disabled
    # NEW: optional global optimum ages per run (Ma)
    optima_ma: Optional[np.ndarray] = None,
    # optional injection of per-run peak lists
    per_run_peaks_list: Optional[List[np.ndarray]] = None,
    **_ignored: Any,
) -> List[Dict]:
    """
    Ensemble picker:
      • candidate humps from a *coarse* ensemble median,
      • at most ONE fine-scale peak per coarse hump,
      • keep only peaks reproducible across runs by local votes,
      • CIs from per-peak optima distribution.
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

    # ---------------- Coarse curve: define the REAL major humps -------------
    # Strong smoothing to wash out shoulders
    sigma_coarse = max(5.0, 0.03 * float(G))  # ≈6 nodes for G=200
    S_med_coarse = gaussian_filter1d(S_med_s, sigma=sigma_coarse, mode="reflect")
    y_coarse = sign * S_med_coarse

    prom_abs = max(float(f_p) * float(Delta), _EPS)

    # First attempt: reasonably strict prominence
    pk_major, _ = find_peaks(
        y_coarse,
        distance=max(1, int(0.10 * G)),   # major peaks must be well separated
        prominence=prom_abs,
    )

    # If that finds nothing, relax once – but NEVER fall back to fine peaks
    if pk_major.size == 0:
        pk_major, _ = find_peaks(
            y_coarse,
            distance=max(1, int(0.10 * G)),
            prominence=prom_abs * 0.5,
        )

    if pk_major.size == 0:
        # Fallback: treat the global max of the fine curve as one hump
        j0 = int(np.nanargmax(y))  # y = sign * S_med_s (fine ensemble curve)
        if j0 <= 0 and G > 2:
            j0 = 1
        elif j0 >= G - 1 and G > 2:
            j0 = G - 2
        pk_major = np.array([j0], dtype=int)

    # ---------------- Fine-scale candidate peaks ----------------------------
    family_nodes = max(1, int(np.ceil(f_d * G)))       # peak-family window (nodes)
    vote_nodes   = max(1, int(np.ceil(f_w * G)))       # half-width of vote window
    edge_guard   = 0

    pk_fine, _ = find_peaks(y, distance=1, width=w_min_nodes, prominence=prom_abs)

    # If no interior fine peaks, fall back to using the coarse hump(s)
    if pk_fine.size == 0:
        pk_fine = np.array(pk_major, dtype=int)

    if pk_fine.size == 0:
        # Still nothing at all → give up
        return []

    # --------- FORCE: at most one fine peak per coarse hump -----------------
    # window size within which a fine peak can represent a coarse hump
    major_rad = max(1, int(0.05 * G))  # ~10 nodes for G=200 (~100 Ma on 1–2000[200])

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
        # coarse humps exist but none supported by a fine peak ⇒ no peaks
        return []

    chosen_idx = sorted(set(chosen_idx))
    pk = pk_fine[chosen_idx]

    # guard against edges
    pk = pk[(pk >= edge_guard) & (pk <= G - 1 - edge_guard)]
    if pk.size == 0:
        return []

    # ---------------- Shoulder merge ---------------------
    prom, left_bases, right_bases = peak_prominences(y, pk)
    order = np.argsort(pk)
    pk          = pk[order]
    prom        = prom[order]
    left_bases  = left_bases[order]
    right_bases = right_bases[order]

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
            pk          = pk[keep]
            prom        = prom[keep]
            left_bases  = left_bases[keep]
            right_bases = right_bases[keep]
            if pk.size == 0:
                return []

    # ---------------- Per-run gates + voting as before ----------------------
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

    out: List[Dict] = []
    optima_ma = np.asarray(optima_ma, float) if optima_ma is not None else None

    for j_idx, j_c in enumerate(pk):
        # refine apex on unoriented ensemble curve
        j_ref   = _crest_index(S_med_s, int(j_c), half_win=2)
        age_ref = float(_parabolic_refine(x, S_med_s, int(j_ref)))

        a = max(1, int(j_c) - vote_nodes)
        b = min(G - 2, int(j_c) + vote_nodes)
        lo_ma_win, hi_ma_win = float(x[a]), float(x[b])

        # adaptive f_r
        if ADAPT_FR:
            runs_with_cand_peaks = sum(
                1 for pr in per_run_peaks_list
                if (np.asarray(pr).size and
                    ((np.asarray(pr) >= lo_ma_win) & (np.asarray(pr) <= hi_ma_win)).any())
            )
            min_voter_runs = max(int(r_min), int(max(1, round(0.05 * R))))
            f_r_eff = float(f_r) if runs_with_cand_peaks >= min_voter_runs else min(float(f_r), 0.10)
        else:
            f_r_eff = float(f_r)

        votes: List[float] = []
        optima_for_peak: List[float] = []

        for r in range(R):
            y_r = sign * S[r]
            pr_list = per_run_peaks_list[r] if r < len(per_run_peaks_list) else None
            pr_list = np.asarray(pr_list, float) if pr_list is not None else np.array([], float)
            cand = pr_list[(pr_list >= lo_ma_win) & (pr_list <= hi_ma_win)]

            if cand.size == 0:
                # fallback : take the local crest in [a,b], but ONLY inside this hump
                j_abs = int(a + np.argmax(y_r[a:b+1]))
                p5, p95 = run_gate[r, 0], run_gate[r, 1]
                thr = p5 + f_r_eff * max(p95 - p5, 0.0)
                if y_r[j_abs] < thr:
                    continue
                age_vote = float(_parabolic_refine(x, S[r], j_abs))
            else:
                j_choices = [int(np.argmin(np.abs(x - p))) for p in cand]
                heights_r = [float(y_r[jj]) for jj in j_choices]
                j_abs     = int(j_choices[int(np.argmax(heights_r))])

                p5, p95 = run_gate[r, 0], run_gate[r, 1]
                thr = p5 + f_r_eff * max(p95 - p5, 0.0)
                if y_r[j_abs] < thr:
                    continue
                age_vote = float(_parabolic_refine(x, S[r], j_abs))

            votes.append(age_vote)

            if optima_ma is not None and 0 <= r < optima_ma.size:
                opt_val = float(optima_ma[r])
                if np.isfinite(opt_val) and (lo_ma_win <= opt_val <= hi_ma_win):
                    optima_for_peak.append(opt_val)

        support = len(votes) / float(R)

        if support < max(float(support_min), float(r_min) / float(max(R, 1))):
            continue

        # CI from optima / votes
        step = float(x[1] - x[0])
        if optima_ma is not None and len(optima_for_peak) >= 3:
            lo_ci, hi_ci = np.nanpercentile(optima_for_peak, [2.5, 97.5])
        elif len(votes) >= 3:
            lo_ci, hi_ci = np.nanpercentile(votes, [2.5, 97.5])
        else:
            lo_ci, hi_ci = age_ref - step, age_ref + step

        if len(votes) >= 1:
            med_votes = float(np.median(votes))
        else:
            med_votes = age_ref
        shift = age_ref - med_votes
        lo_ci = float(lo_ci + shift)
        hi_ci = float(hi_ci + shift)

        lo_ci = max(lo_ci, float(x[0]))
        hi_ci = min(hi_ci, float(x[-1]))
        if (not np.isfinite(lo_ci)) or (not np.isfinite(hi_ci)) or (hi_ci <= lo_ci):
            lo_ci, hi_ci = max(age_ref - step, float(x[0])), min(age_ref + step, float(x[-1]))
        if (hi_ci - lo_ci) < (0.75 * step):
            lo_ci, hi_ci = max(age_ref - step, float(x[0])), min(age_ref + step, float(x[-1]))

        min_steps = 5.0
        min_width = min_steps * step
        if (hi_ci - lo_ci) < min_width:
            lo_ci = max(age_ref - 0.5 * min_width, float(x[0]))
            hi_ci = min(age_ref + 0.5 * min_width, float(x[-1]))

        out.append(dict(
            sample=sample_name,
            peak_no=0,  # set after sort
            age_ma=age_ref,
            ci_low=float(lo_ci),
            ci_high=float(hi_ci),
            support=float(support),
        ))

    out.sort(key=lambda d: d["age_ma"])
    for i, d in enumerate(out, 1):
        d["peak_no"] = i
    return out