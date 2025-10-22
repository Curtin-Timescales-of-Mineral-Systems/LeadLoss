# process/ensemble.py
from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import gaussian_filter1d

_EPS = 1e-12
DEBUG_CAND = False

# Set to 0.0 to keep all nearby apices. Set >0 (Ma) to auto-collapse close peaks later.
_MERGE_WITHIN_MA = 50.0

# -------------------------- utilities ----------------------------------------
def _parabolic_refine(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Quadratic vertex refinement using 3 points around index k; clamps to bracketing x."""
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
    if a == 0.0:
        return x1
    xv = -b / (2.0*a)
    lo, hi = (x0, x2) if x0 <= x2 else (x2, x0)
    return float(min(max(xv, lo), hi))

# -------------------------- per-run peaks (old API) --------------------------
def per_run_peaks(x: np.ndarray, y: np.ndarray, *,
                  prom_frac: float = 0.07,
                  min_dist: int = 3,
                  pad_left: bool = False,
                  min_width_nodes: int = 3,
                  require_full_prom: bool = True,
                  max_keep: Optional[int] = None,
                  fallback_global_max: bool = False,
                  return_details: bool = False):
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

    prom, lb, rb = peak_prominences(y, pk)
    width, _, _, _ = peak_widths(y, pk, rel_height=0.5)

    keep = (prom >= prom_thr) & (width >= float(min_width_nodes))
    if require_full_prom:
        keep &= (pk > 0) & (pk < (y.size - 1))

    pk, prom, width = pk[keep], prom[keep], width[keep]
    if pk.size == 0:
        return (np.array([], float), []) if return_details else np.array([], float)

    refined = np.array([_parabolic_refine(x, y, int(k)) for k in pk], float)
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

# -------------------------- robust ensemble curve (old semantics) ------------
def robust_ensemble_curve(S_runs: np.ndarray, smooth_frac: float = 0.01,
                          *_, **__) -> Tuple[np.ndarray, float, float]:
    """
    OLD API & semantics:
      returns (S_med_s, Delta, sigma_nodes)
        - S_med_s: smoothed median across runs (larger = better)
        - Delta:   robust global scale = p95 - p5 of S_med_s
        - sigma_nodes: Gaussian sigma in *nodes* used for smoothing

    Any extra args/kwargs are accepted & ignored for compatibility with
    newer call-sites that might pass aggregator/trim options.
    """
    S_runs = np.asarray(S_runs, float)
    if S_runs.ndim != 2 or S_runs.shape[1] < 3:
        return np.array([]), 0.0, 0.0
    G = S_runs.shape[1]
    S_med = np.nanmedian(S_runs, axis=0)
    sigma_nodes = max(0.5, float(smooth_frac) * float(G))
    S_med_s = gaussian_filter1d(S_med, sigma=sigma_nodes, mode="reflect")
    q5, q95 = np.nanpercentile(S_med_s, [5, 95])
    Delta = max(q95 - q5, _EPS)
    return S_med_s, float(Delta), float(sigma_nodes)

# -------------------------- old candidate merging helpers --------------------
def _merge_candidates(idx: np.ndarray, prom: np.ndarray,
                      S_curve: np.ndarray, dist_nodes: int, f_v: float
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge shoulder/nearby candidates in index-space using:
      - node distance (dist_nodes)
      - shallow valley check relative to local prominences (fraction f_v)
    Keep the stronger (by prominence; tie-break by higher crest).
    """
    idx = np.asarray(idx, int); prom = np.asarray(prom, float)
    if idx.size == 0:
        return idx, prom
    order = np.argsort(idx); idx, prom = idx[order], prom[order]
    keep_idx, keep_prom = [], []
    i, n = 0, idx.size
    while i < n:
        winner = i; j = i + 1
        while j < n:
            sep = idx[j] - idx[winner]
            lo = min(idx[winner], idx[j]); hi = max(idx[winner], idx[j])
            valley = float(np.min(S_curve[lo:hi+1]))
            hi_w = float(S_curve[idx[winner]]); hi_j = float(S_curve[idx[j]])
            shallow = (min(hi_w, hi_j) - valley) < f_v * min(prom[winner], prom[j])
            if not (sep < dist_nodes or shallow):
                break
            eps = 0.05
            if prom[j] > prom[winner] * (1.0 + eps):
                winner = j
            elif abs(prom[j] - prom[winner]) <= eps * max(prom[j], prom[winner]) and \
                 S_curve[idx[j]] > S_curve[idx[winner]]:
                winner = j
            j += 1
        keep_idx.append(int(idx[winner])); keep_prom.append(float(prom[winner]))
        i = j
    return np.asarray(keep_idx, int), np.asarray(keep_prom, float)

def _suppress_close_by_height_ma(c_idx: np.ndarray, c_prom: np.ndarray,
                                 S_curve: np.ndarray,
                                 age_grid: np.ndarray,
                                 min_sep_ma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    After merging in index-space, optionally collapse peaks closer than `min_sep_ma`
    (measured in Ma) keeping higher crest. Set min_sep_ma <= 0 to disable.
    """
    c_idx = np.asarray(c_idx, int); c_prom = np.asarray(c_prom, float)
    if c_idx.size <= 1 or min_sep_ma <= 0.0:
        return c_idx, c_prom
    order = np.argsort(S_curve[c_idx])[::-1]
    keep_mask = np.ones(c_idx.size, dtype=bool)
    for ii in order:
        if not keep_mask[ii]:
            continue
        ai = float(age_grid[c_idx[ii]])
        for jj in order:
            if jj == ii or not keep_mask[jj]:
                continue
            aj = float(age_grid[c_idx[jj]])
            if abs(aj - ai) <= min_sep_ma:
                keep_mask[jj] = False
    kept = np.where(keep_mask)[0]
    return c_idx[kept], c_prom[kept]

# -------------------------- build ensemble catalogue (old logic) -------------
def build_ensemble_catalogue(
    sample_name: str,
    tier: str,
    # Support BOTH old & new parameter names:
    age_grid: Optional[np.ndarray] = None,
    goodness_runs: Optional[np.ndarray] = None,
    *,
    age_grid_ma: Optional[np.ndarray] = None,     # newer name
    S_runs: Optional[np.ndarray] = None,          # newer name
    grid_step: Optional[float] = None,
    pen_ok_mask=None,
    orientation: str = 'max',
    smooth_frac: float = 0.01,
    f_d: float = 0.05, f_p: float = 0.03, f_v: float = 0.05, f_w: float = 0.08,
    w_min_nodes: int = 1,
    support_min: float = 0.10, r_min: int = 3, f_r: float = 0.35,
    per_run_prom_frac: float = 0.07,
    per_run_min_dist: int = 3,
    per_run_min_width: int = 3,
    per_run_require_full_prom: bool = False,
    vote_group_gap: Optional[float] = None,
    cand_curve: Optional[np.ndarray] = None,
    extra_seed_ages: Optional[np.ndarray] = None,
    _MERGE_WITHIN_MA_override: Optional[float] = None,
    **_ignored: Any,
) -> List[Dict]:
    """
    OLD picker restored verbatim (with a few harmless compat knobs):

      • Candidate curve = smoothed median of runs (unless `cand_curve` is provided
        in which case it is used as-is).
      • Candidates found with a global prominence floor = f_p * Delta, where
        Delta = p95 - p5 of the candidate curve.
      • Merge shoulder/nearby candidates in index space, then optionally suppress
        apices closer than `_MERGE_WITHIN_MA` (in Ma) by crest height.
      • Voting around each candidate window with a LOCAL relative-height gate f_r.
      • Per-run quality checks (relative prominence & width) as in the original.
      • Confidence interval from vote ages; re-centered to the refined apex.

    Returns: list of dicts {sample, peak_no, age_ma, ci_low, ci_high, support}
    """
    # ---- resolve dual naming for compatibility
    if age_grid is None and age_grid_ma is not None:
        age_grid = np.asarray(age_grid_ma, float)
    if goodness_runs is None and S_runs is not None:
        goodness_runs = np.asarray(S_runs, float)

    age_grid = np.asarray(age_grid, float)
    S = np.asarray(goodness_runs, float)
    if S.ndim != 2:
        raise AssertionError("goodness_runs/S_runs must be R x G")
    R, G = S.shape
    if R == 0 or G < 3:
        return []

    # old default grid_step (no _safe_step)
    if grid_step is None:
        dif = np.diff(age_grid)
        grid_step = float(np.median(dif)) if dif.size else 10.0

    dist_nodes  = max(1, int(np.ceil(float(f_d) * G)))
    vote_nodes  = max(1, int(np.ceil(float(f_w) * G)))
    width_floor = max(1, int(w_min_nodes))
    sign = 1.0 if str(orientation).lower().startswith('max') else -1.0

    # Candidate curve (old robust_ensemble_curve semantics)
    if cand_curve is None:
        S_med_s, Delta, _ = robust_ensemble_curve(S, smooth_frac=smooth_frac)
        if S_med_s.size == 0:
            return []
        S_cand = S_med_s
        S_work = sign * S_cand
        q5, q95 = np.nanpercentile(S_work, [5, 95])
        Delta = max(q95 - q5, _EPS)  # re-derive on oriented curve (as in old)
    else:
        S_cand = np.asarray(cand_curve, float).reshape(-1)
        if S_cand.size != G:
            raise AssertionError("cand_curve must match grid length")
        S_work = sign * S_cand
        q5, q95 = np.nanpercentile(S_work, [5, 95])
        Delta = max(q95 - q5, _EPS)

    prom_thr = max(float(f_p) * Delta, _EPS)

    # Initial candidate apices (on oriented curve)
    cand_idx, props = find_peaks(S_work, distance=dist_nodes,
                                 prominence=prom_thr, width=width_floor)
    # interior only if require full prominence later
    interior = (cand_idx >= 1) & (cand_idx <= G - 2)
    cand_idx  = cand_idx[interior]
    cand_prom = (props.get("prominences", np.zeros_like(cand_idx, float))[interior]
                 if cand_idx.size else np.array([], float))

    # Merge shoulders then collapse near-duplicates (in Ma) by crest height
    cand_idx, cand_prom = _merge_candidates(cand_idx, cand_prom, S_work, dist_nodes, float(f_v))
    mw = _MERGE_WITHIN_MA if _MERGE_WITHIN_MA_override is None else float(_MERGE_WITHIN_MA_override)
    cand_idx, cand_prom = _suppress_close_by_height_ma(cand_idx, cand_prom, S_work, age_grid, mw)

    if cand_idx.size == 0:
        return []
    order = np.argsort(cand_idx)
    cand_idx, cand_prom = cand_idx[order], cand_prom[order]
    uniq_idx, uniq_pos = np.unique(cand_idx, return_index=True)
    cand_idx, cand_prom = uniq_idx, cand_prom[uniq_pos]

    # Seed extra candidates from per-run peaks (exactly as the old code)
    votes, run_ids = [], []
    for r in range(R):
        y = S[r]
        if not np.isfinite(y).any():
            continue
        y_work = sign * y
        pk, _ = find_peaks(y_work, distance=int(per_run_min_dist))
        if pk.size:
            rng_r = np.nanpercentile(y_work, 95) - np.nanpercentile(y_work, 5)
            prom_thr_r = float(per_run_prom_frac) * max(rng_r, _EPS)
            prom_r, lb_r, rb_r = peak_prominences(y_work, pk)
            width_r, _, _, _ = peak_widths(y_work, pk, rel_height=0.5)
            keep_r = (prom_r >= prom_thr_r) & (width_r >= float(per_run_min_width))
            if per_run_require_full_prom:
                keep_r &= (lb_r > 0) & (rb_r < (G - 1))
            pk = pk[keep_r]
        if pk.size:
            votes.extend(age_grid[pk].tolist())
            run_ids.extend([r] * pk.size)

    if votes:
        vv = np.asarray(votes, float); rr = np.asarray(run_ids, int)
        order = np.argsort(vv); vv, rr = vv[order], rr[order]
        if vote_group_gap is None:
            vote_group_gap = max(3.0 * float(grid_step), 0.5 * vote_nodes * float(grid_step))

        a = 0
        while a < vv.size:
            b = a + 1
            while b < vv.size and vv[b] - vv[b - 1] <= vote_group_gap:
                b += 1
            runs_here = np.unique(rr[a:b]).size
            support_g = runs_here / float(R)
            if support_g >= max(float(support_min), float(r_min) / float(R)):
                age_med0 = float(np.median(vv[a:b]))
                j0 = int(np.argmin(np.abs(age_grid - age_med0)))
                lo = max(1, j0 - vote_nodes); hi = min(G - 2, j0 + vote_nodes)
                loc, _ = find_peaks(S_work[lo:hi + 1], distance=1)
                if loc.size:
                    win = S_work[lo:hi + 1]
                    abs_loc = lo + loc
                    prom_full, _, _ = peak_prominences(S_work, abs_loc)
                    k_prom = int(np.argmax(prom_full))
                    j_star = int(abs_loc[k_prom])
                    eps = 0.10
                    k_h = int(np.argmax(win[loc]))
                    if prom_full[k_prom] < (1.0 + eps) * prom_full[k_h]:
                        j_star = int(abs_loc[k_h])
                else:
                    j_star = int(np.clip(j0, 1, G - 2))

                if cand_idx.size == 0 or np.min(np.abs(cand_idx - j_star)) > dist_nodes:
                    cand_idx  = np.append(cand_idx, j_star)
                    prom_star = peak_prominences(S_work, np.asarray([j_star]))[0][0]
                    cand_prom = np.append(cand_prom, float(prom_star))
            a = b

        # Re-merge & dedup after seeding; keep prom aligned, then collapse near-duplicates
        cand_idx, cand_prom = _merge_candidates(cand_idx, cand_prom, S_work, dist_nodes, float(f_v))
        uniq_idx, uniq_pos = np.unique(np.sort(cand_idx), return_index=True)
        cand_idx, cand_prom = uniq_idx, cand_prom[uniq_pos]
        cand_idx, cand_prom = _suppress_close_by_height_ma(cand_idx, cand_prom, S_work, age_grid, mw)

    # Voting & CI around each candidate (old logic)
    out, gno = [], 0
    for j_c in cand_idx:
        lo = max(1, j_c - dist_nodes); hi = min(G - 2, j_c + dist_nodes)
        loc, _ = find_peaks(S_work[lo:hi + 1], distance=1)
        if loc.size == 0:
            continue
        win = S_work[lo:hi + 1]
        abs_loc = lo + loc
        prom_full, _, _ = peak_prominences(S_work, abs_loc)
        k_prom = int(np.argmax(prom_full))
        j_peak = int(abs_loc[k_prom])
        eps = 0.10
        k_h = int(np.argmax(win[loc]))
        if prom_full[k_prom] < (1.0 + eps) * prom_full[k_h]:
            j_peak = int(abs_loc[k_h])

        age_ref = float(_parabolic_refine(age_grid, S_cand, int(j_peak)))
        w = max(1, int(vote_nodes)); a = max(1, j_peak - w); b = min(G - 2, j_peak + w)

        vote_ages: List[float] = []
        for r in range(R):
            y = S[r]
            if not np.isfinite(y).any():
                continue

            y_work = sign * y
            wloc = max(1, int(0.5 * vote_nodes))
            aa = max(1, j_peak - wloc)
            bb = min(G - 2, j_peak + wloc)
            win_r = y_work[aa:bb + 1]

            loc_r, _ = find_peaks(win_r, distance=1)
            if loc_r.size == 0:
                continue
            abs_locs = aa + loc_r
            k_closest = int(np.argmin(np.abs(abs_locs - j_peak)))
            jloc = int(abs_locs[k_closest])

            if pen_ok_mask is not None and not pen_ok_mask[r, jloc]:
                continue

            # LOCAL gate only (protects narrow real crests)
            y5, y95 = np.nanpercentile(y_work, [5, 95])
            w5, w95 = np.nanpercentile(win_r,  [5, 95])
            thr_loc = w5 + float(f_r) * max(w95 - w5, 0.0)
            if y_work[jloc] < thr_loc:
                continue

            prom_all, _, _ = peak_prominences(win_r, loc_r)
            prom_sel = float(prom_all[k_closest])
            if prom_sel < 0.03 * max(w95 - w5, 0.0):
                continue
            if prom_sel < 0.02 * max(y95 - y5, 0.0):
                continue

            vote_ages.append(float(_parabolic_refine(age_grid, y, jloc)))

        support = len(vote_ages) / float(R)
        if support < max(float(support_min), float(r_min) / float(max(R, 1))):
            continue

        if len(vote_ages) >= 3:
            lo_ci, hi_ci = np.nanpercentile(vote_ages, [2.5, 97.5])
        else:
            lo_ci, hi_ci = age_ref - float(grid_step), age_ref + float(grid_step)

        # Re-center CI to the refined apex (old behavior)
        med_votes = float(np.median(vote_ages)) if vote_ages else age_ref
        delta = age_ref - med_votes
        lo_ci = min(lo_ci + delta, age_ref - float(grid_step))
        hi_ci = max(hi_ci + delta, age_ref + float(grid_step))

        gno += 1
        out.append(dict(sample=sample_name, peak_no=gno,
                        age_ma=age_ref, ci_low=float(lo_ci),
                        ci_high=float(hi_ci), support=float(support)))

    out.sort(key=lambda d: d["age_ma"])
    for i, d in enumerate(out, 1):
        d["peak_no"] = i
    return out

# -------------------------- (optional) compat wrapper ------------------------
def build_ensemble_catalogue_from_runs(
    runs, ages_y, *,
    use_penalised: bool = True,
    dist_frac: float = 0.05,
    vote_win_frac: float = 0.08,
    prom_frac: float = 0.03,
    per_run_min_width: int = 3,
    support_min: float = 0.10,
    r_min: int = 3,
    per_run_require_full_prom: bool = False,
    plateau_center: str | bool = "auto",
    cand_curve: Optional[np.ndarray] = None,
    smooth_frac: Optional[float] = None,
    valley_frac: Optional[float] = None,   # ignored (old picker has no valley merge)
    f_r: float = 0.35,
    per_run_prom_frac: float = 0.07,
    per_run_min_dist: int = 3,
    **_ignored: Any,
):
    """
    Thin convenience wrapper for newer code paths. It builds the goodness matrix
    from `runs` and delegates to the *old* `build_ensemble_catalogue` logic above.
    This does NOT change any peak picking behavior.
    """
    ages_y  = np.asarray(ages_y, float)
    ages_ma = ages_y / 1e6

    # Build S_runs = 1 - loss/score as in newer code examples
    rows: List[List[float]] = []
    for r in (runs or []):
        row = []
        for a in ages_y:
            st = r.statistics_by_pb_loss_age[a]
            val = (st.score if use_penalised else st.test_statistics[0])
            row.append(1.0 - float(val))
        rows.append(row)
    S_runs = np.asarray(rows, float)

    if smooth_frac is None:
        smooth_frac = 0.01

    return build_ensemble_catalogue(
        sample_name="",
        tier="",
        age_grid=ages_ma,
        goodness_runs=S_runs,
        orientation="max",
        smooth_frac=float(smooth_frac),
        f_d=float(dist_frac), f_p=float(prom_frac), f_v=0.05, f_w=float(vote_win_frac),
        w_min_nodes=int(per_run_min_width),
        support_min=float(support_min), r_min=int(r_min), f_r=float(f_r),
        per_run_prom_frac=float(per_run_prom_frac),
        per_run_min_dist=int(per_run_min_dist),
        per_run_min_width=int(per_run_min_width),
        per_run_require_full_prom=bool(per_run_require_full_prom),
        cand_curve=cand_curve,
        # keep old behavior; no valley merge; no “safe step”
    )
