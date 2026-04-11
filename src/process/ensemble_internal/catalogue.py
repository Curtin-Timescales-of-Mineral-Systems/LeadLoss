"""Catalogue assembly for ensemble CDC peak picking."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from process.ensemble_internal.curve import per_run_peaks, robust_ensemble_curve
from process.ensemble_internal.primitives import (
    _COARSE_SIGMA_GRID_FRAC,
    _DEGENERATE_CI_GRID_FRAC,
    _EPS,
    _append_diagnostic_peak,
    _apply_plateau_onset_adjustment,
    _basin_bounds_from_peaks,
    _crest_index,
    _estimate_window_support,
    _half_prominence_edges,
    _parabolic_refine,
    _step_from_grid,
    find_peaks,
    peak_prominences,
    peak_widths,
)


def _score_candidate_peaks(
    pk,
    x,
    S,
    S_med_s,
    sign,
    R,
    per_run_peaks_list,
    optima_ma,
    run_gate,
    f_r,
    support_min,
    r_min,
    left_bounds,
    right_bounds,
    j_ref_all,
    age_ref_all,
    left_edge_ma_all,
    right_edge_ma_all,
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
                j_abs = int(a + np.argmax(y_r[a:b + 1]))
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
                diagnostic_rows,
                x,
                S_med_s,
                int(j_ref),
                reason="low_support",
                direct_support=float(support),
                winner_support=float(len(optima_for_peak) / float(max(R, 1))) if optima_ma is not None else np.nan,
                ci_low=lo_ma_win,
                ci_high=hi_ma_win,
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

        out.append(
            dict(
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
            )
        )

    return out


def build_ensemble_catalogue(
    sample_name: str,
    tier: str,
    age_grid: np.ndarray,
    goodness_runs: np.ndarray,
    *,
    orientation: str = "max",
    smooth_frac: float = 0.01,
    f_d: float = 0.05,
    f_p: float = 0.03,
    f_v: float = 0.10,
    f_w: float = 0.05,
    w_min_nodes: int = 3,
    support_min: float = 0.10,
    r_min: int = 3,
    f_r: float = 0.25,
    per_run_prom_frac: float = 0.06,
    per_run_min_dist: int = 3,
    per_run_min_width: int = 3,
    per_run_require_full_prom: bool = False,
    plateau_onset_mode: str = "off",
    plateau_onset_min_width_frac: float = 0.20,
    plateau_onset_min_right_left_ratio: float = 1.30,
    plateau_onset_blend_frac: float = 0.50,
    delta_min: float = 0.0,
    height_frac: float = 0.0,
    optima_ma: Optional[np.ndarray] = None,
    per_run_peaks_list: Optional[List[np.ndarray]] = None,
    merge_per_hump: bool = True,
    merge_shoulders: bool = True,
    diagnostic_rows: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Build the final supported-peak catalogue from the ensemble goodness stack."""
    x = np.asarray(age_grid, float)
    S = np.asarray(goodness_runs, float)
    if S.ndim != 2 or S.shape[0] == 0 or S.shape[1] < 3:
        return []

    R, G = S.shape
    sign = 1.0 if str(orientation).lower().startswith("max") else -1.0

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

    family_nodes = max(1, int(np.ceil(f_d * G)))
    edge_guard = 0

    pk_fine, _ = find_peaks(y, distance=1, width=w_min_nodes, prominence=prom_abs)

    if pk_fine.size == 0 and merge_per_hump:
        pk_fine = np.array(pk_major, dtype=int)

    if pk_fine.size == 0:
        return []

    if merge_per_hump:
        chosen_idx: List[int] = []
        for pM in pk_major:
            idx_in_win = np.where(np.abs(pk_fine - int(pM)) <= major_rad)[0]
            if idx_in_win.size == 0:
                continue
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

    pk = pk[(pk >= edge_guard) & (pk <= G - 1 - edge_guard)]
    if pk.size == 0:
        return []

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
        pk = pk[order]
        prom = prom[order]
        left_bases = left_bases[order]
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
                        abs(prom[j] - prom[winner]) <= 0.05 * max(prom[j], prom[winner])
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
        pk = pk[kept_idx_local]
        prom = prom[kept_idx_local]
        left_bases = left_bases[kept_idx_local]
        right_bases = right_bases[kept_idx_local]
        if pk.size == 0:
            return []

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
            pk = pk[keep]
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
                x,
                y_r,
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
        pk,
        x,
        S,
        S_med_s,
        sign,
        R,
        per_run_peaks_list,
        optima_ma,
        run_gate,
        f_r,
        support_min,
        r_min,
        left_bounds,
        right_bounds,
        j_ref_all,
        age_ref_all,
        left_edge_ma_all,
        right_edge_ma_all,
        plateau_onset_mode,
        plateau_onset_min_width_frac,
        plateau_onset_min_right_left_ratio,
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
