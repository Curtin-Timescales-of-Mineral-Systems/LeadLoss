"""Peak-catalogue filtering and deduplication helpers for CDC.

These functions operate on already-detected candidate peaks:
- merge overlapping candidates
- recompute winner-vote support
- apply support thresholds
- deduplicate plateau-like duplicates
- track which candidates were rejected at each step
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from process.cdcConfig import PLATEAU_DEDUPE, PLATEAU_DEDUPE_MIN_OVERLAP_FRAC, PLATEAU_DEDUPE_RADIUS_STEPS


def _collapse_ci_clusters(rows, width_mult: float = 1.0):
    """
    Collapse chains of peaks that are either CI-overlapping or close in age.
    """
    if not rows or len(rows) <= 1:
        return rows

    rows = sorted(rows, key=lambda r: float(r["age_ma"]))
    clusters = []
    current_cluster = [dict(rows[0])]

    def _same_cluster(a, b) -> bool:
        lo1, hi1 = float(a["ci_low"]), float(a["ci_high"])
        lo2, hi2 = float(b["ci_low"]), float(b["ci_high"])
        a1, a2 = float(a["age_ma"]), float(b["age_ma"])
        overlap = (lo2 <= hi1) and (hi2 >= lo1)
        w1 = max(hi1 - lo1, 0.0)
        w2 = max(hi2 - lo2, 0.0)
        sep = abs(a2 - a1)
        close = (w1 > 0.0 or w2 > 0.0) and sep <= width_mult * max(w1, w2)
        return overlap or close

    for r in rows[1:]:
        last = current_cluster[-1]
        if _same_cluster(last, r):
            current_cluster.append(dict(r))
        else:
            clusters.append(current_cluster)
            current_cluster = [dict(r)]
    clusters.append(current_cluster)

    collapsed = []
    for cl in clusters:
        def _score(rr):
            sup = float(rr.get("support", 0.0))
            width = float(rr["ci_high"]) - float(rr["ci_low"])
            return (sup, -width)

        best = max(cl, key=_score)
        collapsed.append(dict(best))

    for i, rr in enumerate(collapsed, 1):
        rr["peak_no"] = i
    return collapsed


def _recompute_winner_support(rows, optima_ma, ages_ma, min_support=None):
    """
    Recompute support as winner-vote fraction from per-run optima.
    """
    if not rows:
        return rows

    rows = sorted([dict(r) for r in rows], key=lambda rr: float(rr["age_ma"]))
    centers = np.array([float(r["age_ma"]) for r in rows], float)
    ci_lo = np.array([float(r["ci_low"]) for r in rows], float)
    ci_hi = np.array([float(r["ci_high"]) for r in rows], float)
    counts = np.zeros(len(rows), float)

    opts = np.asarray(optima_ma, float)
    opts = opts[np.isfinite(opts)]
    if opts.size == 0:
        return rows

    step = float(np.median(np.diff(ages_ma))) if np.asarray(ages_ma).size >= 2 else 5.0
    cap = max(3.0 * step, 30.0)

    for o in opts:
        in_ci = np.where((o >= ci_lo) & (o <= ci_hi))[0]
        if in_ci.size > 0:
            loc = np.argmin(np.abs(centers[in_ci] - o))
            counts[int(in_ci[loc])] += 1.0
            continue

        j = int(np.argmin(np.abs(centers - o)))
        if abs(float(centers[j]) - float(o)) <= cap:
            counts[j] += 1.0

    denom = float(max(opts.size, 1))
    out = []
    for i, r in enumerate(rows):
        winner_sup = float(counts[i] / denom)
        direct_sup = float(r.get("direct_support", r.get("support", float("nan"))))
        if not np.isfinite(direct_sup):
            direct_sup = winner_sup
        rr = dict(
            r,
            direct_support=direct_sup,
            winner_support=winner_sup,
            support=direct_sup,
        )
        if (min_support is not None) and (direct_sup < float(min_support)):
            continue
        out.append(rr)

    out = sorted(out, key=lambda rr: float(rr["age_ma"]))
    for i, rr in enumerate(out, 1):
        rr["peak_no"] = i
    return out


def _support_score(row, mode):
    """Return the support score used for inclusion filtering."""
    mode = str(mode).strip().upper()
    winner = float(row.get("winner_support", row.get("support", 0.0)))
    direct = float(row.get("direct_support", row.get("support", 0.0)))
    if mode == "DIRECT":
        return direct
    if mode == "MAX":
        return max(winner, direct)
    return winner


def _apply_support_filter(rows, min_support, mode):
    """Filter rows by the configured support metric."""
    if not rows:
        return rows
    out = []
    for r in rows:
        score = float(_support_score(r, mode))
        rr = dict(r, filter_support=score)
        if score >= float(min_support):
            out.append(rr)
    out = sorted(out, key=lambda rr: float(rr["age_ma"]))
    for i, rr in enumerate(out, 1):
        rr["peak_no"] = i
    return out


def _step_ma_from_grid(ages_ma):
    ages_ma = np.asarray(ages_ma, float)
    if ages_ma.size >= 2:
        step = float(np.median(np.diff(ages_ma)))
        if np.isfinite(step) and step > 0.0:
            return step
    return 5.0


def _row_match_index(target_row, candidates, used, tol_ma):
    """Return the unmatched candidate index closest in age to `target_row`."""
    tgt = float(target_row.get("age_ma", np.nan))
    if not np.isfinite(tgt):
        return None
    best_i = None
    best_d = None
    for i, row in enumerate(candidates):
        if used[i]:
            continue
        age = float(row.get("age_ma", np.nan))
        if not np.isfinite(age):
            continue
        d = abs(age - tgt)
        if d <= tol_ma and (best_d is None or d < best_d):
            best_i = i
            best_d = d
    return best_i


def _append_rejected_peak(rejected_rows, row, reason_code):
    """Append one rejected candidate row unless a near-identical age already exists."""
    age = float(row.get("age_ma", np.nan))
    if not np.isfinite(age):
        return
    for rr in rejected_rows:
        if abs(float(rr.get("age_ma", np.nan)) - age) <= 1e-6:
            return
    direct = float(row.get("direct_support", row.get("support", np.nan)))
    winner = float(row.get("winner_support", row.get("support", np.nan)))
    rejected_rows.append(
        dict(
            age_ma=age,
            ci_low=float(row.get("ci_low", np.nan)),
            ci_high=float(row.get("ci_high", np.nan)),
            direct_support=direct,
            winner_support=winner,
            reason=str(reason_code),
        )
    )


def _capture_rejected_step(before_rows, after_rows, rejected_rows, reason_code, ages_ma):
    """Record rows lost between two stages as rejected candidates."""
    if not before_rows:
        return
    after_rows = [dict(r) for r in (after_rows or [])]
    used = [False] * len(after_rows)
    tol = max(0.51 * _step_ma_from_grid(ages_ma), 1e-6)
    for row in before_rows:
        j = _row_match_index(row, after_rows, used, tol)
        if j is None:
            _append_rejected_peak(rejected_rows, row, reason_code)
        else:
            used[j] = True


def _plateau_dedupe_rows(rows, ages_ma):
    """
    Collapse near-identical peaks that sit on the same broad/flat crest.
    """
    if (not rows) or len(rows) <= 1:
        return rows

    rows = sorted([dict(r) for r in rows], key=lambda rr: float(rr["age_ma"]))
    step = float(np.median(np.diff(ages_ma))) if np.asarray(ages_ma).size >= 2 else 5.0
    near_ma = max(float(PLATEAU_DEDUPE_RADIUS_STEPS) * step, 20.0)
    min_ov = float(PLATEAU_DEDUPE_MIN_OVERLAP_FRAC)

    def _width(rr):
        return max(0.0, float(rr["ci_high"]) - float(rr["ci_low"]))

    def _overlap_frac(a, b):
        lo = max(float(a["ci_low"]), float(b["ci_low"]))
        hi = min(float(a["ci_high"]), float(b["ci_high"]))
        ov = max(0.0, hi - lo)
        wa, wb = _width(a), _width(b)
        denom = max(min(wa, wb), 1e-9)
        return ov / denom

    def _score(rr):
        win = float(rr.get("winner_support", rr.get("support", 0.0)))
        direct = float(rr.get("direct_support", rr.get("support", 0.0)))
        width = _width(rr)
        return (win, direct, -width)

    deduped = [dict(rows[0])]
    for rr in rows[1:]:
        prev = deduped[-1]
        sep = abs(float(rr["age_ma"]) - float(prev["age_ma"]))
        same_crest = (sep <= near_ma) and (_overlap_frac(prev, rr) >= min_ov)
        if same_crest:
            if _score(rr) > _score(prev):
                deduped[-1] = dict(rr)
        else:
            deduped.append(dict(rr))

    for i, rr in enumerate(deduped, 1):
        rr["peak_no"] = i
    return deduped


def _dedupe_rejected_rows(rejected_rows, ages_ma):
    """Collapse duplicate rejected-peak records generated in adjacent steps."""
    if not rejected_rows:
        return []

    deduped_rejected: List[Dict] = []
    seen = set()
    step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 5.0
    tol = max(0.51 * step, 1e-6)
    for row in rejected_rows:
        age = float(row.get("age_ma", np.nan))
        reason = str(row.get("reason", ""))
        age_key = round(age / tol) if np.isfinite(age) else None
        key = (reason, age_key)
        if key in seen:
            continue
        seen.add(key)
        deduped_rejected.append(row)
    return deduped_rejected


def _refresh_support_filtered_catalogues(
    raw,
    pen,
    rows_for_ui,
    rejected_rows,
    ages_ma,
    support_floor,
    support_filter_mode,
    optima_ma_ui_vote,
    *,
    label=None,
):
    """Recompute winner support and apply the support filter to all catalogues."""
    for surf in (raw, pen):
        surf.rows = _recompute_winner_support(surf.rows, surf.optima_ma, ages_ma, min_support=None)
        surf.rows = _apply_support_filter(surf.rows, support_floor, support_filter_mode)

    rows_for_ui = _recompute_winner_support(rows_for_ui, optima_ma_ui_vote, ages_ma, min_support=None)
    pre_rows = [dict(r) for r in rows_for_ui]
    rows_for_ui = _apply_support_filter(rows_for_ui, support_floor, support_filter_mode)
    if label:
        _capture_rejected_step(pre_rows, rows_for_ui, rejected_rows, label, ages_ma)
    return rows_for_ui, rejected_rows


def _run_filter_pipeline(
    raw,
    pen,
    rows_for_ui,
    rejected_rows,
    ages_ma,
    merge_nearby,
    support_floor,
    support_filter_mode,
    optima_ma_ui_vote,
):
    """Apply overlap merging, support filtering, and plateau deduplication."""
    if merge_nearby:
        pre_merge_ui = [dict(r) for r in rows_for_ui]
        rows_for_ui = _collapse_ci_clusters(rows_for_ui)
        for surf in (raw, pen):
            surf.rows = _collapse_ci_clusters(surf.rows)
        _capture_rejected_step(pre_merge_ui, rows_for_ui, rejected_rows, "merged_overlapping_candidates", ages_ma)

    rows_for_ui, rejected_rows = _refresh_support_filtered_catalogues(
        raw,
        pen,
        rows_for_ui,
        rejected_rows,
        ages_ma,
        support_floor,
        support_filter_mode,
        optima_ma_ui_vote,
        label="low_support",
    )

    if (not merge_nearby) and PLATEAU_DEDUPE:
        for surf in (raw, pen):
            surf.rows = _plateau_dedupe_rows(surf.rows, ages_ma)
        pre_dedupe_ui = [dict(r) for r in rows_for_ui]
        rows_for_ui = _plateau_dedupe_rows(rows_for_ui, ages_ma)
        _capture_rejected_step(pre_dedupe_ui, rows_for_ui, rejected_rows, "plateau_duplicate", ages_ma)
        rows_for_ui, rejected_rows = _refresh_support_filtered_catalogues(
            raw,
            pen,
            rows_for_ui,
            rejected_rows,
            ages_ma,
            support_floor,
            support_filter_mode,
            optima_ma_ui_vote,
            label="low_support",
        )

    return rows_for_ui, rejected_rows
