"""Concordant-anchor helpers for defensible discordant clustering."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from process.cdc_tw import age_ma_from_pb207pb206, age_ma_from_u238pb206


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-6
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sig = 1.4826 * mad
    if not np.isfinite(sig) or sig <= 0.0:
        return 1e-6
    return sig


def concordant_ages_ma(spots):
    """
    Compute practical anchor ages (Ma) for concordant spots.

    Preference order:
    - 207Pb/206Pb age where finite
    - otherwise 238U/206Pb age
    """
    ages = []
    for s in spots:
        t = age_ma_from_pb207pb206(s.pbPbValue)
        if np.isfinite(t):
            ages.append(t)
            continue
        t2 = age_ma_from_u238pb206(s.uPbValue)
        ages.append(t2 if np.isfinite(t2) else np.nan)
    return np.asarray(ages, float)


def discordant_reference_ages_ma(spots):
    """
    Compute approximate ages (Ma) used only for anchor assignment.

    This is not the reported Pb-loss age. It is a simple scalar used to decide
    which concordant anchor is the closest plausible upper-age reference for
    each discordant grain.
    """
    ages = []
    for s in spots:
        t = age_ma_from_pb207pb206(s.pbPbValue)
        if np.isfinite(t):
            ages.append(t)
            continue
        t2 = age_ma_from_u238pb206(s.uPbValue)
        ages.append(t2 if np.isfinite(t2) else np.nan)
    return np.asarray(ages, float)


def _fit_gmm_bic_1d(ages_f: np.ndarray, max_pops: int):
    X = ages_f.reshape(-1, 1)
    from sklearn.mixture import GaussianMixture

    best_gm, best_bic, best_k = None, np.inf, 1
    for k in range(1, min(int(max_pops), ages_f.size) + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
        gm.fit(X)
        bic = gm.bic(X)
        if bic < best_bic:
            best_bic, best_gm, best_k = bic, gm, k
    labels_f = best_gm.predict(X) if best_gm is not None else np.zeros(ages_f.size, int)
    return labels_f.astype(int), int(best_k)


def _collapse_to_single_anchor(n_total: int, ages: np.ndarray):
    labels = np.zeros(int(n_total), int)
    pooled = ages[np.isfinite(ages)]
    med = float(np.median(pooled)) if pooled.size else np.nan
    return labels, 1, np.asarray([med], float)


# Minimum absolute separation required before nearby concordant submodes are
# treated as distinct anchors rather than a single broad concordant population.
MIN_ANCHOR_SEP_MA = 50.0


def cluster_concordant_populations(
    concordantSpots,
    max_pops: int = 6,
    min_frac: float = 0.10,
    min_sep_sigma: float = 2.0,
    min_sep_ma: float = MIN_ANCHOR_SEP_MA,
):
    """
    Estimate a small set of concordant anchor modes.

    Returns
    -------
    labels : np.ndarray[int]
        Label per concordant spot.
    k : int
        Number of retained anchor modes. Falls back to 1 if multimodality is
        weak or under-sized.
    means : np.ndarray[float]
        Retained anchor medians (Ma), sorted from youngest to oldest.
    """
    ages = concordant_ages_ma(concordantSpots)
    mask = np.isfinite(ages)
    ages_f = ages[mask]
    n_total = len(concordantSpots)

    if ages_f.size < 2:
        return _collapse_to_single_anchor(n_total, ages)

    try:
        labels_f, raw_k = _fit_gmm_bic_1d(ages_f, max_pops=max_pops)
    except Exception:
        return _collapse_to_single_anchor(n_total, ages)

    if raw_k <= 1:
        return _collapse_to_single_anchor(n_total, ages)

    min_points = max(3, int(math.ceil(float(min_frac) * ages_f.size)))
    kept: List[Dict] = []
    for k in range(raw_k):
        vals = ages_f[labels_f == k]
        if vals.size < min_points:
            continue
        kept.append(
            dict(
                raw_label=int(k),
                n=int(vals.size),
                median_ma=float(np.median(vals)),
                sigma_ma=float(_robust_sigma(vals)),
            )
        )

    if len(kept) <= 1:
        return _collapse_to_single_anchor(n_total, ages)

    kept.sort(key=lambda item: item["median_ma"])
    for left, right in zip(kept[:-1], kept[1:]):
        abs_sep = abs(float(right["median_ma"]) - float(left["median_ma"]))
        if abs_sep < float(min_sep_ma):
            return _collapse_to_single_anchor(n_total, ages)
        denom = max(float(left["sigma_ma"]), float(right["sigma_ma"]), 1e-6)
        sep = abs_sep / denom
        if sep < float(min_sep_sigma):
            return _collapse_to_single_anchor(n_total, ages)

    labels = np.zeros(n_total, int)
    remap = {item["raw_label"]: i for i, item in enumerate(kept)}
    if mask.any():
        labels_f_out = np.zeros_like(labels_f, int)
        for raw_label, new_label in remap.items():
            labels_f_out[labels_f == raw_label] = int(new_label)
        labels[mask] = labels_f_out

    means = np.asarray([item["median_ma"] for item in kept], float)
    return labels, len(kept), means


def assign_discordant_to_anchors(
    discordantSpots,
    anchor_means_ma,
    ambiguity_ratio: float = 1.25,
):
    """
    Hard-assign each discordant grain to one anchor where clearly preferred.

    Returns
    -------
    labels : np.ndarray[int]
        Anchor index for each discordant grain, or -1 if ambiguous / invalid.
    summary : list[dict]
        One summary row per discordant grain with its assignment status.
    """
    anchors = np.asarray(anchor_means_ma, float)
    labels = np.full(len(discordantSpots), -1, dtype=int)
    summary = []

    if anchors.size == 0 or not np.isfinite(anchors).any():
        return labels, summary

    approx_ages = discordant_reference_ages_ma(discordantSpots)

    if anchors.size == 1:
        for i, age in enumerate(approx_ages):
            ok = bool(np.isfinite(age))
            labels[i] = 0 if ok else -1
            summary.append(
                dict(
                    grain=i,
                    approx_age_ma=float(age) if ok else np.nan,
                    assigned_anchor=0 if ok else None,
                    ambiguous=not ok,
                )
            )
        return labels, summary

    for i, age in enumerate(approx_ages):
        if not np.isfinite(age):
            summary.append(
                dict(
                    grain=i,
                    approx_age_ma=np.nan,
                    assigned_anchor=None,
                    ambiguous=True,
                )
            )
            continue

        distances = np.abs(anchors - float(age))
        order = np.argsort(distances)
        best = int(order[0])
        second = int(order[1])
        d1 = float(distances[best])
        d2 = float(distances[second])
        ratio = np.inf if d1 <= 1e-9 else d2 / d1

        if ratio >= float(ambiguity_ratio):
            labels[i] = best
            summary.append(
                dict(
                    grain=i,
                    approx_age_ma=float(age),
                    assigned_anchor=best,
                    ambiguous=False,
                    best_distance_ma=d1,
                    second_distance_ma=d2,
                    separation_ratio=ratio,
                )
            )
        else:
            summary.append(
                dict(
                    grain=i,
                    approx_age_ma=float(age),
                    assigned_anchor=None,
                    ambiguous=True,
                    best_distance_ma=d1,
                    second_distance_ma=d2,
                    separation_ratio=ratio,
                )
            )

    return labels, summary


def assign_discordant_to_populations(discordantSpots, pop_means_ma):
    """
    Deprecated backward-compatible wrapper.

    The defensible clustering path should call assign_discordant_to_anchors(...)
    directly so ambiguity is visible to the caller.
    """
    labels, _ = assign_discordant_to_anchors(discordantSpots, pop_means_ma)
    return labels
