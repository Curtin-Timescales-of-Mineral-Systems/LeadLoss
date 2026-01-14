"""Concordant-population clustering helpers (optional CDC mode).

Extracted from process/processing.py.
"""

from __future__ import annotations

import numpy as np

from process.cdc_tw import age_ma_from_pb207pb206, age_ma_from_u238pb206


def concordant_ages_ma(spots):
    """
    Compute approximate ages (Ma) for concordant spots, used for population clustering.

    Tries 207Pb/206Pb age first (robust for old SHRIMP data), falling back to
    238U/206Pb if needed. Returns NaN where neither estimate is finite.
    """
    ages = []
    for s in spots:
        t = age_ma_from_pb207pb206(s.pbPbValue)
        if np.isfinite(t):
            ages.append(t)
        else:
            t2 = age_ma_from_u238pb206(s.uPbValue)
            ages.append(t2 if np.isfinite(t2) else np.nan)
    return np.asarray(ages, float)


def cluster_concordant_populations(concordantSpots, max_pops: int = 3):
    """
    Cluster concordant ages into up to max_pops populations using 1-D GMM+BIC.

    Returns
    -------
    labels : np.ndarray[int]
        Label per concordant spot (same length as concordantSpots).
    k : int
        Number of populations chosen.
    means : np.ndarray[float]
        GMM component means (Ma).
    """
    ages = concordant_ages_ma(concordantSpots)
    mask = np.isfinite(ages)
    ages_f = ages[mask]
    if ages_f.size < 2:
        return np.zeros(len(concordantSpots), int), 1, np.array([np.nan])

    X = ages_f.reshape(-1, 1)
    try:
        from sklearn.mixture import GaussianMixture
        best_gm, best_bic, best_k = None, np.inf, 1
        for k in range(1, min(max_pops, ages_f.size) + 1):
            gm = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
            gm.fit(X)
            bic = gm.bic(X)
            if bic < best_bic:
                best_bic, best_gm, best_k = bic, gm, k
        labels_f = best_gm.predict(X)
        labels = np.zeros(len(concordantSpots), int)
        labels[mask] = labels_f
        means = best_gm.means_.ravel()
        return labels, int(best_k), means
    except Exception:
        # fallback: single population
        return np.zeros(len(concordantSpots), int), 1, np.array([np.nan])


def assign_discordant_to_populations(discordantSpots, pop_means_ma):
    """Assign each discordant spot to the nearest concordant population in age."""
    labels = np.zeros(len(discordantSpots), int)
    if pop_means_ma is None or not np.isfinite(pop_means_ma).any():
        return labels
    pops = np.asarray(pop_means_ma, float)
    for i, s in enumerate(discordantSpots):
        t = age_ma_from_pb207pb206(s.pbPbValue)
        if not np.isfinite(t):
            t = age_ma_from_u238pb206(s.uPbValue)
        if not np.isfinite(t):
            labels[i] = 0
        else:
            labels[i] = int(np.argmin(np.abs(pops - t)))
    return labels
