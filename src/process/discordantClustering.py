import math
from typing import Dict, Tuple

import numpy as np
from process import calculations

# -----------------------------------------------------------------------------
# Tunable defaults
# -----------------------------------------------------------------------------
# Minimum absolute number of points per cluster (adaptive gates may raise this)
DC_MIN_POINTS = 4

# Minimum separation (in σ units) used by cluster_proxies_years; actual threshold
# is n-dependent via _adaptive_gates, this is kept for documentation.
DC_MIN_SEP_SIGMA = 1.2

# GMM K upper bound for BIC
DC_MAX_COMPONENTS = 4

def _adaptive_gates(n: int) -> Tuple[int, float, float]:
    """
    Return (min_points, min_frac, sep_sig_thr) tuned to sample size n.

    Parameters
    ----------
    n : int
        Total number of points in the sample.

    Returns
    -------
    min_points : int
        Minimum absolute size of a candidate cluster (points).
    min_frac : float
        Minimum fraction of the sample size that a cluster must represent.
    sep_sig_thr : float
        Required separation between cluster medians in units of the larger
        robust sigma (used as a gate on well-separated clusters).
    """
    n = int(max(0, n))

    # Target ~12 % of n, but clamp between 3 and 6 grains, then respect DC_MIN_POINTS
    base_min = max(3, min(6, int(math.ceil(0.12 * n))))
    min_points = max(DC_MIN_POINTS, base_min)

    # For very small n require a larger fraction, then taper to 10 %
    if n < 8:
        min_frac = 0.40
    elif n < 12:
        min_frac = 0.25
    elif n < 20:
        min_frac = 0.15
    else:
        min_frac = 0.10

    # Separation gate: small n → looser, large n → stricter
    if n < 12:
        sep_sig_thr = 1.05
    elif n < 20:
        sep_sig_thr = 1.10
    else:
        sep_sig_thr = 1.20

    return min_points, min_frac, sep_sig_thr


__all__ = [
    "find_discordant_clusters",
    "cluster_proxies_years",
    "_labels_from_this_run",
    "_soft_accept_labels",
    "_stack_min_across_clusters",
    "fit_gmm1d_bic",
    "assign_labels_gmm1d",
    "_adaptive_gates",
    "stack_goodness_by_cluster",
    "lower_intercept_proxy_wetherill",
    "_labels_from_this_run_wetherill",
]

# -----------------------------------------------------------------------------
# Lightweight EM/BIC GMM
# -----------------------------------------------------------------------------


def fit_gmm1d_bic(x, kmax: int = 3, n_init: int = 3, max_iter: int = 100):
    """
    Small 1-D Gaussian mixture model with BIC-based model selection.

    Parameters
    ----------
    x : array_like
        1-D array of values (YEARS or any real scalar).
    kmax : int, optional
        Maximum number of components to try (inclusive).
    n_init : int, optional
        Number of EM initialisations per K.
    max_iter : int, optional
        Maximum EM iterations per initialisation.

    Returns
    -------
    dict or None
        Best model with keys {K, pi, mu, var, bic, ll} or None if no data.
    """
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return None

    best = None
    log2pi = np.log(2.0 * np.pi)

    for K in range(1, int(kmax) + 1):
        for _ in range(int(n_init)):
            # Init means via quantiles for stability; weights uniform; variances pooled
            if K > 1:
                qs = np.linspace(0.0, 1.0, K + 2)[1:-1]
                mu = np.quantile(x, qs)
            else:
                mu = np.array([np.median(x)], float)

            pi = np.ones(K, float) / K
            v0 = float(np.var(x)) if np.var(x) > 0 else 1e-6
            var = np.full(K, v0, float)

            # EM
            for _it in range(int(max_iter)):
                # E-step: responsibilities
                logp = np.stack(
                    [
                        -0.5
                        * (
                            log2pi
                            + np.log(var[k])
                            + ((x - mu[k]) ** 2) / var[k]
                        )
                        for k in range(K)
                    ],
                    axis=1,
                )
                logp += np.log(pi + 1e-12)
                m = logp.max(axis=1, keepdims=True)
                p = np.exp(logp - m)
                r = p / p.sum(axis=1, keepdims=True)  # (n, K)

                # M-step
                rk = r.sum(axis=0) + 1e-12           # (K,)
                mu = (r.T @ x) / rk                  # (K,)
                diff2 = (x[:, None] - mu[None, :]) ** 2
                var = ((r * diff2).sum(axis=0) / rk).clip(1e-10, None)
                pi = rk / n

            # log-likelihood and BIC
            logp = np.stack(
                [
                    -0.5
                    * (log2pi + np.log(var[k]) + ((x - mu[k]) ** 2) / var[k])
                    for k in range(K)
                ],
                axis=1,
            )
            logp += np.log(pi + 1e-12)
            m = logp.max(axis=1, keepdims=True)
            ll = (m + np.log(np.exp(logp - m).sum(axis=1, keepdims=True))).sum()

            k_params = 3 * K - 1  # weights(K-1) + means(K) + variances(K)
            bic = -2.0 * ll + k_params * np.log(n)
            cand = dict(K=K, pi=pi, mu=mu, var=var, bic=bic, ll=ll)
            if best is None or bic < best["bic"]:
                best = cand

    # Sort components by mean for consistency
    order = np.argsort(best["mu"])
    best["pi"] = best["pi"][order]
    best["mu"] = best["mu"][order]
    best["var"] = best["var"][order]
    return best


def assign_labels_gmm1d(x, model):
    """
    Return hard labels (argmax posterior) for a 1-D GMM model dict.

    Parameters
    ----------
    x : array_like
        1-D array of values.
    model : dict
        Model returned by fit_gmm1d_bic, with keys {mu, var, pi}.

    Returns
    -------
    ndarray or None
        Integer labels for each element of x, or None if model is None.
    """
    if model is None:
        return None
    x = np.asarray(x, float).ravel()
    K = model["mu"].size
    log2pi = np.log(2.0 * np.pi)
    logp = np.stack(
        [
            -0.5
            * (
                log2pi
                + np.log(model["var"][k])
                + ((x - model["mu"][k]) ** 2) / model["var"][k]
            )
            for k in range(K)
        ],
        axis=1,
    )
    logp += np.log(model["pi"] + 1e-12)
    return np.argmax(logp, axis=1)


# -----------------------------------------------------------------------------
# Helpers used by processing / MonteCarloRun
# -----------------------------------------------------------------------------


def _stack_min_across_clusters(runs, ages_y, which: str = "pen") -> np.ndarray:
    """
    Build an R×G matrix of per-run goodness, taking the maximum across clusters.

    For each run and age:
      - if unclustered, use 1 - D (or 1 - score) directly
      - if clustered, compute 1 - D (or 1 - score) per cluster and take the max

    Parameters
    ----------
    runs : list[MonteCarloRun]
        Monte Carlo runs.
    ages_y : array_like
        Age grid in YEARS.
    which : {'raw', 'pen'}
        'raw' -> 1 - KS D ; 'pen' -> 1 - score (penalized D*).

    Returns
    -------
    ndarray
        R × G array of goodness values.
    """
    ages_y = np.asarray(ages_y, float)
    R = len(runs)
    G = len(ages_y)
    S = np.full((R, G), np.nan, float)

    for r_i, r in enumerate(runs):
        stats_map = getattr(r, "_stats_by_age_by_cluster", None)

        # Fallback: non-clustered statistics
        if not stats_map and hasattr(r, "statistics_by_pb_loss_age"):
            if which == "raw":
                arr = np.array(
                    [1.0 - r.statistics_by_pb_loss_age[a].test_statistics[0] for a in ages_y],
                    float,
                )
            else:
                ds = np.array([r.statistics_by_pb_loss_age[a].score for a in ages_y], float)
                ds = np.clip(ds, 0.0, 1.0)
                arr = 1.0 - ds
            S[r_i] = arr
            continue

        # Clustered: take across-cluster minimum D (=> max goodness)
        per_c = []
        if stats_map:
            for _, m in stats_map.items():
                if which == "raw":
                    per_c.append(
                        np.array([1.0 - m[a].test_statistics[0] for a in ages_y], float)
                    )
                else:
                    ds = np.array([m[a].score for a in ages_y], float)
                    ds = np.clip(ds, 0.0, 1.0)
                    per_c.append(1.0 - ds)

        if per_c:
            S[r_i] = np.nanmax(np.vstack(per_c), axis=0)

    return S


def stack_goodness_by_cluster(runs, ages_y, which: str = "pen") -> Dict[int, np.ndarray]:
    """
    Build per-cluster goodness matrices without collapsing across clusters.

    Parameters
    ----------
    runs : list[MonteCarloRun]
        Monte Carlo runs.
    ages_y : array_like
        Age grid in YEARS.
    which : {'raw', 'pen'}
        'raw' -> 1 - KS D ; 'pen' -> 1 - score.

    Returns
    -------
    dict
        Mapping cluster_id -> R × G goodness matrix. If no run has clustering,
        returns {0: _stack_min_across_clusters(...)}.
    """
    ages_y = np.asarray(ages_y, float)
    R = len(runs)
    G = len(ages_y)

    cluster_ids = set()
    has_any_clustered = False
    for r in runs:
        stats_map = getattr(r, "_stats_by_age_by_cluster", None)
        if stats_map:
            has_any_clustered = True
            cluster_ids.update(stats_map.keys())

    if not has_any_clustered:
        # No clustering: behave like a single cluster 0
        return {0: _stack_min_across_clusters(runs, ages_y, which=which)}

    out: Dict[int, np.ndarray] = {}
    for cid in sorted(cluster_ids):
        S = np.full((R, G), np.nan, float)
        for r_i, r in enumerate(runs):
            stats_map = getattr(r, "_stats_by_age_by_cluster", None)
            if not stats_map:
                continue
            m = stats_map.get(cid)
            if m is None:
                continue

            if which == "raw":
                arr = np.array([1.0 - m[a].test_statistics[0] for a in ages_y], float)
            else:
                ds = np.array([m[a].score for a in ages_y], float)
                ds = np.clip(ds, 0.0, 1.0)
                arr = 1.0 - ds
            S[r_i] = arr
        out[cid] = S

    return out


def _labels_from_this_run(discordantUPb_row: np.ndarray,
                          discordantPbPb_row: np.ndarray,
                          ages_grid_y: np.ndarray) -> np.ndarray:
    """
    Per-run relabelling helper: compute 1-D LI proxies and cluster them.

    Parameters
    ----------
    discordantUPb_row : array_like
        U/Pb ratios (TW x-coordinate) for discordant grains in this run.
    discordantPbPb_row : array_like
        Pb/Pb ratios (TW y-coordinate) for discordant grains in this run.
    ages_grid_y : array_like
        Rim-age grid in YEARS.

    Returns
    -------
    ndarray
        Integer labels per discordant point; all zeros if clustering fails.
    """
    proxy_ma, keep_idx = [], []
    ages_grid_y = np.asarray(ages_grid_y, float)

    for idx, (du, dp) in enumerate(zip(discordantUPb_row, discordantPbPb_row)):
        proxy_y = lower_intercept_proxy(float(du), float(dp), ages_grid_y)
        if proxy_y is not None and np.isfinite(proxy_y):
            proxy_ma.append(proxy_y / 1e6)  # convert years to Ma
            keep_idx.append(idx)

    labels = np.zeros(len(discordantUPb_row), dtype=int)
    if not keep_idx:
        return labels

    up_ma = np.asarray(proxy_ma, float)
    keep_idx = np.asarray(keep_idx, int)
    n = up_ma.size

    # Tiny n → nothing stable to fit
    if n < 3:
        return labels

    min_pts, min_frac, sep_sig = _adaptive_gates(n)
    if n < min_pts:
        return labels

    core_labels, *_ = find_discordant_clusters(up_ma, max_k=min(DC_MAX_COMPONENTS, n))

    soft = _soft_accept_labels(
        core_labels,
        up_ma,
        min_points=min_pts,
        min_frac=min_frac,
        sep_sig_thr=sep_sig,
    )
    if soft.max() >= 1:
        labels[keep_idx] = soft
    return labels


def _soft_accept_labels(core_labels, up_ma, *, min_points: int = 5,
                        min_frac: float = 0.10, sep_sig_thr: float = 1.5) -> np.ndarray:
    """
    Apply size and separation gates to raw 1-D GMM labels and merge components.

    Parameters
    ----------
    core_labels : array_like
        Initial component labels from a 1-D GMM fit.
    up_ma : array_like
        Ages in Ma for each point.
    min_points : int, optional
        Minimum absolute size for a component.
    min_frac : float, optional
        Minimum fraction of the total sample for a component.
    sep_sig_thr : float, optional
        Required separation between component medians in units of the larger
        robust sigma.

    Returns
    -------
    ndarray
        Labels remapped to 0..M-1 for kept components, or all zeros if no
        multi-component solution survives the gates.
    """
    core_labels = np.asarray(core_labels, int)
    up_ma = np.asarray(up_ma, float)
    K = int(core_labels.max()) + 1
    if K <= 1 or up_ma.size == 0:
        return np.zeros_like(core_labels, int)

    size = np.array([(core_labels == k).sum() for k in range(K)], int)
    mu = np.array(
        [
            np.median(up_ma[core_labels == k]) if size[k] else np.nan
            for k in range(K)
        ],
        float,
    )
    sigma = np.array(
        [
            1.4826
            * np.median(
                np.abs(up_ma[core_labels == k] - np.median(up_ma[core_labels == k]))
            )
            if size[k]
            else 1e-6
            for k in range(K)
        ],
        float,
    )

    size_thr = max(min_points, int(math.ceil(min_frac * up_ma.size)))
    good = [k for k in range(K) if size[k] >= size_thr]
    if len(good) <= 1:
        return np.zeros_like(core_labels, int)

    # Enforce separation among 'good'; iteratively merge under-separated components
    good = sorted(good, key=lambda k: size[k], reverse=True)
    changed = True
    while changed:
        changed = False
        for i in range(len(good)):
            for j in range(i + 1, len(good)):
                ki, kj = good[i], good[j]

                # Safe denominator for separation in σ units
                denom = max(sigma[ki], sigma[kj])
                if not np.isfinite(denom) or denom <= 0.0:
                    denom = 1e-6
                sep = abs(mu[ki] - mu[kj]) / denom

                if sep < sep_sig_thr:
                    keep, drop = (ki, kj) if size[ki] >= size[kj] else (kj, ki)

                    mask_keep = (core_labels == keep) | (core_labels == drop)
                    merged = up_ma[mask_keep]
                    size[keep] = int(merged.size)
                    mu[keep] = float(np.median(merged)) if merged.size else np.nan
                    mad = np.median(np.abs(merged - mu[keep])) if merged.size else 0.0
                    sigma[keep] = 1.4826 * mad if mad > 0 else 1e-6

                    if drop in good:
                        good.remove(drop)
                    changed = True
                    break
            if changed:
                break

    if len(good) <= 1:
        return np.zeros_like(core_labels, int)

    kept_sorted = sorted(good, key=lambda k: size[k], reverse=True)
    map_keep = {k: i for i, k in enumerate(kept_sorted)}

    out = np.zeros_like(core_labels, int)
    for k in kept_sorted:
        out[core_labels == k] = map_keep[k]

    # Merge any leftover (small) components into the nearest kept one by μ, favouring larger size
    others = [k for k in range(K) if k not in kept_sorted and size[k] > 0]
    for k in others:
        nearest = min(kept_sorted, key=lambda g: (abs(mu[g] - mu[k]), -size[g]))
        out[core_labels == k] = map_keep[nearest]
    return out


def lower_intercept_proxy(u_pb, pb_pb, ages_y):
    """
    Find the best lower-intercept age on a TW concordia grid for a single point.

    Parameters
    ----------
    u_pb : float
        238U/206Pb ratio (TW x-coordinate) of the discordant point.
    pb_pb : float
        207Pb/206Pb ratio (TW y-coordinate) of the discordant point.
    ages_y : array_like
        Candidate ages (YEARS) along the concordia.

    Returns
    -------
    float or None
        Best lower-intercept age in YEARS for this point, or None if none valid.
    """
    best_age = None
    best_resid = np.inf
    x2, y2 = float(u_pb), float(pb_pb)

    for age in ages_y:
        x0 = calculations.u238pb206_from_age(age)
        y0 = calculations.pb207pb206_from_age(age)
        ui = calculations.discordant_age(x0, y0, x2, y2)
        if ui is None or not np.isfinite(ui):
            continue

        x1 = calculations.u238pb206_from_age(ui)
        y1 = calculations.pb207pb206_from_age(ui)
        denom_u = x1 - x0
        denom_p = y1 - y0
        if abs(denom_u) < 1e-12 or abs(denom_p) < 1e-12:
            continue

        lam_u = (x2 - x0) / denom_u
        lam_p = (y2 - y0) / denom_p
        resid = abs(lam_u - lam_p)
        if resid < best_resid:
            best_resid = resid
            best_age = age
    return best_age

def lower_intercept_proxy_wetherill(u_pb, pb_pb, ages_y):
    """
    Wetherill analogue of lower_intercept_proxy, but accepts TW inputs and converts internally.

    Inputs (TW):
      u_pb  = 238U/206Pb
      pb_pb = 207Pb/206Pb

    Internally converts to Wetherill:
      x2 = 207Pb/235U = (pb_pb * U) / u_pb
      y2 = 206Pb/238U = 1 / u_pb

    Then uses calculations.discordant_age_wetherill(...) against the Wetherill concordia.
    Returns best lower-intercept age in YEARS, or None.
    """
    try:
        u_pb = float(u_pb)
        pb_pb = float(pb_pb)
        if not (np.isfinite(u_pb) and np.isfinite(pb_pb)) or u_pb <= 0.0:
            return None
        U = float(calculations.U238U235_RATIO)
        if not np.isfinite(U) or U <= 0.0:
            return None
        x2 = pb_pb * U / u_pb
        y2 = 1.0 / u_pb
        if not (np.isfinite(x2) and np.isfinite(y2)) or y2 <= 0.0:
            return None
    except Exception:
        return None

    best_age = None
    best_resid = np.inf

    for age in ages_y:
        x0 = calculations.pb207u235_from_age(age)
        y0 = calculations.pb206u238_from_age(age)
        ui = calculations.discordant_age_wetherill(x0, y0, x2, y2)
        if ui is None or not np.isfinite(ui):
            continue

        x1 = calculations.pb207u235_from_age(ui)
        y1 = calculations.pb206u238_from_age(ui)

        denom_x = x1 - x0
        denom_y = y1 - y0
        if abs(denom_x) < 1e-12 or abs(denom_y) < 1e-12:
            continue

        lam_x = (x2 - x0) / denom_x
        lam_y = (y2 - y0) / denom_y
        resid = abs(lam_x - lam_y)

        if resid < best_resid:
            best_resid = resid
            best_age = age

    return best_age


def _labels_from_this_run_wetherill(discordantUPb_row: np.ndarray,
                                    discordantPbPb_row: np.ndarray,
                                    ages_grid_y: np.ndarray) -> np.ndarray:
    """
    Per-run relabelling helper for Wetherill-mode:
    compute LI proxies using lower_intercept_proxy_wetherill (TW inputs, Weth geometry),
    then cluster proxies exactly like the TW helper does.
    """
    proxy_ma, keep_idx = [], []
    ages_grid_y = np.asarray(ages_grid_y, float)

    for idx, (du, dp) in enumerate(zip(discordantUPb_row, discordantPbPb_row)):
        proxy_y = lower_intercept_proxy_wetherill(float(du), float(dp), ages_grid_y)
        if proxy_y is not None and np.isfinite(proxy_y):
            proxy_ma.append(proxy_y / 1e6)
            keep_idx.append(idx)

    labels = np.zeros(len(discordantUPb_row), dtype=int)
    if not keep_idx:
        return labels

    up_ma = np.asarray(proxy_ma, float)
    keep_idx = np.asarray(keep_idx, int)
    n = up_ma.size
    if n < 3:
        return labels

    min_pts, min_frac, sep_sig = _adaptive_gates(n)
    if n < min_pts:
        return labels

    core_labels, *_ = find_discordant_clusters(up_ma, max_k=min(DC_MAX_COMPONENTS, n))

    soft = _soft_accept_labels(
        core_labels,
        up_ma,
        min_points=min_pts,
        min_frac=min_frac,
        sep_sig_thr=sep_sig,
    )
    if soft.max() >= 1:
        labels[keep_idx] = soft
    return labels


def cluster_proxies_years(proxies_y):
    """
    Cluster 1D lower-intercept proxies (YEARS) for the UI helper.

    Uses the same adaptive gates as the per-run logic; primarily for visualisation
    and summary of discordant group structure.

    Parameters
    ----------
    proxies_y : array_like
        Lower-intercept proxy ages in YEARS.

    Returns
    -------
    labels : ndarray
        Integer labels per proxy (0..K-1; 0 if no robust clustering).
    summary : list[dict]
        One dict per kept cluster with keys {k, n, median_ma}.
    """
    y = np.asarray(proxies_y, float)
    mask = np.isfinite(y)
    t = y[mask]
    n = t.size

    min_pts, min_frac, sep_sig = _adaptive_gates(n)
    if n < max(5, min_pts):
        labels = np.zeros_like(y, int)
        return labels, [dict(k=0, n=int(n), median_ma=float(np.median(t) / 1e6) if n else np.nan)]

    try:
        from sklearn.mixture import GaussianMixture

        X = t.reshape(-1, 1)
        best = (None, np.inf, 1)
        for k in range(1, min(DC_MAX_COMPONENTS, n) + 1):
            g = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=0,
            )
            g.fit(X)
            bic = g.bic(X)
            if bic < best[1]:
                best = (g, bic, k)
        gmm, _, K = best
        core = gmm.predict(X) if gmm else np.zeros(n, int)
    except Exception:
        core, K = np.zeros(n, int), 1

    sizes, med, sig = [], [], []
    for k in range(K):
        vals = t[core == k]
        sizes.append(int(vals.size))
        med.append(float(np.median(vals)) if vals.size else np.nan)
        mad = np.median(np.abs(vals - np.median(vals))) if vals.size else 0.0
        sig.append(1.4826 * mad if mad > 0 else 1e-6)

    min_size = max(min_pts, int(math.ceil(min_frac * n)))
    kept = [k for k in range(K) if sizes[k] >= min_size]

    accept = True
    if len(kept) >= 2:
        sep_min = min(
            abs(med[i] - med[j]) / max(sig[i], sig[j])
            for i in kept
            for j in kept
            if i < j
        )
        accept = sep_min >= float(sep_sig)
    elif len(kept) == 0:
        accept = False

    if accept and K > 1:
        kept_sorted = sorted(kept, key=lambda k: med[k])
        remap = {k: i for i, k in enumerate(kept_sorted)}
        out = np.zeros_like(y, int)
        out[mask] = np.array([remap.get(c, 0) for c in core], int)
        summary = [
            dict(k=int(remap[k]), n=int(sizes[k]), median_ma=float(med[k] / 1e6))
            for k in kept_sorted
        ]
        return out, summary

    labels = np.zeros_like(y, int)
    return labels, [
        dict(k=0, n=int(n), median_ma=float(np.median(t) / 1e6) if n else np.nan)
    ]


# -----------------------------------------------------------------------------
# Legacy-compatible clusterer used by processing._performRimAgeSampling
# -----------------------------------------------------------------------------


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    m = float(np.nanmean(x))
    s = float(np.nanstd(x))
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return (x - m) / s


def find_discordant_clusters(upper_ages, max_k: int = DC_MAX_COMPONENTS,
                             random_state: int = 0):
    """
    Legacy-compatible entry point used by processing.py.

    Accepts 1-D ages (Ma OR years), fits 1..max_k Gaussian components by BIC,
    and returns (labels, n_components, model_or_none).

    NOTE: This does *not* apply size/separation gates; those are handled by
    _soft_accept_labels(...) at the call site to match your historical pipeline.
    """
    a = np.asarray(upper_ages, float).ravel()
    a = a[np.isfinite(a)]
    n = a.size
    if n == 0:
        return np.empty(0, dtype=int), 0, None
    if n == 1:
        return np.zeros(1, dtype=int), 1, None

    k_max = min(int(max_k), n)

    # Prefer sklearn if available (stable, fast). Otherwise, fall back to local EM/BIC.
    try:
        from sklearn.mixture import GaussianMixture

        Z = _zscore(a).reshape(-1, 1)
        best_gm, best_bic = None, np.inf
        for k in range(1, k_max + 1):
            gm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=1e-6,
                random_state=random_state,
            )
            gm.fit(Z)
            bic = gm.bic(Z)
            if bic < best_bic:
                best_gm, best_bic = gm, bic
        labels = best_gm.predict(Z).astype(int)
        return labels, int(best_gm.n_components), best_gm
    except Exception:
        model = fit_gmm1d_bic(a, kmax=k_max, n_init=3, max_iter=100)
        if model is None:
            return np.zeros(n, dtype=int), 1, None
        labels = assign_labels_gmm1d(a, model).astype(int)
        return labels, int(model["K"]), model
