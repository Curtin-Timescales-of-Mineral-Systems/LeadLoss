import math
from typing import Dict, Tuple

import numpy as np
from process import calculations
from process.cdc_population import (
    assign_discordant_to_anchors,
    cluster_concordant_populations,
)

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

    # For very small n require a larger fraction, then taper to 15 %.
    if n < 8:
        min_frac = 0.40
    elif n < 12:
        min_frac = 0.25
    else:
        min_frac = 0.15

    # Separation gate: small n → looser, large n → stricter
    if n < 12:
        sep_sig_thr = 1.05
    elif n < 20:
        sep_sig_thr = 1.10
    else:
        sep_sig_thr = 1.20

    return min_points, min_frac, sep_sig_thr


__all__ = [
    "build_fixed_discordant_labels",
    "find_discordant_clusters",
    "cluster_proxies_years",
    "_labels_from_this_run",
    "_hard_accept_labels",
    "_stack_min_across_clusters",
    "fit_gmm1d_bic",
    "assign_labels_gmm1d",
    "_adaptive_gates",
    "stack_goodness_by_cluster",
    "anchored_lower_intercept_proxy",
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
    Deprecated helper retained only for backward compatibility.

    The defensible clustering path does not relabel clusters per Monte Carlo run.
    """
    return np.zeros(len(discordantUPb_row), dtype=int)


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


def anchored_lower_intercept_proxy(anchor_age_y: float, u_pb: float, pb_pb: float):
    """
    Compute the younger concordia intercept defined by a fixed upper anchor age
    and one discordant grain in Tera-Wasserburg space.
    """
    anchor_age_y = float(anchor_age_y)
    u_pb = float(u_pb)
    pb_pb = float(pb_pb)
    if not (np.isfinite(anchor_age_y) and np.isfinite(u_pb) and np.isfinite(pb_pb)):
        return None

    try:
        x_anchor = calculations.u238pb206_from_age(anchor_age_y)
        y_anchor = calculations.pb207pb206_from_age(anchor_age_y)
    except Exception:
        return None

    if not (np.isfinite(x_anchor) and np.isfinite(y_anchor)):
        return None

    if abs(u_pb - x_anchor) < 1e-12:
        return None

    m = (pb_pb - y_anchor) / (u_pb - x_anchor)
    c = y_anchor - m * x_anchor

    lower_age = float(getattr(calculations, "LOWER_AGE", 1.0e6))
    upper_age = max(lower_age + 1.0, anchor_age_y - 1.0e3)
    if upper_age <= lower_age:
        return None

    def func(t):
        return calculations.pb207pb206_from_age(t) - (m * calculations.u238pb206_from_age(t) + c)

    scan = np.linspace(lower_age, upper_age, 256)
    prev_t = float(scan[0])
    try:
        prev_v = float(func(prev_t))
    except Exception:
        return None

    for t in scan[1:]:
        try:
            cur_v = float(func(float(t)))
        except Exception:
            prev_t, prev_v = float(t), np.nan
            continue

        if not (np.isfinite(prev_v) and np.isfinite(cur_v)):
            prev_t, prev_v = float(t), cur_v
            continue

        if prev_v == 0.0:
            return prev_t
        if cur_v == 0.0:
            return float(t)
        if prev_v * cur_v < 0.0:
            try:
                result = calculations.root_scalar(func, bracket=(prev_t, float(t)), method="brentq")
                return float(result.root) if result.converged else None
            except Exception:
                return None
        prev_t, prev_v = float(t), cur_v

    return None


def _hard_accept_labels(core_labels, proxy_ma, *, min_points: int = 5,
                        min_frac: float = 0.10, sep_sig_thr: float = 1.5):
    """
    Apply size and separation gates to a 1-D proxy clustering solution.

    Returns
    -------
    tuple[np.ndarray | None, list[dict]]
        Accepted labels remapped to 0..K-1 with rejected components marked -1,
        plus one summary row per retained cluster. Returns (None, []) if no
        robust multi-cluster solution survives.
    """
    core_labels = np.asarray(core_labels, int)
    proxy_ma = np.asarray(proxy_ma, float)
    if proxy_ma.size == 0:
        return None, []

    K = int(core_labels.max()) + 1
    if K <= 1:
        return None, []

    size_thr = max(int(min_points), int(math.ceil(float(min_frac) * proxy_ma.size)))
    clusters = []
    for k in range(K):
        vals = proxy_ma[core_labels == k]
        if vals.size < size_thr:
            continue
        clusters.append(
            dict(
                raw_label=int(k),
                n=int(vals.size),
                median_ma=float(np.median(vals)),
                sigma_ma=float(_robust_sigma(vals)),
            )
        )

    if len(clusters) <= 1:
        return None, []

    clusters.sort(key=lambda item: item["median_ma"])
    for left, right in zip(clusters[:-1], clusters[1:]):
        denom = max(float(left["sigma_ma"]), float(right["sigma_ma"]), 1e-6)
        sep = abs(float(right["median_ma"]) - float(left["median_ma"])) / denom
        if sep < float(sep_sig_thr):
            return None, []

    out = np.full(core_labels.shape, -1, dtype=int)
    remap = {row["raw_label"]: i for i, row in enumerate(clusters)}
    for raw_label, new_label in remap.items():
        out[core_labels == raw_label] = int(new_label)

    summary = [
        dict(k=int(remap[row["raw_label"]]), n=int(row["n"]), median_ma=float(row["median_ma"]))
        for row in clusters
    ]
    return out, summary


def build_fixed_discordant_labels(concordant_spots, discordant_spots):
    """
    Build one fixed clustering solution for the full sample, or reject clustering.

    Returns
    -------
    tuple[bool, np.ndarray | None, dict]
        (accepted, labels_or_none, summary_dict)
    """
    labels_full = np.full(len(discordant_spots), -1, dtype=int)
    anchor_labels, n_anchors, anchor_means = cluster_concordant_populations(concordant_spots, max_pops=6)
    assigned_anchors, assignment_rows = assign_discordant_to_anchors(
        discordant_spots,
        anchor_means,
        ambiguity_ratio=1.25,
    )

    assigned_mask = assigned_anchors >= 0
    n_disc = len(discordant_spots)
    n_assigned = int(np.sum(assigned_mask))
    assigned_frac = (float(n_assigned) / float(n_disc)) if n_disc else 0.0
    anchor_rows = []
    for anchor_id, anchor_age in enumerate(np.asarray(anchor_means, float)):
        anchor_rows.append(
            dict(
                anchor_id=int(anchor_id),
                age_ma=float(anchor_age),
                n_concordant=int(np.sum(np.asarray(anchor_labels, int) == int(anchor_id))),
            )
        )

    summary = dict(
        accepted=False,
        reason=None,
        anchor_means_ma=np.asarray(anchor_means, float).tolist(),
        n_anchors=int(n_anchors),
        anchors=anchor_rows,
        n_discordant=int(n_disc),
        n_assigned=int(n_assigned),
        n_ambiguous=int(n_disc - n_assigned),
        assigned_fraction=float(assigned_frac),
        assignments=assignment_rows,
        clusters=[],
    )

    if assigned_frac < 0.70:
        summary["reason"] = "too_many_ambiguous_assignments"
        return False, None, summary

    proxy_ma = []
    proxy_idx = []
    for idx, (spot, anchor_id) in enumerate(zip(discordant_spots, assigned_anchors)):
        if int(anchor_id) < 0:
            continue
        proxy_y = anchored_lower_intercept_proxy(
            float(anchor_means[int(anchor_id)]) * 1e6,
            float(spot.uPbValue),
            float(spot.pbPbValue),
        )
        if proxy_y is None or not np.isfinite(proxy_y):
            continue
        proxy_ma.append(float(proxy_y) / 1e6)
        proxy_idx.append(idx)

    summary["n_valid_proxies"] = int(len(proxy_idx))
    if len(proxy_idx) < 3:
        summary["reason"] = "too_few_valid_anchored_proxies"
        return False, None, summary

    proxy_ma = np.asarray(proxy_ma, float)
    proxy_idx = np.asarray(proxy_idx, int)
    min_pts, min_frac, sep_sig = _adaptive_gates(proxy_ma.size)
    core_labels, *_ = find_discordant_clusters(proxy_ma, max_k=min(DC_MAX_COMPONENTS, proxy_ma.size))
    accepted_labels, cluster_rows = _hard_accept_labels(
        core_labels,
        proxy_ma,
        min_points=min_pts,
        min_frac=min_frac,
        sep_sig_thr=sep_sig,
    )

    if accepted_labels is None:
        summary["n_clustered_proxies"] = 0
        summary["n_unclustered_valid_proxies"] = int(len(proxy_idx))
        summary["reason"] = "no_robust_proxy_split"
        return False, None, summary

    labels_full[proxy_idx] = accepted_labels
    n_clustered = int(np.count_nonzero(np.asarray(accepted_labels, int) >= 0))
    summary["n_clustered_proxies"] = n_clustered
    summary["n_unclustered_valid_proxies"] = int(len(proxy_idx) - n_clustered)
    summary["accepted"] = True
    summary["clusters"] = cluster_rows
    summary["reason"] = "accepted"
    return True, labels_full, summary


def _soft_accept_labels(core_labels, up_ma, *, min_points: int = 5,
                        min_frac: float = 0.10, sep_sig_thr: float = 1.5) -> np.ndarray:
    """
    Legacy compatibility shim.

    The defensible clustering path no longer soft-merges components. Callers that
    still import this helper receive the hard-gated labels, or all zeros on reject.
    """
    accepted, _summary = _hard_accept_labels(
        core_labels,
        up_ma,
        min_points=min_points,
        min_frac=min_frac,
        sep_sig_thr=sep_sig_thr,
    )
    if accepted is None:
        return np.zeros_like(np.asarray(core_labels, int), int)
    return accepted


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
