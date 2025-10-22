import os
import math
import numpy as np
from process import calculations

# -----------------------------------------------------------------------------
# Tunable defaults (env overrides preserved)
# -----------------------------------------------------------------------------
DC_MIN_POINTS      = 10      # used by cluster_proxies_years (UI helper)
DC_MIN_SEP_SIGMA   = 1.2    # used by cluster_proxies_years 1.15
DC_MAX_COMPONENTS  = 4       # GMM K upper bound for BIC

# Legacy GMM acceptance for discordant clusters (used by per-run relabelling)
MIN_POINTS_GMM = int(os.environ.get("CDC_MIN_POINTS_GMM", "10"))     # ≥ 5
SEP_SIG_THR    = float(os.environ.get("CDC_GMM_SEP_SIG", "1.2"))    # ≥ 1.5 σ̂
MIN_FRAC       = float(os.environ.get("CDC_GMM_MIN_FRAC", "0.10"))  # ≥ 10%

__all__ = [
    "find_discordant_clusters",
    "cluster_proxies_years",
    "_labels_from_this_run",
    "_soft_accept_labels",
    "_stack_min_across_clusters",
    "fit_gmm1d_bic",
    "assign_labels_gmm1d",
]

# -----------------------------------------------------------------------------
# Lightweight EM/BIC GMM (sklearn-free fallback)
# -----------------------------------------------------------------------------
def fit_gmm1d_bic(x, kmax=3, n_init=3, max_iter=100):
    """
    Return best {K, pi, mu, var, bic, ll} by BIC over K=1..kmax.
    x: 1-D array of finite values (YEARS or any 1-D real).
    """
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return None

    best = None
    log2pi = np.log(2 * np.pi)

    for K in range(1, int(kmax) + 1):
        for _ in range(int(n_init)):
            # Init means via quantiles for stability; weights uniform; variances pooled
            if K > 1:
                qs = np.linspace(0, 1, K + 2)[1:-1]
                mu = np.quantile(x, qs)
            else:
                mu = np.array([np.median(x)], float)
            pi = np.ones(K, float) / K
            v0 = float(np.var(x)) if np.var(x) > 0 else 1e-6
            var = np.full(K, v0, float)

            for _it in range(int(max_iter)):
                # E-step
                logp = np.stack(
                    [-0.5 * (log2pi + np.log(var[k]) + ((x - mu[k]) ** 2) / var[k]) for k in range(K)],
                    axis=1
                )
                logp += np.log(pi + 1e-12)
                m = logp.max(axis=1, keepdims=True)
                p = np.exp(logp - m)
                r = p / p.sum(axis=1, keepdims=True)      # (n, K)

                # M-step
                rk = r.sum(axis=0) + 1e-12                # (K,)
                mu = (r.T @ x) / rk                       # (K,)
                diff2 = (x[:, None] - mu[None, :]) ** 2   # (n, K)
                var = ((r * diff2).sum(axis=0) / rk).clip(1e-10, None)  # (K,)
                pi = rk / n

            # Log-likelihood for BIC
            logp = np.stack(
                [-0.5 * (log2pi + np.log(var[k]) + ((x - mu[k]) ** 2) / var[k]) for k in range(K)],
                axis=1
            )
            logp += np.log(pi + 1e-12)
            m = logp.max(axis=1, keepdims=True)
            ll = (m + np.log(np.exp(logp - m).sum(axis=1, keepdims=True))).sum()

            k_params = 3 * K - 1  # weights(K-1) + means(K) + variances(K)
            bic = -2 * ll + k_params * np.log(n)
            cand = dict(K=K, pi=pi, mu=mu, var=var, bic=bic, ll=ll)
            if best is None or bic < best["bic"]:
                best = cand

    # Sort components by mean
    order = np.argsort(best["mu"])
    best["pi"]  = best["pi"][order]
    best["mu"]  = best["mu"][order]
    best["var"] = best["var"][order]
    return best


def assign_labels_gmm1d(x, model):
    """Return hard labels (argmax posterior) for a 1-D GMM model dict."""
    if model is None:
        return None
    x = np.asarray(x, float).ravel()
    K = model["mu"].size
    log2pi = np.log(2 * np.pi)
    logp = np.stack(
        [-0.5 * (log2pi + np.log(model["var"][k]) + ((x - model["mu"][k]) ** 2) / model["var"][k]) for k in range(K)],
        axis=1
    )
    logp += np.log(model["pi"] + 1e-12)
    return np.argmax(logp, axis=1)

# -----------------------------------------------------------------------------
# Helpers used by processing / MonteCarloRun
# -----------------------------------------------------------------------------
def _stack_min_across_clusters(runs, ages_y, which='pen'):
    """
    Returns an R×G matrix of goodness after taking the min across clusters
    (equivalently, max across cluster-wise goodness).
    which: 'raw' -> 1 - KS-D ; 'pen' -> 1 - score
    """
    R = len(runs)
    G = len(ages_y)
    S = np.full((R, G), np.nan, float)
    for r_i, r in enumerate(runs):
        stats_map = getattr(r, "_stats_by_age_by_cluster", None)
        if not stats_map:
            # fall back to non-clustered stats if needed
            if hasattr(r, "statistics_by_pb_loss_age"):
                if which == 'raw':
                    S[r_i] = np.array([1.0 - r.statistics_by_pb_loss_age[a].test_statistics[0] for a in ages_y], float)
                else:
                    S[r_i] = np.array([1.0 - r.statistics_by_pb_loss_age[a].score for a in ages_y], float)
            continue
        # collect per-cluster goodness
        per_c = []
        for _, m in stats_map.items():
            if which == 'raw':
                per_c.append(np.array([1.0 - m[a].test_statistics[0] for a in ages_y], float))
            else:
                per_c.append(np.array([1.0 - m[a].score for a in ages_y], float))
        if per_c:
            S[r_i] = np.nanmax(np.vstack(per_c), axis=0)  # across-cluster min(D) => max(goodness)
    return S


# def _labels_from_this_run(discordantUPb_row, discordantPbPb_row, mid_UPb, mid_PbPb):
#     """
#     Compute 1-D preliminary discordant ages (Ma) for THIS run at the mid-age,
#     cluster them, then 'soft-accept' components by size and separation.
#     Returns integer labels with acceptance gates; falls back to 1 cluster.
#     """
#     up_ma, keep_idx = [], []
#     for idx, (du, dp) in enumerate(zip(discordantUPb_row, discordantPbPb_row)):
#         ui = calculations.discordant_age(mid_UPb, mid_PbPb, float(du), float(dp))
#         if ui is not None and np.isfinite(ui):
#             up_ma.append(ui / 1e6)
#             keep_idx.append(idx)

#     labels = np.zeros(len(discordantUPb_row), dtype=int)  # default single cluster
#     if not keep_idx:
#         return labels

#     up_ma    = np.asarray(up_ma, float)
#     keep_idx = np.asarray(keep_idx, int)

#     if up_ma.size < MIN_POINTS_GMM:
#         return labels

#     core_labels, *_ = find_discordant_clusters(up_ma)
#     soft = _soft_accept_labels(core_labels, up_ma,
#                                min_points=MIN_POINTS_GMM,
#                                min_frac=MIN_FRAC,
#                                sep_sig_thr=SEP_SIG_THR)
#     if soft.max() >= 1:
#         labels[keep_idx] = soft
#     return labels

def _labels_from_this_run(discordantUPb_row, discordantPbPb_row, ages_grid_y):
    """
+    Compute per-run 1-D proxies by scanning the rim-age GRID (YEARS) and
+    picking the best lower-intercept for each grain. Then cluster and
+    soft-accept components (size + separation). Returns int labels of
+    length = len(discordantUPb_row); falls back to a single cluster.
+    """
    proxy_ma, keep_idx = [], []
    ages_grid_y = np.asarray(ages_grid_y, float)
    for idx, (du, dp) in enumerate(zip(discordantUPb_row, discordantPbPb_row)):
        proxy_y = lower_intercept_proxy(float(du), float(dp), ages_grid_y)
        if proxy_y is not None and np.isfinite(proxy_y):
            proxy_ma.append(proxy_y / 1e6)  # store in Ma for clustering metric
            keep_idx.append(idx)

    labels = np.zeros(len(discordantUPb_row), dtype=int)  # default: 1 cluster
    if not keep_idx:
        return labels  # nothing finite
    up_ma    = np.asarray(proxy_ma, float)
    keep_idx = np.asarray(keep_idx, int)
 
    if up_ma.size < MIN_POINTS_GMM:
         return labels

    core_labels, *_ = find_discordant_clusters(up_ma)
    soft = _soft_accept_labels(core_labels, up_ma,
                               min_points=MIN_POINTS_GMM,
                               min_frac=MIN_FRAC,
                               sep_sig_thr=SEP_SIG_THR)
    if soft.max() >= 1:
        labels[keep_idx] = soft
    return labels

def _soft_accept_labels(core_labels, up_ma, *, min_points=5, min_frac=0.10, sep_sig_thr=1.5):
    """
    Take raw 1-D GMM labels (core_labels) and their ages (up_ma, Ma).
    Keep components that meet size AND mutual separation; merge failing
    components into the nearest kept component. Returns 0..M-1 ids (M>=2) or all zeros.
    """
    core_labels = np.asarray(core_labels, int)
    up_ma       = np.asarray(up_ma, float)
    K = int(core_labels.max()) + 1
    if K <= 1 or up_ma.size == 0:
        return np.zeros_like(core_labels, int)

    size  = np.array([(core_labels == k).sum() for k in range(K)], int)
    mu    = np.array([np.median(up_ma[core_labels == k]) if size[k] else np.nan for k in range(K)], float)
    sigma = np.array([
        1.4826*np.median(np.abs(up_ma[core_labels == k] - np.median(up_ma[core_labels == k]))) if size[k] else 1e-6
        for k in range(K)
    ], float)

    size_thr = max(min_points, int(np.ceil(min_frac * up_ma.size)))
    good = [k for k in range(K) if size[k] >= size_thr]
    if len(good) <= 1:
        return np.zeros_like(core_labels, int)

    # enforce separation among 'good'
    good = sorted(good, key=lambda k: size[k], reverse=True)
    changed = True
    while changed:
        changed = False
        for i in range(len(good)):
            for j in range(i+1, len(good)):
                ki, kj = good[i], good[j]
                sep = abs(mu[ki] - mu[kj]) / max(sigma[ki], sigma[kj])
                if sep < sep_sig_thr:
                    drop = kj if size[kj] <= size[ki] else ki
                    keep = ki if drop == kj else kj
                    size[keep] += size[drop]
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
    others = [k for k in range(K) if k not in kept_sorted and size[k] > 0]
    for k in others:
        nearest = min(kept_sorted, key=lambda g: (abs(mu[g] - mu[k]), -size[g]))
        out[core_labels == k] = map_keep[nearest]
    return out

# -----------------------------------------------------------------------------
# UI helper (unchanged behavior): cluster arbitrary 1D proxies in YEARS
# -----------------------------------------------------------------------------
def _summarize(proxies_y, labels, finite_mask):
    years_to_ma = 1e-6
    K = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    summary = []
    for k in range(K):
        k_vals = proxies_y[(labels == k) & finite_mask] * years_to_ma
        summary.append(dict(
            cluster_id=int(k),
            n=int(k_vals.size),
            median_ma=float(np.median(k_vals)) if k_vals.size else float("nan"),
        ))
    return summary


def lower_intercept_proxy(u_pb, pb_pb, ages_y):
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


def cluster_proxies_years(proxies_y,
                          min_points=MIN_POINTS_GMM,
                          min_frac=MIN_FRAC,
                          sep_sig=SEP_SIG_THR):
    """
    Cluster 1D proxies (YEARS) with size & separation gates. Returns (labels, summary).
    Sklearn GMM if available; otherwise falls back to 1 cluster.
    """
    y = np.asarray(proxies_y, float)
    mask = np.isfinite(y); t = y[mask]
    n = t.size
    if n < max(5, min_points):
        labels = np.zeros_like(y, int)
        return labels, [dict(k=0, n=int(n), median_ma=float(np.median(t)/1e6) if n else np.nan)]

    # Fit 1..3 components by BIC (sklearn if available; else K=1)
    try:
        from sklearn.mixture import GaussianMixture
        X = t.reshape(-1, 1)
        best = (None, np.inf, 1)
        for k in (1, 2, 3):
            g = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
            g.fit(X)
            bic = g.bic(X)
            if bic < best[1]:
                best = (g, bic, k)
        gmm, _, K = best
        core = gmm.predict(X) if gmm else np.zeros(n, int)
    except Exception:
        core, K = np.zeros(n, int), 1

    # Gates: size & robust separation
    sizes = []; med = []; sig = []
    for k in range(K):
        vals = t[core == k]
        sizes.append(int(vals.size))
        med.append(float(np.median(vals)) if vals.size else np.nan)
        mad = np.median(np.abs(vals - np.median(vals))) if vals.size else 0.0
        sig.append(1.4826 * mad if mad > 0 else 1e-6)
    min_size = max(min_points, int(np.ceil(min_frac * n)))
    kept = [k for k in range(K) if sizes[k] >= min_size]

    accept = True
    if len(kept) >= 2:
        sep_min = min(abs(med[i] - med[j]) / max(sig[i], sig[j]) for i in kept for j in kept if i < j)
        accept = sep_min >= float(sep_sig)
    elif len(kept) == 0:
        accept = False

    out = np.zeros_like(y, int)
    if accept and K > 1:
        out[mask] = core
        summary = [dict(k=int(k), n=int(sizes[k]), median_ma=float(med[k]/1e6)) for k in kept]
        return out, summary
    else:
        return out, [dict(k=0, n=int(n), median_ma=float(np.median(t)/1e6))]

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


def find_discordant_clusters(upper_ages, max_k: int = DC_MAX_COMPONENTS, random_state: int = 0):
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
                random_state=random_state
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
