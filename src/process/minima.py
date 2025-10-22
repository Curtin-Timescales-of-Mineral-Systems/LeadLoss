# process/minima.py
import numpy as np
from typing import Optional
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from process import calculations  # for estimate_proxies_for_clustering
from typing import Optional

def detect_significant_minima(
    ks_d_curve: np.ndarray,
    *,
    ages_y: Optional[np.ndarray] = None,   # YEARS
    sigma_nodes: float = 1.0,
    min_separation_ma: float = 90.0,
    min_separation_frac: Optional[float] = None,  # fallback if ages_y is None
    k_mad: float = 1.3,
    depth_q: float = 0.35,
    depth_eps: float = 0.006,
):
    s_raw = np.asarray(ks_d_curve, float)
    G = int(s_raw.size)
    if G < 3 or not np.isfinite(s_raw).any():
        g = int(np.nanargmin(s_raw))
        return np.array([g], int), np.array([1.0], float)

    s = gaussian_filter1d(s_raw, sigma=float(sigma_nodes), mode="nearest")

    # node distance from Ma separation where possible
    if ages_y is not None and G > 1:
        ages_y = np.asarray(ages_y, float)
        step_ma = float((ages_y[-1] - ages_y[0]) / (G - 1)) / 1e6
        step_ma = max(step_ma, 1e-9)
        distance = max(2, int(round(float(min_separation_ma) / step_ma)))
    else:
        frac = 0.03 if (min_separation_frac is None) else float(min_separation_frac)
        distance = max(2, int(round(frac * G)))

    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med))
    prom_abs = max(0.006, float(k_mad) * float(mad))

    J, _ = find_peaks(-s, prominence=prom_abs, distance=distance)

    g = int(np.nanargmin(s))
    if g not in J:
        J = np.unique(np.append(J, g))

    qd = np.nanpercentile(s, 100.0 * float(depth_q))
    J = J[s[J] <= (qd - float(depth_eps))]
    if J.size == 0:
        J = np.array([g], int)

    d95 = np.nanpercentile(s, 95)
    depths = np.maximum(0.0, d95 - s[J]).astype(float)
    return J.astype(int), depths


def _collect_minima_on_curve(
    ages_y: np.ndarray,
    curve: np.ndarray,
    *,
    sigma_nodes: float,
    min_separation_ma: float,
    k_mad: float,
    depth_q: float,
    depth_eps: float,
):
    """Return (ages_y_selected, weights) for minima on a given curve."""
    idxs, depths = detect_significant_minima(
        curve,
        ages_y=ages_y,
        sigma_nodes=sigma_nodes,
        min_separation_ma=min_separation_ma,
        k_mad=k_mad,
        depth_q=depth_q,
        depth_eps=depth_eps,
    )
    if idxs.size == 0:
        return np.array([], float), np.array([], float)
    yy = ages_y[idxs].astype(float)
    ww = depths.astype(float)
    s  = float(np.nansum(ww))
    ww = (ww / s) if (s > 0 and np.isfinite(s)) else np.ones_like(ww, float) / float(ww.size)
    return yy, ww
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths, peak_prominences

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths, peak_prominences

def attach_run_minima(
    run,
    ages_y,
    *,
    sigma_nodes: float = 1.0,
    score_min_sep_ma: float = 90.0,
    score_k_mad: float = 1.3,
    score_depth_q: float = 0.35,
    score_depth_eps: float = 0.006,
    include_raw_aux: bool = False,
    per_run_min_width: int = 3,
    **_,
):
    ages_y = np.asarray(ages_y, float)
    G = ages_y.size
    if G < 3:
        run.local_minima_ages_y = np.array([], float)
        run.local_minima_weights = np.array([], float)
        return

    S = np.array([
        (1.0 - run.statistics_by_pb_loss_age.get(float(a)).score)
        if run.statistics_by_pb_loss_age.get(float(a)) is not None else np.nan
        for a in ages_y
    ], float)
    if not np.isfinite(S).any():
        run.local_minima_ages_y = np.array([], float)
        run.local_minima_weights = np.array([], float)
        return

    S_filled = np.where(np.isfinite(S), S, np.nanmedian(S))
    S_s = gaussian_filter1d(S_filled, sigma=float(sigma_nodes), mode="nearest") if sigma_nodes > 0 else S_filled

    q5, q95 = np.nanpercentile(S_s, [5, 95]); delta = max(q95 - q5, 1e-12)
    med = np.nanmedian(S_s); mad = np.nanmedian(np.abs(S_s - med))
    prom_abs = max(float(score_k_mad) * mad, float(score_depth_eps) * delta, 1e-12)

    step_y = float(np.median(np.diff(ages_y))) if G > 1 else 1.0
    dist_nodes = max(2, int(round((float(score_min_sep_ma) * 1e6) / max(step_y, 1.0))))

    pk, _ = find_peaks(S_s, distance=dist_nodes, prominence=prom_abs)
    pk = pk[(pk > 0) & (pk < G - 1)]
    if pk.size:
        widths = peak_widths(S_s, pk, rel_height=0.5)[0]
        keep = widths >= float(per_run_min_width)
        pk = pk[keep]

    if pk.size == 0:
        j = int(np.nanargmax(S_s))
        run.local_minima_ages_y = np.array([float(ages_y[j])], float)
        run.local_minima_weights = np.array([1.0], float)
        if include_raw_aux: run._S_pen_last = S_s
        return

    prom = peak_prominences(S_s, pk)[0]
    w = prom / prom.sum() if prom.sum() > 0 else np.ones(pk.size, float) / float(pk.size)
    run.local_minima_ages_y = ages_y[pk]
    run.local_minima_weights = w
    if include_raw_aux: run._S_pen_last = S_s



def aggregate_winners_from_runs(runs, ages_y, use_score_weighted_voting: bool):
    winners_y, weights = [], []
    ages_y = np.asarray(ages_y, float)  # only used for sanity; not strictly needed
    for r in (runs or []):
        yy = getattr(r, "local_minima_ages_y", None)
        ww = getattr(r, "local_minima_weights", None)
        if yy is None or ww is None or len(yy) != len(ww) or len(yy) == 0:
            by = float(getattr(r, "optimal_pb_loss_age", float("nan")))
            if np.isfinite(by):
                winners_y.append(by); weights.append(1.0)
            continue
        run_w = 1.0
        if use_score_weighted_voting:
            sc = float(getattr(getattr(r, "optimal_statistic", None), "score", float("nan")))
            sc = sc if np.isfinite(sc) else 1.0
            run_w = max(0.0, min(1.0, 1.0 - sc)) ** 2
        for y_i, w_i in zip(np.asarray(yy, float), np.asarray(ww, float)):
            winners_y.append(float(y_i))
            weights.append(float(run_w * w_i))
    return np.asarray(winners_y, float), np.asarray(weights, float)


def estimate_proxies_for_clustering(sample, ages_y, take: str = "rightmost"):
    """
    Robust 'upper-age' proxies for discordant spots using each run’s defensible
    KS minimum as the anchor. Aggregates per spot by median across runs.
    """
    runs = getattr(sample, "monteCarloRuns", []) or []
    disc_spots = sample.discordantSpots()
    m = len(disc_spots)
    out = np.full(m, np.nan, float)
    if m == 0 or len(runs) == 0:
        return out

    per_spot = [[] for _ in range(m)]
    ages_y = np.asarray(ages_y, float)

    for r in runs:
        mins_y = getattr(r, "local_minima_ages_y", None)
        if mins_y is None or len(mins_y) == 0:
            ks_d = np.array([r.statistics_by_pb_loss_age[a].test_statistics[0] for a in ages_y], float)
            idxs, _depths = detect_significant_minima(
                ks_d, sigma_nodes=1.0, min_separation_frac=0.04, k_mad=1.3, depth_q=0.20, depth_eps=0.010
            )
            if idxs.size == 0:
                continue
            mins_y = ages_y[idxs]

        k = int(np.argmax(mins_y)) if take == "rightmost" else int(np.argmin(mins_y))
        t = float(mins_y[k])  # YEARS

        x1 = calculations.u238pb206_from_age(t)
        y1 = calculations.pb207pb206_from_age(t)
        for i, spot in enumerate(disc_spots):
            try:
                age = calculations.discordant_age(x1, y1, spot.uPbValue, spot.pbPbValue)
                if isinstance(age, (int, float)) and np.isfinite(age):
                    per_spot[i].append(float(age))
            except Exception:
                pass

    for i, vals in enumerate(per_spot):
        if vals:
            out[i] = float(np.median(vals))
    return out