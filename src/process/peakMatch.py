# process/peakmatch.py
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

from process import calculations

def _reconstruct_discordant_ages_at_anchor(sample, anchor_age_y):
    x1 = calculations.u238pb206_from_age(anchor_age_y)
    y1 = calculations.pb207pb206_from_age(anchor_age_y)
    out = []
    for spot in sample.discordantSpots():
        try:
            age = calculations.discordant_age(x1, y1, spot.uPbValue, spot.pbPbValue)
            out.append(age if (isinstance(age, (int, float)) and np.isfinite(age)) else np.nan)
        except Exception:
            out.append(np.nan)
    return np.asarray(out, float)

def _invert_pb207pb206_to_age_years(r_ratio, lo=1e6, hi=4500e6, max_iter=80):
    """
    Numeric invert of pb207pb206_from_age(age) -> age (YEARS).
    Assumes monotonic increase in the valid geologic range.
    """
    # Simple bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = calculations.pb207pb206_from_age(mid) - r_ratio
        flo = calculations.pb207pb206_from_age(lo) - r_ratio
        # If the sign doesn't bracket (noisy ratio), just return mid after a few iters
        if fmid == 0:
            return mid
        if np.sign(fmid) == np.sign(flo):
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e4:  # 0.01 Ma resolution
            break
    return 0.5 * (lo + hi)


def build_concordant_reference(sample, *, grid_nodes=1024):
    """
    Build a simple KDE-based reference from concordant ages (YEARS).
    Returns dict: {"peaks_y": np.ndarray, "bw_y": float, "tol_y": float, "span": (ymin, ymax)}
    """
    # Extract approximate concordant ages from measured 207Pb/206Pb ratios
    ratios = [s.pbPbValue for s in sample.concordantSpots()]
    ages_y = []
    for r in ratios:
        try:
            if np.isfinite(r):
                ages_y.append(_invert_pb207pb206_to_age_years(float(r)))
        except Exception:
            pass
    ages_y = np.asarray(ages_y, float)
    ages_y = ages_y[np.isfinite(ages_y)]
    if ages_y.size == 0:
        return dict(peaks_y=np.array([], float), bw_y=0.0, tol_y=0.0, span=(np.nan, np.nan))

    ymin, ymax = float(np.min(ages_y)), float(np.max(ages_y))
    if ymax <= ymin:
        ymax = ymin + 1.0

    bw_y = 1.06 * np.std(ages_y, ddof=1) * np.power(ages_y.size, -1/5) if ages_y.size >= 2 else max(1e6, 0.02 * (ymax - ymin))
    kde  = gaussian_kde(ages_y, bw_method=bw_y / np.std(ages_y, ddof=1))

    xs = np.linspace(ymin, ymax, grid_nodes)
    ys = kde(xs)

    # Smoothness already handled by KDE; detect peaks
    J, _ = find_peaks(ys, distance=max(2, grid_nodes // 200))
    peaks_y = xs[J]

    # Tolerance window in YEARS (>= 15 Ma, scaled by bandwidth)
    tol_y = max(15e6, 0.75 * bw_y)

    return dict(peaks_y=peaks_y, bw_y=float(bw_y), tol_y=float(tol_y), span=(ymin, ymax))


def dpm_fraction_in_windows(pm_ref, reconstructed_ages_y):
    """
    Distance-of-peak-match: 1 - fraction of reconstructed ages that lie within +/- tol_y
    of any reference peak. Returns (dpm in [0,1], aux dict).
    """
    peaks_y = np.asarray(pm_ref.get("peaks_y", []), float)
    tol_y = float(pm_ref.get("tol_y", 0.0))
    rec = np.asarray(reconstructed_ages_y, float)
    rec = rec[np.isfinite(rec)]
    if rec.size == 0 or peaks_y.size == 0 or tol_y <= 0:
        return 1.0, dict(frac_in=0.0, tol_y=tol_y, n_ref=int(peaks_y.size), n_test=int(rec.size))
    # mark any rec within any peak window
    mask = np.zeros(rec.shape, dtype=bool)
    for p in peaks_y:
        mask |= (np.abs(rec - p) <= tol_y)
    frac_in = float(np.mean(mask)) if rec.size else 0.0
    dpm = float(1.0 - max(0.0, min(1.0, frac_in)))
    return dpm, dict(frac_in=frac_in, tol_y=tol_y, n_ref=int(peaks_y.size), n_test=int(rec.size))


def dpm_curve_over_grid(sample, ages_grid_y):
    pm_ref = build_concordant_reference(sample)
    if pm_ref["peaks_y"].size == 0:
        return pm_ref, np.full(len(ages_grid_y), 1.0, float)

    dpm_vals = np.empty(len(ages_grid_y), float)
    for i, a in enumerate(ages_grid_y):
        rec_y = _reconstruct_discordant_ages_at_anchor(sample, float(a))
        dpm, _ = dpm_fraction_in_windows(pm_ref, rec_y)
        dpm_vals[i] = dpm
    return pm_ref, dpm_vals
