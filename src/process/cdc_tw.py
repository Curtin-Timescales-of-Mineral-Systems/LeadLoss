"""Tera-Wasserburg (TW) helper functions used by CDC.

This is extracted from process/processing.py to keep that module focused on
workflow orchestration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from process import calculations
from process.cdc_config import REV_TOL_X, REV_TOL_Y

# ======================  DECAY CONSTANTS & TW HELPERS  ======================

# These are in units of 1/Ma
LAMBDA_238 = 1.55125e-4
LAMBDA_235 = 9.8485e-4
R_U = 137.818  # 238U/235U


def age_ma_from_u238pb206(u: float) -> float:
    """Convert 238U/206Pb (TW x-axis) to age in Ma."""
    try:
        u = float(u)
        if not np.isfinite(u) or u <= 0.0:
            return float("nan")
        return (1.0 / LAMBDA_238) * np.log1p(1.0 / u)
    except Exception:
        return float("nan")


def age_ma_from_pb207pb206(v: float) -> float:
    """Convert 207Pb/206Pb (TW y-axis) to age in Ma via bisection."""
    try:
        v = float(v)
        if not np.isfinite(v) or v <= 0.0:
            return float("nan")

        lo, hi = 1e-9, 5000.0  # avoid 0/0 at t=0
        def f(t):
            num = np.expm1(LAMBDA_235 * t)
            den = np.expm1(LAMBDA_238 * t)
            if den == 0.0:
                # limit t→0 of (e^{λ235 t}-1)/(e^{λ238 t}-1) = λ235/λ238
                ratio = LAMBDA_235 / LAMBDA_238
            else:
                ratio = num / den
            return ratio - (v * R_U)

        flo = f(lo)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if np.sign(fm) == np.sign(flo):
                lo, flo = mid, fm
            else:
                hi = mid
        return 0.5 * (lo + hi)
    except Exception:
        return float("nan")


def is_reverse_discordant(u: float, v: float, tol_y: float = REV_TOL_Y, tol_x: float = REV_TOL_X) -> bool:
    """
    TW coordinates: x = 238U/206Pb (= u), y = 207Pb/206Pb (= v)

    A discordant point is 'reverse' if EITHER:
      • below the curve at the same x   : v < y_concordia(u) - tol_y
      • left  of the curve at the same y: u < x_concordia(v) - tol_x
    """
    try:
        if not (np.isfinite(u) and np.isfinite(v)) or u <= 0.0 or v <= 0.0:
            return False

        # Below-at-same-x
        v_conc = calculations.pb207pb206_from_u238pb206(float(u))
        if np.isfinite(v_conc) and (float(v) < (v_conc - tol_y)):
            return True

        # Left-of-at-same-y: invert y→t, then t→x on concordia
        t_ma = age_ma_from_pb207pb206(float(v))
        if np.isfinite(t_ma):
            x_conc = calculations.u238pb206_from_age(float(t_ma) * 1e6)  # expects years
            if np.isfinite(x_conc) and (float(u) < (x_conc - tol_x)):
                return True
    except Exception:
        pass
    return False
