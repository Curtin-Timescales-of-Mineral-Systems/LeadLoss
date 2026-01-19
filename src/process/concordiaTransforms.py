"""
process.concordia_transforms

Utilities to convert between concordia coordinate systems used in this app.

Coordinate systems
------------------
Tera–Wasserburg ("TW")
  x_tw = 238U / 206Pb
  y_tw = 207Pb / 206Pb

Wetherill
  x_w  = 207Pb / 235U
  y_w  = 206Pb / 238U

Transform relations
-------------------
Let U = 238U/235U (a constant used elsewhere in the codebase).

From TW -> Wetherill:
  y_w = 1 / x_tw
  x_w = (y_tw * U) / x_tw

From Wetherill -> TW:
  x_tw = 1 / y_w
  y_tw = x_w / (U * y_w)

Error propagation here assumes independence (no covariance to match 
the current TW pipeline assumption.
"""

from __future__ import annotations

import math
from typing import Tuple

from process.calculations import U238U235_RATIO


def tw_to_wetherill(
    u238pb206: float,
    pb207pb206: float,
    *,
    u_ratio: float = U238U235_RATIO
) -> Tuple[float, float]:
    """
    Convert TW ratios -> Wetherill ratios.

    Returns:
      (pb207u235, pb206u238)
    """
    if u238pb206 == 0 or not math.isfinite(u238pb206) or not math.isfinite(pb207pb206):
        return (float("nan"), float("nan"))

    pb206u238 = 1.0 / u238pb206
    pb207u235 = (pb207pb206 * u_ratio) / u238pb206
    return pb207u235, pb206u238


def wetherill_to_tw(
    pb207u235: float,
    pb206u238: float,
    *,
    u_ratio: float = U238U235_RATIO
) -> Tuple[float, float]:
    """
    Convert Wetherill ratios -> TW ratios.

    Returns:
      (u238pb206, pb207pb206)
    """
    if pb206u238 == 0 or not math.isfinite(pb206u238) or not math.isfinite(pb207u235):
        return (float("nan"), float("nan"))

    u238pb206 = 1.0 / pb206u238
    pb207pb206 = pb207u235 / (u_ratio * pb206u238)
    return u238pb206, pb207pb206


def tw_stdevs_to_wetherill(
    u238pb206: float,
    u238pb206_stdev: float,
    pb207pb206: float,
    pb207pb206_stdev: float,
    *,
    u_ratio: float = U238U235_RATIO
) -> Tuple[float, float]:
    """
    Propagate 1σ uncertainties from TW space to Wetherill space,
    assuming independence (covariance = 0).

    Returns:
      (pb207u235_stdev, pb206u238_stdev)
    """
    if u238pb206 == 0 or not all(map(math.isfinite, [u238pb206, u238pb206_stdev, pb207pb206, pb207pb206_stdev])):
        return (float("nan"), float("nan"))

    # y_w = 1/u
    pb206u238_stdev = abs(u238pb206_stdev / (u238pb206 ** 2))

    # x_w = (U*v)/u
    dx_dv = u_ratio / u238pb206
    dx_du = -(u_ratio * pb207pb206) / (u238pb206 ** 2)
    pb207u235_stdev = math.sqrt((dx_dv * pb207pb206_stdev) ** 2 + (dx_du * u238pb206_stdev) ** 2)

    return pb207u235_stdev, pb206u238_stdev


def wetherill_stdevs_to_tw(
    pb207u235: float,
    pb207u235_stdev: float,
    pb206u238: float,
    pb206u238_stdev: float,
    *,
    u_ratio: float = U238U235_RATIO
) -> Tuple[float, float]:
    """
    Propagate 1σ uncertainties from Wetherill space to TW space,
    assuming independence (covariance = 0).

    Returns:
      (u238pb206_stdev, pb207pb206_stdev)
    """
    if pb206u238 == 0 or not all(map(math.isfinite, [pb207u235, pb207u235_stdev, pb206u238, pb206u238_stdev])):
        return (float("nan"), float("nan"))

    # u = 1/y
    u238pb206_stdev = abs(pb206u238_stdev / (pb206u238 ** 2))

    # v = x / (U*y)
    dv_dx = 1.0 / (u_ratio * pb206u238)
    dv_dy = -(pb207u235) / (u_ratio * (pb206u238 ** 2))
    pb207pb206_stdev = math.sqrt((dv_dx * pb207u235_stdev) ** 2 + (dv_dy * pb206u238_stdev) ** 2)

    return u238pb206_stdev, pb207pb206_stdev
