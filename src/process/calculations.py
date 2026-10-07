import math

from scipy.optimize import root_scalar, minimize_scalar

import utils.errorUtils as errors

###############
## Constants ##
###############

from process.reconstructedAge import ReconstructedAge

U238_DECAY_CONSTANT = 1.55125*(10**-10)
U235_DECAY_CONSTANT = 9.8485*(10**-10)
U238U235_RATIO = 137.818

UPPER_AGE = 6000 * (10 ** 6)
LOWER_AGE = 1 * (10 ** 6)

################
## Geological ##
################

def _require_positive_finite(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return value

def age_from_u238pb206(u238pb206):
    u238pb206 = _require_positive_finite(u238pb206, "u238pb206")
    return errors.log(1 / u238pb206 + 1) / U238_DECAY_CONSTANT

def age_from_pb206u238(pb206u238):
    pb206u238 = _require_positive_finite(pb206u238, "pb206u238")
    return errors.log(pb206u238 + 1) / U238_DECAY_CONSTANT

def age_from_pb207pb206(pb207pb206):
    pb207pb206 = _require_positive_finite(pb207pb206, "pb207pb206")

    lower_age = 1.0
    upper_age = 10.0 ** 10

    def _objective(age):
        return pb207pb206_from_age(age) - pb207pb206

    lower_value = _objective(lower_age)
    upper_value = _objective(upper_age)
    if lower_value == 0:
        return lower_age
    if upper_value == 0:
        return upper_age
    if lower_value * upper_value > 0:
        raise ValueError(
            "pb207pb206 is outside invertible concordia bounds "
            f"for ages [{lower_age}, {upper_age}] years: {pb207pb206!r}"
        )

    result = root_scalar(_objective, bracket=[lower_age, upper_age], method="brentq")
    if not result.converged:
        raise ValueError(f"Failed to invert pb207pb206 value {pb207pb206!r}")
    return result.root

def age_from_pb207u235(pb207u235):
    pb207u235 = _require_positive_finite(pb207u235, "pb207u235")
    return errors.log(pb207u235 + 1) / U235_DECAY_CONSTANT

def pb206u238_from_age(age):
    return errors.exp(U238_DECAY_CONSTANT * age) - 1

def u238pb206_from_age(age):
    return 1/(pb206u238_from_age(age))

def pb207u235_from_age(age):
    return errors.exp(U235_DECAY_CONSTANT * age) - 1

def pb207pb206_from_age(age):
    pb207u235 = pb207u235_from_age(age)
    u238pb206 = u238pb206_from_age(age)
    return pb207u235*(1/U238U235_RATIO)*u238pb206

def pb207pb206_from_u238pb206(u238pb206):
    age = age_from_u238pb206(u238pb206)
    return pb207pb206_from_age(age)

def u238pb206_from_pb207pb206(pb207pb206):
    age = age_from_pb207pb206(pb207pb206)
    return u238pb206_from_age(age)

def _space_key(space):
    value = getattr(space, "value", space)
    text = str(value or "").strip().lower()
    if text == "wetherill":
        return "wetherill"
    return "tera-wasserburg"

def is_wetherill_space(space):
    return _space_key(space) == "wetherill"

def concordia_xy(age, ratio_space):
    if is_wetherill_space(ratio_space):
        return pb207u235_from_age(age), pb206u238_from_age(age)
    return u238pb206_from_age(age), pb207pb206_from_age(age)

def tw_to_wetherill(u238pb206, pb207pb206):
    """Convert TW coordinates to Wetherill coordinates.

    TW:        x = 238U/206Pb, y = 207Pb/206Pb
    Wetherill: x = 207Pb/235U, y = 206Pb/238U
    """
    u238pb206 = _require_positive_finite(u238pb206, "u238pb206")
    pb207pb206 = _require_positive_finite(pb207pb206, "pb207pb206")
    pb206u238 = 1.0 / u238pb206
    pb207u235 = pb207pb206 * U238U235_RATIO * pb206u238
    return pb207u235, pb206u238

def wetherill_to_tw(pb207u235, pb206u238):
    """Convert Wetherill coordinates to TW coordinates."""
    pb207u235 = _require_positive_finite(pb207u235, "pb207u235")
    pb206u238 = _require_positive_finite(pb206u238, "pb206u238")
    u238pb206 = 1.0 / pb206u238
    pb207pb206 = pb207u235 / (U238U235_RATIO * pb206u238)
    return u238pb206, pb207pb206

def convert_ratio_xy(x, y, from_space, to_space):
    if _space_key(from_space) == _space_key(to_space):
        return float(x), float(y)
    if is_wetherill_space(to_space):
        return tw_to_wetherill(x, y)
    return wetherill_to_tw(x, y)

def convert_ratio_xy_array(x, y, from_space, to_space):
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out_x = np.full_like(x, np.nan, dtype=float)
    out_y = np.full_like(y, np.nan, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if _space_key(from_space) == _space_key(to_space):
        out_x[mask] = x[mask]
        out_y[mask] = y[mask]
    elif is_wetherill_space(to_space):
        out_y[mask] = 1.0 / x[mask]
        out_x[mask] = y[mask] * U238U235_RATIO * out_y[mask]
    else:
        out_x[mask] = 1.0 / y[mask]
        out_y[mask] = x[mask] / (U238U235_RATIO * y[mask])
    return out_x, out_y

def _covariance_from_stdevs(x_stdev, y_stdev, rho=0.0):
    import numpy as np

    sx = 0.0 if x_stdev is None else float(x_stdev)
    sy = 0.0 if y_stdev is None else float(y_stdev)
    if not math.isfinite(sx) or sx < 0.0:
        sx = 0.0
    if not math.isfinite(sy) or sy < 0.0:
        sy = 0.0
    rho = 0.0 if rho is None else float(rho)
    if not math.isfinite(rho):
        rho = 0.0
    rho = max(-0.999999, min(0.999999, rho))
    return np.array([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]], dtype=float)

def _stdevs_from_covariance(cov):
    import numpy as np

    cov = np.asarray(cov, dtype=float)
    sx = math.sqrt(max(float(cov[0, 0]), 0.0))
    sy = math.sqrt(max(float(cov[1, 1]), 0.0))
    if sx > 0.0 and sy > 0.0:
        rho = float(cov[0, 1]) / (sx * sy)
        rho = max(-0.999999, min(0.999999, rho))
    else:
        rho = 0.0
    return sx, sy, rho

def convert_ratio_covariance(x, y, x_stdev, y_stdev, rho, from_space, to_space):
    import numpy as np

    if _space_key(from_space) == _space_key(to_space):
        return float(x_stdev or 0.0), float(y_stdev or 0.0), float(rho or 0.0)

    x = _require_positive_finite(x, "x ratio")
    y = _require_positive_finite(y, "y ratio")
    cov = _covariance_from_stdevs(x_stdev, y_stdev, rho)
    if is_wetherill_space(to_space):
        # TW (u, p) -> Wetherill (r75, r68)
        jac = np.array([
            [-(U238U235_RATIO * y) / (x * x), U238U235_RATIO / x],
            [-1.0 / (x * x), 0.0],
        ], dtype=float)
    else:
        # Wetherill (r75, r68) -> TW (u, p)
        jac = np.array([
            [0.0, -1.0 / (y * y)],
            [1.0 / (U238U235_RATIO * y), -x / (U238U235_RATIO * y * y)],
        ], dtype=float)
    return _stdevs_from_covariance(jac @ cov @ jac.T)

def discordance(u238pb206, pb207pb206):
    try:
        uPbAge = age_from_u238pb206(u238pb206)
        pbPbAge = age_from_pb207pb206(pb207pb206)
    except Exception:
        return float("inf")

    if not (math.isfinite(uPbAge) and math.isfinite(pbPbAge)) or pbPbAge == 0:
        return float("inf")

    result = (pbPbAge - uPbAge) / pbPbAge
    if not math.isfinite(result):
        return float("inf")

    # Get rid of floating point inaccuracies
    if result > 10 ** -10:
        return result
    return 0.0

def concordant_age(u238pb206, pb207pb206):
    def distance(t):
        x = u238pb206 - u238pb206_from_age(t)
        y = pb207pb206 - pb207pb206_from_age(t)
        d = math.hypot(x, y)
        return d
    result = minimize_scalar(distance, method='Bounded', bounds=[LOWER_AGE, UPPER_AGE])
    return result.x

def concordant_age_wetherill(pb207u235, pb206u238):
    def distance(t):
        x = pb207u235 - pb207u235_from_age(t)
        y = pb206u238 - pb206u238_from_age(t)
        d = math.hypot(x, y)
        return d
    result = minimize_scalar(distance, method='Bounded', bounds=[LOWER_AGE, UPPER_AGE])
    return result.x

def concordant_age_for_space(x, y, ratio_space):
    if is_wetherill_space(ratio_space):
        return concordant_age_wetherill(x, y)
    return concordant_age(x, y)

def concordant_age_wetherill_from_tw(u238pb206, pb207pb206):
    x, y = tw_to_wetherill(u238pb206, pb207pb206)
    return concordant_age_wetherill(x, y)

def discordant_age(x1, y1, x2, y2):
    if x1 <= x2:
        return None

    m = (y2 - y1)/(x2 - x1)
    c = y1 - m*x1

    lower_limit = age_from_u238pb206(min(errors.value(x1), errors.value(x2)))
    upper_limit = UPPER_AGE

    def func(t):
        curve_pb207pb206_value = pb207pb206_from_age(t)
        line_pb207pb206 = m*u238pb206_from_age(t) + c
        line_pb207pb206_value = line_pb207pb206
        return curve_pb207pb206_value - line_pb207pb206_value

    v1 = func(lower_limit)
    v2 = func(upper_limit)
    if (v1 > 0 and v2 > 0) or (v1 < 0 and v2 < 0):
        return None

    result = root_scalar(func, bracket=(lower_limit, upper_limit))
    return result.root

def discordant_age_wetherill(x1, y1, x2, y2):
    """
    Return the upper concordia intercept age in Wetherill space.

    Arguments are Wetherill coordinates:
      x = 207Pb/235U, y = 206Pb/238U.
    """
    try:
        x1 = _require_positive_finite(x1, "lower pb207u235")
        y1 = _require_positive_finite(y1, "lower pb206u238")
        x2 = _require_positive_finite(x2, "analysis pb207u235")
        y2 = _require_positive_finite(y2, "analysis pb206u238")
    except Exception:
        return None

    if x1 >= x2 or y1 >= y2:
        return None
    if x1 == x2:
        return None

    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1

    try:
        lower_limit = max(age_from_pb206u238(y2), age_from_pb207u235(x2), LOWER_AGE)
    except Exception:
        return None
    upper_limit = UPPER_AGE
    if lower_limit >= upper_limit:
        return None

    def func(t):
        curve_y = pb206u238_from_age(t)
        line_y = m * pb207u235_from_age(t) + c
        return curve_y - line_y

    lo = min(upper_limit, lower_limit + max(1.0, lower_limit * 1e-9))
    try:
        v1 = func(lo)
        v2 = func(upper_limit)
        if not (math.isfinite(v1) and math.isfinite(v2)):
            return None
        if v1 == 0:
            return lo
        if v2 == 0:
            return upper_limit
        if (v1 > 0 and v2 < 0) or (v1 < 0 and v2 > 0):
            result = root_scalar(func, bracket=(lo, upper_limit))
            return result.root
    except Exception:
        return None

    return None

def discordant_age_for_space(leadLossAge, x, y, ratio_space):
    if is_wetherill_space(ratio_space):
        x_low, y_low = concordia_xy(float(leadLossAge), ratio_space)
        return discordant_age_wetherill(x_low, y_low, x, y)
    x_low, y_low = concordia_xy(float(leadLossAge), ratio_space)
    return discordant_age(x_low, y_low, x, y)

def discordant_age_wetherill_from_tw(leadLossAge, u238pb206, pb207pb206):
    try:
        x2, y2 = tw_to_wetherill(u238pb206, pb207pb206)
    except Exception:
        return None
    x1, y1 = concordia_xy(float(leadLossAge), "Wetherill")
    return discordant_age_wetherill(x1, y1, x2, y2)


def mahalanobisRadius(sigmas):
    if sigmas == 1:
        p = 0.6827
    elif sigmas == 2:
        p = 0.9545
    else:
        raise Exception("Unable to handle " + str(sigmas) + " sigmas")
    return -2 * math.log(1 - p)

def isConcordantErrorEllipse(uPbValue, uPbError, pbPbValue, pbPbError, ellipseSigmas):
    """
    See https://www.xarg.org/2018/04/how-to-plot-a-covariance-error-ellipse/

    Keyword arguments:
    uPbValue -- the coordinate in TW concordia space
    uPbError -- the error associated with the U/Pb value (1 standard deviation)
    pbPbValue -- the x coordinate in TW concordia space
    pbPbError -- the error associated with the Pb/Pb value (1 standard deviation)
    """
    s = mahalanobisRadius(ellipseSigmas)

    # Handle degenerate cases
    if uPbError == 0:
        try:
            localPbPb = pb207pb206_from_u238pb206(uPbValue)
        except Exception:
            return False
        if not math.isfinite(localPbPb):
            return False
        root_s = math.sqrt(s)
        return abs(localPbPb-pbPbValue) <= pbPbError*root_s
    if pbPbError == 0:
        try:
            localUPb = u238pb206_from_pb207pb206(pbPbValue)
        except Exception:
            return False
        if not math.isfinite(localUPb):
            return False
        root_s = math.sqrt(s)
        return abs(localUPb-uPbValue) <= uPbError*root_s

    # Otherwise minimise for distance in elliptical space
    def distanceToEllipse(t):
        x = u238pb206_from_age(t)
        y = pb207pb206_from_age(t)
        value = ((uPbValue-x)/uPbError)**2 + ((pbPbValue-y)/pbPbError)**2
        return value

    result = minimize_scalar(distanceToEllipse, bracket=(1,5.*(10**9)))
    if not result.success:
        raise Exception("Exception occurred while minimising distance to error ellipse:\n\n" + result.message)
    return result.fun <= s

def isConcordantErrorEllipseForSpace(xValue, xError, yValue, yError, ellipseSigmas, ratio_space, rho=0.0):
    try:
        xValue = _require_positive_finite(xValue, "x ratio")
        yValue = _require_positive_finite(yValue, "y ratio")
    except Exception:
        return False

    s = mahalanobisRadius(ellipseSigmas)
    rho = 0.0 if rho is None else float(rho)
    if not math.isfinite(rho):
        rho = 0.0
    rho = max(-0.999999, min(0.999999, rho))

    xError = 0.0 if xError is None else float(xError)
    yError = 0.0 if yError is None else float(yError)
    if (not math.isfinite(xError)) or (not math.isfinite(yError)) or xError < 0.0 or yError < 0.0:
        return False

    if xError == 0.0 and yError == 0.0:
        try:
            x_c, y_c = concordia_xy(concordant_age_for_space(xValue, yValue, ratio_space), ratio_space)
            return x_c == xValue and y_c == yValue
        except Exception:
            return False

    if (rho == 0.0) and (not is_wetherill_space(ratio_space)):
        return isConcordantErrorEllipse(xValue, xError, yValue, yError, ellipseSigmas)

    def distanceToEllipse(t):
        x, y = concordia_xy(t, ratio_space)
        dx = float(xValue) - x
        dy = float(yValue) - y
        if xError == 0.0:
            return (dy / yError) ** 2 if dx == 0.0 else float("inf")
        if yError == 0.0:
            return (dx / xError) ** 2 if dy == 0.0 else float("inf")
        return (
            ((dx / xError) ** 2)
            - (2.0 * rho * dx * dy / (xError * yError))
            + ((dy / yError) ** 2)
        ) / (1.0 - rho * rho)

    result = minimize_scalar(distanceToEllipse, method='Bounded', bounds=[LOWER_AGE, UPPER_AGE])
    if not result.success:
        raise Exception("Exception occurred while minimising distance to error ellipse:\n\n" + result.message)
    return result.fun <= s

#############
## General ##
#############

def to1StdDev(value, error, form, sigmas):
    if form == "Percentage":
        error = (error/100.0) * value
    return error/sigmas

def from1StdDev(value, error, form, sigmas):
    res = error*sigmas
    if form == "Percentage":
        return 100.0*res/value
    return res

def convert_from_stddev_without_sigmas(value, error, form):
    if form == "Percentage":
        return 100.0*error/value
    return error
