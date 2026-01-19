import math
from scipy.optimize import root_scalar, minimize_scalar

import utils.errorUtils as errors

# --- Constants (match current codebase constants) ---
U238_DECAY_CONSTANT = 1.55125e-10
U235_DECAY_CONSTANT = 9.8485e-10
U238U235_RATIO = 137.818

UPPER_AGE = 6000 * (10 ** 6)
LOWER_AGE = 1


# ----------------
# Age/ratio helpers
# ----------------
def age_from_pb206u238(ratio206_238: float) -> float:
    return errors.log(ratio206_238 + 1.0) / U238_DECAY_CONSTANT


def age_from_pb207u235(ratio207_235: float) -> float:
    return errors.log(ratio207_235 + 1.0) / U235_DECAY_CONSTANT


def pb206u238_from_age(age: float) -> float:
    return errors.exp(U238_DECAY_CONSTANT * age) - 1.0


def pb207u235_from_age(age: float) -> float:
    return errors.exp(U235_DECAY_CONSTANT * age) - 1.0


# ------------------------
# Concordia distance logic
# ------------------------
def discordance_wetherill(r207_235: float, r206_238: float) -> float:
    """
    Signed discordance in Wetherill space, defined using 207/235 and 206/238 ages:
        (t207 - t206) / t207

    Positive => "normal discordance" (t207 > t206)
    Negative => "reverse discordance" (t207 < t206)
    """
    t207 = age_from_pb207u235(r207_235)
    t206 = age_from_pb206u238(r206_238)

    if t207 == 0 or not math.isfinite(t207) or not math.isfinite(t206):
        return float("nan")

    result = (t207 - t206) / t207

    # Only squash tiny floating-point noise
    if abs(result) < 1e-12:
        return 0.0
    return result


def concordant_age_wetherill(r206_238: float, r207_235: float) -> float:
    """
    Returns the age t that minimises Euclidean distance in Wetherill (x,y) space to the concordia curve,
    where x(t)=207/235 and y(t)=206/238.

    Note: this is a geometric closest-point-on-curve estimate.
    """
    def distance(t: float) -> float:
        x_theory = pb207u235_from_age(t)
        y_theory = pb206u238_from_age(t)
        dx = r207_235 - x_theory
        dy = r206_238 - y_theory
        return math.hypot(dx, dy)

    res = minimize_scalar(distance, method="bounded", bounds=(LOWER_AGE, UPPER_AGE))
    return res.x


def discordant_age_wetherill(x1: float, y1: float, x2: float, y2: float):
    """
    Upper-intercept reconstruction for a discordant spot in Wetherill space.

    The line between:
      anchor point (x1,y1) on concordia (from Pb-loss age)
    and
      observed point (x2,y2)
    intersects concordia at (at least) the anchor age; we solve for the *other* intersection (older).

    Returns:
      age in years, or None if no valid intersection found.
    """
    # In Wetherill, x increases with age. Anchor should be younger => x1 < x2.
    if x1 >= x2:
        return None

    anchor_age = age_from_pb207u235(x1)

    # Skip the trivial root at the anchor age
    bracket_start = anchor_age + 1e3  # 1000 years
    if bracket_start >= UPPER_AGE:
        return None

    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1

    def func(t: float) -> float:
        x_t = pb207u235_from_age(t)
        y_t = pb206u238_from_age(t)
        return y_t - (m * x_t + c)

    v1 = func(bracket_start)
    v2 = func(UPPER_AGE)

    # Need a sign change to bracket a root
    if (v1 > 0 and v2 > 0) or (v1 < 0 and v2 < 0):
        return None

    res = root_scalar(func, bracket=(bracket_start, UPPER_AGE))
    if not res.converged:
        return None
    return res.root


# ----------------------------
# Error ellipse concordance test
# ----------------------------
def mahalanobisRadius(sigmas: int) -> float:
    if sigmas == 1:
        p = 0.6827
    elif sigmas == 2:
        p = 0.9545
    else:
        raise ValueError(f"Cannot handle ellipse sigmas={sigmas}")
    return -2.0 * math.log(1.0 - p)


def isConcordantErrorEllipseWetherill(
    ratio207_235: float,
    error207_235: float,
    ratio206_238: float,
    error206_238: float,
    ellipseSigmas: int
) -> bool:
    """
    Returns True if the concordia curve intersects the axis-aligned error ellipse
    (no covariance) around the measured point.

    This mirrors the TW ellipse approach in your app, but in Wetherill coordinates.
    """
    s = mahalanobisRadius(ellipseSigmas)

    # Degenerate ellipse handling (one axis error == 0)
    if error206_238 == 0:
        try:
            root_s = math.sqrt(s)
            t_guess = age_from_pb206u238(ratio206_238)
            x_theory = pb207u235_from_age(t_guess)
            dx = abs(ratio207_235 - x_theory)
            return dx <= error207_235 * root_s
        except Exception:
            return False

    if error207_235 == 0:
        try:
            root_s = math.sqrt(s)
            t_guess = age_from_pb207u235(ratio207_235)
            y_theory = pb206u238_from_age(t_guess)
            dy = abs(ratio206_238 - y_theory)
            return dy <= error206_238 * root_s
        except Exception:
            return False

    def distanceToEllipse(t: float) -> float:
        x_th = pb207u235_from_age(t)
        y_th = pb206u238_from_age(t)
        dx = (ratio207_235 - x_th) / error207_235
        dy = (ratio206_238 - y_th) / error206_238
        return dx * dx + dy * dy

    res = minimize_scalar(distanceToEllipse, bracket=(LOWER_AGE, UPPER_AGE))
    if not res.success:
        # If this ever happens, we *want* to know loudly in a PhD pipeline.
        raise RuntimeError("Error minimising Wetherill ellipse distance: " + str(res.message))

    return res.fun <= s
