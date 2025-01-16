import unittest
import math
from scipy.optimize import minimize_scalar

U235_DECAY_CONSTANT = 9.8485e-10
U238_DECAY_CONSTANT = 1.55125e-10

def pb207u235_from_age(t):
    return math.exp(U235_DECAY_CONSTANT * t) - 1

def pb206u238_from_age(t):
    return math.exp(U238_DECAY_CONSTANT * t) - 1

def mahalanobisRadius(sigmas):
    if sigmas == 2:
        return -2 * math.log(1 - 0.9545)  # ~4.606
    elif sigmas == 1:
        return -2 * math.log(1 - 0.6827)
    else:
        raise ValueError("Unsupported ellipse sigmas")

def isConcordantErrorEllipseWetherill(r206_238, e206_238,
                                      r207_235, e207_235,
                                      ellipseSigmas):
    """
    Minimal version
    """
    from math import sqrt
    s = mahalanobisRadius(ellipseSigmas)
    root_s = sqrt(s)

    # handle degenerate
    if e206_238 == 0:
        t_guess = math.log(r206_238+1)/U238_DECAY_CONSTANT
        x_theory = pb207u235_from_age(t_guess)
        diff = abs(x_theory - r207_235)
        return (diff <= e207_235 * root_s)
    if e207_235 == 0:
        t_guess = math.log(r207_235+1)/U235_DECAY_CONSTANT
        y_theory = pb206u238_from_age(t_guess)
        diff = abs(y_theory - r206_238)
        return (diff <= e206_238 * root_s)

    def distanceToEllipse(t):
        x_theory = pb207u235_from_age(t)
        y_theory = pb206u238_from_age(t)
        dx = (r207_235 - x_theory)/e207_235
        dy = (r206_238 - y_theory)/e206_238
        return dx*dx + dy*dy

    res = minimize_scalar(distanceToEllipse, bracket=(1e5, 6e9))
    if not res.success:
        return False
    return (res.fun <= s)

class WetherillEllipseTests(unittest.TestCase):

    def testBasicConcordantCase(self):
        is_conc = isConcordantErrorEllipseWetherill(
            r206_238=0.168, e206_238=0.01,
            r207_235=1.68,  e207_235=0.02,
            ellipseSigmas=2
        )
        self.assertTrue(is_conc)

    def testClearlyOffCurve(self):
        is_conc = isConcordantErrorEllipseWetherill(
            r206_238=0.50, e206_238=0.01,
            r207_235=0.10, e207_235=0.02,
            ellipseSigmas=2
        )
        self.assertFalse(is_conc)

    def testDegenerateXErrorZero(self):
        is_conc = isConcordantErrorEllipseWetherill(
            r206_238=0.168, e206_238=0.01,
            r207_235=1.68,  e207_235=0.0,
            ellipseSigmas=2
        )
        self.assertTrue(is_conc)

    def testDegenerateYErrorZero(self):
        is_conc = isConcordantErrorEllipseWetherill(
            r206_238=0.593, e206_238=0.0,
            r207_235=18.2,  e207_235=0.2,
            ellipseSigmas=2
        )
        self.assertTrue(is_conc)

if __name__ == "__main__":
    unittest.main()
