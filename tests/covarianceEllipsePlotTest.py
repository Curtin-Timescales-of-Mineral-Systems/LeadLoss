import math
import unittest

from utils.covarianceEllipsePlot import ellipse_geometry


class CovarianceEllipsePlotTest(unittest.TestCase):
    def test_uncorrelated_axes_match_input_standard_deviations(self):
        width, height, angle = ellipse_geometry(2.0, 1.0, 0.0, 2.5)
        self.assertAlmostEqual(width, 10.0)
        self.assertAlmostEqual(height, 5.0)
        self.assertTrue(math.isclose(abs(angle), 180.0) or math.isclose(angle, 0.0))

    def test_correlation_rotates_the_ellipse(self):
        _, _, angle = ellipse_geometry(1.0, 1.0, 0.8, 2.5)
        equivalent = ((angle + 180.0) % 180.0)
        self.assertAlmostEqual(equivalent, 45.0)


if __name__ == "__main__":
    unittest.main()
