import unittest
import numpy as np

from process import calculations
from process.concordiaTransforms import (
    tw_to_wetherill,
    wetherill_to_tw,
    tw_stdevs_to_wetherill,
    wetherill_stdevs_to_tw,
)


class ConcordiaTransformsTest(unittest.TestCase):
    def test_tw_to_wetherill_matches_concordia_param(self):
        ages = [10e6, 100e6, 500e6, 1500e6, 3000e6]
        for t in ages:
            u = calculations.u238pb206_from_age(t)
            v = calculations.pb207pb206_from_age(t)

            xw, yw = tw_to_wetherill(u, v)

            xw_true = calculations.pb207u235_from_age(t)
            yw_true = calculations.pb206u238_from_age(t)

            self.assertTrue(np.isfinite(xw) and np.isfinite(yw))
            self.assertAlmostEqual(xw, xw_true, places=10)
            self.assertAlmostEqual(yw, yw_true, places=10)

    def test_wetherill_to_tw_roundtrip(self):
        ages = [10e6, 100e6, 500e6, 1500e6, 3000e6]
        for t in ages:
            xw = calculations.pb207u235_from_age(t)
            yw = calculations.pb206u238_from_age(t)

            u, v = wetherill_to_tw(xw, yw)
            u_true = calculations.u238pb206_from_age(t)
            v_true = calculations.pb207pb206_from_age(t)

            self.assertTrue(np.isfinite(u) and np.isfinite(v))
            self.assertAlmostEqual(u, u_true, places=10)
            self.assertAlmostEqual(v, v_true, places=10)

    def test_stdev_propagation_roundtrip_is_finite(self):
        # Not asserting equality (nonlinear), just finiteness + sanity
        u, v = 10.0, 0.2
        su, sv = 0.1, 0.01

        sxw, syw = tw_stdevs_to_wetherill(u, su, v, sv)
        self.assertTrue(np.isfinite(sxw) and np.isfinite(syw))

        su2, sv2 = wetherill_stdevs_to_tw(
            *tw_to_wetherill(u, v),
            sxw,
            syw
        )
        self.assertTrue(np.isfinite(su2) and np.isfinite(sv2))


if __name__ == "__main__":
    unittest.main()
