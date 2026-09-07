import unittest

from model.settings.ratio import ConcordiaRatioSpace
from process.calculations import (
    concordant_age,
    concordant_age_for_space,
    concordia_xy,
    convert_ratio_covariance,
    discordant_age_for_space,
    pb206u238_from_age,
    pb207pb206_from_age,
    pb207u235_from_age,
    tw_to_wetherill,
    u238pb206_from_age,
    wetherill_to_tw,
)
from utils.stringUtils import round_to_sf


class EllipseTests(unittest.TestCase):

    def testConcordantAge(self):
        t = 1*(10**9)
        uPb = u238pb206_from_age(t)
        pbPb = pb207pb206_from_age(t)

        self.assertAlmostEqual(t, round_to_sf(concordant_age(uPb, pbPb), 7))

    def testWetherillRoundTrip(self):
        t = 1250 * (10**6)
        tw_x = u238pb206_from_age(t)
        tw_y = pb207pb206_from_age(t)
        w_x, w_y = tw_to_wetherill(tw_x, tw_y)

        self.assertAlmostEqual(w_x, pb207u235_from_age(t))
        self.assertAlmostEqual(w_y, pb206u238_from_age(t))

        back_x, back_y = wetherill_to_tw(w_x, w_y)
        self.assertAlmostEqual(back_x, tw_x)
        self.assertAlmostEqual(back_y, tw_y)
        self.assertAlmostEqual(t, round_to_sf(concordant_age_for_space(w_x, w_y, ConcordiaRatioSpace.WETHERILL), 7))

    def testWetherillCovarianceRoundTrip(self):
        t = 1800 * (10**6)
        tw_x = u238pb206_from_age(t)
        tw_y = pb207pb206_from_age(t)
        sx = 0.04
        sy = 0.002
        rho = 0.35

        w_x, w_y = tw_to_wetherill(tw_x, tw_y)
        w_sx, w_sy, w_rho = convert_ratio_covariance(
            tw_x,
            tw_y,
            sx,
            sy,
            rho,
            ConcordiaRatioSpace.TERA_WASSERBURG,
            ConcordiaRatioSpace.WETHERILL,
        )
        back_sx, back_sy, back_rho = convert_ratio_covariance(
            w_x,
            w_y,
            w_sx,
            w_sy,
            w_rho,
            ConcordiaRatioSpace.WETHERILL,
            ConcordiaRatioSpace.TERA_WASSERBURG,
        )

        self.assertAlmostEqual(back_sx, sx)
        self.assertAlmostEqual(back_sy, sy)
        self.assertAlmostEqual(back_rho, rho)

    def testWetherillDiscordiaUpperIntercept(self):
        lower_age = 500 * (10**6)
        upper_age = 2500 * (10**6)
        lower_x, lower_y = concordia_xy(lower_age, ConcordiaRatioSpace.WETHERILL)
        upper_x, upper_y = concordia_xy(upper_age, ConcordiaRatioSpace.WETHERILL)
        point_x = lower_x + 0.4 * (upper_x - lower_x)
        point_y = lower_y + 0.4 * (upper_y - lower_y)

        recovered = discordant_age_for_space(
            lower_age,
            point_x,
            point_y,
            ConcordiaRatioSpace.WETHERILL,
        )

        self.assertAlmostEqual(round_to_sf(recovered, 7), upper_age)

if __name__ == '__main__':
    unittest.main()
