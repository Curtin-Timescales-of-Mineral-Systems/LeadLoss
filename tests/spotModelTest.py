import unittest

from model.settings.imports import LeadLossImportSettings
from model.settings.ratio import ConcordiaRatioSpace, ConcordiaSpaceSelection
from model.spot import Spot
from process import calculations


class SpotModelTest(unittest.TestCase):
    def setUp(self):
        self.settings = LeadLossImportSettings()
        self.row = ["S1", "2.5", "0.1", "0.15", "0.01"]

    def test_discordance_display_column_does_not_grow_across_runs(self):
        s = Spot(self.row, self.settings)
        base_len = len(s.displayStrings)
        self.assertEqual(base_len, 4)

        s.updateConcordance(False, 0.10, reverse=False)
        self.assertEqual(len(s.displayStrings), base_len + 1)
        first = s.displayStrings[-1]

        s.updateConcordance(False, 0.20, reverse=False)
        self.assertEqual(len(s.displayStrings), base_len + 1)
        second = s.displayStrings[-1]
        self.assertNotEqual(first, second)

        s.clear()
        self.assertEqual(len(s.displayStrings), base_len)

    def test_error_ellipse_mode_keeps_base_display_columns(self):
        s = Spot(self.row, self.settings)
        base_len = len(s.displayStrings)
        s.updateConcordance(True, None, reverse=False)
        self.assertEqual(len(s.displayStrings), base_len)

    def test_native_wetherill_ratios_and_rho_are_cached(self):
        self.settings.inputRatioSpace = ConcordiaRatioSpace.WETHERILL
        self.settings.displayRatioSpace = ConcordiaSpaceSelection.WETHERILL
        self.settings.rhoColumn = 5

        age = 1100 * (10**6)
        pb207u235 = calculations.pb207u235_from_age(age)
        pb206u238 = calculations.pb206u238_from_age(age)
        row = ["S1", str(pb207u235), "0.02", str(pb206u238), "0.004", "0.42"]

        s = Spot(row, self.settings)

        self.assertTrue(s.valid)
        self.assertEqual(len(s.displayStrings), 5)
        self.assertAlmostEqual(s.pb207u235Value, pb207u235)
        self.assertAlmostEqual(s.pb206u238Value, pb206u238)
        self.assertAlmostEqual(s.uPbValue, 1.0 / pb206u238)
        self.assertAlmostEqual(s.pbPbValue, pb207u235 / (calculations.U238U235_RATIO * pb206u238))
        self.assertAlmostEqual(s.getRatioStDevs(ConcordiaRatioSpace.WETHERILL)[2], 0.42)

    def test_native_tw_rho_is_cached_and_propagated(self):
        self.settings.inputRatioSpace = ConcordiaRatioSpace.TERA_WASSERBURG
        self.settings.displayRatioSpace = ConcordiaSpaceSelection.TERA_WASSERBURG
        self.settings.rhoColumn = 5
        row = ["S1", "2.5", "0.1", "0.15", "0.01", "-0.35"]

        s = Spot(row, self.settings)

        self.assertTrue(s.valid)
        self.assertEqual(len(s.displayStrings), 5)
        self.assertAlmostEqual(s.getRatioStDevs(ConcordiaRatioSpace.TERA_WASSERBURG)[2], -0.35)
        self.assertTrue(
            -1.0 <= s.getRatioStDevs(ConcordiaRatioSpace.WETHERILL)[2] <= 1.0
        )



if __name__ == "__main__":
    unittest.main()
