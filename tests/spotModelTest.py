import unittest

from model.settings.imports import LeadLossImportSettings
from model.spot import Spot


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


if __name__ == "__main__":
    unittest.main()
