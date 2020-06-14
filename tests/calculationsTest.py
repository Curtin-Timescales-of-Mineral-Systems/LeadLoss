import unittest

from process.calculations import concordant_age, u238pb206_from_age, pb207pb206_from_age
from utils.stringUtils import round_to_sf


class EllipseTests(unittest.TestCase):

    def testConcordantAge(self):
        t = 1*(10**9)
        uPb = u238pb206_from_age(t)
        pbPb = pb207pb206_from_age(t)

        self.assertAlmostEqual(t, round_to_sf(concordant_age(uPb, pbPb), 7))

if __name__ == '__main__':
    unittest.main()