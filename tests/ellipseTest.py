import unittest

from process import calculations


class EllipseTests(unittest.TestCase):

    def testXDimExamples(self):
        self.assertTrue(calculations.isConcordantErrorEllipse(
            uPbValue=0.8,
            uPbError=0.1,
            pbPbValue=1,
            pbPbError=0.05,
            ellipseSigmas=2))

        self.assertFalse(calculations.isConcordantErrorEllipse(
            uPbValue=1.1,
            uPbError=0.1,
            pbPbValue=1,
            pbPbError=0.05,
            ellipseSigmas=2))

        self.assertFalse(calculations.isConcordantErrorEllipse(
            uPbValue=0.5,
            uPbError=0.1,
            pbPbValue=1,
            pbPbError=0.05,
            ellipseSigmas=2))


    def testXDimDegenerateExamples(self):
        self.assertTrue(calculations.isConcordantErrorEllipse(
            uPbValue=0.8,
            uPbError=0.1,
            pbPbValue=1,
            pbPbError=0,
            ellipseSigmas=2))

        self.assertFalse(calculations.isConcordantErrorEllipse(
            uPbValue=1.1,
            uPbError=0.1,
            pbPbValue=1,
            pbPbError=0,
            ellipseSigmas=2))

        self.assertFalse(calculations.isConcordantErrorEllipse(
            uPbValue=0.5,
            uPbError=0.1,
            pbPbValue=1,
            pbPbError=0,
            ellipseSigmas=2))

    def testYDimExamples(self):
        self.assertTrue(calculations.isConcordantErrorEllipse(
            uPbValue=1,
            uPbError=0.1,
            pbPbValue=0.5,
            pbPbError=0.05,
            ellipseSigmas=2))

        self.assertFalse(calculations.isConcordantErrorEllipse(
            uPbValue=1,
            uPbError=0.1,
            pbPbValue=1.5,
            pbPbError=0.05,
            ellipseSigmas=2))

        self.assertFalse(calculations.isConcordantErrorEllipse(
            uPbValue=1,
            uPbError=0.1,
            pbPbValue=0.3,
            pbPbError=0.05,
            ellipseSigmas=2))

    def testYDimDegenerateExamples(self):
        self.assertTrue(calculations.isConcordantErrorEllipse(
            uPbValue=1,
            uPbError=0,
            pbPbValue=0.5,
            pbPbError=0.05,
            ellipseSigmas=2))

        self.assertFalse(calculations.isConcordantErrorEllipse(
            uPbValue=1,
            uPbError=0,
            pbPbValue=1.5,
            pbPbError=0.05,
            ellipseSigmas=2))

        self.assertFalse(calculations.isConcordantErrorEllipse(
            uPbValue=1,
            uPbError=0,
            pbPbValue=0.3,
            pbPbError=0.05,
            ellipseSigmas=2))

if __name__ == '__main__':
    unittest.main()