
import sys

# Add path to sys.path
path = r'C:\Users\20793801\OneDrive - Curtin University of Technology Australia\Documents\GitHub\LeadLoss\src'
sys.path.append(path)

import unittest
from process.processing import _calculateConcordantAges, ProgressType
from model.sample import Sample, SampleSignals
from model.settings.calculation import LeadLossCalculationSettings


class SampleSpot:
    def __init__(self, valid, uPbValue, pbPbValue, uPbStDev, pbPbStDev, concordant=True):
        self.valid = valid  # Define the 'valid' attribute here
        self.uPbValue = uPbValue
        self.pbPbValue = pbPbValue
        self.uPbStDev = uPbStDev
        self.pbPbStDev = pbPbStDev
        self.concordant = concordant  # Set to True for concordant spots, False for discordant spots

        
class TestCalculateConcordantAges(unittest.TestCase):
    def test_calculate_concordant_ages(self):
        # Create a sample with sample spots and calculation settings
        spots = [SampleSpot(valid=True,uPbValue=100, pbPbValue=50, uPbStDev=2, pbPbStDev=1, concordant=True)]
        calculation_settings = LeadLossCalculationSettings()
        sample = Sample(id=1, name="Sample1", spots=spots)
        sample.calculationSettings = calculation_settings

        # Set up SampleSignals
        signals = SampleSignals()
        sample.signals = signals

        # Call the _calculateConcordantAges function
        result = _calculateConcordantAges(signals, sample)

        # Assert that the function returns True (processing continues)
        self.assertTrue(result)

        # Assert that the progress is set to 1.0 (no discordant points)
        signals.progress.assert_called_with(ProgressType.OPTIMAL, 1.0, sample.name, None)
        
if __name__ == '__main__':
    unittest.main()


