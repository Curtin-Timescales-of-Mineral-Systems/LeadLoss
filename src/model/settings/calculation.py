import math
from enum import Enum
import numpy as np

from process.dissimilarityTests import DissimilarityTest
from model.settings.type import SettingsType

class DiscordanceClassificationMethod(Enum):
    PERCENTAGE = "Percentage"
    ERROR_ELLIPSE = "Error ellipse"

    def __eq__(self, other):
        return self.value == other.value

class LeadLossCalculationSettings:

    KEY = SettingsType.CALCULATION

    def __init__(self):
        self.discordanceClassificationMethod = DiscordanceClassificationMethod.PERCENTAGE
        self.discordancePercentageCutoff = 0.1
        self.discordanceEllipseSigmas = 2

        self.minimumRimAge = 500*(10**6)
        self.maximumRimAge = 4500*(10**6)
        self.rimAgesSampled = 100

        self.monteCarloRuns = 50

        self.dissimilarityTest = DissimilarityTest.KOLMOGOROV_SMIRNOV

        self.penaliseInvalidAges = True

    def rimAges(self):
        return np.linspace(start=self.minimumRimAge, stop=self.maximumRimAge, num=self.rimAgesSampled)

    def getNearestSampledAge(self, targetAge):
        return min(self.rimAges(), key=lambda v: abs(v - targetAge))

    def validate(self):
        if self.discordanceClassificationMethod == DiscordanceClassificationMethod.PERCENTAGE:
            if not self.discordancePercentageCutoff:
                return "Please enter a discordance percentage cutoff"

            if self.discordancePercentageCutoff < 0 or self.discordancePercentageCutoff > 1.0:
                return "Discordance percentage cutoff must be between 0 and 100%"

        if not self.minimumRimAge:
            return "Please enter a minimum time for radiogenic-Pb loss"

        if not self.maximumRimAge:
            return "Please enter a maximum time for radiogenic-Pb loss"

        if self.minimumRimAge >= self.maximumRimAge:
            return "The minimum rim age must be strictly less than the maximum rim age"

        if not self.rimAgesSampled:
            return "Please enter a number of samples"

        if self.rimAgesSampled < 2:
            return "The number of samples must be >= 2"

        if not self.monteCarloRuns:
            return "Please enter a number of Monte Carlo runs"

        if self.monteCarloRuns < 1:
            return "The number of Monte Carlo runs must be >= 1"

        return None

    @staticmethod
    def getDefaultHeaders():
        return [
            "Concordant",
            "Discordance (%)",
            "Age (Ma)",
        ]