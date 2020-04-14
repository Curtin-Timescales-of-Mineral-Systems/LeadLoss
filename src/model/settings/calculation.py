from enum import Enum

from process.dissimilarityTests import DissimilarityTest
from model.settings.type import SettingsType


class DiscordanceClassificationMethod(Enum):
    PERCENTAGE="Percentage"
    ERROR_ELLIPSE="Error ellipse"

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

        self.dissimilarityTest = DissimilarityTest.KOLMOGOROV_SMIRNOV


    def validate(self):
        if self.discordanceClassificationMethod == DiscordanceClassificationMethod.PERCENTAGE:
            if self.discordancePercentageCutoff < 0 or self.discordancePercentageCutoff > 1.0:
                return "Discordance percentage cutoff must be between 0 and 100%"

        if self.minimumRimAge >= self.maximumRimAge:
            return "The minimum rim age must be strictly less than the maximum rim age"

        if self.rimAgesSampled < 2:
            return "The number of samples must be >= 2"

        return None

    def getHeaders(self):
        headers = []
        headers.append("Concordant")
        if self.discordanceClassificationMethod.value == DiscordanceClassificationMethod.PERCENTAGE.value:
            headers.append("Discordance (%)")
        headers.append("Age (Ma)")
        return headers

    @staticmethod
    def getDefaultHeaders():
        return [
            "Concordant",
            "Discordance (%)",
            "Age (Ma)",
        ]