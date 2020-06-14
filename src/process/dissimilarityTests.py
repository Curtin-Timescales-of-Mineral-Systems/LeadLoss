from enum import Enum


class DissimilarityTest(Enum):
    KOLMOGOROV_SMIRNOV = "Kolmogorov-Smirnov"

    # KUIPER = "Kuiper"
    # CRAMER_VON_MISES = "Cramér-von-Mises"

    def __eq__(self, other):
        return self.value == other.value

    def getComparisonValue(self, statistic):
        if self == DissimilarityTest.KOLMOGOROV_SMIRNOV:
            d, p = statistic
            return d

    def getPValue(self, statistic):
        if self == DissimilarityTest.KOLMOGOROV_SMIRNOV:
            d, p = statistic
            return p
