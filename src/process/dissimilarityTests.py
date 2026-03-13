from enum import Enum

from scipy.stats import ks_2samp


class DissimilarityTest(Enum):
    KOLMOGOROV_SMIRNOV = "Kolmogorov-Smirnov"

    # KUIPER = "Kuiper"
    # CRAMER_VON_MISES = "Cramér-von-Mises"

    def __eq__(self, other):
        return self.value == getattr(other, "value", other)

    def perform(self, distribution1, distribution2):
        if self == DissimilarityTest.KOLMOGOROV_SMIRNOV:
            if not distribution1 or not distribution2:
                return (1.0, 0.0)
            return ks_2samp(distribution1, distribution2)

    def getComparisonValue(self, statistic):
        if self == DissimilarityTest.KOLMOGOROV_SMIRNOV:
            d, p = statistic
            return d

    def getPValue(self, statistic):
        if self == DissimilarityTest.KOLMOGOROV_SMIRNOV:
            d, p = statistic
            return p
