from enum import Enum


class DissimilarityTest(Enum):
    KOLMOGOROV_SMIRNOV = "Kolmogorov-Smirnov"
    KUIPER = "Kuiper"
    CRAMER_VON_MISES = "Cramér-von-Mises"
