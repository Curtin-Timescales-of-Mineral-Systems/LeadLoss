from enum import Enum
import numpy as np

from process.dissimilarityTests import DissimilarityTest
from model.settings.type import SettingsType

class ConcordiaMode(Enum):
    TW = "TW"
    WETHERILL = "Wetherill"

    @staticmethod
    def coerce(value):
        if isinstance(value, ConcordiaMode):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if "weth" in v:
                return ConcordiaMode.WETHERILL
            return ConcordiaMode.TW
        return ConcordiaMode.TW

class DiscordanceClassificationMethod(Enum):
    PERCENTAGE = "Percentage"
    ERROR_ELLIPSE = "Error ellipse"

    def __eq__(self, other):
        return self.value == getattr(other, "value", other)

class LeadLossCalculationSettings:
    KEY = SettingsType.CALCULATION

    def __init__(self):

        # Concordia space
        # (TW = Tera–Wasserburg; Wetherill = conventional 207/235 vs 206/238)
        self.concordiaMode = ConcordiaMode.TW

        # Discordance
        self.discordanceClassificationMethod = DiscordanceClassificationMethod.PERCENTAGE
        self.discordancePercentageCutoff = 0.10     # 0..1 (10% default)
        self.discordanceEllipseSigmas = 2

        # Pb-loss time grid (YEARS internally)
        self.minimumRimAge = 1 * 10**6
        self.maximumRimAge = 2500 * 10**6
        self.rimAgesSampled = 250

        # MC
        self.monteCarloRuns = 100

        # Comparison
        self.dissimilarityTest = DissimilarityTest.KOLMOGOROV_SMIRNOV
        self.penaliseInvalidAges = True

        # Legacy multi-peak (UI toggles preserved)
        self.useSummedKS = False
        self.summedKSSmoothSigma = 1.0

        # Clustering toggles (old semantics)
        self.use_discordant_clustering = False
        self.relabel_clusters_per_run  = False

        # Experimental extras
        self.enable_ensemble_peak_picking = False
        self.use_score_weighted_voting    = False
        self.use_hdi_top_peak_ci          = False

        # Population-aware CDC
        self.split_by_concordant_population = False

    def rimAges(self):
        return np.linspace(start=self.minimumRimAge, stop=self.maximumRimAge, num=self.rimAgesSampled)

    def getNearestSampledAge(self, targetAge):
        return min(self.rimAges(), key=lambda v: abs(v - targetAge))

    def validate(self):
        if self.discordanceClassificationMethod == DiscordanceClassificationMethod.PERCENTAGE:
            if self.discordancePercentageCutoff is None:
                return "Please enter a discordance percentage cutoff"
            if not (0.0 <= float(self.discordancePercentageCutoff) <= 1.0):
                return "Discordance percentage cutoff must be between 0.0 and 1.0"

        if self.minimumRimAge is None:
            return "Please enter a minimum time for radiogenic-Pb loss"
        if self.maximumRimAge is None:
            return "Please enter a maximum time for radiogenic-Pb loss"
        if float(self.minimumRimAge) >= float(self.maximumRimAge):
            return "The minimum rim age must be strictly less than the maximum rim age"

        if self.rimAgesSampled is None:
            return "Please enter a number of samples"
        if int(self.rimAgesSampled) < 2:
            return "The number of samples must be ≥ 2"

        if self.monteCarloRuns is None:
            return "Please enter a number of Monte Carlo runs"
        if int(self.monteCarloRuns) < 1:
            return "The number of Monte Carlo runs must be ≥ 1"

        return None

    @staticmethod
    def getDefaultHeaders():
        return ["Concordant", "Discordance (%)", "Age (Ma)"]
