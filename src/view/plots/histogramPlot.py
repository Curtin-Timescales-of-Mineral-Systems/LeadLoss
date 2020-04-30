from utils import config


class HistogramPlot:

    _age_xlim = (0, 5000)

    _barResolution = 100
    _barMax = 5000
    _barMin = 0
    _bars = int((_barMax - _barMin) / _barResolution)

    def __init__(self, axis):
        self.axis = axis

        self.calculatedXLimits = self._age_xlim
        self.concordantAges = []

        self._setupAxis()

    def _setupAxis(self):
        self.axis.set_title("CDF")
        self.axis.set_xlabel("Age (Ma)")
        self.axis.set_ylim(0, 1.0)
        self.axis.set_xlim(*self._age_xlim)

    #############################
    ## Reconstructed age range ##
    #############################

    def setReconstructedAgeRange(self, reconstructedAgeRange):
        self.reconstructedAgeRange = [v / (10 ** 6) for v in reconstructedAgeRange]

    #############################
    ## Concordant distribution ##
    #############################

    # Haven't yet found a way of neatly updating histograms so we just crudely clear and redraw

    def plotConcordantDistribution(self, concordantAges):
        self.concordantAges = concordantAges
        self._clearDistributions()
        self._drawDistributions(self.concordantAges, [])

    def clearConcordantDistribution(self):
        self.reconstructedAgeRange = self._age_xlim
        self.concordantAges = []
        self._clearDistributions()

    ################################
    ## Reconstructed distribution ##
    ################################

    def plotOptimalReconstructedDistribution(self, reconstructedAges):
        self.optimalReconstructedAges = reconstructedAges
        self._clearDistributions()
        self._drawDistributions(self.concordantAges, [])

    def plotReconstructedDistribution(self, reconstructedAges):
        self._clearDistributions()
        self._drawDistributions(self.concordantAges, reconstructedAges)

    def clearReconstructedDistribution(self):
        self._clearDistributions()
        if self.concordantAges:
            self._drawDistributions(self.concordantAges, None)

    #########
    ## All ##
    #########

    def _clearDistributions(self):
        self.axis.clear()
        self._setupAxis()

    def _drawDistributions(self, concordantAges, reconstructedAges):
        self.axis.hist(
            [v/(10**6) for v in concordantAges],
            bins=self._bars,
            cumulative=True,
            density=True,
            histtype='step',
            edgecolor=config.CONCORDANT_COLOUR_1,
            facecolor=(0, 0, 0, 0)
        )

        if reconstructedAges is not None:
            edgecolor=config.PREDICTION_COLOUR_1
        else:
            reconstructedAges = self.optimalReconstructedAges
            edgecolor=config.OPTIMAL_COLOUR_1

        self.axis.hist(
            [v/(10**6) for v in reconstructedAges],
            bins=self._bars,
            cumulative=True,
            density=True,
            histtype='step',
            edgecolor=edgecolor,
            facecolor=(0, 0, 0, 0)
        )
        self.axis.set_xlim(*self.reconstructedAgeRange)