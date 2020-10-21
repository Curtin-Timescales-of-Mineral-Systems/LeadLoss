from utils import config


class HistogramAxis:

    _age_xlim = (0, 5000)

    _barResolution = 100
    _barMax = 5000
    _barMin = 0
    _bars = int((_barMax - _barMin) / _barResolution)

    def __init__(self, axis):
        self.axis = axis

        self.calculatedXLimits = self._age_xlim

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

    def plotDistributions(self, concordantAges, reconstructedAges, selectedAges=None):
        self._clearDistributions()
        self._drawDistributions(concordantAges, reconstructedAges, selectedAges)

    #########
    ## All ##
    #########

    def _clearDistributions(self):
        self.axis.clear()
        self._setupAxis()

    def _drawDistributions(self, concordantAges, reconstructedAges, selectedAges):
        allAges = [concordantAges, reconstructedAges, selectedAges]
        allColours = [config.CONCORDANT_COLOUR_1, config.OPTIMAL_COLOUR_1, config.PREDICTION_COLOUR_1]

        for ages, colour in zip(allAges, allColours):
            if ages is None:
                continue

            self.axis.hist(
                [v/(10**6) for v in ages],
                bins=self._bars,
                cumulative=True,
                density=True,
                histtype='step',
                edgecolor=colour,
                facecolor=(0, 0, 0, 0)
            )