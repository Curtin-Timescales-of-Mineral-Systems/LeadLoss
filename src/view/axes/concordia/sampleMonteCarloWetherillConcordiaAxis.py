from matplotlib.collections import LineCollection

from process import calculations
from utils import config
from utils.errorbarPlot import Errorbars
from view.axes.concordia.abstractConcordiaAxis import ConcordiaAxis


class SampleMonteCarloWetherillConcordiaAxis(ConcordiaAxis):
    """
    Identical structure and naming to SampleMonteCarloConcordiaAxis,
    but adapted for Wetherill ratios:
      X = 207Pb/235U
      Y = 206Pb/238U
    """

    def __init__(self, axis):
        super().__init__(axis)

        self.concordantData = self.axis.plot(
            [], [], marker='x', linewidth=0, color=config.CONCORDANT_COLOUR_1
        )[0]
        self.discordantData = self.axis.plot(
            [], [], marker='x', linewidth=0, color=config.DISCORDANT_COLOUR_1
        )[0]
        self.leadLossAge = self.axis.plot(
            [], [], marker='o', linewidth=0, color=config.OPTIMAL_COLOUR_1
        )[0]

        self.optimalAge = self.axis.plot(
            [], [], marker='o', color=config.PREDICTION_COLOUR_1
        )[0]
        self.selectedAge = self.axis.plot(
            [], [], marker='o', color=config.PREDICTION_COLOUR_1
        )[0]
        self.reconstructedLines = None

    ######################
    ## Internal actions ##
    ######################

    def plotMonteCarloRun(self, monteCarloRun):

        self.concordantData.set_xdata(monteCarloRun.concordant_207_235)
        self.concordantData.set_ydata(monteCarloRun.concordant_206_238)
        self.discordantData.set_xdata(monteCarloRun.discordant_207_235)
        self.discordantData.set_ydata(monteCarloRun.discordant_206_238)
        self.leadLossAge.set_xdata([monteCarloRun.optimal_207_235])
        self.leadLossAge.set_ydata([monteCarloRun.optimal_206_238])

        # upper_xlim = max(
        # max(monteCarloRun.concordant_uPb),
        # max(monteCarloRun.discordant_uPb),
        # monteCarloRun.optimal_uPb)
        # self.axis.set_xlim(0, 1.2*upper_xlim)

    def plotSelectedAge(self, selectedAge, reconstructedAges):
        self.clearSelectedAge()

        # In Wetherill space:
        xVal = calculations.pb207u235_from_age(selectedAge)
        yVal = calculations.pb206u238_from_age(selectedAge)
        self.selectedAge.set_xdata([xVal])
        self.selectedAge.set_ydata([yVal])

        lines = []
        for reconstructedAge in reconstructedAges:
            if reconstructedAge is None:
                line = []
            else:
                xRec = calculations.pb207u235_from_age(reconstructedAge)
                yRec = calculations.pb206u238_from_age(reconstructedAge)
                line = [(xVal, yVal), (xRec, yRec)]
            lines.append(line)

        self.reconstructedLines = LineCollection(
            lines, linewidths=1, colors=config.PREDICTION_COLOUR_1
        )
        self.axis.add_collection(self.reconstructedLines)

    def clearSelectedAge(self):
        self.concordantData.set_xdata([])
        self.concordantData.set_ydata([])
        self.discordantData.set_xdata([])
        self.discordantData.set_ydata([])
        self.leadLossAge.set_xdata([])
        self.leadLossAge.set_ydata([])

    #############
    ## Actions ##
    #############

    def clearInputData(self):
        self.clearSelectedAge()