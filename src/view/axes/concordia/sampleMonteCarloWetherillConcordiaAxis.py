from matplotlib.collections import LineCollection

from process import calculations
from utils import config
from utils.errorbarPlot import Errorbars
from view.axes.concordia.abstractConcordiaAxis import ConcordiaAxis
from model import monteCarloRunWetherill
from process import calculationsWetherill

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

    def plotMonteCarloRun(self, MonteCarloRunWetherill):

        self.concordantData.set_xdata(MonteCarloRunWetherill.concordant_207_235)
        self.concordantData.set_ydata(MonteCarloRunWetherill.concordant_206_238)
        self.discordantData.set_xdata(MonteCarloRunWetherill.discordant_207_235)
        self.discordantData.set_ydata(MonteCarloRunWetherill.discordant_206_238)
        self.leadLossAge.set_xdata([MonteCarloRunWetherill.optimal_207_235])
        self.leadLossAge.set_ydata([MonteCarloRunWetherill.optimal_206_238])

        max_x = max(
            max(MonteCarloRunWetherill.concordant_207_235) if len(MonteCarloRunWetherill.concordant_207_235)>0 else 0,
            max(MonteCarloRunWetherill.discordant_207_235) if len(MonteCarloRunWetherill.discordant_207_235)>0 else 0,
            MonteCarloRunWetherill.optimal_207_235
        )
        self.axis.set_xlim(0, 1.2 * max_x)

    def plotSelectedAge(self, selectedAge, reconstructedAges):
        self.clearSelectedAge()

        # In Wetherill space:
        xVal = calculationsWetherill.pb207u235_from_age(selectedAge)
        yVal = calculationsWetherill.pb206u238_from_age(selectedAge)
        self.selectedAge.set_xdata([xVal])
        self.selectedAge.set_ydata([yVal])

        lines = []
        for reconstructedAge in reconstructedAges:
            if reconstructedAge is None:
                line = []
            else:
                xRec = calculationsWetherill.pb207u235_from_age(reconstructedAge)
                yRec = calculationsWetherill.pb206u238_from_age(reconstructedAge)
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