from matplotlib.collections import LineCollection

from process import calculations
from utils import config
from utils.errorbarPlot import Errorbars
from view.axes.concordia.abstractConcordiaAxis import ConcordiaAxis


class SampleMonteCarloConcordiaAxis(ConcordiaAxis):

    def __init__(self, axis):
        super().__init__(axis)

        self.concordantData = self.axis.plot([], [], marker='x', linewidth=0, color=config.CONCORDANT_COLOUR_1)[0]
        self.discordantData = self.axis.plot([], [], marker='x', linewidth=0, color=config.DISCORDANT_COLOUR_1)[0]
        self.leadLossAge = self.axis.plot([], [], marker='o', linewidth=0, color=config.OPTIMAL_COLOUR_1)[0]

        self.optimalAge = self.axis.plot([], [], marker='o', color=config.PREDICTION_COLOUR_1)[0]
        self.selectedAge = self.axis.plot([], [], marker='o', color=config.PREDICTION_COLOUR_1)[0]
        self.reconstructedLines = None

    ######################
    ## Internal actions ##
    ######################

    def plotMonteCarloRun(self, monteCarloRun):
        self.concordantData.set_xdata(monteCarloRun.concordant_uPb)
        self.concordantData.set_ydata(monteCarloRun.concordant_pbPb)
        self.discordantData.set_xdata(monteCarloRun.discordant_uPb)
        self.discordantData.set_ydata(monteCarloRun.discordant_pbPb)
        self.leadLossAge.set_xdata([monteCarloRun.optimal_uPb])
        self.leadLossAge.set_ydata([monteCarloRun.optimal_pbPb])

    def plotSelectedAge(self, selectedAge, reconstructedAges):
        self.clearSelectedAge()

        uPb = calculations.u238pb206_from_age(selectedAge)
        pbPb = calculations.pb207pb206_from_age(selectedAge)
        self.selectedAge.set_xdata([uPb])
        self.selectedAge.set_ydata([pbPb])

        lines = []
        for reconstructedAge in reconstructedAges:
            if reconstructedAge is None:
                line = []
            else:
                line = [
                    (calculations.u238pb206_from_age(selectedAge), calculations.pb207pb206_from_age(selectedAge)),
                    (calculations.u238pb206_from_age(reconstructedAge), calculations.pb207pb206_from_age(reconstructedAge))
                ]
            lines.append(line)

        self.reconstructedLines = LineCollection(
            lines,
            linewidths=1,
            colors=config.PREDICTION_COLOUR_1
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
