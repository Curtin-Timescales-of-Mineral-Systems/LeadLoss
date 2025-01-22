from matplotlib.collections import LineCollection

from process import calculations
from utils import config
from utils.errorbarPlot import Errorbars
from view.axes.concordia.abstractConcordiaAxis import ConcordiaAxis
from model import monteCarloRunWetherill
from process import calculationsWetherill
import numpy as np

class SampleMonteCarloWetherillConcordiaAxis(ConcordiaAxis):

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

        self.axis.set_title("Monte Carlo Wetherill Concordia")
        self.axis.set_xlabel("207Pb/235U")
        self.axis.set_ylabel("206Pb/238U")

    ######################
    ## Internal actions ##
    ######################

    def _plotConcordiaCurve(self):

        minAge = calculationsWetherill.LOWER_AGE
        maxAge = calculationsWetherill.UPPER_AGE
        ages = np.linspace(minAge, maxAge, 200)

        xvals = [calculationsWetherill.pb207u235_from_age(t) for t in ages]
        yvals = [calculationsWetherill.pb206u238_from_age(t) for t in ages]

        self.axis.plot(xvals, yvals, color='blue')

    def plotMonteCarloRun(self, monteCarloRun):

        self._plotConcordiaCurve()
        self.concordantData.set_xdata(monteCarloRun.concordant_207_235)
        self.concordantData.set_ydata(monteCarloRun.concordant_206_238)
        self.discordantData.set_xdata(monteCarloRun.discordant_207_235)
        self.discordantData.set_ydata(monteCarloRun.discordant_206_238)
        self.leadLossAge.set_xdata([monteCarloRun.optimal_207_235])
        self.leadLossAge.set_ydata([monteCarloRun.optimal_206_238])

        max_x = max(
            max(monteCarloRun.concordant_207_235) if len(monteCarloRun.concordant_207_235)>0 else 0,
            max(monteCarloRun.discordant_207_235) if len(monteCarloRun.discordant_207_235)>0 else 0,
            monteCarloRun.optimal_207_235
        )
        self.axis.set_xlim(0, 1.2 * max_x)

    def plotSelectedAge(self, selectedAge, reconstructedAges):

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

    def clearSelectedAge(self):
        if self.reconstructedLines:
            self.reconstructedLines.remove()
            self.reconstructedLines = None

        self.selectedAge.set_xdata([])
        self.selectedAge.set_ydata([])


    #############
    ## Actions ##
    #############

    def clearInputData(self):
        self.clearSelectedAge()