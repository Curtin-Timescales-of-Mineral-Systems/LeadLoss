from matplotlib.collections import LineCollection
import numpy as np

from model.settings.ratio import ratio_space_from_value
from process import calculations
from utils import config
from view.axes.concordia.abstractConcordiaAxis import ConcordiaAxis


class SampleMonteCarloConcordiaAxis(ConcordiaAxis):

    def __init__(self, axis):
        super().__init__(axis)
        self._initArtists()

    def _initArtists(self):

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
        if self.setRatioSpace(ratio_space_from_value(getattr(monteCarloRun, "modelRatioSpace", self.ratio_space))):
            self._initArtists()
        self.concordantData.set_xdata(monteCarloRun.concordant_uPb)
        self.concordantData.set_ydata(monteCarloRun.concordant_pbPb)
        self.discordantData.set_xdata(monteCarloRun.discordant_uPb)
        self.discordantData.set_ydata(monteCarloRun.discordant_pbPb)
        self.leadLossAge.set_xdata([monteCarloRun.optimal_uPb])
        self.leadLossAge.set_ydata([monteCarloRun.optimal_pbPb])

        values_x = list(monteCarloRun.concordant_uPb) + list(monteCarloRun.discordant_uPb) + [monteCarloRun.optimal_uPb]
        finite_x = [float(v) for v in values_x if v is not None and np.isfinite(float(v))]
        if finite_x:
            self.axis.set_xlim(0, 1.2 * max(finite_x))
        values_y = list(monteCarloRun.concordant_pbPb) + list(monteCarloRun.discordant_pbPb) + [monteCarloRun.optimal_pbPb]
        finite_y = [float(v) for v in values_y if v is not None and np.isfinite(float(v))]
        if finite_y:
            self.axis.set_ylim(0, 1.2 * max(finite_y))

    def plotSelectedAge(self, selectedAge, reconstructedAges):
        self.clearSelectedAge()

        uPb, pbPb = calculations.concordia_xy(selectedAge, self.ratio_space)
        self.selectedAge.set_xdata([uPb])
        self.selectedAge.set_ydata([pbPb])

        lines = []
        for reconstructedAge in reconstructedAges:
            if reconstructedAge is None:
                line = []
            else:
                line = [
                    calculations.concordia_xy(selectedAge, self.ratio_space),
                    calculations.concordia_xy(reconstructedAge, self.ratio_space)
                ]
            lines.append(line)

        self.reconstructedLines = LineCollection(
            lines,
            linewidths=1,
            colors=config.PREDICTION_COLOUR_1
        )
        self.axis.add_collection(self.reconstructedLines)

    def clearSelectedAge(self):
        self.selectedAge.set_xdata([])
        self.selectedAge.set_ydata([])
        if self.reconstructedLines is not None:
            try:
                self.reconstructedLines.remove()
            except Exception:
                pass
            self.reconstructedLines = None

    def clearRunData(self):
        self.concordantData.set_xdata([])
        self.concordantData.set_ydata([])
        self.discordantData.set_xdata([])
        self.discordantData.set_ydata([])
        self.leadLossAge.set_xdata([])
        self.leadLossAge.set_ydata([])
        self.clearSelectedAge()


    #############
    ## Actions ##
    #############

    def clearInputData(self):
        self.clearRunData()
