from matplotlib.collections import LineCollection

from process import calculations
from utils import config
from utils.errorbarPlot import Errorbars
from view.axes.concordia.abstractConcordiaAxis import ConcordiaAxis
import math
from model.settings.calculation import ConcordiaMode

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

    def _wetherill_to_tw(self, pb207u235, pb206u238):
        try:
            x = float(pb207u235)   # 207/235
            y = float(pb206u238)   # 206/238
            if not math.isfinite(x) or not math.isfinite(y) or y <= 0.0:
                return (math.nan, math.nan)

            U = float(calculations.U238U235_RATIO)
            if not math.isfinite(U) or U <= 0.0:
                return (math.nan, math.nan)

            u = 1.0 / y                 # 238/206
            v = x / (U * y)             # 207/206
            return (u, v)
        except Exception:
            return (math.nan, math.nan)

    def plotMonteCarloRun(self, monteCarloRun):
        mode = ConcordiaMode.coerce(getattr(monteCarloRun, "concordiaMode", ConcordiaMode.TW))

        if mode == ConcordiaMode.WETHERILL:
            # Convert Wetherill -> TW for plotting on TW axis
            cx, cy = [], []
            for x, y in zip(monteCarloRun.concordant_uPb, monteCarloRun.concordant_pbPb):
                u, v = self._wetherill_to_tw(x, y)
                cx.append(u); cy.append(v)

            dx, dy = [], []
            for x, y in zip(monteCarloRun.discordant_uPb, monteCarloRun.discordant_pbPb):
                u, v = self._wetherill_to_tw(x, y)
                dx.append(u); dy.append(v)

            ox, oy = self._wetherill_to_tw(monteCarloRun.optimal_uPb, monteCarloRun.optimal_pbPb)

        else:
            # Already TW
            cx = list(monteCarloRun.concordant_uPb)
            cy = list(monteCarloRun.concordant_pbPb)
            dx = list(monteCarloRun.discordant_uPb)
            dy = list(monteCarloRun.discordant_pbPb)
            ox, oy = float(monteCarloRun.optimal_uPb), float(monteCarloRun.optimal_pbPb)

        self.concordantData.set_xdata(cx)
        self.concordantData.set_ydata(cy)
        self.discordantData.set_xdata(dx)
        self.discordantData.set_ydata(dy)
        self.leadLossAge.set_xdata([ox])
        self.leadLossAge.set_ydata([oy])

        finite_u = [u for u in (cx + dx + [ox]) if isinstance(u, (int, float)) and math.isfinite(u)]
        if finite_u:
            self.axis.set_xlim(0, 1.2 * max(finite_u))


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
