import math

from process import calculations
from utils import config
from view.axes.concordia.abstractWetherillConcordiaAxis import WetherillConcordiaAxis


class SampleMonteCarloWetherillConcordiaAxis(WetherillConcordiaAxis):

    def __init__(self, axis):
        super().__init__(axis)

        self.concordantData = self.axis.plot([], [], marker='x', linewidth=0, color=config.CONCORDANT_COLOUR_1)[0]
        self.discordantData = self.axis.plot([], [], marker='x', linewidth=0, color=config.DISCORDANT_COLOUR_1)[0]
        self.leadLossAge    = self.axis.plot([], [], marker='o', linewidth=0, color=config.OPTIMAL_COLOUR_1)[0]

        # These exist in the TW version; keep them for parity even if you don't currently use them.
        self.optimalAge = self.axis.plot([], [], marker='o', color=config.PREDICTION_COLOUR_1)[0]
        self.selectedAge = self.axis.plot([], [], marker='o', color=config.PREDICTION_COLOUR_1)[0]
        self.reconstructedLines = None

    def _tw_to_wetherill(self, u238pb206, pb207pb206):
        """
        TW coords:
          u = 238U/206Pb
          v = 207Pb/206Pb

        Wetherill coords:
          x = 207Pb/235U = v * (238/235) / u  = v * U238U235 / u
          y = 206Pb/238U = 1/u
        """
        try:
            u = float(u238pb206)
            v = float(pb207pb206)
            if not math.isfinite(u) or not math.isfinite(v) or u == 0.0:
                return (math.nan, math.nan)

            U = calculations.U238U235_RATIO
            y = 1.0 / u
            x = v * U / u
            return (x, y)
        except Exception:
            return (math.nan, math.nan)

    def plotMonteCarloRun(self, monteCarloRun):
        # Convert concordant points
        cx, cy = [], []
        for u, v in zip(monteCarloRun.concordant_uPb, monteCarloRun.concordant_pbPb):
            x, y = self._tw_to_wetherill(u, v)
            cx.append(x); cy.append(y)

        # Convert discordant points
        dx, dy = [], []
        for u, v in zip(monteCarloRun.discordant_uPb, monteCarloRun.discordant_pbPb):
            x, y = self._tw_to_wetherill(u, v)
            dx.append(x); dy.append(y)

        # Convert optimal Pb-loss age point
        ox, oy = self._tw_to_wetherill(monteCarloRun.optimal_uPb, monteCarloRun.optimal_pbPb)

        self.concordantData.set_xdata(cx)
        self.concordantData.set_ydata(cy)

        self.discordantData.set_xdata(dx)
        self.discordantData.set_ydata(dy)

        self.leadLossAge.set_xdata([ox])
        self.leadLossAge.set_ydata([oy])

        # Set x-limits similarly to TW (but in Wetherill x-space)
        finite_x = [x for x in (cx + dx + [ox]) if isinstance(x, (int, float)) and math.isfinite(x)]
        if finite_x:
            upper_xlim = max(finite_x)
            self.axis.set_xlim(0, 1.2 * upper_xlim)

    def clearSelectedAge(self):
        self.concordantData.set_xdata([])
        self.concordantData.set_ydata([])
        self.discordantData.set_xdata([])
        self.discordantData.set_ydata([])
        self.leadLossAge.set_xdata([])
        self.leadLossAge.set_ydata([])

    def clearInputData(self):
        self.clearSelectedAge()
