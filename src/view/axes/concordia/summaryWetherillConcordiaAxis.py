import math
import numpy as np

from process import calculations
from utils import config
from view.axes.concordia.abstractWetherillConcordiaAxis import WetherillConcordiaAxis
from utils.errorEllipsePlot import ErrorEllipses


class SummaryWetherillConcordiaAxis(WetherillConcordiaAxis):
    """
    Summary concordia in Wetherill space:
      x = 207Pb/235U
      y = 206Pb/238U

    Mirrors SummaryConcordiaAxis (TW) structure:
      - per-sample SamplePlot storing Errorbars buckets
      - legend entries from example plot
      - selectSamples() clears unselected samples
    """

    def __init__(self, axis, samples):
        super().__init__(axis)

        self.selectedSamples = samples
        self.unselectedSamples = []
        self.samples = {}

        for sample in samples:
            self.plotSample(sample)

        if not samples:
            return
        examplePlot = self.samples[samples[0]]


        legendEntries = [
            (examplePlot.unclassified.line, "Unclassified"),
            (examplePlot.concordant.line,   "Concordant"),
            (examplePlot.discordant.line,   "Discordant"),
            (examplePlot.reverse.line,      "Reverse discordant"),
            (examplePlot.pbLossAge,         "Pb-loss age"),
        ]
        self.axis.legend(*zip(*legendEntries), frameon=False)

    def plotSample(self, sample):
        self.samples[sample] = SamplePlot(self.axis, sample)

    def refreshSample(self, sample):
        if sample in self.selectedSamples:
            self.samples[sample].plotInputData(sample)

    def selectSamples(self, selectedSamples, unselectedSamples):
        self.selectedSamples = selectedSamples
        self.unselectedSamples = unselectedSamples

        for sample in selectedSamples:
            self.samples[sample].plotInputData(sample)
        for sample in unselectedSamples:
            self.samples[sample].clearData()


class SamplePlot:
    def __init__(self, axis, sample):
        self.axis = axis
        self.sample = sample

        reverse_col = getattr(config, "REVERSE_DISCORDANT_COLOUR_1", getattr(config, "DISCORDANT_COLOUR_1", "orange"))

        self.unclassified = ErrorEllipses(axis, color=config.UNCLASSIFIED_COLOUR_1, zorder=2, lw=1.0, alpha=0.20, marker="o", markersize=2)
        self.concordant   = ErrorEllipses(axis, color=config.CONCORDANT_COLOUR_1,   zorder=3, lw=1.0, alpha=0.20, marker="o", markersize=2)
        self.discordant   = ErrorEllipses(axis, color=config.DISCORDANT_COLOUR_1,   zorder=3, lw=1.0, alpha=0.20, marker="o", markersize=2)
        self.reverse      = ErrorEllipses(axis, color=reverse_col,                  zorder=4, lw=1.0, alpha=0.20, marker="o", markersize=2)

                
        self.pbLossAge   = self.axis.plot([], [], marker='o', color=config.OPTIMAL_COLOUR_1)[0]
        self.pbLossRange = self.axis.plot([], [], color=config.OPTIMAL_COLOUR_1)[0]

        self.plotInputData(sample)

    def plotInputData(self, sample):
        rs = math.sqrt(calculations.mahalanobisRadius(2))

        concordantData   = []
        discordantData   = []
        reverseData      = []
        unclassifiedData = []

        upper_x = 0.0
        upper_y = 0.0

        for spot in sample.validSpots:
            x = getattr(spot, "pb207U235Value", None)
            y = getattr(spot, "pb206U238Value", None)
            sx = getattr(spot, "pb207U235StDev", None)
            sy = getattr(spot, "pb206U238StDev", None)

            if x is None or y is None:
                continue

            semi_x = (sx or 0.0) * rs
            semi_y = (sy or 0.0) * rs

            upper_x = max(upper_x, x + semi_x)
            upper_y = max(upper_y, y + semi_y)

            # Same bucket logic as TW
            if not spot.processed:
                bucket = unclassifiedData
            elif spot.concordant:
                bucket = concordantData
            elif getattr(spot, "reverseDiscordant", False):
                bucket = reverseData
            else:
                bucket = discordantData

            bucket.append((x, y, semi_x, semi_y))

        if concordantData:   self.concordant.set_data(*zip(*concordantData))
        else:                self.concordant.clear_data()
        if discordantData:   self.discordant.set_data(*zip(*discordantData))
        else:                self.discordant.clear_data()
        if reverseData:      self.reverse.set_data(*zip(*reverseData))
        else:                self.reverse.clear_data()
        if unclassifiedData: self.unclassified.set_data(*zip(*unclassifiedData))
        else:                self.unclassified.clear_data()

        # Pb-loss age range curve (parameterised by age)
        if getattr(sample, "optimalAge", None):
            t0 = sample.optimalAgeUpperBound
            t1 = sample.optimalAgeLowerBound
            if t0 is None or t1 is None:
                self.pbLossAge.set_xdata([]);   self.pbLossAge.set_ydata([])
                self.pbLossRange.set_xdata([]); self.pbLossRange.set_ydata([])
            else:
                t_min = min(t0, t1)
                t_max = max(t0, t1)
                ts = np.linspace(t_min, t_max, 200)

                xs = [calculations.pb207u235_from_age(t) for t in ts]
                ys = [calculations.pb206u238_from_age(t) for t in ts]

                self.pbLossAge.set_xdata([xs[0], xs[-1]])
                self.pbLossAge.set_ydata([ys[0], ys[-1]])
                self.pbLossRange.set_xdata(xs)
                self.pbLossRange.set_ydata(ys)

                upper_x = max(upper_x, max(xs) if xs else upper_x)
                upper_y = max(upper_y, max(ys) if ys else upper_y)
        else:
            self.pbLossAge.set_xdata([]);   self.pbLossAge.set_ydata([])
            self.pbLossRange.set_xdata([]); self.pbLossRange.set_ydata([])

        # Limits (basic but stable)
        self.axis.set_xlim(0, 1.2 * (upper_x or 1.0))
        self.axis.set_ylim(0, 1.2 * (upper_y or 1.0))

    def clearData(self):
        self.concordant.clear_data()
        self.discordant.clear_data()
        self.reverse.clear_data()
        self.unclassified.clear_data()
        self.pbLossAge.set_xdata([]);   self.pbLossAge.set_ydata([])
        self.pbLossRange.set_xdata([]); self.pbLossRange.set_ydata([])
