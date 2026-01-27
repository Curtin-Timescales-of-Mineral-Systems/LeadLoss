import math

import numpy as np

from process import calculations
from utils import config
from utils.errorbarPlot import Errorbars
from view.axes.concordia.abstractConcordiaAxis import ConcordiaAxis
from utils.errorEllipsePlot import ErrorEllipses

class SummaryConcordiaAxis(ConcordiaAxis):

    def __init__(self, axis, samples):
        super().__init__(axis)

        self.selectedSamples = samples
        self.unselectedSamples = []

        for sample in samples:
            self.plotSample(sample)

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

        self.unclassified = ErrorEllipses(axis, color=config.UNCLASSIFIED_COLOUR_1, zorder=2, alpha=0.2, lw=0.8)
        self.concordant   = ErrorEllipses(axis, color=config.CONCORDANT_COLOUR_1,   zorder=3, alpha=0.2, lw=0.8)
        self.discordant   = ErrorEllipses(axis, color=config.DISCORDANT_COLOUR_1,   zorder=3, alpha=0.2, lw=0.8)
        self.reverse      = ErrorEllipses(axis, color=config.REVERSE_DISCORDANT_COLOUR_1, zorder=4, alpha=0.2, lw=0.8)


        self.pbLossAge   = self.axis.plot([], [], marker='o', color=config.OPTIMAL_COLOUR_1)[0]
        self.pbLossRange = self.axis.plot([], [], color=config.OPTIMAL_COLOUR_1)[0]

        self.plotInputData(sample)

    def plotInputData(self, sample):
        rs = math.sqrt(calculations.mahalanobisRadius(2))

        concordantData   = []
        discordantData   = []
        reverseData      = []
        unclassifiedData = []

        upper_xlim = 0.0

        for spot in sample.validSpots:
            semi_minor = (spot.uPbStDev or 0.0) * rs
            semi_major = (spot.pbPbStDev or 0.0) * rs
            if spot.uPbValue is not None:
                upper_xlim = max(upper_xlim, spot.uPbValue + semi_minor)

            # bucket selection order matters
            if not spot.processed:
                bucket = unclassifiedData
            elif spot.concordant:
                bucket = concordantData           # honour user’s threshold FIRST
            elif getattr(spot, "reverseDiscordant", False):
                bucket = reverseData              # reverse among discordant only
            else:
                bucket = discordantData


            bucket.append((spot.uPbValue, spot.pbPbValue, semi_minor, semi_major))

        if concordantData:   self.concordant.set_data(*zip(*concordantData))
        else:                self.concordant.clear_data()
        if discordantData:   self.discordant.set_data(*zip(*discordantData))
        else:                self.discordant.clear_data()
        if reverseData:      self.reverse.set_data(*zip(*reverseData))
        else:                self.reverse.clear_data()
        if unclassifiedData: self.unclassified.set_data(*zip(*unclassifiedData))
        else:                self.unclassified.clear_data()

        xlim_max = max(1.0, 1.2 * (upper_xlim or 1.0))
        self.axis.set_xlim(0.0, xlim_max)

        if sample.optimalAge:
            t0 = sample.optimalAgeLowerBound
            t1 = sample.optimalAgeUpperBound

            if t0 is None or t1 is None:
                self.pbLossAge.set_xdata([]);   self.pbLossAge.set_ydata([])
                self.pbLossRange.set_xdata([]); self.pbLossRange.set_ydata([])
            else:
                t_min = min(t0, t1)
                t_max = max(t0, t1)

                # hard floor avoids near-zero ages blowing up u238/pb206
                try:
                    t_min = max(t_min, float(getattr(calculations, "LOWER_AGE", 1_000_000.0)))
                except Exception:
                    pass

                ts = np.linspace(t_min, t_max, 200)
                xs = np.asarray([calculations.u238pb206_from_age(t) for t in ts], float)
                ys = np.asarray([calculations.pb207pb206_from_age(t) for t in ts], float)

                m = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0.0) & (xs <= xlim_max)
                xs = xs[m]; ys = ys[m]

                if xs.size:
                    self.pbLossRange.set_xdata(xs)
                    self.pbLossRange.set_ydata(ys)

                    # endpoints marker/segment (kept, but now clipped)
                    self.pbLossAge.set_xdata([xs[0], xs[-1]])
                    self.pbLossAge.set_ydata([ys[0], ys[-1]])
                else:
                    # nothing visible within the plot window
                    self.pbLossAge.set_xdata([]);   self.pbLossAge.set_ydata([])
                    self.pbLossRange.set_xdata([]); self.pbLossRange.set_ydata([])
        else:
            self.pbLossAge.set_xdata([]);   self.pbLossAge.set_ydata([])
            self.pbLossRange.set_xdata([]); self.pbLossRange.set_ydata([])

    def clearData(self):
        self.concordant.clear_data()
        self.discordant.clear_data()
        self.reverse.clear_data()
        self.unclassified.clear_data()
        self.pbLossAge.set_xdata([]);   self.pbLossAge.set_ydata([])
        self.pbLossRange.set_xdata([]); self.pbLossRange.set_ydata([])