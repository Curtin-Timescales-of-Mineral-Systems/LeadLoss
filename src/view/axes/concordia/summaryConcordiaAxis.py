import math

import numpy as np

from model.settings.ratio import ConcordiaRatioSpace, ratio_space_from_value
from process import calculations
from utils import config
from utils.covarianceEllipsePlot import CovarianceEllipses
from view.axes.concordia.abstractConcordiaAxis import ConcordiaAxis


class SummaryConcordiaAxis(ConcordiaAxis):

    def __init__(self, axis, samples):
        self.allSamples = list(samples)
        self.selectedSamples = list(samples)
        self.unselectedSamples = []
        super().__init__(axis, self._targetRatioSpace())

        self._buildSamplePlots()

    def _targetRatioSpace(self):
        if self.selectedSamples:
            return self.selectedSamples[0].getDisplayRatioSpace()
        return ConcordiaRatioSpace.TERA_WASSERBURG

    def _buildSamplePlots(self):
        self.samples = {}
        for sample in self.allSamples:
            self.plotSample(sample)
        if self.allSamples:
            examplePlot = self.samples[self.allSamples[0]]
            legendEntries = [
                (examplePlot.unclassified.line, "Unclassified"),
                (examplePlot.concordant.line,   "Concordant"),
                (examplePlot.discordant.line,   "Discordant"),
                (examplePlot.reverse.line,      "Reverse discordant"),
                (examplePlot.pbLossAge,         "Pb-loss age"),
            ]
            self.axis.legend(*zip(*legendEntries), frameon=False)

    def _syncRatioSpace(self):
        if self.setRatioSpace(self._targetRatioSpace()):
            self._buildSamplePlots()
            return True
        return False


    def plotSample(self, sample):
        self.samples[sample] = SamplePlot(self.axis, sample, self.ratio_space)

    def refreshSample(self, sample):
        rebuilt = self._syncRatioSpace()
        if rebuilt:
            for unselectedSample in self.unselectedSamples:
                self.samples[unselectedSample].clearData()
        if sample in self.selectedSamples:
            self.samples[sample].plotInputData(sample)

    def selectSamples(self, selectedSamples, unselectedSamples):
        self.selectedSamples = list(selectedSamples)
        self.unselectedSamples = list(unselectedSamples)
        self._syncRatioSpace()
        
        for sample in selectedSamples:
            self.samples[sample].plotInputData(sample)
        for sample in unselectedSamples:
            self.samples[sample].clearData()

class SamplePlot:
    def __init__(self, axis, sample, ratio_space):
        self.axis = axis
        self.sample = sample
        self.ratio_space = ratio_space_from_value(ratio_space)

        self.unclassified = CovarianceEllipses(axis, config.UNCLASSIFIED_COLOUR_1, zorder=2)
        self.concordant   = CovarianceEllipses(axis, config.CONCORDANT_COLOUR_1, zorder=3)
        self.discordant   = CovarianceEllipses(axis, config.DISCORDANT_COLOUR_1, zorder=3)
        self.reverse      = CovarianceEllipses(axis, config.REVERSE_DISCORDANT_COLOUR_1, zorder=4)



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
        upper_ylim = 0.0

        for spot in sample.validSpots:
            x, y = spot.getRatioValues(self.ratio_space)
            sx, sy, rho = spot.getRatioStDevs(self.ratio_space)
            semi_minor = (sx or 0.0) * rs
            semi_major = (sy or 0.0) * rs
            if x is not None:
                upper_xlim = max(upper_xlim, x + semi_minor)
            if y is not None:
                upper_ylim = max(upper_ylim, y + semi_major)

            # bucket selection order matters
            if not spot.processed:
                bucket = unclassifiedData
            elif spot.concordant:
                bucket = concordantData           # honour user’s threshold FIRST
            elif getattr(spot, "reverseDiscordant", False):
                bucket = reverseData              # reverse among discordant only
            else:
                bucket = discordantData


            bucket.append((x, y, sx or 0.0, sy or 0.0, rho or 0.0))

        if concordantData:   self.concordant.set_data(*zip(*concordantData), confidence_radius=rs)
        else:                self.concordant.clear_data()
        if discordantData:   self.discordant.set_data(*zip(*discordantData), confidence_radius=rs)
        else:                self.discordant.clear_data()
        if reverseData:      self.reverse.set_data(*zip(*reverseData), confidence_radius=rs)
        else:                self.reverse.clear_data()
        if unclassifiedData: self.unclassified.set_data(*zip(*unclassifiedData), confidence_radius=rs)
        else:                self.unclassified.clear_data()

        if sample.optimalAge:
            ages = np.linspace(sample.optimalAgeLowerBound, sample.optimalAgeUpperBound, 100)
            xy = [calculations.concordia_xy(age, self.ratio_space) for age in ages]
            xs, ys = zip(*xy)
            upper_xlim = max(upper_xlim, max(xs))
            upper_ylim = max(upper_ylim, max(ys))
            self.pbLossAge.set_xdata([xs[0], xs[-1]])
            self.pbLossAge.set_ydata([ys[0], ys[-1]])
            self.pbLossRange.set_xdata(xs)
            self.pbLossRange.set_ydata(ys)
        else:
            self.pbLossAge.set_xdata([]);   self.pbLossAge.set_ydata([])
            self.pbLossRange.set_xdata([]); self.pbLossRange.set_ydata([])

        self.axis.set_xlim(0, 1.2 * (upper_xlim or 1.0))
        if self.ratio_space == ConcordiaRatioSpace.WETHERILL:
            self.axis.set_ylim(0, 1.2 * (upper_ylim or 1.0))

    def clearData(self):
        self.concordant.clear_data()
        self.discordant.clear_data()
        self.reverse.clear_data()
        self.unclassified.clear_data()
        self.pbLossAge.set_xdata([]);   self.pbLossAge.set_ydata([])
        self.pbLossRange.set_xdata([]); self.pbLossRange.set_ydata([])
