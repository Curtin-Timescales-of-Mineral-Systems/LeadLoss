import math

import numpy as np

from process import calculations
from utils import config
from utils.errorbarPlot import Errorbars
from view.axes.concordia.abstractConcordiaAxis import ConcordiaAxis


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
            (examplePlot.concordant.line, "Concordant"),
            (examplePlot.discordant.line, "Discordant"),
            (examplePlot.pbLossAge, "Pb-loss age"),
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

        self.unclassified = Errorbars(axis.errorbar([], [], xerr=[], yerr=[], fmt='+', linestyle='', color=config.UNCLASSIFIED_COLOUR_1))
        self.concordant = Errorbars(axis.errorbar([], [], xerr=[], yerr=[], fmt='+', linestyle='', color=config.CONCORDANT_COLOUR_1))
        self.discordant = Errorbars(axis.errorbar([], [], xerr=[], yerr=[], fmt='+', linestyle='', color=config.DISCORDANT_COLOUR_1))
        self.pbLossAge = self.axis.plot([], [], marker='o', color=config.OPTIMAL_COLOUR_1)[0]
        self.pbLossRange = self.axis.plot([], [], color=config.OPTIMAL_COLOUR_1)[0]

        self.plotInputData(sample)

    def plotInputData(self, sample):
        rs = math.sqrt(calculations.mahalanobisRadius(2))

        concordantData = []
        discordantData = []
        unclassifiedData = []

        for spot in sample.validSpots:
            semi_minor = spot.uPbStDev * rs
            semi_major = spot.pbPbStDev * rs

            if not spot.processed:
                data = unclassifiedData
            elif spot.concordant:
                data = concordantData
            else:
                data = discordantData

            data.append((spot.uPbValue, spot.pbPbValue, semi_minor, semi_major))

        if concordantData:
            self.concordant.set_data(*zip(*concordantData))
        else:
            self.concordant.clear_data()

        if discordantData:
            self.discordant.set_data(*zip(*discordantData))
        else:
            self.discordant.clear_data()

        if unclassifiedData:
            self.unclassified.set_data(*zip(*unclassifiedData))
        else:
            self.unclassified.clear_data()

        if sample.optimalAge:
            xMin = calculations.u238pb206_from_age(sample.optimalAgeUpperBound)
            xMax = calculations.u238pb206_from_age(sample.optimalAgeLowerBound)
            if xMin == xMax:
                xs = [xMin]
            else:
                xs = np.arange(xMin, xMax, 0.1)
            ys = [calculations.pb207pb206_from_u238pb206(x) for x in xs]

            self.pbLossAge.set_xdata([xs[0], xs[-1]])
            self.pbLossAge.set_ydata([ys[0], ys[-1]])
            self.pbLossRange.set_xdata(xs)
            self.pbLossRange.set_ydata(ys)
        else:
            self.pbLossAge.set_xdata([])
            self.pbLossAge.set_ydata([])
            self.pbLossRange.set_xdata([])
            self.pbLossRange.set_ydata([])

    def clearData(self):
        self.concordant.clear_data()
        self.discordant.clear_data()
        self.unclassified.clear_data()
        self.pbLossAge.set_xdata([])
        self.pbLossAge.set_ydata([])
        self.pbLossRange.set_xdata([])
        self.pbLossRange.set_ydata([])