import math

from process import calculations
from utils import config
from utils.errorbarPlot import Errorbars
from view.axes.concordia.abstractConcordiaAxis import ConcordiaAxis


class SummaryConcordiaAxis(ConcordiaAxis):

    def __init__(self, axis, samples):
        super().__init__(axis)

        for sample in samples:
            self.plotSample(sample)

    def plotSample(self, sample):
        self.samples[sample] = SamplePlot(self.axis, sample)




class SamplePlot:

    def __init__(self, axis, sample):
        self.axis = axis
        self.sample = sample

        self.unclassified = Errorbars(axis.errorbar([], [], xerr=[], yerr=[], fmt='+', linestyle='', color=config.UNCLASSIFIED_COLOUR_1))
        self.concordant = Errorbars(axis.errorbar([], [], xerr=[], yerr=[], fmt='+', linestyle='', color=config.CONCORDANT_COLOUR_1))
        self.discordant = Errorbars(axis.errorbar([], [], xerr=[], yerr=[], fmt='+', linestyle='', color=config.DISCORDANT_COLOUR_1))

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
            """
            ellipse = Ellipse(
                xy=(row.uPbValue(), row.pbPbValue()),
                width=semi_minor*2,
                height=semi_major*2,
                lw=2,
                edgecolor=rgba,
                facecolor=rgba,
                alpha=1,
                clip_box=self.axis.bbox
            )
            self.axis.add_artist(ellipse)
            self.errorEllipses.append(ellipse)
            """

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
