import math

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse

from model.settings.type import SettingsType
from process import calculations
from utils import config
from utils.errorbarPlot import Errorbars
from utils.settings import Settings


class ConcordiaPlot:

    _default_xlim = (-1, 18)
    _default_ylim = (0, 0.6)

    def __init__(self, axis):
        self.axis = axis
        self._setupAxis()

    def _setupAxis(self):
        self.axis.set_title("TW concordia plot")
        self.axis.set_xlabel("${}^{238}U/{}^{206}Pb$")
        self.axis.set_ylabel("${}^{207}Pb/{}^{206}Pb$")
        self.axis.set_xlim(*self._default_xlim)
        self.axis.set_ylim(*self._default_ylim)

        maxAge = 4500
        minAge = 200
        xMin = calculations.u238pb206_from_age(maxAge * (10 ** 6))
        xMax = calculations.u238pb206_from_age(minAge * (10 ** 6))

        # Plot concordia curve
        xs = np.arange(xMin, xMax, 0.1)
        ys = [calculations.pb207pb206_from_u238pb206(x) for x in xs]
        self.axis.plot(xs, ys)

        # Plot concordia times
        ts2 = list(range(100, minAge, 100)) + list(range(500, maxAge + 1, 500))
        xs2 = [calculations.u238pb206_from_age(t * (10 ** 6)) for t in ts2]
        ys2 = [calculations.pb207pb206_from_age(t * (10 ** 6)) for t in ts2]
        self.axis.scatter(xs2, ys2)
        for i, txt in enumerate(ts2):
            self.axis.annotate(
                str(txt) + " ",
                (xs2[i], ys2[i]),
                horizontalalignment="right",
                verticalalignment="top",
                fontsize="small"
            )

        self.optimalAge = self.axis.plot([], [], marker='o', color=config.PREDICTION_COLOUR_1)[0]
        self.selectedAge = self.axis.plot([], [], marker='o', color=config.PREDICTION_COLOUR_1)[0]
        self.unclassifiedDataPoints = Errorbars(self.axis.errorbar([],[],xerr=[], yerr=[], fmt='+', linestyle='',color=config.UNCLASSIFIED_COLOUR_1))
        self.concordantDataPoints = Errorbars(self.axis.errorbar([],[],xerr=[], yerr=[], fmt='+', linestyle='',color=config.CONCORDANT_COLOUR_1))
        self.discordantDataPoints = Errorbars(self.axis.errorbar([],[],xerr=[], yerr=[], fmt='+', linestyle='',color=config.DISCORDANT_COLOUR_1))
        self.reconstructedLines = None
        self.errorEllipses = []

    ################
    ## Input data ##
    ################

    def plotInputData(self, rows):
        importSettings = Settings.get(SettingsType.IMPORT)
        calculationSettings = Settings.get(SettingsType.CALCULATION)

        rs = math.sqrt(calculations.mahalanobisRadius(calculationSettings.discordanceEllipseSigmas))

        concordantData = []
        discordantData = []
        unclassifiedData = []

        for row in rows:
            semi_minor = row.uPbStDev(importSettings) * rs
            semi_major = row.pbPbStDev(importSettings) * rs

            if semi_minor == 0 or semi_major == 0:
                if not row.processed:
                    data = unclassifiedData
                elif row.concordant:
                    data = concordantData
                else:
                    data = discordantData
                data.append((row.uPbValue(), row.pbPbValue(), semi_minor, semi_major))
            else:
                if not row.processed:
                    rgba = config.UNCLASSIFIED_COLOUR_1
                elif row.concordant:
                    rgba = config.CONCORDANT_COLOUR_1
                else:
                    rgba = config.DISCORDANT_COLOUR_1

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

        if concordantData:
            self.concordantDataPoints.set_data(*zip(*concordantData))
        else:
            self.concordantDataPoints.clear_data()

        if discordantData:
            self.discordantDataPoints.set_data(*zip(*discordantData))
        else:
            self.discordantDataPoints.clear_data()

        if unclassifiedData:
            self.unclassifiedDataPoints.set_data(*zip(*unclassifiedData))
        else:
            self.unclassifiedDataPoints.clear_data()

    def clearInputData(self):
        self.clearSelectedAge()
        for ellipse in self.errorEllipses:
            ellipse.remove()
        self.errorEllipses = []

    ##################
    ## Selected age ##
    ##################

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

    def plotOptimalAge(self, optimalAge, discordantAges):
        pass

    def clearSelectedAge(self):
        self.selectedAge.set_xdata([])
        self.selectedAge.set_ydata([])
        if self.reconstructedLines is not None:
            self.reconstructedLines.remove()
            self.reconstructedLines = None