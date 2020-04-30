import math

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse

from model.settings.type import SettingsType
from process import calculations
from utils import config
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
        self.unclassifiedDataPoints = self.axis.plot([],[],label='toto',marker='o',ls='',color='k')[0]
        self.concordantDataPoints = self.axis.plot([],[],label='toto',marker='o',ls='',color=config.CONCORDANT_COLOUR_1)[0]
        self.discordantDataPoints = self.axis.plot([],[],label='toto',marker='o',ls='',color=config.DISCORDANT_COLOUR_1)[0]
        self.reconstructedLines = None

    ################
    ## Input data ##
    ################

    def plotInputData(self, rows):
        importSettings = Settings.get(SettingsType.IMPORT)
        calculationSettings = Settings.get(SettingsType.CALCULATION)

        rs = math.sqrt(calculations.mahalanobisRadius(calculationSettings.discordanceEllipseSigmas))

        for row in rows:
            semi_minor = row.uPbStDev(importSettings) * rs
            semi_major = row.pbPbStDev(importSettings) * rs

            if not row.processed:
                rgba = config.UNCLASSIFIED_COLOUR_1
            elif row.concordant:
                rgba = config.CONCORDANT_COLOUR_1
            else:
                rgba = config.DISCORDANT_COLOUR_1

            if semi_minor == 0 or semi_major == 0:
                self.axis.errorbar(x=row.uPbValue(), y=row.pbPbValue(), xerr=semi_minor, yerr=semi_major, fmt='+', linestyle='', color=rgba)
            else:
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

    def clearInputData(self):
        pass

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