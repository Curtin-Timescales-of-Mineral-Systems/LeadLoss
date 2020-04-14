import math

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *

import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse

from model.settings.calculation import DiscordanceClassificationMethod
from model.settings.type import SettingsType
from utils import config
from utils.settings import Settings

matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import process.calculations as calculations


class LeadLossGraphPanel(QGroupBox):
    _default_xlim = (-1, 18)
    _default_ylim = (0, 0.6)

    _age_xlim = (0, 5000)
    _statistic_ymax = 1.1

    _barResolution = 100
    _barMax = 5000
    _barMin = 0
    _bars = int((_barMax - _barMin) / _barResolution)

    _age_color=(1,0,0,1)

    def __init__(self, controller, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QVBoxLayout()
        layout.addWidget(self.createGraph())
        layout.addWidget(self._createCitation())
        self.setLayout(layout)

        self.controller = controller

        controller.signals.allRowsUpdated.connect(self.plotDataOnConcordiaAxis)
        controller.signals.rimAgeSelected.connect(self.onRimAgeSelected)
        controller.signals.statisticsUpdated.connect(self.onNewStatistics)

        self.mouseOnStatisticsAxes = False

    def _setupConcordiaPlot(self, axis):
        axis.set_xlabel("${}^{238}U/{}^{206}Pb$")
        axis.set_ylabel("${}^{207}Pb/{}^{206}Pb$")

        maxAge = 4500
        minAge = 200
        xMin = calculations.u238pb206_from_age(maxAge * (10 ** 6))
        xMax = calculations.u238pb206_from_age(minAge * (10 ** 6))

        # Plot concordia curve
        xs = np.arange(xMin, xMax, 0.1)
        ys = [calculations.pb207pb206_from_u238pb206(x) for x in xs]
        axis.plot(xs, ys)

        # Plot concordia times
        ts2 = list(range(100, minAge, 100)) + list(range(500, maxAge + 1, 500))
        xs2 = [calculations.u238pb206_from_age(t * (10 ** 6)) for t in ts2]
        ys2 = [calculations.pb207pb206_from_age(t * (10 ** 6)) for t in ts2]
        axis.scatter(xs2, ys2)
        for i, txt in enumerate(ts2):
            axis.annotate(str(txt) + " ", (xs2[i], ys2[i]), horizontalalignment="right", verticalalignment="top",
                         fontsize="small")

    def _createCitation(self):
        label = QLabel(self._getCitationText())
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return label

    def _getCitationText(self):
        return "Hugo K.H. Olierook, Christopher L. Kirkland, ???, " \
               "Matthew L. Daggitt, ??? " \
               "<b>PAPER TITLE</b>, 2020"


    def createGraph(self):
        widget = QWidget()
        fig = plt.figure()
        self.concordiaAxis = plt.subplot(211)
        self.statisticAxis = plt.subplot(223)
        self.histogramAxis = plt.subplot(224)

        plt.subplots_adjust(hspace = 0.7, wspace=0.4)

        # plot
        self.canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        layout.addWidget(toolbar)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()
        widget.setLayout(layout)

        self.cid = fig.canvas.mpl_connect('motion_notify_event', self.onHover)
        self.cid2 = fig.canvas.mpl_connect('axes_enter_event', self.onMouseEnterAxes)
        self.cid3 = fig.canvas.mpl_connect('axes_leave_event', self.onMouseExitAxes)

        self.plotConcordia()
        self.plotHistogram([], [], None)
        self.plotStatistics({})
        return widget

    def plotConcordia(self):
        self.concordiaAxis.set_xlim(*self._default_xlim)
        self.concordiaAxis.set_ylim(*self._default_ylim)
        self._setupConcordiaPlot(self.concordiaAxis)
        self.concordiaAxis.set_title("TW concordia plot")

        self.concordiaAxis.chosenAgePoint = self.concordiaAxis.plot([], [], marker='o', color=self._age_color)[0]
        self.concordiaAxis.unclassifiedDataPoints = self.concordiaAxis.plot([],[],label='toto',marker='o',ls='',color='k')[0]
        self.concordiaAxis.concordantDataPoints = self.concordiaAxis.plot([],[],label='toto',marker='o',ls='')[0]
        self.concordiaAxis.discordantDataPoints = self.concordiaAxis.plot([],[],label='toto',marker='o',ls='')[0]
        self.concordiaAxis.reconstructedLines = None

    def plotDataOnConcordiaAxis(self, rows):
        importSettings = Settings.get(SettingsType.IMPORT)
        calculationSettings = Settings.get(SettingsType.CALCULATION)

        if calculationSettings.discordanceClassificationMethod == DiscordanceClassificationMethod.PERCENTAGE:
            self.plotDataPointsOnConcordiaAxis(rows)
        else:
            self.plotDataEllipsesOnConcordiaAxis(rows, importSettings, calculationSettings)

    def plotDataPointsOnConcordiaAxis(self, rows):
        uxs = []
        uys = []
        cxs = []
        cys = []
        dxs = []
        dys = []
        for row in rows:
            if not row.processed:
                xs, ys = uxs, uys
            elif row.concordant:
                xs, ys = cxs, cys
            else:
                xs, ys = dxs, dys
            xs.append(row.uPbValue())
            ys.append(row.pbPbValue())

        self.concordiaAxis.unclassifiedDataPoints.set_data(uxs, uys)
        self.concordiaAxis.concordantDataPoints.set_data(cxs,cys)
        self.concordiaAxis.discordantDataPoints.set_data(dxs,dys)

    def plotDataEllipsesOnConcordiaAxis(self, rows, importSettings, calculationSettings):
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

            ellipse = Ellipse(
                xy=(row.uPbValue(), row.pbPbValue()),
                width=semi_minor*2,
                height=semi_major*2,
                lw=2,
                edgecolor="k",
                facecolor=rgba,
                alpha=1,
                clip_box=self.concordiaAxis.bbox
            )
            self.concordiaAxis.add_artist(ellipse)

    def plotReconstructedLinesOnConcordia(self, rimAge, concordantAges, discordantAges):
        if self.concordiaAxis.reconstructedLines is not None:
            self.concordiaAxis.reconstructedLines.remove()

        lines = []
        for c, d in zip(concordantAges, discordantAges):
            if d is None:
                line = []
            else:
                line = [
                    (calculations.u238pb206_from_age(rimAge), calculations.pb207pb206_from_age(rimAge)),
                    (calculations.u238pb206_from_age(d), calculations.pb207pb206_from_age(d))
                ]
            lines.append(line)

        self.concordiaAxis.reconstructedLines = LineCollection(
            lines,
            linewidths=2,
            colors=self._age_color
        )
        self.concordiaAxis.add_collection(self.concordiaAxis.reconstructedLines)

    def _scale(self, values):
        scaledValues = [v/(10**6) for v in values]
        weights = [1/len(values)]*len(values) if values else []
        return scaledValues, weights

    def plotHistogram(self, concordantValues, discordantValues, calculationSettings):
        self.histogramAxis.clear()
        self.histogramAxis.set_title("CDF")
        self.histogramAxis.set_xlabel("Age (Ma)")
        self.histogramAxis.set_ylim(0, 1.0)
        if not concordantValues:
            self.histogramAxis.set_xlim(*self._age_xlim)
            return

        self.histogramAxis.hist(
            [v/(10**6) for v in concordantValues],
            bins=self._bars,
            cumulative=True,
            density=True,
            histtype='step',
            edgecolor=config.CONCORDANT_COLOUR_1,
            facecolor=(0, 0, 0, 0)
        )
        self.histogramAxis.hist(
            [v/(10**6) for v in discordantValues],
            bins=self._bars,
            cumulative=True,
            density=True,
            histtype='step',
            edgecolor=config.DISCORDANT_COLOUR_1,
            facecolor=(0, 0, 0, 0)
        )
        self.histogramAxis.set_xlim(*self.histogramAxis.calculatedXLims)

        self.canvas.draw()

    def plotStatistics(self, statistics):
        self.statisticAxis.clear()
        self.statisticAxis.set_title("KS statistic")
        self.statisticAxis.set_xlabel("Age (Ma)")
        self.statisticAxis.set_ylabel("p value")
        self.statisticAxis.chosenAgeLine = self.statisticAxis.plot([],[],color=self._age_color)[0]

        if not statistics:
            self.statisticAxis.set_xlim(*self._age_xlim)
            self.statisticAxis.set_ylim((0,self._statistic_ymax))
            return

        xs = [age/(10**6) for age in statistics.keys()]
        ys = list(statistics.values())
        self.statisticAxis.plot(xs, ys)
        self.canvas.draw()

    def plotAgeComparison(self, rimAge):
        if rimAge is None:
            return
        scaledAge = rimAge/(10**6)
        self.statisticAxis.chosenAgeLine.set_xdata([scaledAge, scaledAge])
        self.statisticAxis.chosenAgeLine.set_ydata([0, self._statistic_ymax])

        uPb = calculations.u238pb206_from_age(rimAge)
        pbPb = calculations.pb207pb206_from_age(rimAge)
        self.concordiaAxis.chosenAgePoint.set_xdata([uPb])
        self.concordiaAxis.chosenAgePoint.set_ydata([pbPb])
        self.canvas.draw()

    #######################
    ## Mouse interaction ##
    #######################

    def onRimAgeSelected(self, rimAge, rows, reconstructedAges):
        calculationSettings = Settings.get(SettingsType.CALCULATION)

        discordantAges = []
        for reconstructedAge in reconstructedAges:
            if reconstructedAge is not None:
                discordantAges.append(reconstructedAge.values[0])
        concordantAges = [row.concordantAge for row in rows if row.concordant]

        self.plotHistogram(concordantAges, discordantAges, calculationSettings)
        self.plotAgeComparison(rimAge)
        self.plotReconstructedLinesOnConcordia(rimAge, concordantAges, discordantAges)

    def onNewlyClassifiedPoints(self, rows):
        self.plotDataPointsOnConcordiaAxis(rows)

    def onNewStatistics(self, statisticsByAge, ageRange):
        self.plotStatistics(statisticsByAge)
        if ageRange:
            self.histogramAxis.calculatedXLims = [v/(10**6) for v in ageRange]

    def onAllRowsUpdated(self, rows):
        self.plotDataPointsOnConcordiaAxis(rows)

    def onMouseEnterAxes(self, event):
        self.mouseOnStatisticsAxes = event.inaxes == self.statisticAxis

    def onMouseExitAxes(self, event):
        if self.mouseOnStatisticsAxes:
            self.mouseOnStatisticsAxes = False
            self.controller.selectAgeToCompare(None)

    def onHover(self, event):
        if not self.mouseOnStatisticsAxes:
            return

        x, y = self.statisticAxis.transData.inverted().transform([(event.x, event.y)]).ravel()
        chosenAge = x * (10**6)
        self.controller.selectAgeToCompare(chosenAge)
