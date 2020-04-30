from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *

import matplotlib

from model.settings.type import SettingsType
from utils.settings import Settings
from view.plots.concordiaPlot import ConcordiaPlot
from view.plots.histogramPlot import HistogramPlot
from view.plots.statisticPlot import StatisticPlot

matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class LeadLossGraphPanel(QGroupBox):

    def __init__(self, controller, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.processingComplete = False
        self.mouseOnStatisticsAxes = False

        graphWidget = self.createGraph()
        citationWidget = self._createCitation()
        graphWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout()
        layout.addWidget(graphWidget)
        layout.addWidget(citationWidget)
        self.setLayout(layout)

        self.controller = controller
        signals = controller.signals

        signals.inputDataLoaded.connect(self.onInputDataLoaded)
        signals.inputDataCleared.connect(self.onInputDataCleared)

        signals.processingCleared.connect(self.onProcessingCleared)
        signals.processingStarted.connect(self.onProcessingStarted)

        signals.concordancyClassification.connect(self.onConcordancyClassification)
        signals.allStatisticsUpdated.connect(self.onNewStatistics)
        signals.optimalAgeFound.connect(self.onOptimalAgeFound)

        signals.ageSelected.connect(self.onAgeSelected)
        signals.ageDeselected.connect(self.onAgeDeselected)

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
        fig = plt.figure()

        self.concordiaPlot = ConcordiaPlot(plt.subplot(211))
        self.statisticPlot = StatisticPlot(plt.subplot(223))
        self.histogramPlot = HistogramPlot(plt.subplot(224))

        plt.subplots_adjust(hspace = 0.7, wspace=0.4)

        self.canvas = FigureCanvas(fig)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()
        toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        layout.addWidget(toolbar)

        fig.canvas.mpl_connect('motion_notify_event', self.onHover)
        fig.canvas.mpl_connect('axes_enter_event', self.onMouseEnterAxes)
        fig.canvas.mpl_connect('axes_leave_event', self.onMouseExitAxes)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def clearAgeSelected(self):
        self.statisticPlot.clearSelectedAge()
        self.histogramPlot.clearReconstructedDistribution()
        self.concordiaPlot.clearSelectedAge()

    def clearProcessingResults(self):
        self.processingComplete = False

        self.clearAgeSelected()
        self.statisticPlot.clearStatisticData()
        self.statisticPlot.clearOptimalAge()
        self.histogramPlot.clearConcordantDistribution()

    def clearInputData(self):
        self.clearProcessingResults()
        self.concordiaPlot.clearInputData()

    #######################
    ## Input data events ##
    #######################

    def onInputDataLoaded(self, inputFile, headers, rows):
        self.concordiaPlot.plotInputData(rows)
        self.canvas.draw()

    def onInputDataCleared(self):
        self.clearInputData()
        self.canvas.draw()

    ########################
    ## Processing events ##
    ########################

    def onProcessingStarted(self):
        calculationSettings = Settings.get(SettingsType.CALCULATION)
        xmin = calculationSettings.minimumRimAge/(10**6)
        xmax = calculationSettings.maximumRimAge/(10**6)
        buffer = (xmax - xmin)*0.05
        self.statisticPlot.setXLimits(xmin-buffer, xmax+buffer)

    def onProcessingCleared(self):
        self.clearProcessingResults()
        self.canvas.draw()

    ############################
    ## Processing data events ##
    ############################

    def onConcordancyClassification(self, rows):
        self.concordiaPlot.plotInputData(rows)
        concordantAges = [row.concordantAge for row in rows if row.concordant]
        self.histogramPlot.plotConcordantDistribution(concordantAges)
        self.canvas.draw()

    def onNewStatistics(self, statisticsByAge):
        self.statisticPlot.plotStatisticData(statisticsByAge)
        self.canvas.draw()

    def onOptimalAgeFound(self, optimalRimAge, pValue, dValue, reconstructedAgeObjects, reconstructedAgeRange):
        reconstructedAges = [age.values[0] for age in reconstructedAgeObjects if age is not None]

        self.processingComplete = True
        self.statisticPlot.plotOptimalAge(optimalRimAge)
        self.histogramPlot.plotOptimalReconstructedDistribution(reconstructedAges)
        self.concordiaPlot.plotOptimalAge(optimalRimAge, reconstructedAges)
        self.histogramPlot.setReconstructedAgeRange(reconstructedAgeRange)
        self.canvas.draw()

    ###################
    ## Age selection ##
    ###################

    def onAgeSelected(self, selectedAge, reconstructedAgeObjects):
        reconstructedAges = [age.values[0] for age in reconstructedAgeObjects if age is not None]

        self.histogramPlot.plotReconstructedDistribution(reconstructedAges)
        self.concordiaPlot.plotSelectedAge(selectedAge, reconstructedAges)
        self.statisticPlot.plotSelectedAge(selectedAge)
        self.canvas.draw()

    def onAgeDeselected(self):
        self.clearAgeSelected()
        self.canvas.draw()

    #######################
    ## Mouse interaction ##
    #######################

    def onMouseEnterAxes(self, event):
        if not self.processingComplete:
            return

        self.mouseOnStatisticsAxes = event.inaxes == self.statisticPlot.axis

    def onMouseExitAxes(self, event):
        if not self.processingComplete:
            return

        if self.mouseOnStatisticsAxes:
            self.mouseOnStatisticsAxes = False
            self.controller.selectAgeToCompare(None)

    def onHover(self, event):
        if not self.processingComplete:
            return

        if not self.mouseOnStatisticsAxes:
            return

        x, y = self.statisticPlot.axis.transData.inverted().transform([(event.x, event.y)]).ravel()
        chosenAge = x * (10**6)

        self.controller.selectAgeToCompare(chosenAge)
