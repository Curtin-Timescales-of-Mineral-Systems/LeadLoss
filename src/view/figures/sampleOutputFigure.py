import time

from model.settings.type import SettingsType
from utils.settings import Settings
from view.axes.concordia.sampleMonteCarloConcordiaAxis import SampleMonteCarloConcordiaAxis
from view.axes.heatmapAxis import HeatmapAxis
from view.axes.histogramAxis import HistogramAxis
from view.axes.statisticAxis import StatisticAxis
from view.figures.abstractFigure import AbstractFigure

class SampleOutputFigure(AbstractFigure):

    def __init__(self, controller, sample):
        super().__init__()

        self.sample = sample
        self.lastDrawTime = None

        self.heatmapAxis = HeatmapAxis(self.fig.add_subplot(111), self.canvas, self.fig)

        self.fig.set_constrained_layout(True)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onHover)
        self.fig.canvas.mpl_connect('axes_enter_event', self.onMouseEnterAxes)
        self.fig.canvas.mpl_connect('axes_leave_event', self.onMouseExitAxes)

        sample.signals.concordancyCalculated.connect(self._onSampleConcordancyCalculated)
        sample.signals.monteCarloRunAdded.connect(self._onMonteCarloRunAdded)
        sample.signals.processingCleared.connect(self._onProcessingCleared)
        """
        #signals.processingCleared.connect(self.onProcessingCleared)

        signals.allStatisticsUpdated.connect(self.onNewStatistics)
        signals.optimalAgeFound.connect(self.onOptimalAgeFound)

        signals.ageSelected.connect(self.onAgeSelected)
        signals.ageDeselected.connect(self.onAgeDeselected)
        """

        self.processingComplete = False
        self.mouseOnStatisticsAxes = False

    def clearProcessingResults(self):
        self.processingComplete = False
        self.lastDrawTime = None
        self.heatmapAxis.clearAll()

    def clearInputData(self):
        self.clearProcessingResults()
        self.concordiaPlot.clearInputData()

    ########################
    ## Processing events ##
    ########################

    def onProcessingStarted(self):
        calculationSettings = Settings.get(SettingsType.CALCULATION)
        xmin = calculationSettings.minimumRimAge / (10 ** 6)
        xmax = calculationSettings.maximumRimAge / (10 ** 6)
        buffer = (xmax - xmin) * 0.05
        self.statisticPlot.setXLimits(xmin - buffer, xmax + buffer)

    def onProcessingCleared(self):
        self.clearProcessingResults()
        self.canvas.draw()

    ############################
    ## Processing data events ##
    ############################

    def _onSampleConcordancyCalculated(self):
        pass
        """
        self.concordiaPlot.plotSample(sample)
        concordantAges = [row.concordantAge for row in rows if row.concordant]
        self.histogramPlot.plotConcordantDistribution(concordantAges)
        self.canvas.draw()
        """

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
        chosenAge = x * (10 ** 6)

        self.controller.selectAgeToCompare(chosenAge)

    def _onProcessingCleared(self):
        self.clearProcessingResults()
        self.canvas.draw()

    def _onMonteCarloRunAdded(self):
        settings = Settings.get(SettingsType.CALCULATION)
        currentTime = time.time()

        if len(self.sample.monteCarloRuns) == settings.monteCarloRuns or\
            self.lastDrawTime is None or\
            currentTime - self.lastDrawTime >= 2:

            self.lastDrawTime = currentTime
            self.heatmapAxis.plotRuns(self.sample.monteCarloRuns, settings)
            self.canvas.draw()