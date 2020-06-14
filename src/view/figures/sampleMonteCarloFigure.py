from view.axes.concordia.sampleMonteCarloConcordiaAxis import SampleMonteCarloConcordiaAxis
from view.axes.histogramAxis import HistogramAxis
from view.axes.statisticAxis import StatisticAxis
from view.figures.abstractFigure import AbstractFigure


class SampleMonteCarloFigure(AbstractFigure):

    def __init__(self, sample):
        super().__init__()
        self.sample = sample

        self.concordiaPlot = SampleMonteCarloConcordiaAxis(self.fig.add_subplot(211))
        self.statisticPlot = StatisticAxis(self.fig.add_subplot(223))
        self.histogramPlot = HistogramAxis(self.fig.add_subplot(224))

        self.fig.subplots_adjust(hspace=0.7, wspace=0.4)

        self.fig.canvas.mpl_connect('motion_notify_event', self.onHover)
        self.fig.canvas.mpl_connect('axes_enter_event', self.onMouseEnterAxes)
        self.fig.canvas.mpl_connect('axes_leave_event', self.onMouseExitAxes)

        self.processingComplete = True
        """
        signals = controller.signals
        signals.inputDataLoaded.connect(self.onInputDataLoaded)
        signals.inputDataCleared.connect(self.onInputDataCleared)
        signals.processingCleared.connect(self.onProcessingCleared)
        signals.processingStarted.connect(self.onProcessingStarted)
        signals.allStatisticsUpdated.connect(self.onNewStatistics)
        signals.optimalAgeFound.connect(self.onOptimalAgeFound)
        signals.ageSelected.connect(self.onAgeSelected)
        signals.ageDeselected.connect(self.onAgeDeselected)
        """
        self.mouseOnStatisticsAxes = False

    #############
    ## Actions ##
    #############

    def selectRun(self, run):

        statistics = [(age, d) for age ,(d, p) in run.statistics_by_pb_loss_age.items()]
        self.statisticPlot.plotOptimalAge(run.optimal_pb_loss_age)
        self.statisticPlot.plotStatisticData(*zip(*statistics))
        self.concordiaPlot.plotMonteCarloRun(run)

        #self.concordiaPlot.plotOptimalAge(run.optimal_pb_loss_age, reconstructedAges)

        reconstructedAges = run.reconstructed_ages_by_pb_loss_age[run.optimal_pb_loss_age]
        reconstructedAges = [age if age else 0 for age in reconstructedAges]

        self.histogramPlot.plotDistributions(run.concordant_ages, reconstructedAges)
        #self.histogramPlot.setReconstructedAgeRange(reconstructedAgeRange)
        self.canvas.draw()

    def _clearAgeSelected(self):
        self.statisticPlot.clearSelectedAge()
        self.histogramPlot.clearReconstructedDistribution()
        self.concordiaPlot.clearSelectedAge()

    ############
    ## Events ##
    ############

    def onProcessingStarted(self):
        calculationSettings = self.sample.calculationSettings
        xmin = calculationSettings.minimumRimAge / (10 ** 6)
        xmax = calculationSettings.maximumRimAge / (10 ** 6)
        buffer = (xmax - xmin) * 0.05
        self.statisticPlot.setXLimits(xmin - buffer, xmax + buffer)

    def onProcessingCleared(self):
        self.clearProcessingResults()
        self.canvas.draw()

    def onNewStatistics(self, statisticsByAge):
        self.statisticPlot.plotStatisticData(statisticsByAge)
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
            #self.controller.selectAgeToCompare(None)

    def onHover(self, event):
        if not self.processingComplete:
            return

        if not self.mouseOnStatisticsAxes:
            return

        x, y = self.statisticPlot.axis.transData.inverted().transform([(event.x, event.y)]).ravel()
        chosenAge = x * (10 ** 6)

        #self.controller.selectAgeToCompare(chosenAge)
