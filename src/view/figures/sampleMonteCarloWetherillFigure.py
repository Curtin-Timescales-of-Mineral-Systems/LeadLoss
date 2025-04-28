from view.figures.abstractFigure import AbstractFigure
from view.axes.concordia.sampleMonteCarloWetherillConcordiaAxis import SampleMonteCarloWetherillConcordiaAxis
from view.axes.histogramAxis import HistogramAxis
from view.axes.statisticAxis import StatisticAxis

class SampleMonteCarloWetherillFigure(AbstractFigure):
    
    def __init__(self, sample, parent):
        super().__init__()

        self.sample = sample
        self.parent = parent 

        self.concordiaPlot = SampleMonteCarloWetherillConcordiaAxis(self.fig.add_subplot(211))
        self.statisticPlot = StatisticAxis(self.fig.add_subplot(223))
        self.histogramPlot = HistogramAxis(self.fig.add_subplot(224))

        self.fig.subplots_adjust(hspace=0.7, wspace=0.4)

        self.fig.canvas.mpl_connect('motion_notify_event', self.onHover)
        self.fig.canvas.mpl_connect('axes_enter_event', self.onMouseEnterAxes)
        self.fig.canvas.mpl_connect('axes_leave_event', self.onMouseExitAxes)

        self.currentRun = None
        self.optimalReconstructedAges = None
        self.processingComplete = True
        self.mouseOnStatisticsAxes = False

    #############
    ## Actions ##
    #############

    def selectRun(self, run):
        self.currentRun = run
        self.concordiaPlot.plotMonteCarloRun(run)
        statistics = [(age, stat.score) for age, stat in run.statistics_by_pb_loss_age.items()]
        self.statisticPlot.plotOptimalAge(run.optimal_pb_loss_age)
        self.statisticPlot.plotStatisticData(*zip(*statistics))
        

        self.optimalReconstructedAges = run.optimal_statistic.valid_discordant_ages

        self.histogramPlot.plotDistributions(run.concordant_ages, self.optimalReconstructedAges, None)

        self.canvas.draw()

    def selectAge(self, age):
        if self.currentRun is None:
            return

        selectedAge = self.sample.calculationSettings.getNearestSampledAge(age)
        selectedDistribution = self.currentRun.statistics_by_pb_loss_age[selectedAge].valid_discordant_ages
        self.concordiaPlot.plotSelectedAge(age, self.optimalReconstructedAges)
        self.histogramPlot.plotDistributions(self.currentRun.concordant_ages, self.optimalReconstructedAges, selectedDistribution)
        self.statisticPlot.plotSelectedAge(selectedAge)
        self.canvas.draw()

    def deselectAge(self):
        self.statisticPlot.clearSelectedAge()
        self.concordiaPlot.clearSelectedAge()
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
            self.parent.deselectAge()

    def onHover(self, event):
        if not self.processingComplete:
            return

        if not self.mouseOnStatisticsAxes:
            return

        x, y = self.statisticPlot.axis.transData.inverted().transform([(event.x, event.y)]).ravel()
        chosenAge = x * (10 ** 6)

        self.parent.selectAge(chosenAge)
