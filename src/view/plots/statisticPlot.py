from model.settings.type import SettingsType
from utils import config
from utils.settings import Settings


class StatisticPlot:

    _statistic_ymax = 1.1
    _age_xlim = (0, 5000)

    def __init__(self, axis):
        self.axis = axis
        self._setupAxis()

    def _setupAxis(self):
        self.axis.clear()
        self.axis.set_title("KS statistic")
        self.axis.set_xlabel("Age (Ma)")
        self.axis.set_ylabel("D value")

        self.optimalAgeLine = self.axis.plot([], [], color=config.OPTIMAL_COLOUR_1)[0]
        self.selectedAgeLine = self.axis.plot([], [], color=config.PREDICTION_COLOUR_1)[0]
        self.statisticDataPoints = self.axis.plot([], [])[0]

        self.axis.set_xlim(*self._age_xlim)
        self.axis.set_ylim((0,self._statistic_ymax))

    ##############
    ## X limits ##
    ##############

    def setXLimits(self, xmin, xmax):
        self.axis.set_xlim(xmin, xmax)

    ####################
    ## Statistic data ##
    ####################

    def plotStatisticData(self, statistics):
        xs = [age/(10**6) for age in statistics.keys()]
        ys = list(statistics.values())
        self.statisticDataPoints.set_xdata(xs)
        self.statisticDataPoints.set_ydata(ys)

    def clearStatisticData(self):
        self.statisticDataPoints.set_xdata([])
        self.statisticDataPoints.set_ydata([])

    #################
    ## Optimal age ##
    #################

    def plotOptimalAge(self, optimalAge):
        optimalAge = optimalAge/(10**6)
        self.optimalAgeLine.set_xdata([optimalAge, optimalAge])
        self.optimalAgeLine.set_ydata([0, self._statistic_ymax])

    def clearOptimalAge(self):
        self.optimalAgeLine.set_xdata([])
        self.optimalAgeLine.set_ydata([])

    ##################
    ## Selected age ##
    ##################

    def plotSelectedAge(self, selectedAge):
        self.clearSelectedAge()

        selectedAge = selectedAge/(10**6)
        self.selectedAgeLine.set_xdata([selectedAge, selectedAge])
        self.selectedAgeLine.set_ydata([0, self._statistic_ymax])

    def clearSelectedAge(self):
        self.selectedAgeLine.set_xdata([])
        self.selectedAgeLine.set_ydata([])