import numpy as np

class HeatmapAxis:

    def __init__(self, axis, calculationSettings):
        self.axis = axis
        self._setupAxis(calculationSettings)

        self.resolution = calculationSettings.rimAgesSampled

    def _setupAxis(self, settings):
        self.axis.clear()
        self.axis.set_title("KS statistic")
        self.axis.set_xlabel("Age (Ma)")
        self.axis.set_ylabel("D value")
        self.axis.set_ylim(0.0, 1.0)

    ##############
    ## X limits ##
    ##############

    def setXLimits(self, xmin, xmax):
        self.axis.set_xlim(xmin, xmax)

    ####################
    ## Statistic data ##
    ####################

    def plotRuns(self, runs):
        data = np.zeros((self.resolution, self.resolution))
        for run in runs:
            for col, age in enumerate(run.pb_loss_ages):
                value = run.statistics_by_pb_loss_age[age][0]
                row = int(value * (self.resolution-1))
                data[row][col] += 1/len(runs)

        X = [age/(10**6) for age in run.pb_loss_ages]
        Y = np.linspace(0.0, 1.0, self.resolution)

        self.axis.set_xlim(X[0], X[-1])
        self.axis.pcolor(X, Y, data, cmap='viridis')

    def clearStatisticData(self):
        self.statisticDataPoints.set_xdata([])
        self.statisticDataPoints.set_ydata([])

    #################
    ## Optimal age ##
    #################

    def plotOptimalAge(self, optimalAge):
        optimalAge = optimalAge / (10 ** 6)
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

        selectedAge = selectedAge / (10 ** 6)
        self.selectedAgeLine.set_xdata([selectedAge, selectedAge])
        self.selectedAgeLine.set_ydata([0, self._statistic_ymax])

    def clearSelectedAge(self):
        self.selectedAgeLine.set_xdata([])
        self.selectedAgeLine.set_ydata([])
