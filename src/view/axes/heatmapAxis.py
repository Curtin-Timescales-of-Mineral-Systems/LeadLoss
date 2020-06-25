import numpy as np
import scipy as sp
from PyQt5.QtCore import pyqtSignal, QObject

from controller.signals import ProcessingSignals
from process import processing
from utils.async import AsyncTask


class HeatmapAxis:

    RESOLUTION = 100

    def __init__(self, axis, canvas):
        self.canvas = canvas
        self.axis = axis
        self.clearAll()
        self.processingSignals = ProcessingSignals()
        self.processingSignals.processingProgress.connect(self._plotRuns)

    ##############
    ## X limits ##
    ##############

    def setXLimits(self, xmin, xmax):
        self.axis.set_xlim(xmin, xmax)

    ####################
    ## Statistic data ##
    ####################

    def clearAll(self):
        self.axis.clear()
        self.axis.set_title("KS statistic")
        self.axis.set_xlabel("Age (Ma)")
        self.axis.set_ylabel("D value")
        self.axis.set_ylim(0.0, 1.0)

    def plotRuns(self, runs, settings):
        self.worker = AsyncTask(self.processingSignals, processing.calculateHeatmapData, self.RESOLUTION, runs, settings)
        self.worker.start()

    def _plotRuns(self, args):
        data, settings = args
        minAge = settings.minimumRimAge
        maxAge = settings.maximumRimAge

        X = np.linspace(minAge/10**6, maxAge/10**6, self.RESOLUTION)
        Y = np.linspace(0.0, 1.0, self.RESOLUTION)

        self.axis.set_xlim(X[0], X[-1])
        self.axis.pcolor(X, Y, data, cmap='viridis')
        self.canvas.draw()

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

class HeatmapSignals(QObject):
    dataCalculated = pyqtSignal(object)