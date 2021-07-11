import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject

from controller.signals import ProcessingSignals
from process import processing
from utils import config
from utils.asynchronous import AsyncTask


class HeatmapAxis:

    def __init__(self, axis, canvas, figure):
        self.canvas = canvas
        self.figure = figure
        self.axis = axis
        self.colorbar = None

        self.processingSignals = ProcessingSignals()
        self.processingSignals.processingProgress.connect(self._plotRuns)

        self.clearAll()

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
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None

        self.axis.set_title("KS statistic")
        self.axis.set_xlabel("Age (Ma)")
        self.axis.set_ylabel("Score")
        self.axis.set_ylim(0.0, 1.0)

    def plotRuns(self, runs, settings):
        self.worker = AsyncTask(self.processingSignals, processing.calculateHeatmapData, runs, settings)
        self.worker.start()

    def _plotRuns(self, args):
        data, settings = args
        minAge = settings.minimumRimAge/10**6
        maxAge = settings.maximumRimAge/10**6
        resolution = config.HEATMAP_RESOLUTION

        X = np.linspace(minAge, maxAge, resolution)
        Y = np.linspace(0.0, 1.0, resolution)

        self.clearAll()
        self.axis.set_xlim(X[0], X[-1])
        colourmap = self.axis.pcolorfast(X, Y, data, cmap='viridis')
        self.colorbar = self.figure.colorbar(colourmap, ax=self.axis, label="Probability of score")
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