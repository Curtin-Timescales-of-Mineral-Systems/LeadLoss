import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject

from controller.signals import ProcessingSignals
from process import processing
from utils import config
from utils.asynchronous import AsyncTask

_MARKER_STYLE = dict(
    marker="v", s=60, facecolors="none",
    edgecolors="white", linewidths=1.5
)

class HeatmapAxis:
    """
    Renders the KS heat-map. Also supports overlaying peak markers at y≈0
    (for visual alignment with the goodness pane).
    """

    def __init__(self, axis, canvas, figure):
        self.canvas = canvas
        self.figure = figure
        self.axis = axis
        self.colorbar = None

        # cache for peaks/curve to repaint after heatmap refresh
        self._peaks_ma = None
        self._last_ages_ma = None
        self._last_S_view = None

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
        # clear and safely remove any previous colorbar
        self.axis.clear()
        if self.colorbar:
            try:
                if self.colorbar.ax is not None:
                    self.colorbar.remove()
            except Exception:
                pass
            finally:
                self.colorbar = None
        self.axis.set_title("KS statistic")
        self.axis.set_xlabel("Age (Ma)")
        self.axis.set_ylabel("Score")
        self.axis.set_ylim(0.0, 1.0)

    def plotRuns(self, runs, settings):
        self._worker = AsyncTask(self.processingSignals, processing.calculateHeatmapData, runs, settings)
        self._worker.start()

    def _plotRuns(self, args):
        if not (isinstance(args, tuple) and len(args) == 2):
            return
        data, settings = args
        minAge = settings.minimumRimAge/1e6
        maxAge = settings.maximumRimAge/1e6
        resolution = config.HEATMAP_RESOLUTION

        X = np.linspace(minAge, maxAge, resolution)
        Y = np.linspace(0.0, 1.0, resolution)

        self.clearAll()
        self.axis.set_xlim(X[0], X[-1])
        colourmap = self.axis.pcolorfast(X, Y, data, cmap='viridis')
        self.colorbar = self.figure.colorbar(colourmap, ax=self.axis, label="Probability of score")

        # if we already know peaks, paint them near the baseline
        if isinstance(self._peaks_ma, (list, tuple, np.ndarray)) and len(self._peaks_ma):
            self.axis.scatter(self._peaks_ma, [0.02]*len(self._peaks_ma), zorder=20, **_MARKER_STYLE)

        self.canvas.draw_idle()

    # --------- external hooks from the figure ---------

    def set_curve(self, ages_ma, S_view):
        """Optional cache of the 1-D goodness to re-draw overlays after heatmap refresh."""
        try:
            self._last_ages_ma = np.asarray(ages_ma, float)
            self._last_S_view  = np.asarray(S_view,  float)
        except Exception:
            self._last_ages_ma = None
            self._last_S_view  = None

    def set_peaks(self, peaks_ma):
        """Overlay triangle markers at the current heatmap baseline."""
        if peaks_ma is None:
            self._peaks_ma = None
            return
        self._peaks_ma = [float(p) for p in peaks_ma]
        # draw immediately if the heatmap is already on canvas
        try:
            self.axis.scatter(self._peaks_ma, [0.02]*len(self._peaks_ma), zorder=20, **_MARKER_STYLE)
            self.canvas.draw_idle()
        except Exception:
            pass

    # legacy stubs retained for API compatibility with callers that still set these lines
    def clearStatisticData(self):
        pass

    def plotOptimalAge(self, optimalAge):
        pass

    def clearOptimalAge(self):
        pass

    def plotSelectedAge(self, selectedAge):
        pass

    def clearSelectedAge(self):
        pass


class HeatmapSignals(QObject):
    dataCalculated = pyqtSignal(object)
