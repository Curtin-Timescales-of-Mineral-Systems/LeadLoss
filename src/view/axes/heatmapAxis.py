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
        # Display true KS/penalised dissimilarity space (D*):
        # lower values are better.
        self._display_goodness = False

        # cache for peaks to repaint after heatmap refresh
        self._peaks_ma = None
        # cache for the ensemble curve overlay to keep top/bottom in sync
        self._curve_ages_ma = None
        self._curve_vals = None
        self._curve_line = None

        self.processingSignals = ProcessingSignals()
        self.processingSignals.processingProgress.connect(self._plotRuns)
        self._plot_seq = 0
        self._worker = None

        self.clearAll()

    ##############
    ## X limits ##
    ##############

    def setXLimits(self, xmin, xmax):
        self.axis.set_xlim(xmin, xmax)

    ####################
    ## Statistic data ##
    ####################

    def clearAll(self, preserve_cache: bool = False):
        # clear and safely remove any previous colorbar
        curve_ages = self._curve_ages_ma
        curve_vals = self._curve_vals
        peaks_ma = self._peaks_ma
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
        self.axis.set_ylabel("D* (lower is better)")
        self.axis.set_ylim(0.0, 1.0)
        self._curve_line = None
        if preserve_cache:
            self._curve_ages_ma = curve_ages
            self._curve_vals = curve_vals
            self._peaks_ma = peaks_ma
        else:
            self._curve_ages_ma = None
            self._curve_vals = None
            self._peaks_ma = None

    def _draw_curve_overlay(self):
        """Draw cached ensemble curve on heatmap in matching coordinates."""
        if self._curve_ages_ma is None or self._curve_vals is None:
            return

        x = np.asarray(self._curve_ages_ma, float)
        v = np.asarray(self._curve_vals, float)
        m = np.isfinite(x) & np.isfinite(v)
        if not m.any():
            return

        # Incoming curve is S (goodness); heatmap y-axis is D* unless toggled.
        y = v[m] if self._display_goodness else (1.0 - v[m])
        y = np.clip(y, 0.0, 1.0)

        if self._curve_line is None:
            (self._curve_line,) = self.axis.plot(
                x[m], y, ls="--", lw=1.6, color="white", alpha=0.95, zorder=30
            )
        else:
            self._curve_line.set_data(x[m], y)

    def plotRuns(self, runs, settings):
        # Cancel any in-flight heatmap worker before starting a new one.
        if self._worker is not None:
            try:
                if self._worker.isRunning():
                    self._worker.halt()
            except Exception:
                pass
        self._plot_seq += 1
        seq = int(self._plot_seq)
        self._worker = AsyncTask(
            self.processingSignals,
            processing.calculateHeatmapData,
            runs,
            settings,
            seq,
        )
        self._worker.start()

    def _plotRuns(self, args):
        if not isinstance(args, tuple):
            return

        # New payload shape with request sequence:
        #   (seq, data, settings)
        # Legacy payload shape:
        #   (data, settings)
        if len(args) == 3:
            seq, data, settings = args
            try:
                if int(seq) != int(self._plot_seq):
                    return
            except Exception:
                return
        elif len(args) == 2:
            data, settings = args
        else:
            return

        minAge = settings.minimumRimAge/1e6
        maxAge = settings.maximumRimAge/1e6
        resolution = config.HEATMAP_RESOLUTION

        self.clearAll(preserve_cache=True)
        self.axis.set_xlim(minAge, maxAge)
        self.axis.set_ylim(0.0, 1.0)

        # Use extents (xmin,xmax), (ymin,ymax) to avoid edge-length mismatch across platforms.
        data_arr = np.asarray(data, float)
        if data_arr.ndim != 2:
            return
        # Internal heatmap bins are D* (dissimilarity).
        if self._display_goodness:
            data_arr = np.flipud(data_arr)
        colourmap = self.axis.pcolorfast((minAge, maxAge), (0.0, 1.0), data_arr, cmap='viridis')
        self._draw_curve_overlay()


        # if we already know peaks, paint them near the baseline
        if isinstance(self._peaks_ma, (list, tuple, np.ndarray)) and len(self._peaks_ma):
            self.axis.scatter(self._peaks_ma, [0.02]*len(self._peaks_ma), zorder=20, **_MARKER_STYLE)

        self.canvas.draw_idle()

    # --------- external hooks from the figure ---------

    def set_curve(self, ages_ma, S_view):
        """
        Cache and overlay the same ensemble curve shown in the lower panel.
        This keeps heatmap and goodness views visually synchronized.
        """
        try:
            self._curve_ages_ma = np.asarray(ages_ma, float)
            self._curve_vals = np.asarray(S_view, float)
            self._draw_curve_overlay()
            self.canvas.draw_idle()
        except Exception:
            pass

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
