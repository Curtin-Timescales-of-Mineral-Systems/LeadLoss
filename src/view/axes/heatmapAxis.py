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
_BOUNDARY_STYLE = dict(
    marker="<", s=75, facecolors="none",
    edgecolors="orange", linewidths=1.6
)

class HeatmapAxis:
    """
    Renders the empirical per-run KS heat-map.
    Peak and boundary markers can be overlaid near y≈0 for alignment with the
    catalogue, but the top panel does not draw the ensemble summary curve.
    """

    def __init__(self, axis, canvas, figure):
        self.canvas = canvas
        self.figure = figure
        self.axis = axis
        self.colorbar = None
        # Display true KS/penalised dissimilarity space (D*):
        # lower values are better.
        self._display_goodness = False
        self._show_curve_overlay = False

        # cache for peaks to repaint after heatmap refresh
        self._peaks_ma = None
        self._boundary_rows = []
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
        boundary_rows = list(self._boundary_rows)
        self.axis.clear()
        if self.colorbar:
            try:
                if self.colorbar.ax is not None:
                    self.colorbar.remove()
            except Exception:
                pass
            finally:
                self.colorbar = None
        self.axis.set_title("KS run density")
        self.axis.set_xlabel("Age (Ma)")
        self.axis.set_ylabel("D* (lower is better)")
        self.axis.set_ylim(0.0, 1.0)
        self._curve_line = None
        if preserve_cache:
            self._curve_ages_ma = curve_ages
            self._curve_vals = curve_vals
            self._peaks_ma = peaks_ma
            self._boundary_rows = boundary_rows
        else:
            self._curve_ages_ma = None
            self._curve_vals = None
            self._peaks_ma = None
            self._boundary_rows = []

    def _draw_curve_overlay(self):
        """Draw cached ensemble curve on heatmap in matching coordinates."""
        if not self._show_curve_overlay:
            return
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

    def _draw_boundary_markers(self):
        rows = list(self._boundary_rows or [])
        if not rows:
            return
        xs = []
        ys = []
        for row in rows:
            try:
                xs.append(float(row.get("age_ma", np.nan)))
                ys.append(0.02)
            except Exception:
                continue
        if xs:
            self.axis.scatter(xs, ys, zorder=32, **_BOUNDARY_STYLE)

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
        self._draw_boundary_markers()


        # if we already know peaks, paint them near the baseline
        if isinstance(self._peaks_ma, (list, tuple, np.ndarray)) and len(self._peaks_ma):
            self.axis.scatter(self._peaks_ma, [0.02]*len(self._peaks_ma), zorder=20, **_MARKER_STYLE)

        self.canvas.draw_idle()

    # --------- external hooks from the figure ---------

    def set_curve(self, ages_ma, S_view):
        """
        Cache the ensemble curve shown in the lower panel. The top panel is an
        empirical density view, so the cached curve is only drawn when the
        explicit overlay flag is enabled.
        """
        try:
            self._curve_ages_ma = np.asarray(ages_ma, float)
            self._curve_vals = np.asarray(S_view, float)
            if self._show_curve_overlay:
                self._draw_curve_overlay()
                self.canvas.draw_idle()
        except Exception:
            pass

    def set_boundary_rows(self, rows):
        self._boundary_rows = [dict(r) for r in (rows or [])]
        try:
            self._draw_boundary_markers()
            self.canvas.draw_idle()
        except Exception:
            pass

    def plotMatrix(self, ages_ma, S_runs):
        ages_ma = np.asarray(ages_ma, float)
        S_runs = np.asarray(S_runs, float)
        if ages_ma.ndim != 1 or S_runs.ndim != 2 or S_runs.shape[1] != ages_ma.size or ages_ma.size == 0:
            return

        resolution = config.HEATMAP_RESOLUTION
        D_runs = 1.0 - S_runs
        D_runs = np.clip(D_runs, 0.0, 1.0)
        y_edges = np.linspace(0.0, 1.0, resolution + 1)
        data = np.zeros((resolution, ages_ma.size), float)
        prev_hist = None
        for col in range(ages_ma.size):
            vals = np.asarray(D_runs[:, col], float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                if prev_hist is None:
                    hist = np.zeros(resolution, float)
                    hist[resolution // 2] = 1.0
                else:
                    hist = prev_hist.copy()
            else:
                hist, _ = np.histogram(vals, bins=y_edges)
                hist = hist.astype(float)
                total = float(np.sum(hist))
                if total > 0.0:
                    hist /= total
                elif prev_hist is not None:
                    hist = prev_hist.copy()
                else:
                    hist[resolution // 2] = 1.0
            prev_hist = hist
            data[:, col] = hist

        if ages_ma.size == 1:
            step = 1.0
        else:
            step = float(np.median(np.diff(ages_ma)))
            if not np.isfinite(step) or step <= 0.0:
                step = 1.0
        x_edges = np.empty(ages_ma.size + 1, float)
        x_edges[1:-1] = 0.5 * (ages_ma[:-1] + ages_ma[1:])
        x_edges[0] = ages_ma[0] - 0.5 * step
        x_edges[-1] = ages_ma[-1] + 0.5 * step

        self.clearAll(preserve_cache=True)
        self.axis.set_xlim(float(x_edges[0]), float(x_edges[-1]))
        self.axis.set_ylim(0.0, 1.0)
        self.axis.pcolormesh(x_edges, y_edges, data, cmap="viridis", shading="auto")
        self._draw_curve_overlay()
        self._draw_boundary_markers()
        if isinstance(self._peaks_ma, (list, tuple, np.ndarray)) and len(self._peaks_ma):
            self.axis.scatter(self._peaks_ma, [0.02] * len(self._peaks_ma), zorder=20, **_MARKER_STYLE)
        self.canvas.draw_idle()

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
