# utils/errorEllipsePlot.py
import numpy as np
from matplotlib.collections import EllipseCollection
from matplotlib.colors import to_rgba

class ErrorEllipses:
    def __init__(self, axis, color="k", zorder=3, lw=0.8, alpha=0.20,
                 marker="o", markersize=2, edge_alpha=1.0):
        self.axis = axis

        self.line = axis.plot(
            [], [], linestyle="",
            marker=marker, markersize=markersize,
            color=color, zorder=zorder + 1
        )[0]

        face = to_rgba(color, alpha)
        edge = to_rgba(color, edge_alpha)

        self._ell = EllipseCollection(
            widths=np.array([0.0]),
            heights=np.array([0.0]),
            angles=np.array([0.0]),
            units="xy",
            offsets=np.array([[0.0, 0.0]]),
            transOffset=axis.transData,
            facecolors=[face],
            edgecolors=[edge],
            linewidths=lw,
            zorder=zorder,
        )
        self._ell.set_visible(False)
        axis.add_collection(self._ell)

    def set_data(self, xs, ys, xErrors, yErrors):
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        xErrors = np.asarray(xErrors, float)
        yErrors = np.asarray(yErrors, float)

        # Basic sanity filter
        m = (
            np.isfinite(xs) & np.isfinite(ys) &
            np.isfinite(xErrors) & np.isfinite(yErrors)
        )
        xs, ys, xErrors, yErrors = xs[m], ys[m], xErrors[m], yErrors[m]

        self.line.set_xdata(xs)
        self.line.set_ydata(ys)

        if xs.size == 0:
            self._ell.set_visible(False)

            self._ell.set_offsets(np.array([[0.0, 0.0]]))
            self._ell._widths  = np.array([0.0])
            self._ell._heights = np.array([0.0])
            self._ell._angles  = np.array([0.0])
            self._ell.stale = True
            return

        offsets = np.column_stack([xs, ys])

        widths  = 2.0 * np.maximum(xErrors, 0.0)
        heights = 2.0 * np.maximum(yErrors, 0.0)
        angles  = np.zeros_like(widths)

        self._ell.set_visible(True)
        self._ell.set_offsets(offsets)

        self._ell._widths  = widths
        self._ell._heights = heights
        self._ell._angles  = angles
        self._ell.stale = True

    def clear_data(self):
        self.set_data([], [], [], [])
