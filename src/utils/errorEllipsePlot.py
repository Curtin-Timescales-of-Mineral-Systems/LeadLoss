import numpy as np

try:
    from matplotlib.collections import EllipseCollection
    _HAS_ELLIPSE_COLLECTION = True
except Exception:
    EllipseCollection = None
    _HAS_ELLIPSE_COLLECTION = False


class ErrorEllipses:
    def __init__(self, axis, color, zorder=3, lw=1.0, alpha=0.20, marker="o", markersize=2):
        self.axis = axis

        self.line = axis.plot(
            [], [], linestyle="",
            marker=marker, markersize=markersize,
            color=color, zorder=zorder + 1
        )[0]

        self._ell = EllipseCollection(
            widths=np.array([], float),
            heights=np.array([], float),
            angles=np.array([], float),
            units="xy",
            offsets=np.empty((0, 2), float),
            transOffset=axis.transData,
            facecolors=color,
            edgecolors=color,
            linewidths=lw,
            alpha=alpha,
            zorder=zorder,
            clip_on=True,
        )

        self._ell.set_visible(False)   # ← IMPORTANT
        axis.add_collection(self._ell)


    def set_data(self, xs, ys, xErrors, yErrors):
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        xErrors = np.asarray(xErrors, float)
        yErrors = np.asarray(yErrors, float)

        m = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(xErrors) & np.isfinite(yErrors)
        xs, ys, xErrors, yErrors = xs[m], ys[m], xErrors[m], yErrors[m]

        self.line.set_xdata(xs)
        self.line.set_ydata(ys)

        if xs.size == 0:
            self._ell.set_visible(False)
            return

        offsets = np.column_stack([xs, ys])
        widths  = 2.0 * np.maximum(xErrors, 0.0)
        heights = 2.0 * np.maximum(yErrors, 0.0)
        angles  = np.zeros_like(widths)

        self._ell.set_offsets(offsets)

        # Matplotlib 3.7 compatibility
        self._ell._widths  = widths
        self._ell._heights = heights
        self._ell._angles  = angles

        self._ell.set_visible(True)
        self._ell.stale = True


    def clear_data(self):
        self.line.set_xdata([])
        self.line.set_ydata([])
        self._ell.set_visible(False)
