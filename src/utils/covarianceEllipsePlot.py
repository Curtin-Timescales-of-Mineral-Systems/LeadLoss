"""Small Matplotlib helper for plotting correlated analytical uncertainty."""

from __future__ import annotations

import math

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Ellipse


def ellipse_geometry(x_stdev, y_stdev, rho, confidence_radius):
    """Return full width, height and angle for a covariance ellipse."""
    sx = max(float(x_stdev or 0.0), 0.0)
    sy = max(float(y_stdev or 0.0), 0.0)
    rho = max(-0.999999, min(0.999999, float(rho or 0.0)))
    covariance = np.array(
        [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]],
        dtype=float,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    major = float(confidence_radius) * math.sqrt(float(eigenvalues[0]))
    minor = float(confidence_radius) * math.sqrt(float(eigenvalues[1]))
    angle = math.degrees(math.atan2(float(eigenvectors[1, 0]), float(eigenvectors[0, 0])))
    return 2.0 * major, 2.0 * minor, angle


class CovarianceEllipses:
    """Point markers plus uncertainty ellipses that can be refreshed in place."""

    def __init__(self, axis, color, zorder=2):
        self.axis = axis
        self.color = color
        self.zorder = zorder
        self.line = axis.plot(
            [], [], marker="o", markersize=2.8, linestyle="", color=color, zorder=zorder + 0.2
        )[0]
        self.collection = None

    def set_data(self, xs, ys, x_stdevs, y_stdevs, rhos, confidence_radius):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        x_stdevs = np.asarray(x_stdevs, dtype=float)
        y_stdevs = np.asarray(y_stdevs, dtype=float)
        rhos = np.asarray(rhos, dtype=float)

        self.line.set_xdata(xs)
        self.line.set_ydata(ys)
        if self.collection is not None:
            self.collection.remove()
            self.collection = None

        patches = []
        for x, y, sx, sy, rho in zip(xs, ys, x_stdevs, y_stdevs, rhos):
            if not all(np.isfinite(v) for v in (x, y, sx, sy, rho)):
                continue
            width, height, angle = ellipse_geometry(sx, sy, rho, confidence_radius)
            if width <= 0.0 and height <= 0.0:
                continue
            patches.append(Ellipse((x, y), width=width, height=height, angle=angle))

        if patches:
            edge = to_rgba(self.color, 0.82)
            face = to_rgba(self.color, 0.10)
            self.collection = PatchCollection(
                patches,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.75,
                zorder=self.zorder,
            )
            self.axis.add_collection(self.collection)

    def clear_data(self):
        self.line.set_xdata([])
        self.line.set_ydata([])
        if self.collection is not None:
            self.collection.remove()
            self.collection = None
