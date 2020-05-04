import numpy as np


class Errorbars:

    def __init__(self, errorbarContainer):
        self.line = errorbarContainer.lines[0]
        self.barsx, self.barsy = errorbarContainer.lines[2]
        #self.vertices = errorbarContainer.lines[2]

    def set_data(self, xs, ys, xErrors, yErrors):
        xs = np.array(xs)
        ys = np.array(ys)
        xErrors = np.array(xErrors)
        yErrors = np.array(yErrors)

        self.line.set_xdata(xs)
        self.line.set_ydata(ys)

        xerr_top = xs + xErrors
        xerr_bot = xs - xErrors
        yerr_top = ys + yErrors
        yerr_bot = ys - yErrors

        new_segments_x = [np.array([[xt, y], [xb, y]]) for xt, xb, y in zip(xerr_top, xerr_bot, ys)]
        new_segments_y = [np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(xs, yerr_top, yerr_bot)]
        self.barsx.set_segments(new_segments_x)
        self.barsy.set_segments(new_segments_y)

    def clear_data(self):
        self.set_data([], [], [], [])

