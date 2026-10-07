import numpy as np

from model.settings.ratio import ConcordiaRatioSpace, ratio_space_from_value
from process import calculations


class ConcordiaAxis:

    _default_xlim = (-1, 18)
    _default_ylim = (0, 0.6)

    def __init__(self, axis, ratio_space=ConcordiaRatioSpace.TERA_WASSERBURG):
        self.axis = axis
        self.samples = {}
        self.ratio_space = ratio_space_from_value(ratio_space)
        self._setupAxis()

    def _setupAxis(self):
        self.axis.cla()
        if self.ratio_space == ConcordiaRatioSpace.WETHERILL:
            self.axis.set_title("Wetherill concordia plot")
            self.axis.set_xlabel("${}^{207}Pb/{}^{235}U$")
            self.axis.set_ylabel("${}^{206}Pb/{}^{238}U$")
        else:
            self.axis.set_title("TW concordia plot")
            self.axis.set_xlabel("${}^{238}U/{}^{206}Pb$")
            self.axis.set_ylabel("${}^{207}Pb/{}^{206}Pb$")

        maxAge = calculations.UPPER_AGE // (10 ** 6)
        minAge = calculations.LOWER_AGE // (10 ** 6)

        # Plot concordia curve
        ages = np.linspace(minAge * (10 ** 6), maxAge * (10 ** 6), 600)
        coords = [calculations.concordia_xy(age, self.ratio_space) for age in ages]
        xs, ys = zip(*coords)
        self.axis.plot(xs, ys)

        # Plot concordia times
        time = maxAge
        i = 0
        increments = [500, 100, 50, 10, 5, 1]
        ts2 = []
        while i < len(increments) and time >= minAge:
            increment = increments[i]
            while time > increment and time >= minAge:
                ts2.append(time)
                time -= increment
            i += 1
        if time == minAge:
            ts2.append(time)
        xy2 = [calculations.concordia_xy(t * (10 ** 6), self.ratio_space) for t in ts2]
        xs2, ys2 = zip(*xy2)
        self.axis.scatter(xs2, ys2, s=8)
        for i, txt in enumerate(ts2):
            self.axis.annotate(
                str(txt) + " ",
                (xs2[i], ys2[i]),
                horizontalalignment="right",
                verticalalignment="top",
                fontsize="small"
            )

        if self.ratio_space == ConcordiaRatioSpace.WETHERILL:
            self.axis.set_xlim(0, max(xs) * 1.05)
            self.axis.set_ylim(0, max(ys) * 1.05)
        else:
            self.axis.set_xlim(*self._default_xlim)
            self.axis.set_ylim(*self._default_ylim)

    def setRatioSpace(self, ratio_space):
        ratio_space = ratio_space_from_value(ratio_space)
        if ratio_space == self.ratio_space:
            return False
        self.ratio_space = ratio_space
        self._setupAxis()
        return True
