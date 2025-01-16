import numpy as np

from process import calculations


class TeraWasserburgConcordiaAxis:

    _default_xlim = (-1, 18)
    _default_ylim = (0, 0.6)

    def __init__(self, axis):
        self.axis = axis
        self.samples = {}
        self._setupTWAxis()

    def _setupTWAxis(self):
        self.axis.set_title("TW concordia plot")
        self.axis.set_xlabel("${}^{238}U/{}^{206}Pb$")
        self.axis.set_ylabel("${}^{207}Pb/{}^{206}Pb$")
        self.axis.set_xlim(*self._default_xlim)
        self.axis.set_ylim(*self._default_ylim)

        maxAge = calculations.UPPER_AGE // (10 ** 6)
        minAge = calculations.LOWER_AGE // (10 ** 6)
        xMin = calculations.u238pb206_from_age(maxAge * (10 ** 6))
        xMax = calculations.u238pb206_from_age(minAge * (10 ** 6))

        # Plot concordia curve
        xs = np.arange(xMin, xMax, 0.1)
        ys = [calculations.pb207pb206_from_u238pb206(x) for x in xs]
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
        xs2 = [calculations.u238pb206_from_age(t * (10 ** 6)) for t in ts2]
        ys2 = [calculations.pb207pb206_from_age(t * (10 ** 6)) for t in ts2]
        self.axis.scatter(xs2, ys2, s=8)
        for i, txt in enumerate(ts2):
            self.axis.annotate(
                str(txt) + " ",
                (xs2[i], ys2[i]),
                horizontalalignment="right",
                verticalalignment="top",
                fontsize="small"
            )

