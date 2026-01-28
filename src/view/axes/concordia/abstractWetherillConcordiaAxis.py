import numpy as np
import math
from utils import resourceUtils

from process import calculations


class WetherillConcordiaAxis:
    """
    Wetherill concordia coordinates:
      x = 207Pb/235U
      y = 206Pb/238U
    """

    _default_xlim = (-1, 18)
    _default_ylim = (0, 0.6)

    def __init__(self, axis):
        self.axis = axis
        self.samples = {}
        self._setupAxis()

    def _setupAxis(self):
        self.axis.figure.set_facecolor("none")
        self.axis.figure.patch.set_alpha(0)     # transparent figure background
        self.axis.set_facecolor("white")   
        
        self.axis.set_title("Wetherill concordia plot")
        self.axis.set_xlabel("${}^{207}Pb/{}^{235}U$")
        self.axis.set_ylabel("${}^{206}Pb/{}^{238}U$")

        self.axis.set_xlim(*self._default_xlim)
        self.axis.set_ylim(*self._default_ylim)

        self._plot_concordia_curve()
        self._plot_concordia_times()

    def _max_age_ma_for_default_view(self) -> int:
        minAgeMa = int(calculations.LOWER_AGE // 1e6)

        x_right = float(self._default_xlim[1])
        y_top   = float(self._default_ylim[1])

        tmax_x = calculations.UPPER_AGE
        tmax_y = calculations.UPPER_AGE

        try:
            if math.isfinite(x_right) and x_right > 0.0:
                tmax_x = float(calculations.age_from_pb207u235(x_right))
        except Exception:
            pass

        try:
            if math.isfinite(y_top) and y_top > 0.0:
                tmax_y = float(calculations.age_from_pb206u238(y_top))
        except Exception:
            pass

        tmax = min(float(calculations.UPPER_AGE), tmax_x, tmax_y)
        if not math.isfinite(tmax) or tmax <= calculations.LOWER_AGE:
            return minAgeMa

        return int(tmax // 1e6)

    def _plot_concordia_curve(self):
        minAgeMa = int(calculations.LOWER_AGE // 1e6)

        maxAgeMa = min(4500, int(calculations.UPPER_AGE // 1e6))

        ts_ma = np.arange(minAgeMa, maxAgeMa + 1, 1, dtype=int)
        xs = [calculations.pb207u235_from_age(float(t) * 1e6) for t in ts_ma]
        ys = [calculations.pb206u238_from_age(float(t) * 1e6) for t in ts_ma]
        self.axis.plot(xs, ys, linestyle='-')

    def _plot_concordia_times(self):
        minAgeMa = int(calculations.LOWER_AGE // 1e6)
        maxAgeMa = min(4500, int(calculations.UPPER_AGE // 1e6))

        # Same “round number” scheme as TW, but avoid clutter near 0 by skipping <500 Ma
        increments = [500, 100, 50, 10, 5, 1]
        time = maxAgeMa

        ts2 = []
        i = 0
        while i < len(increments) and time >= minAgeMa:
            inc = increments[i]
            while time > inc and time >= minAgeMa:
                ts2.append(time)
                time -= inc
            i += 1
        if time == minAgeMa:
            ts2.append(time)

        # Wetherill gets unreadable at tiny ages near origin; keep only round, useful labels
        ts2 = [t for t in ts2 if t >= 500]

        xs2 = [calculations.pb207u235_from_age(float(t) * 1e6) for t in ts2]
        ys2 = [calculations.pb206u238_from_age(float(t) * 1e6) for t in ts2]

        self.axis.scatter(xs2, ys2, s=8)
        for t, x, y in zip(ts2, xs2, ys2):
            self.axis.annotate(
                f"{t} ",
                (x, y),
                horizontalalignment="right",
                verticalalignment="top",
                fontsize="small",
            )