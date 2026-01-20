import numpy as np

from process import calculations


class WetherillConcordiaAxis:
    """
    Wetherill concordia coordinates:
      x = 207Pb/235U
      y = 206Pb/238U

    Mirrors the structure of view.axes.concordia.abstractConcordiaAxis.ConcordiaAxis
    so downstream plot classes can stay consistent.
    """

    def __init__(self, axis):
        self.axis = axis
        self.samples = {}

        self.axis.set_title("Wetherill concordia plot")
        self.axis.set_xlabel("${}^{207}Pb/{}^{235}U$")
        self.axis.set_ylabel("${}^{206}Pb/{}^{238}U$")

        # Match the existing TW plot framing style (tuned to ~0–3 Ga)
        self.axis.set_xlim(-1, 18)
        self.axis.set_ylim(0, 0.6)

        self._plot_concordia_curve()

    def _plot_concordia_curve(self):
        maxAgeMa = calculations.UPPER_AGE // (10 ** 6)
        minAgeMa = calculations.LOWER_AGE // (10 ** 6)

        # Same pattern as TW: loop in Ma, convert to years for calculations.*
        xs = []
        ys = []
        for tMa in range(minAgeMa, maxAgeMa):
            tYr = tMa * (10 ** 6)
            xs.append(calculations.pb207u235_from_age(tYr))
            ys.append(calculations.pb206u238_from_age(tYr))

        self.axis.plot(xs, ys, linestyle='-')
