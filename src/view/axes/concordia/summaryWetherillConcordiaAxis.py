import math

import numpy as np

from process import calculations
from utils import config
from utils.errorbarPlot import Errorbars
from view.axes.concordia.abstractWetherillConcordiaAxis import WetherillConcordiaAxis


class SummaryWetherillConcordiaAxis(WetherillConcordiaAxis):

    def __init__(self, axis, samples):
        super().__init__(axis)
        self.samples = {}
        self.refreshSamples(samples)

    def refreshSamples(self, samples):
        for sample in samples:
            self.refreshSample(sample)

    def refreshSample(self, sample):
        name = sample.name

        # Remove existing plots for this sample
        if name in self.samples:
            for artist in self.samples[name].artists:
                try:
                    artist.remove()
                except Exception:
                    pass

        self.samples[name] = SamplePlot(self.axis, sample)

    def selectSamples(self, selectedSamples, unselectedSamples):
        for sample in selectedSamples:
            if sample.name in self.samples:
                self.samples[sample.name].setSelected(True)
        for sample in unselectedSamples:
            if sample.name in self.samples:
                self.samples[sample.name].setSelected(False)


class SamplePlot:
    def __init__(self, axis, sample):
        self.axis = axis
        self.sample = sample
        self.artists = []
        self.isSelected = True
        self._plot()

    def _plot(self):
        sample = self.sample

        xValues = [s.pb207U235Value for s in sample.validSpots if s.pb207U235Value is not None]
        yValues = [s.pb206U238Value for s in sample.validSpots if s.pb206U238Value is not None]

        # Nothing to plot
        if not xValues or not yValues:
            return

        # Main points
        (line,) = self.axis.plot(
            xValues,
            yValues,
            marker="o",
            linestyle="",
            picker=4,
            color=config.UNCLASSIFIED_COLOUR_1,
            alpha=config.PLOT_ALPHA,
        )
        self.artists.append(line)

        # Error bars (match TW: stdev * rs where rs = sqrt(radius))
        rs = math.sqrt(calculations.mahalanobisRadius(2))
        xerr = [
            (s.pb207U235StDev * rs) if (s.pb207U235StDev is not None) else 0.0
            for s in sample.validSpots
            if s.pb207U235Value is not None
        ]
        yerr = [
            (s.pb206U238StDev * rs) if (s.pb206U238StDev is not None) else 0.0
            for s in sample.validSpots
            if s.pb206U238Value is not None
        ]

        errorContainer = self.axis.errorbar(
            xValues,
            yValues,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            alpha=config.PLOT_ALPHA,
            ecolor=config.UNCLASSIFIED_COLOUR_1,
        )
        errors = Errorbars(errorContainer)
        self.artists.extend(errors.plots)

        # Pb-loss age range curve (parametrised by age)
        if getattr(sample, "optimalAgeUpperBound", None) is not None and getattr(sample, "optimalAgeLowerBound", None) is not None:
            t0 = min(sample.optimalAgeUpperBound, sample.optimalAgeLowerBound)
            t1 = max(sample.optimalAgeUpperBound, sample.optimalAgeLowerBound)

            ts = np.linspace(t0, t1, 200)
            xs = [calculations.pb207u235_from_age(t) for t in ts]
            ys = [calculations.pb206u238_from_age(t) for t in ts]

            (pbLossRange,) = self.axis.plot(xs, ys, linewidth=2, label="Pb loss age range")
            self.artists.append(pbLossRange)

        # Expand x-axis to include errors (similar spirit to TW)
        upper_xlim = max([x + xe for x, xe in zip(xValues, xerr)]) if xValues else None
        if upper_xlim is not None:
            self.axis.set_xlim(0, 1.2 * upper_xlim)

    def setSelected(self, selected):
        self.isSelected = bool(selected)
        alpha = config.PLOT_ALPHA if self.isSelected else config.PLOT_ALPHA_UNSELECTED
        for artist in self.artists:
            try:
                artist.set_alpha(alpha)
            except Exception:
                pass
