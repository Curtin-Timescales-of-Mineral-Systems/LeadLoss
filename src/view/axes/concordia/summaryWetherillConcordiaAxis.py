import math
import numpy as np
import math
from process import calculations
from utils import config
from utils.errorbarPlot import Errorbars
from view.axes.concordia.abstractConcordiaAxis import ConcordiaAxis

class SummaryWetherillConcordiaAxis(ConcordiaAxis):
    """
    Mirrors SummaryConcordiaAxis, but uses Wetherill (x=207Pb/235U, y=206Pb/238U).
    Also draws a Wetherill concordia curve.
    """

    def __init__(self, axis, samples):
        super().__init__(axis)

        self.selectedSamples = samples
        self.unselectedSamples = []
        self.samples = {}  # store a SamplePlotWetherill for each sample

        # Adjust the axis labels for Wetherill
        self.axis.set_title("Wetherill Concordia Plot")
        self.axis.set_xlabel(r"${}^{207}\mathrm{Pb}/{}^{235}\mathrm{U}$")
        self.axis.set_ylabel(r"${}^{206}\mathrm{Pb}/{}^{238}\mathrm{U}$")

        # Plot the “Wetherill” curve first
        self._plotConcordiaCurve()

        maxAge = calculations.UPPER_AGE // (10 ** 6)
        minAge = calculations.LOWER_AGE // (10 ** 6)
        # Plot concordia times
        time = maxAge
        i = 0
        increments = [500, 100, 50]
        ts2 = []
        while i < len(increments) and time >= minAge:
            increment = increments[i]
            while time > increment and time >= minAge:
                ts2.append(time)
                time -= increment
            i += 1
        if time == minAge:
            ts2.append(time)
        xs2 = [calculations.pb207u235_from_age(t * (10 ** 6)) for t in ts2]
        ys2 = [calculations.pb206u238_from_age(t * (10 ** 6)) for t in ts2]
        self.axis.scatter(xs2, ys2, s=8)
        for i, txt in enumerate(ts2):
            self.axis.annotate(
                str(txt) + " ",
                (xs2[i], ys2[i]),
                horizontalalignment="right",
                verticalalignment="top",
                fontsize="small"
            )

        # Create and plot each sample
        for sample in samples:
            self.plotSample(sample)

        # Build the legend
        examplePlot = self.samples[samples[0]]
        legendEntries = [
            (examplePlot.unclassified.line, "Unclassified"),
            (examplePlot.concordant.line,   "Concordant"),
            (examplePlot.discordant.line,   "Discordant"),
            (examplePlot.pbLossAge,         "Pb-loss age"),
        ]
        self.axis.legend(*zip(*legendEntries), frameon=False)

    def plotSample(self, sample):
        self.samples[sample] = SamplePlotWetherill(self.axis, sample)

    def refreshSample(self, sample):
        # If sample is among selected, re-plot; otherwise clear it
        if sample in self.selectedSamples:
            self.samples[sample].plotInputData(sample)
        else:
            self.samples[sample].clearData()

    def selectSamples(self, selectedSamples, unselectedSamples):
        self.selectedSamples = selectedSamples
        self.unselectedSamples = unselectedSamples
        
        for sample in selectedSamples:
            self.samples[sample].plotInputData(sample)
        for sample in unselectedSamples:
            self.samples[sample].clearData()

    def _plotConcordiaCurve(self):
        """
        Plot the Wetherill curve: X=207Pb/235U, Y=206Pb/238U
        using your existing calculations.* functions.
        """
        minAge = calculations.LOWER_AGE
        maxAge = calculations.UPPER_AGE
        ages = np.linspace(minAge, maxAge, 200)

        xvals = [calculations.pb207u235_from_age(t) for t in ages]
        yvals = [calculations.pb206u238_from_age(t) for t in ages]

        self.axis.plot(xvals, yvals, color='blue')

class SamplePlotWetherill:
    """
    Equivalent to SamplePlot, but uses Wetherill ratios for X and Y.
    Also same color logic for unclassified, concordant, discordant.
    """

    def __init__(self, axis, sample):
        self.axis = axis
        self.sample = sample

        # The 3 errorbars (unclassified, concordant, discordant)
        self.unclassified = Errorbars(axis.errorbar([], [], xerr=[], yerr=[],
            fmt='+', linestyle='', color=config.UNCLASSIFIED_COLOUR_1))
        self.concordant = Errorbars(axis.errorbar([], [], xerr=[], yerr=[],
            fmt='+', linestyle='', color=config.CONCORDANT_COLOUR_1))
        self.discordant = Errorbars(axis.errorbar([], [], xerr=[], yerr=[],
            fmt='+', linestyle='', color=config.DISCORDANT_COLOUR_1))

        self.pbLossAge   = self.axis.plot([], [], marker='o', color=config.OPTIMAL_COLOUR_1)[0]
        self.pbLossRange = self.axis.plot([], [], color=config.OPTIMAL_COLOUR_1)[0]

        self.plotInputData(sample)

    def plotInputData(self, sample):

        rs = math.sqrt(calculations.mahalanobisRadius(2))

        concordantData = []
        discordantData = []
        unclassifiedData = []

        upper_x = 0
        upper_y = 0

        for spot in sample.validSpots:
            if (spot.pb207U235Value is None) or (spot.pb206U238Value is None):
                continue

            # 1) Mahalanobis radius approach, same as Tera-W
            semi_minor = (spot.pb207U235Error or 0) * rs
            semi_major = (spot.pb206U238Error or 0) * rs

            upper_x = max(upper_x, spot.pb207U235Value + semi_minor)
            upper_y = max(upper_y, spot.pb206U238Value + semi_major)

            if not spot.processed:
                unclassifiedData.append( (spot.pb207U235Value, spot.pb206U238Value, semi_minor, semi_major) )
            else:
                if spot.concordant:
                    concordantData.append( (spot.pb207U235Value, spot.pb206U238Value, semi_minor, semi_major) )
                else:
                    discordantData.append( (spot.pb207U235Value, spot.pb206U238Value, semi_minor, semi_major) )

        if concordantData:
            self.concordant.set_data(*zip(*concordantData))
        else:
            self.concordant.clear_data()

        if discordantData:
            self.discordant.set_data(*zip(*discordantData))
        else:
            self.discordant.clear_data()

        if unclassifiedData:
            self.unclassified.set_data(*zip(*unclassifiedData))
        else:
            self.unclassified.clear_data()

        if sample.optimalAge:
            a_low = sample.optimalAgeLowerBound
            a_high = sample.optimalAgeUpperBound
            n_points = 100
            if a_low == a_high:
                x_vals = [calculations.pb207u235_from_age(a_low)]
                y_vals = [calculations.pb206u238_from_age(a_low)]
            else:
                ages_lin = np.linspace(a_low, a_high, n_points)
                x_vals = [calculations.pb207u235_from_age(a) for a in ages_lin]
                y_vals = [calculations.pb206u238_from_age(a) for a in ages_lin]

            self.pbLossAge.set_xdata([x_vals[0], x_vals[-1]])
            self.pbLossAge.set_ydata([y_vals[0], y_vals[-1]])
            self.pbLossRange.set_xdata(x_vals)
            self.pbLossRange.set_ydata(y_vals)

            upper_x = max(upper_x, max(x_vals))
            upper_y = max(upper_y, max(y_vals))
        else:
            self.pbLossAge.set_xdata([])
            self.pbLossAge.set_ydata([])
            self.pbLossRange.set_xdata([])
            self.pbLossRange.set_ydata([])

        self.axis.set_xlim(0, 1.2 * upper_x)
        self.axis.set_ylim(0, 1.2 * upper_y)

        self.axis.set_xlim(0, 1.2 * upper_x)
        self.axis.set_ylim(0, 1.2 * upper_y)

    def clearData(self):
        self.concordant.clear_data()
        self.discordant.clear_data()
        self.unclassified.clear_data()
        self.pbLossAge.set_xdata([])
        self.pbLossAge.set_ydata([])
        self.pbLossRange.set_xdata([])
        self.pbLossRange.set_ydata([])
