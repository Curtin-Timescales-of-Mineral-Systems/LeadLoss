from process import calculations
from view.axes.concordia.abstractWetherillConcordiaAxis import WetherillConcordiaAxis
from utils.errorbarPlot import Errorbars

class SummaryWetherillConcordiaAxis(WetherillConcordiaAxis):

    def __init__(self, axis, samples):
        super().__init__(axis)
        self.refreshSamples(samples)

    def refreshSamples(self, samples):
        for sample in samples:
            self.refreshSample(sample)

    def refreshSample(self, sample):
        # Remove existing plots for this sample if present
        if sample.name in self.samples:
            for artist in self.samples[sample.name]:
                try:
                    artist.remove()
                except Exception:
                    pass

        xValues = [spot.pb207U235Value for spot in sample.validSpots if spot.pb207U235Value is not None]
        yValues = [spot.pb206U238Value for spot in sample.validSpots if spot.pb206U238Value is not None]

        line, = self.axis.plot(xValues, yValues, marker='o', linestyle='', picker=4)

        # Error bars (use stdevs in Wetherill space)
        r = calculations.mahalanobisRadius(2)
        xerr = [
            (spot.pb207U235StDev * r) if spot.pb207U235StDev is not None else 0.0
            for spot in sample.validSpots
            if spot.pb207U235Value is not None
        ]
        yerr = [
            (spot.pb206U238StDev * r) if spot.pb206U238StDev is not None else 0.0
            for spot in sample.validSpots
            if spot.pb206U238Value is not None
        ]

        errors = Errorbars(self.axis, xValues, yValues, xerr=xerr, yerr=yerr, fmt="none")

        # Expand x-axis to fit points + errors (matches TW logic)
        if xValues:
            upper_xlim = max([x + (xe if xe is not None else 0.0) for x, xe in zip(xValues, xerr)])
            self.axis.set_xlim(0, 1.2 * upper_xlim)

        # Pb-loss age range (parametrized by age, not x->y conversion)
        pbLossAgeRange = None
        if getattr(sample, "optimalAgeUpperBound", None) is not None and getattr(sample, "optimalAgeLowerBound", None) is not None:
            a0 = min(sample.optimalAgeUpperBound, sample.optimalAgeLowerBound)
            a1 = max(sample.optimalAgeUpperBound, sample.optimalAgeLowerBound)

            # Step at 1 Ma
            a0Ma = int(a0 // (10 ** 6))
            a1Ma = int(a1 // (10 ** 6))

            xs = []
            ys = []
            for tMa in range(a0Ma, a1Ma + 1):
                tYr = tMa * (10 ** 6)
                xs.append(calculations.pb207u235_from_age(tYr))
                ys.append(calculations.pb206u238_from_age(tYr))

            pbLossAgeRange, = self.axis.plot(xs, ys, linewidth=2, label="Pb loss age range")
            self.samples[sample.name] = (line, errors, pbLossAgeRange)
        else:
            self.samples[sample.name] = (line, errors)
