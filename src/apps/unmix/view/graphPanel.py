import matplotlib
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from apps.abstract.view.graphPanel import AbstractGraphPanel

matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import numpy as np

from utils import config, stringUtils
import utils.calculations as calculations


class UnmixGraphPanel(AbstractGraphPanel):
    _default_xlim = (-1, 18)
    _default_ylim = (0, 0.6)

    def __init__(self, controller, *args, **kwargs):
        super().__init__("Concordia plot", *args, **kwargs)

    def _getCitationText(self):
        return "Hugo K.H. Olierook, Christopher L. Kirkland, Milo Barham, " \
               "Matthew L. Daggitt, Julie Hollis, Michael Hartnady" \
               "<b>Unmixing U-Pb ages from core–rim mixtures</b>, 2020"

    def createGraph(self):
        widget = QWidget()

        fig, self.axis = plt.subplots()
        plt.xlim(*self._default_xlim)
        plt.ylim(*self._default_ylim)
        plt.tight_layout(rect=[0.05, 0.08, 1, 0.95])
        self.axis.spines['top'].set_visible(False)
        self.axis.spines['right'].set_visible(False)

        self._setupConcordiaPlot(self.axis)

        # plot
        self.canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        layout.addWidget(toolbar)
        widget.setLayout(layout)
        return widget

    def displayRows(self, rows, calculationSettings):
        self.axis.clear()
        self._setupConcordiaPlot(self.axis)

        if len(rows) == 1:
            self._displayRow(rows[0], calculationSettings)
        elif rows:
            self._displayRows(rows, calculationSettings)

        self.canvas.draw()

    def _displayRow(self, row, calculationSettings):

        if row is None or not row.validImports or not row.processed:
            self.axis.set_xlim(*self._default_xlim)
            self.axis.set_ylim(*self._default_ylim)

            if row is not None and not row.processed:
                label = "(data not yet processed)"
            elif row is not None and not row.validImports:
                label = "(imported data is invalid)"
            else:
                label = ""
            self.axis.text(0.95, 0.95, label, horizontalalignment='right', verticalalignment='top', transform=self.axis.transAxes)
            return

        x1 = row.rimUPbValue
        y1 = row.rimPbPbValue
        x1_error = row.rimUPbStDev * calculationSettings.outputErrorSigmas
        y1_error = row.rimPbPbStDev * calculationSettings.outputErrorSigmas

        x2 = row.mixedUPbValue
        y2 = row.mixedPbPbValue
        x2_error = row.mixedUPbStDev * calculationSettings.outputErrorSigmas
        y2_error = row.mixedPbPbStDev * calculationSettings.outputErrorSigmas

        # Data
        best_fit_line_xs = [x1, x2]
        best_fit_line_ys = [y1, y2]

        reconstructedAge = row.reconstructedAgeObj

        if reconstructedAge is None:
            x3s = []
            y3s = []
            t_error_xs = []
            t_error_ys = []
            x3_errors = []
            y3_errors = []
            t3_fmt = 'none'
            xlim = self._default_xlim
            ylim = self._default_ylim
            label_text = "(no intercept with concordia curve found)"
        else:
            t, x3, y3 = reconstructedAge.values
            t_min, x3_min, y3_min = reconstructedAge.minValues
            t_max, x3_max, y3_max = reconstructedAge.maxValues

            best_fit_line_xs += [x3]
            best_fit_line_ys += [y3]

            label_text = "all errors at " + stringUtils.get_error_sigmas_str(calculationSettings.outputErrorSigmas)
            if not t_min:
                t_min = t
                label_text += "\n (no lower error bar available for reconstructed age)"
            if not t_max:
                t_max = t
                label_text += "\n (no upper error bar available for reconstructed age)"
            t3_fmt = 'none' if reconstructedAge.fullyValid else "^"

            if not x3_min:
                x3_min = x3
            if not x3_max:
                x3_max = x3
            if not y3_min:
                y3_min = y3
            if not y3_max:
                y3_max = y3

            x3s = [x3]
            y3s = [y3]
            x3_errors = [[x3-x3_min], [x3_max-x3]]
            y3_errors = [[y3-y3_min], [y3_max-y3]]

            ts = list(np.linspace(start=t_min, stop=t_max, num=20))
            t_error_xs = [calculations.u238pb206_from_age(t) for t in ts]
            t_error_ys = [calculations.pb207pb206_from_age(t) for t in ts]

            xlim, ylim = self._calculateMargin(x3_min, x1 + x1_error, y1 - y1_error, y3_max)

        self.axis.set_xlim(*xlim)
        self.axis.set_ylim(*ylim)

        # Best fit line
        self.axis.plot(best_fit_line_xs, best_fit_line_ys, linestyle='--')
        # Rim points
        self.axis.errorbar([x1], [y1], xerr=[x1_error], yerr=[y1_error], fmt='none', color=config.COLOUR_RIM_AGE)
        # Discordant points
        self.axis.errorbar([x2], [y2], xerr=[x2_error], yerr=[y2_error], fmt='none', color=config.COLOUR_MIXED_POINT)
        # Reconstructed age error line
        self.axis.plot(t_error_xs, t_error_ys, color=config.COLOUR_RECONSTRUCTED_AGE, linewidth=2)[0]
        # Reconstructed age xy points
        self.axis.errorbar(x3s, y3s, xerr=x3_errors, yerr=y3_errors, fmt=t3_fmt, color=config.COLOUR_RECONSTRUCTED_AGE)
        # Text
        self.axis.text(0.95, 0.95, label_text, horizontalalignment='right', verticalalignment='top', transform=self.axis.transAxes)
        #plt.text(12, 0.4, t_label_text)



    def _displayRows(self, rows, calculationSettings):
        good_xs = []
        good_ys = []
        good_xs_error_min = []
        good_xs_error_max = []
        good_ys_error_min = []
        good_ys_error_max = []

        bad_xs = []
        bad_ys = []
        bad_xs_error_min = []
        bad_xs_error_max = []
        bad_ys_error_min = []
        bad_ys_error_max = []

        xmax, xmin = self._default_xlim
        ymax, ymin = self._default_ylim

        anyRows = False
        anyInvalidRows = False
        for row in rows:
            if not row.validImports or not row.processed or row.reconstructedAgeObj is None:
                continue

            anyRows = True
            reconstructedAge = row.reconstructedAgeObj
            (t, x3, y3) = reconstructedAge.values
            t_min, x3_min, y3_min = reconstructedAge.minValues
            t_max, x3_max, y3_max = reconstructedAge.maxValues

            if not x3_min:
                x3_min = x3
            if not x3_max:
                x3_max = x3
            if not y3_min:
                y3_min = y3
            if not y3_max:
                y3_max = y3

            if reconstructedAge.fullyValid:
                xs = good_xs
                ys = good_ys
                xs_error_min = good_xs_error_min
                xs_error_max = good_xs_error_max
                ys_error_min = good_ys_error_min
                ys_error_max = good_ys_error_max
            else:
                anyInvalidRows = True
                xs = bad_xs
                ys = bad_ys
                xs_error_min = bad_xs_error_min
                xs_error_max = bad_xs_error_max
                ys_error_min = bad_ys_error_min
                ys_error_max = bad_ys_error_max

            xs += [x3]
            ys += [y3]

            xmin = min(xmin, x3_min)
            xmax = max(xmax, x3_max)
            ymin = min(ymin, y3_min)
            ymax = max(ymax, y3_max)

            xs_error_min += [x3 - x3_min]
            xs_error_max += [x3_max - x3]
            ys_error_min += [y3 - y3_min]
            ys_error_max += [y3_max - y3]

        good_xs_error = good_xs_error_min, good_xs_error_max
        good_ys_error = good_ys_error_min, good_ys_error_max
        bad_xs_error = bad_xs_error_min, bad_xs_error_max
        bad_ys_error = bad_ys_error_min, bad_ys_error_max

        if anyRows:
            xlim, ylim = self._calculateMargin(xmin, xmax, ymin, ymax)
        else:
            xlim, ylim = self._default_xlim, self._default_ylim

        if not any(row.processed for row in rows):
            label_text = "(data not yet processed)"
        else:
            label_text = "all errors at " + stringUtils.get_error_sigmas_str(calculationSettings.outputErrorSigmas)

        if anyInvalidRows:
            label_text += "\n (some error bars unavailable)"

        self.axis.errorbar(good_xs, good_ys, xerr=good_xs_error, yerr=good_ys_error, fmt='none', color=config.COLOUR_RECONSTRUCTED_AGE)
        self.axis.errorbar(bad_xs, bad_ys, xerr=bad_xs_error, yerr=bad_ys_error, fmt='^', color=config.COLOUR_RECONSTRUCTED_AGE)

        self.axis.set_xlim(*xlim)
        self.axis.set_ylim(*ylim)
        self.axis.text(0.95, 0.95, label_text, horizontalalignment='right', verticalalignment='top', transform=self.axis.transAxes)

    def _calculateMargin(self, xmin, xmax, ymin, ymax):
        xmargin = xmax * 0.1
        ymargin = ymax * 0.1

        xlim = (max(0, xmin - xmargin), xmax + xmargin)
        ylim = (max(0, ymin - ymargin), ymax + ymargin)

        return xlim, ylim

    def updatePlot(self):
        print("Update!")
