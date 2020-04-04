from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QFormLayout, QVBoxLayout, QLabel

from utils import calculations
import numpy as np


class AbstractGraphPanel(QGroupBox):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QVBoxLayout()
        layout.addWidget(self.createGraph())
        layout.addWidget(self._createCitation())
        self.setLayout(layout)

    def _setupConcordiaPlot(self, axis):
        axis.set_xlabel("${}^{238}U/{}^{206}Pb$")
        axis.set_ylabel("${}^{207}Pb/{}^{206}Pb$")

        maxAge = 4500
        minAge = 200
        xMin = calculations.u238pb206_from_age(maxAge * (10 ** 6))
        xMax = calculations.u238pb206_from_age(minAge * (10 ** 6))

        # Plot concordia curve
        xs = np.arange(xMin, xMax, 0.1)
        ys = [calculations.pb207pb206_from_u238pb206(x) for x in xs]
        axis.plot(xs, ys)

        # Plot concordia times
        ts2 = list(range(100, minAge, 100)) + list(range(500, maxAge + 1, 500))
        xs2 = [calculations.u238pb206_from_age(t * (10 ** 6)) for t in ts2]
        ys2 = [calculations.pb207pb206_from_age(t * (10 ** 6)) for t in ts2]
        axis.scatter(xs2, ys2)
        for i, txt in enumerate(ts2):
            axis.annotate(str(txt) + " ", (xs2[i], ys2[i]), horizontalalignment="right", verticalalignment="top",
                         fontsize="small")

    def _createCitation(self):
        label = QLabel(self._getCitationText())
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return label