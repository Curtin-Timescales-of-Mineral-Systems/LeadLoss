import matplotlib
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QSizePolicy, QWidget

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

matplotlib.use('QT5Agg')

class AbstractFigure(QWidget):

    def __init__(self):
        super().__init__()

        self.fig = plt.figure()

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()
        toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        layout.addWidget(toolbar)

        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)