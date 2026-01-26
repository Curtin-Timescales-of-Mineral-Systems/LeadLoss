from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy, QWidget

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class AbstractFigure(QWidget):

    def __init__(self):
        super().__init__()

        self.fig = Figure()
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

    def redraw(self):
        # Safe redraw for Qt; avoids hard crashes if canvas is gone
        try:
            self.canvas.draw_idle()
        except RuntimeError:
            pass