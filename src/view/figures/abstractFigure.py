# from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy, QWidget

# from matplotlib.figure import Figure
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


# class AbstractFigure(QWidget):

#     def __init__(self):
#         super().__init__()

#         self.fig = Figure()
#         self.canvas = FigureCanvas(self.fig)

#         self.canvas.setFocusPolicy(Qt.ClickFocus)
#         self.canvas.setFocus()

#         toolbar = NavigationToolbar(self.canvas, self)

#         layout = QVBoxLayout()
#         layout.setContentsMargins(0, 0, 0, 0)
#         layout.addWidget(self.canvas)
#         layout.addWidget(toolbar)

#         self.setLayout(layout)
#         self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

#     def redraw(self):
#         # Safe redraw for Qt; avoids hard crashes if canvas is gone
#         try:
#             self.canvas.draw_idle()
#         except RuntimeError:
#             pass

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy, QWidget, QFrame

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class AbstractFigure(QWidget):

    def __init__(self):
        super().__init__()

        # Make the FIGURE background transparent (this affects the margin around the axes)
        self.fig = Figure(facecolor="none")
        self.fig.patch.set_alpha(0)

        self.canvas = FigureCanvas(self.fig)

        # Make the CANVAS widget itself transparent so what's behind it shows through
        self.canvas.setStyleSheet("background: transparent;")
        self.canvas.setAttribute(Qt.WA_TranslucentBackground, True)
        self.canvas.setAutoFillBackground(False)

        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()

        # Host widget behind the canvas (this is where we can paint / stylesheet a background image)
        self.canvasHost = QFrame()
        self.canvasHost.setObjectName("FigureHost")  # you can override per-plot, e.g. "ConcordiaHost"
        self.canvasHost.setAttribute(Qt.WA_StyledBackground, True)

        host_layout = QVBoxLayout(self.canvasHost)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.addWidget(self.canvas)

        toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvasHost)
        layout.addWidget(toolbar)

        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_watermark(self, png_path: str):
        """
        Set a background image behind the matplotlib canvas.
        Use a PNG with alpha for “transparent watermark” effect.
        """
        path = png_path.replace("\\", "/")
        # border-image will scale to fill; good for a full-panel watermark
        self.canvasHost.setStyleSheet(
            f'QFrame#{self.canvasHost.objectName()} {{'
            f'  border-image: url("{path}") 0 0 0 0 stretch stretch;'
            f'  background-color: transparent;'
            f'}}'
        )

    def redraw(self):
        try:
            self.canvas.draw_idle()
        except RuntimeError:
            pass
