import multiprocessing
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QStyleFactory, QMessageBox, QTabWidget, QDialog, QVBoxLayout


class AbstractApplication:
    def __init__(self):
        # Necessary for building executable with Pyinstaller correctly on Windows
        # (see https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing)
        multiprocessing.freeze_support()

    def exceptionHook(self, exctype, value, traceback):
        sys.__excepthook__(exctype, value, traceback)
        QMessageBox.critical(None, "Error", str(value))

    def createGUI(self):
        # Reroute exceptions to display a message box to the user
        sys.excepthook = self.exceptionHook

        app = QApplication(sys.argv)
        app.setStyle(QStyleFactory.create('Fusion'))
        app.setWindowIcon(QIcon(self.getIcon()))

        self.mainWindow = GUI(self)
        sys.exit(app.exec_())

class GUI(QDialog):

    def __init__(self, application):
        super().__init__()

        self.left = 10
        self.top = 10
        self.width = 1220
        self.height = 500

        title = application.getTitle()
        version = application.getVersion()
        controllers = application.getControllers()

        self.setWindowTitle(title + " (v" + version + ")")
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowFlags(self.windowFlags() & Qt.WindowMaximizeButtonHint)

        if len(controllers) > 1:
            widget = QTabWidget()
            for controller in controllers:
                widget.addTab(controller.view, controller.name)
            widget.setCurrentWidget(controllers[0].view)
        else:
            widget = controllers[0].view

        layout = QVBoxLayout()
        layout.addWidget(widget)
        self.setLayout(layout)
        self.showMaximized()