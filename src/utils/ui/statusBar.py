import os
from pathlib import Path

from PyQt5.QtGui import QPixmap, QDesktopServices, QCursor
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from utils.ui import uiUtils


class StatusBarWidget(QWidget):

    def __init__(self, signals, *args, **kwargs):
        super(StatusBarWidget, self).__init__(*args, **kwargs)
        self.initUI()

        signals.taskStarted.connect(self.onTaskStarted)
        signals.taskProgress.connect(self.onTaskProgress)
        signals.taskComplete.connect(self.onTaskComplete)


    def initUI(self):
        self.text = QLabel("No data imported")
        self.icon = QLabel()
        self._setIcon(False)

        logoPix = QPixmap('../resources/logo-linear.png')
        logo = QLabel()
        logo.setPixmap(logoPix)
        logo.setCursor(QCursor(Qt.PointingHandCursor))
        logo.mouseReleaseEvent = self.openLink

        self.progressBar = QProgressBar()
        self.progressBar.setGeometry(30, 40, 200, 25)
        self.progressBar.hide()
        self.progressBar.setRange(0, 100)
        uiUtils.retainSizeWhenHidden(self.progressBar)

        layout = QHBoxLayout()
        layout.addWidget(logo)
        layout.addWidget(self.progressBar)
        layout.addSpacing(10)
        layout.addWidget(self.icon)
        layout.addWidget(self.text)
        layout.setAlignment(Qt.AlignRight)
        layout.setContentsMargins(0,0,0,0)

        self.setLayout(layout)

    def onTaskStarted(self, text):
        self.progressBar.show()
        self.text.setText(text)
        self.icon.hide()
        self.onTaskProgress(0)

    def onTaskProgress(self, progress):
        self.progressBar.setValue(int(progress * 100))

    def onTaskComplete(self, success, text):
        self.progressBar.hide()
        self.text.setText(text)
        self.text.adjustSize()
        self.icon.show()
        self._setIcon(success)

    def _setIcon(self, success):
        style = "SP_DialogApplyButton" if success else "SP_BrowserStop"
        icon = self.style().standardIcon(getattr(QStyle, style))
        self.icon.setPixmap(icon.pixmap(QSize(20, 20)))

    def openLink(self, e):
        QDesktopServices.openUrl(QUrl("https://github.com/Curtin-Timescales-of-Mineral-Systems/ConcordiaApplications"))