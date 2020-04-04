from PyQt5.QtCore import pyqtSignal, QObject

from apps.abstract.model.row import AbstractRow


class Signals(QObject):
    # Success, file name
    csvImported = pyqtSignal([bool, str])

    # Row index, row
    rowUpdated = pyqtSignal([int, AbstractRow])
    # Rows
    allRowsUpdated = pyqtSignal(list)
    # Headers
    headersUpdated = pyqtSignal(list)

    # Task description
    taskStarted = pyqtSignal(str)
    # Progress (0.0 - 1.0)
    taskProgress = pyqtSignal(float)
    # Success, success description
    taskComplete = pyqtSignal([bool, str])

    processingStarted = pyqtSignal()
    processingProgress = pyqtSignal(object)
    processingCompleted = pyqtSignal(object)
    processingCancelled = pyqtSignal()
    processingErrored = pyqtSignal(object)