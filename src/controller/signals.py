from PyQt5.QtCore import pyqtSignal, QObject

from model.row import Row


class Signals(QObject):
    inputDataLoaded = pyqtSignal(str, list, list)  # Input file, headers, rows
    inputDataCleared = pyqtSignal()

    headersUpdated = pyqtSignal(list)  # Headers
    rowUpdated = pyqtSignal([int, Row])  # Row index, row
    allRowsUpdated = pyqtSignal(list)  # Rows

    taskStarted = pyqtSignal(str)  # Task description
    taskProgress = pyqtSignal(float)  # Progress (0.0 - 1.0)
    taskComplete = pyqtSignal([bool, str])  # Success, success description

    processingCleared = pyqtSignal()
    processingStarted = pyqtSignal()

    concordancyClassification = pyqtSignal(list) # rows
    statisticUpdated = pyqtSignal(int, float, float) # rowNumber, pValue, dValue
    allStatisticsUpdated = pyqtSignal(dict)
    optimalAgeFound = pyqtSignal(float, float, float, list, tuple) # age, pValue, dValue, reconstructedAges, maximumRangeOfReconstructedAges
    ageDeselected = pyqtSignal()
    ageSelected = pyqtSignal([float, list]) # age, reconstructedAges


class ProcessingSignals(QObject):
    processingProgress = pyqtSignal(object)
    processingCompleted = pyqtSignal(object)
    processingCancelled = pyqtSignal()
    processingErrored = pyqtSignal(object)
