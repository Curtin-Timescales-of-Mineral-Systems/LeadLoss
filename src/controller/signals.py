from PyQt5.QtCore import pyqtSignal, QObject

from model.sample import Sample


class Signals(QObject):
    inputDataLoaded = pyqtSignal(str, list)  # Input file, samples
    inputDataCleared = pyqtSignal()

    taskStarted = pyqtSignal(str)  # Task description
    taskProgress = pyqtSignal(float)  # Progress (0.0 - 1.0)
    taskComplete = pyqtSignal([bool, str])  # Success, success description

    allProcessingCleared = pyqtSignal()
    processingStarted = pyqtSignal()
    processingFinished = pyqtSignal()

    statisticUpdated = pyqtSignal(int, float, float) # rowNumber, pValue, dValue
    allStatisticsUpdated = pyqtSignal(dict)
    optimalAgeFound = pyqtSignal(float, float, float, list, tuple) # age, pValue, dValue, reconstructedAges, maximumRangeOfReconstructedAges
    ageDeselected = pyqtSignal()
    ageSelected = pyqtSignal([float, list]) # age, reconstructedAges


class ProcessingSignals(QObject):
    processingNewTask = pyqtSignal(object)
    processingProgress = pyqtSignal(object)
    processingCompleted = pyqtSignal(object)
    processingCancelled = pyqtSignal()
    processingErrored = pyqtSignal(object)
