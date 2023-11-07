from copy import deepcopy

from PyQt5.QtCore import pyqtSignal, QObject


class Sample:

    def __init__(self, id, name, spots):
        self.id = id
        self.name = name
        self.spots = spots

        self.validSpots = [spot for spot in self.spots if spot.valid]
        self.invalidSpots = [spot for spot in self.spots if not spot.valid]

        self.signals = SampleSignals()

        self.calculationSettings = None
        self.optimalAge = None
        self.optimalAgeLowerBound = None
        self.optimalAgeUpperBound = None
        self.optimalAgeDValue = None
        self.optimalAgePValue = None
        self.optimalAgeNumberOfInvalidPoints = None
        self.optimalAgeScore = None
        self.monteCarloRuns = []

    def concordantSpots(self):
        return [spot for spot in self.validSpots if spot.concordant]

    def discordantSpots(self):
        return [spot for spot in self.validSpots if not spot.concordant]

    ##################
    ## Calculations ##
    ##################

    def startCalculation(self, calculationSettings):
        self.calculationSettings = calculationSettings

    def clearCalculation(self):
        self.optimalAge = None
        self.monteCarloRuns = []
        for spot in self.spots:
            spot.clear()
        self.signals.processingCleared.emit()

    def updateConcordance(self, concordancy, discordances):
        for spot, concordant, discordance in zip(self.validSpots, concordancy, discordances):
            spot.updateConcordance(concordant, discordance)

        if self.signals:
            self.signals.concordancyCalculated.emit()

    def addMonteCarloRun(self, run):
        self.monteCarloRuns.append(run)
        if self.signals:
            self.signals.monteCarloRunAdded.emit()

    def setOptimalAge(self, args):
        self.optimalAge = args[0]
        self.optimalAgeLowerBound = args[1]
        self.optimalAgeUpperBound = args[2]
        self.optimalAgeDValue = args[3]
        self.optimalAgePValue = args[4]
        self.optimalAgeNumberOfInvalidPoints = args[5]
        self.optimalAgeScore = args[6]

        if self.signals:
            self.signals.optimalAgeCalculated.emit()

    def createProcessingCopy(self):
        signals = self.signals
        self.signals = None
        copy = deepcopy(self)
        self.signals = signals
        return copy
    
    def getMonteCarloRuns(self):
        return self.monteCarloRuns   

class SampleSignals(QObject):
    processingCleared = pyqtSignal()

    concordancyCalculated = pyqtSignal()
    monteCarloRunAdded = pyqtSignal()
    optimalAgeCalculated = pyqtSignal()
