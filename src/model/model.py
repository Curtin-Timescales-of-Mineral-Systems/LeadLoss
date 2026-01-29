import time
import csv
import numpy as np

from collections import defaultdict

from model.sample import Sample
from model.spot import Spot
from process import processing


from PyQt5.QtWidgets import QFileDialog

class LeadLossModel:

    UPDATE_INTERVAL = 0.5

    def __init__(self, signals):
        self.signals = signals
        self.headers = []
        self.samples = []
        self.samplesByName = {}

        # legacy state used by getters/exports in tools
        self.rows = []
        self.concordantRows = []
        self.discordantRows = []

        self.dValuesByAge = {}
        self.pValuesByAge = {}
        self.reconstructedAges = {}
        self.optimalAge = None

        self.lastUpdateTime = 0

    ################
    ## Input data ##
    ################

    def loadInputData(self, inputFile, importSettings, rawHeaders, rawSpotData):
        self.headers = rawHeaders

        spotsBySampleName = defaultdict(list)
        for row in rawSpotData:
            spot = Spot(row, importSettings)
            spotsBySampleName[spot.sampleName].append(spot)

        self.samples = []
        self.samplesByName = {}
        for id, (sampleName, sampleRows) in enumerate(spotsBySampleName.items()):
            sample = Sample(id, sampleName, sampleRows)
            self.samples.append(sample)
            self.samplesByName[sampleName] = sample

        self.signals.inputDataLoaded.emit(inputFile, self.samples)
        self.signals.taskComplete.emit(True, "Successfully imported CSV file")

    def clearInputData(self):
        self.headers = []
        self.rows = []
        self.concordantRows = []
        self.discordantRows = []

        self.signals.inputDataCleared.emit()
        
    def emitSummedKS(self, sampleName, payload):
        if sampleName in self.samplesByName:   # or self.samplesByName.keys() in your code
            sample = self.samplesByName[sampleName]

            # payload is: (ages_ma_list, y_curve_list, peaks_age, peaks_ci, support)
            try:
                ages_ma = payload[0]
                y_curve = payload[1]
                sample.summedKS_ages_Ma = np.asarray(ages_ma, dtype=float)
                sample.summedKS_goodness = np.asarray(y_curve, dtype=float)
            except Exception:
                sample.summedKS_ages_Ma = None
                sample.summedKS_goodness = None

            if sample.signals:
                sample.signals.summedKS.emit(payload)


    #################
    ## Calculation ##
    #################

    def clearCalculation(self):
        for sample in self.samples:
            sample.clearCalculation()

        self.lastUpdateTime = time.time()

    def getProcessingFunction(self):
        return processing.processSamples

    def getProcessingData(self):
        return [sample.createProcessingCopy() for sample in self.samples]

    def updateConcordance(self, sampleName, concordantAges, discordances, reverse_flags=None):
        sample = next((s for s in self.samples if s.name == sampleName), None)
        if not sample:
            return
        sample.updateConcordance(concordantAges, discordances, reverse_flags)


    def addMonteCarloRun(self, sampleName, run):
        sample = self.samplesByName[sampleName]
        sample.addMonteCarloRun(run)

    def setOptimalAge(self, sampleName, args):
        sample = self.samplesByName[sampleName]
        sample.setOptimalAge(args)

    #############
    ## Getters ##
    #############

    def addRimAgeStats(self, rimAge, discordantAges, dValue, pValue):
        self.dValuesByAge[rimAge] = dValue
        self.pValuesByAge[rimAge] = pValue
        self.reconstructedAges[rimAge] = discordantAges

        self.signals.statisticUpdated.emit(len(self.dValuesByAge)-1, dValue, pValue)
        now = time.time()
        if now - self.lastUpdateTime > self.UPDATE_INTERVAL:
            self.signals.allStatisticsUpdated.emit(self.dValuesByAge)
            self.lastUpdateTime = now

    def getAgeRange(self):
        concordantAges = [row.concordantAge for row in self.rows if getattr(row, "concordant", False)]
        recAges = [recAge for ages in self.reconstructedAges.values() for recAge in ages]
        discordantAges = [recAge.values[0] for recAge in recAges if recAge]
        allAges = concordantAges + discordantAges
        return (min(allAges), max(allAges)) if allAges else (None, None)

    def getNearestSampledAge(self, requestedAge):
        if not self.dValuesByAge:
            return None, []

        if requestedAge is not None:
            actualAge = min(self.dValuesByAge, key=lambda a: abs(a-requestedAge))
        else:
            actualAge = self.optimalAge

        return actualAge, self.dValuesByAge[actualAge], self.pValuesByAge[actualAge], self.reconstructedAges[actualAge]