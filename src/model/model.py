import time
import csv
from collections import defaultdict

from model.sample import Sample
from model.spot import Spot
from model.settings.type import SettingsType
from process import processing

from model.settings.calculation import LeadLossCalculationSettings
from utils.settings import Settings

from utils.csvUtils import write_monte_carlo_output
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
        sample = self.samplesByName.get(sampleName)
        if not sample:
            return

        # Optional: keep legacy attrs in sync for code that still reads them
        try:
            ages_ma, y_curve, peaks, *rest = payload
            if isinstance(peaks, (list, tuple)):
                import numpy as np
                sample.summedKS_peaks_Ma = np.asarray(peaks, float)
                if rest and isinstance(rest[0], (list, tuple)):
                    ci_pairs = rest[0]
                    sample.summedKS_ci_low_Ma  = np.asarray([lo for lo, _ in ci_pairs], float)
                    sample.summedKS_ci_high_Ma = np.asarray([hi for _, hi in ci_pairs], float)
        except Exception:
            pass

        # Emit to the original UI sample; SampleOutputFigure is already connected to this
        if hasattr(sample.signals, "summedKS"):
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

    def updateConcordance(self, sampleName, concordancy, discordances):
        sample = self.samplesByName[sampleName]
        sample.updateConcordance(concordancy, discordances)

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
    
    ################
    ## Export data ##
    ################

    def exportMonteCarloRuns(self, append=False):
        filename = QFileDialog.getSaveFileName(
            caption='Save CSV file',
            directory='.',
            options=QFileDialog.DontUseNativeDialog
        )[0]
        if not filename:
            return
        mode = 'a' if append else 'w'
        with open(filename, mode, newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for sample in self.samples:
                for run in sample.getMonteCarloRuns():
                    run.calculateOptimalAge()
                    writer.writerow(run.toList())
