import sys
import traceback

from PyQt5.QtWidgets import QMessageBox

from view.helpDialog import LeadLossHelpDialog
from process.processing import ProgressType
from controller.signals import Signals, ProcessingSignals
from utils import csvUtils
from utils.async import AsyncTask
from utils.settings import Settings
from model.model import LeadLossModel
from view.view import LeadLossView
from model.settings.type import SettingsType


class LeadLossTabController:

    def __init__(self):
        self.worker = None

        self.signals = Signals()

        self.processingSignals = ProcessingSignals()
        self.processingSignals.processingProgress.connect(self.onProcessingProgress)
        self.processingSignals.processingCompleted.connect(self.onProcessingCompleted)
        self.processingSignals.processingCancelled.connect(self.onProcessingCancelled)
        self.processingSignals.processingErrored.connect(self.onProcessingErrored)

        self.name = "Pb loss"
        self.model = LeadLossModel(self.signals)
        self.view = LeadLossView(self)

        #self.cheatLoad()

    ############
    ## Events ##
    ############

    def onProcessingCancelled(self):
        self.signals.taskComplete.emit(False, "Cancelled processing of data")

    def onProcessingErrored(self, exception):
        self.signals.taskComplete.emit(False, "Error whilst processing data")
        message = exception.__class__.__name__ + ": " + str(exception)
        QMessageBox.critical(None, "Error", "An error occurred during processing: \n\n" + message)

    def onProcessingProgress(self, progressArgs):
        type = progressArgs[0]

        if type == ProgressType.ERRORS:
            progress, i = progressArgs[1:]
            self.signals.taskProgress.emit(progress)
            if progress == 1.0:
                self.signals.taskStarted.emit("Identifying concordant points...")
        elif type == ProgressType.CONCORDANCE:
            progress, i, concordantAge, discordance = progressArgs[1:]
            self.model.updateConcordance(i, discordance, concordantAge)
            self.signals.taskProgress.emit(progress)
            if progress == 1.0:
                self.signals.concordancyClassification.emit(self.model.rows)
                self.signals.taskStarted.emit("Sampling rim age distribution...")
        elif type == ProgressType.SAMPLING:
            progress, i, rimAge, discordantAges, dValue, pValue = progressArgs[1:]
            self.model.addRimAgeStats(rimAge, discordantAges, dValue, pValue)
            self.signals.taskProgress.emit(progress)

    def onProcessingCompleted(self, output):
        optimalAge = output[0]
        self.model.setOptimalAge(optimalAge)
        _, dValue, pValue, reconstructedAges = self.model.getNearestSampledAge(optimalAge)

        self.signals.taskComplete.emit(True, "Processing complete")
        self.signals.optimalAgeFound.emit(optimalAge, dValue, pValue, reconstructedAges, self.model.getAgeRange())
        self.signals.allStatisticsUpdated.emit(self.model.dValuesByAge)
        self.selectAgeToCompare(optimalAge)

    #############
    ## Actions ##
    #############

    def importCSV(self):
        self.inputFile = self.view.getInputFile()
        if not self.inputFile:
            return

        self.view.getSettings(
            settingsType=SettingsType.IMPORT,
            rows=self.model.rows,
            callback=self._importCSVWithSettings
        )

    def _importCSVWithSettings(self, importSettings):
        if not importSettings:
            return
        Settings.update(importSettings)

        self.model.clearInputData()
        self._importCSV(self.inputFile, importSettings)

    def _importCSV(self, inputFile, importSettings):
        results = csvUtils.read_input(inputFile, importSettings)
        success = results is not None
        if success:
            self.model.loadInputData(inputFile, importSettings, *results)

    ################
    ## Processing ##
    ################

    def process(self):
        self.view.getSettings(
            settingsType=SettingsType.CALCULATION,
            rows=self.model.rows,
            callback=self._processWithSettings
        )

    def _processWithSettings(self, processingSettings):
        if not processingSettings:
            return
        Settings.update(processingSettings)

        self.model.resetCalculation()
        self.signals.taskStarted.emit("Pre-calculating error distributions...")
        self.signals.processingStarted.emit()
        self._process(processingSettings)

    def _process(self, calculationSettings):
        importSettings = Settings.get(SettingsType.IMPORT)
        self.worker = AsyncTask(self.processingSignals, self.model.getProcessingFunction(), self.model.getProcessingData(), importSettings, calculationSettings)
        self.worker.start()

    def cancelProcessing(self):
        if self.worker is not None:
            self.worker.halt()

    ############
    ## Export ##
    ############

    def exportCSV(self):
        pass

    ###########
    ## Other ##
    ###########

    def showHelp(self):
        dialog = LeadLossHelpDialog()
        dialog.exec_()

    def selectAgeToCompare(self, requestedRimAge):
        if requestedRimAge is None:
            self.signals.ageDeselected.emit()
            return

        actualRimAge, dValue, pValue, reconstructedAges = self.model.getNearestSampledAge(requestedRimAge)
        self.signals.ageSelected.emit(actualRimAge, reconstructedAges)

    def cheatLoad(self):
        try:
            inputFile = "../tests/hugoTest.csv"
            self._importCSV(inputFile, Settings.get(SettingsType.IMPORT))
        except:
            print(traceback.format_exc(), file=sys.stderr)
