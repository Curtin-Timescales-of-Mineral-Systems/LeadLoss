import sys
import traceback

from view.helpDialog import LeadLossHelpDialog
from process.processing import ProgressType
from controller.signals import Signals
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
        self.signals.processingProgress.connect(self.onProcessingProgress)
        self.signals.processingCompleted.connect(self.onProcessingCompleted)
        self.signals.processingCancelled.connect(self.onProcessingCancelled)
        self.signals.processingErrored.connect(self.onProcessingErrored)

        self.name = "Pb loss"
        self.model = LeadLossModel(self.signals)
        self.view = LeadLossView(self)

        self.cheatLoad()

    ############
    ## Events ##
    ############

    def onProcessingCompleted(self, output):
        self.signals.taskComplete.emit(True, "Processing complete")
        self.model.addProcessingOutput(output)

    def onProcessingCancelled(self):
        self.signals.taskComplete.emit(False, "Cancelled processing of data")

    def onProcessingErrored(self, exception):
        self.signals.taskComplete.emit(False, "Error whilst processing data")

    #############
    ## Actions ##
    #############

    def importCSV(self):
        self.inputFile = self.view.getInputFile()
        if not self.inputFile:
            return

        self.view.getSettings(SettingsType.IMPORT, self._importCSVWithSettings)

    def _importCSVWithSettings(self, importSettings):
        if not importSettings:
            return
        Settings.update(importSettings)

        self._importCSV(self.inputFile, importSettings)

    def _importCSV(self, inputFile, importSettings):
        results = csvUtils.read_input(inputFile, importSettings)
        success = results is not None
        if success:
            self.model.loadRawData(importSettings, *results)
        self.view.onCSVImportFinished(success, inputFile)

    def selectAgeToCompare(self, age):
        self.model.selectAgeToCompare(age)

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
                self.signals.allRowsUpdated.emit(self.model.rows)
                self.signals.taskStarted.emit("Sampling rim age distribution...")
        elif type == ProgressType.SAMPLING:
            progress, i = progressArgs[1:]
            self.signals.taskProgress.emit(progress)

    def process(self):
        self.view.getSettings(SettingsType.CALCULATION, self._processWithSettings)

    def _processWithSettings(self, processingSettings):
        if not processingSettings:
            return
        Settings.update(processingSettings)

        self.signals.processingStarted.emit()
        self.signals.taskStarted.emit("Processing data...")

        self.model.resetCalculations()
        self._process(processingSettings)

    def _process(self, calculationSettings):
        importSettings = Settings.get(SettingsType.IMPORT)
        self.worker = AsyncTask(self.signals, self.model.getProcessingFunction(), self.model.getProcessingData(), importSettings, calculationSettings)
        self.worker.start()

    def showHelp(self):
        dialog = LeadLossHelpDialog()
        dialog.exec_()

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


    def cheatLoad(self):
        try:
            inputFile = "../tests/leadLossTest2.csv"
            self._importCSV(inputFile, Settings.get(SettingsType.IMPORT))
        except:
            print(traceback.format_exc(), file=sys.stderr)
