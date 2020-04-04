from apps.abstract.model.signals import Signals
from utils import csvUtils
from utils.async import AsyncTask
from utils.settings import Settings
from apps.type import SettingsType


class AbstractTabController:

    def __init__(self, applicationType):
        self.applicationType = applicationType
        self.worker = None

        self.signals = Signals()
        self.signals.processingProgress.connect(self.onProcessingProgress)
        self.signals.processingCompleted.connect(self.onProcessingCompleted)
        self.signals.processingCancelled.connect(self.onProcessingCancelled)
        self.signals.processingErrored.connect(self.onProcessingErrored)

    ############
    ## Import ##
    ############

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

    #############
    ## Process ##
    #############

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

        importSettings = Settings.get(self.applicationType, SettingsType.IMPORT)
        self.worker = AsyncTask(self.signals, self.model.getProcessingFunction(), self.model.getProcessingData(), importSettings, calculationSettings)
        self.worker.start()

    def onProcessingCompleted(self, output):
        self.signals.taskComplete.emit(True, "Processing complete")
        self.model.addProcessingOutput(output)

    def cancelProcessing(self):
        if self.worker is not None:
            self.worker.halt()

    ############
    ## Export ##
    ############

    def exportCSV(self):
        pass

    ###########
    ## Utils ##
    ###########

    def onProcessingCancelled(self):
        self.signals.taskComplete.emit(False, "Cancelled processing of data")

    def onProcessingErrored(self, exception):
        self.signals.taskComplete.emit(False, "Error whilst processing data")