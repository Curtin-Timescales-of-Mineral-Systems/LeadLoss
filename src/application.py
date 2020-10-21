import multiprocessing
import sys
import traceback

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox, QApplication, QStyleFactory

from controller.signals import Signals, ProcessingSignals
from model.model import LeadLossModel
from model.settings.type import SettingsType
from process.processing import ProgressType
from utils import config, resourceUtils, csvUtils
from utils.async import AsyncTask
from utils.settings import Settings
from view.dialogs.help import LeadLossHelpDialog
from view.view import LeadLossView


class LeadLossApplication:

    @staticmethod
    def getTitle():
        return config.LEAD_LOSS_TITLE

    @staticmethod
    def getIcon():
        return resourceUtils.getResourcePath("icon.png")

    @staticmethod
    def exceptionHook(exctype, value, traceback):
        sys.__excepthook__(exctype, value, traceback)
        QMessageBox.critical(None, "Error", str(value))

    def __init__(self):
        # Necessary for building executable with Pyinstaller correctly on Windows
        # (see https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing)
        multiprocessing.freeze_support()

        # Reroute exceptions to display a message box to the user
        sys.excepthook = self.exceptionHook

        self.worker = None
        self.signals = Signals()

        self.processingSignals = ProcessingSignals()
        self.processingSignals.processingNewTask.connect(self.onProcessingNewTask)
        self.processingSignals.processingProgress.connect(self.onProcessingProgress)
        self.processingSignals.processingCompleted.connect(self.onProcessingCompleted)
        self.processingSignals.processingCancelled.connect(self.onProcessingCancelled)
        self.processingSignals.processingErrored.connect(self.onProcessingErrored)

        app = QApplication(sys.argv)
        app.setStyle(QStyleFactory.create('Fusion'))
        app.setWindowIcon(QIcon(self.getIcon()))

        self.model = LeadLossModel(self.signals)
        self.view = LeadLossView(self, self.getTitle(), config.VERSION)

        #self.cheatLoad()

        self.view.showMaximized()
        sys.exit(app.exec_())


    #############
    ## Actions ##
    #############

    def importCSV(self):
        self.inputFile = self.view.getInputFile()
        if not self.inputFile:
            return

        Settings.setCurrentFile(self.inputFile)
        self.view.getImportSettings(callback=self._importCSVWithSettings)

    def _importCSVWithSettings(self, importSettings):
        if not importSettings:
            return
        Settings.update(importSettings)

        self.model.clearInputData()
        self._importCSV(self.inputFile, importSettings)

    def _importCSV(self, inputFile, importSettings):
        results = csvUtils.read_input(inputFile, importSettings)
        success = results is not None

        if not success:
            return

        self.model.loadInputData(inputFile, importSettings, *results)

    ################
    ## Processing ##
    ################

    def processSample(self, sample):
        defaultSettings = sample.calculationSettings
        if defaultSettings is None:
            defaultSettings = Settings.get(SettingsType.CALCULATION)
        samples = [sample]
        callback = lambda settings : self._processSamples(samples, settings)
        self.view.getCalculationSettings(samples, defaultSettings, callback)

    def processAllSamples(self):
        defaultSettings = Settings.get(SettingsType.CALCULATION)
        samples = self.model.samples
        callback = lambda settings : self._processSamples(samples, settings)
        self.view.getCalculationSettings(samples, defaultSettings, callback)

    def _processSamples(self, samples, settings):
        if not settings:
            return

        Settings.update(settings)
        for sample in samples:
            sample.clearCalculation()

        clonedSamples = []
        for sample in samples:
            sample.startCalculation(settings)
            clonedSamples.append(sample.createProcessingCopy())

        self.worker = AsyncTask(self.processingSignals, self.model.getProcessingFunction(), clonedSamples)
        self.worker.start()
        self.signals.processingStarted.emit()

    def cancelProcessing(self):
        if self.worker is not None:
            self.worker.halt()

    ############
    ## Events ##
    ############

    def onProcessingNewTask(self, taskDescription):
        self.signals.taskStarted.emit(taskDescription)

    def onProcessingProgress(self, progressArgs):
        type = progressArgs[0]
        progress = progressArgs[1]

        self.signals.taskProgress.emit(progress)
        if type == ProgressType.CONCORDANCE and progress == 1.0:
            sampleName, concordantAges, discordances = progressArgs[2:]
            self.model.updateConcordance(sampleName, concordantAges, discordances)
        if type == ProgressType.SAMPLING:
            sampleName, run = progressArgs[2:]
            self.model.addMonteCarloRun(sampleName, run)
        if type == ProgressType.OPTIMAL:
            sampleName, args = progressArgs[2:]
            self.model.setOptimalAge(sampleName, args)

    def onProcessingCancelled(self):
        self.signals.taskComplete.emit(False, "Cancelled processing of data")
        self.signals.processingFinished.emit()

    def onProcessingErrored(self, exception):
        self.signals.taskComplete.emit(False, "Error whilst processing data")
        message = exception.__class__.__name__ + ": " + str(exception)
        QMessageBox.critical(None, "Error", "An error occurred during processing: \n\n" + message)
        self.signals.processingFinished.emit()

    def onProcessingCompleted(self, output):
        self.signals.taskComplete.emit(True, "Processing complete")
        self.signals.processingFinished.emit()

    ############
    ## Export ##
    ############

    def exportCSV(self, headers, rows):
        outputFile = self.view.getOutputFile()
        if outputFile:
            csvUtils.write_output(headers, rows, outputFile)
        self.signals.taskComplete.emit(True, "Export complete")

    ###########
    ## Other ##
    ###########

    def showHelp(self):
        dialog = LeadLossHelpDialog()
        dialog.exec_()

    def selectSamples(self, selectedIndices):
        selectedSamples = [sample for sample in self.model.samples if sample.id in selectedIndices]
        unselectedSamples = [sample for sample in self.model.samples if sample.id not in selectedIndices]
        if not selectedSamples:
            selectedSamples = self.model.samples
            unselectedSamples = []

        self.signals.samplesSelected.emit(selectedSamples, unselectedSamples)

    def selectAgeToCompare(self, requestedRimAge):
        if requestedRimAge is None:
            self.signals.ageDeselected.emit()
            return

        actualRimAge, dValue, pValue, reconstructedAges = self.model.getNearestSampledAge(requestedRimAge)
        self.signals.ageSelected.emit(actualRimAge, reconstructedAges)

    def cheatLoad(self):
        try:
            #inputFile = "/home/matthew/Downloads/Haughton.csv"
            inputFile = "/home/matthew/Code/concordia-applications/LeadLoss/tests/leadLossTest_with_errors.csv"
            Settings.setCurrentFile(inputFile)
            settings = Settings.get(SettingsType.IMPORT)
            self._importCSV(inputFile, settings)
        except:
            print(traceback.format_exc(), file=sys.stderr)


if __name__ == '__main__':
    app = LeadLossApplication()
