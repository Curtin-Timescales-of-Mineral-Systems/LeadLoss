import multiprocessing
import sys
import traceback

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox, QApplication, QStyleFactory

from utils.csvUtils import write_monte_carlo_output
from controller.signals import Signals, ProcessingSignals
from model.model import LeadLossModel
from model.settings.type import SettingsType
from process.processing import ProgressType
from utils import config, resourceUtils, csvUtils
from utils.asynchronous import AsyncTask
from utils.settings import Settings
from view.dialogs.help import LeadLossHelpDialog
from view.view import LeadLossView


class LeadLossApplication:

    @staticmethod
    def get_title():
        return config.LEAD_LOSS_TITLE

    @staticmethod
    def get_icon():
        return resourceUtils.getResourcePath("icon.png")

    @staticmethod
    def exception_hook(exctype, value, traceback):
        sys.__excepthook__(exctype, value, traceback)
        QMessageBox.critical(None, "Error", str(value))

    def __init__(self):
        # Necessary for building executable with Pyinstaller correctly on Windows
        # (see https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing)
        multiprocessing.freeze_support()

        # Reroute exceptions to display a message box to the user
        sys.excepthook = self.exception_hook

        self.worker = None
        self.signals = Signals()

        self.processing_signals = ProcessingSignals()
        self.processing_signals.processingNewTask.connect(self.onProcessingNewTask)
        self.processing_signals.processingProgress.connect(self.onProcessingProgress)
        self.processing_signals.processingCompleted.connect(self.onProcessingCompleted)
        self.processing_signals.processingCancelled.connect(self.onProcessingCancelled)
        self.processing_signals.processingErrored.connect(self.onProcessingErrored)
        self.processing_signals.processingSkipped.connect(self.onProcessingSkipped)


        app = QApplication(sys.argv)
        app.setStyle(QStyleFactory.create('Fusion'))
        app.setWindowIcon(QIcon(self.get_icon()))

        self.model = LeadLossModel(self.signals)
        self.view = LeadLossView(self, self.get_title(), config.VERSION)

        #self.cheatLoad()

        self.view.showMaximized()
        self.cancelProcessing()
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

        self.worker = AsyncTask(self.processing_signals, self.model.getProcessingFunction(), clonedSamples)
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

    def onProcessingSkipped(self, sample_name, skip_reason):
        # Find the sample by name
        sample = next((s for s in self.model.samples if s.name == sample_name), None)
        if sample:
            sample.setSkipReason(skip_reason)


    ############
    ## Export ##
    ############

    def exportCSV(self, headers, rows, is_monte_carlo=False):
        outputFile = self.view.getOutputFile()
        if outputFile:
            csvUtils.write_output(headers, rows, outputFile)
        self.signals.taskComplete.emit(True, "Export complete")

    def exportMonteCarloRuns(self):

        # Get all samples
        samples = self.model.samples

        # Get the output file
        output_file = self.view.getOutputFile()

        # Check if the user canceled the dialog
        if not output_file:
            self.signals.taskComplete.emit(False, "Export canceled by user")
            return  # Exit the function early

        # Initialize an empty list for the distribution
        distribution = []

        for sample in samples:
            # Get the Monte Carlo runs from the sample
            monte_carlo_runs = sample.getMonteCarloRuns()

            # Convert each MonteCarloRun object to a list
            distribution.extend([run.toList() for run in monte_carlo_runs])

        # Write the Monte Carlo runs to the output file
        write_monte_carlo_output(distribution, output_file, write_headers=True)

        self.signals.taskComplete.emit(True, "Export Monte Carlo runs complete")

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
