import multiprocessing
import sys
import traceback

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox, QApplication, QStyleFactory
from process.cdc_config import PER_RUN_PROM_FRAC, PER_RUN_MIN_DIST, PER_RUN_MIN_WIDTH

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

from PyQt5.QtGui import QFont, QPalette, QColor
from utils import resourceUtils

from model.settings.calculation import ConcordiaMode

class LeadLossApplication:
    """
    Main Qt application entry point for the Pb-loss modelling tool.

    - Handles CSV import/export
    - Launches CDC + ensemble processing in a worker process
    - Bridges worker progress to the Qt GUI via Signals/ProcessingSignals
    """
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
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor("#DCE8FF"))
        pal.setColor(QPalette.WindowText, QColor("#111827"))
        pal.setColor(QPalette.Base, QColor("#FFFFFF"))
        pal.setColor(QPalette.AlternateBase, QColor("#FBFCFF"))
        pal.setColor(QPalette.Text, QColor("#111827"))
        pal.setColor(QPalette.Button, QColor("#FBFCFF"))
        pal.setColor(QPalette.ButtonText, QColor("#111827"))
        pal.setColor(QPalette.Highlight, QColor("#D0DCFF"))
        pal.setColor(QPalette.HighlightedText, QColor("#0B1F3A"))
        app.setPalette(pal)

        app.setWindowIcon(QIcon(self.get_icon()))
        app.setFont(QFont("Helvetica Neue", 11))

        try:
            with open(resourceUtils.getResourcePath("theme.qss"), "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())
        except Exception as e:
            print("Theme not loaded:", e)

        self.model = LeadLossModel(self.signals)
        self.view = LeadLossView(self, self.get_title(), config.VERSION)

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
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(None, "Processing", "Processing is already running.")
            return

        if not settings:
            return

        Settings.update(settings)

        clonedSamples = []
        for sample in samples:
            sample.startCalculation(settings)
            sample.clearCalculation()
            clonedSamples.append(sample.createProcessingCopy())

        worker = AsyncTask(self.processing_signals, self.model.getProcessingFunction(), clonedSamples)

        worker.finished.connect(lambda w=worker: self._onWorkerFinished(w))
        worker.finished.connect(worker.deleteLater)

        self.worker = worker
        self.signals.processingStarted.emit()
        worker.start()

    def _onWorkerFinished(self, finished_worker):
        if finished_worker is self.worker:
            self.worker = None


    def cancelProcessing(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.halt()

    ############
    ## Events ##
    ############

    def onProcessingNewTask(self, taskDescription):
        self.signals.taskStarted.emit(taskDescription)


    def onProcessingProgress(self, progressArgs):
        # Basic shape guard
        if not isinstance(progressArgs, tuple) or len(progressArgs) < 2:
            return

        kind = progressArgs[0]
        progress = progressArgs[1]

        if not isinstance(kind, (ProgressType, str)):
            return

        if not isinstance(progress, (int, float)):
            return

        self.signals.taskProgress.emit(float(progress))

        if kind == "summedKS":
            sampleName, payload = progressArgs[2:]
            self.model.emitSummedKS(sampleName, payload)
            return

        if kind == ProgressType.CONCORDANCE and float(progress) == 1.0:
            payload = list(progressArgs[2:])
            sampleName       = payload[0]
            concordantAges   = payload[1]
            discordances     = payload[2]
            reverse_flags    = payload[3] if len(payload) > 3 else None
            self.model.updateConcordance(sampleName, concordantAges, discordances, reverse_flags)
            return

        if kind == ProgressType.SAMPLING:
            sampleName, run = progressArgs[2:]
            self.model.addMonteCarloRun(sampleName, run)
            return

        if kind == ProgressType.OPTIMAL:
            sampleName, args = progressArgs[2:]
            self.model.setOptimalAge(sampleName, args)
            return

    def onProcessingCancelled(self):
        self.signals.taskComplete.emit(False, "Cancelled processing of data")
        self.signals.processingFinished.emit()

    def onProcessingErrored(self, payload):
        self.signals.taskComplete.emit(False, "Error whilst processing data")
        if isinstance(payload, (tuple, list)) and payload:
            payload = payload[0]
        message = payload if isinstance(payload, str) else (payload.__class__.__name__ + ": " + str(payload))
        QMessageBox.critical(None, "Error", "An error occurred during processing:\n\n" + message)
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
        samples = self.model.samples
        output_file = self.view.getOutputFile()
        if not output_file:
            self.signals.taskComplete.emit(False, "Export canceled by user")
            return

        distribution = []
        for sample in samples:
            for mode in (ConcordiaMode.TW, ConcordiaMode.WETHERILL):
                runs = sample.getMonteCarloRuns(mode)
                if not runs:
                    continue
                mode_str = "WETHERILL" if mode == ConcordiaMode.WETHERILL else "TW"
                distribution.extend([[sample.name, run.run_number, mode_str, float(run.optimal_pb_loss_age) / 1e6] for run in runs])


        write_monte_carlo_output(distribution, output_file, write_headers=True)

        self.signals.taskComplete.emit(True, "Export Monte Carlo runs complete")

    def exportPerRunPeaks(self):
        from process.ensemble import per_run_peaks
        import numpy as np

        output_file = self.view.getOutputFile()
        if not output_file:
            self.signals.taskComplete.emit(False, "Export canceled by user")
            return

        headers = ["Sample", "Mode", "Run #", "Surface", "Peak #", "Peak age (Ma)", "Weight"]
        rows = []

        for sample in self.model.samples:
            for mode in (ConcordiaMode.TW, ConcordiaMode.WETHERILL):
                runs = sample.getMonteCarloRuns(mode)
                if not runs:
                    continue

                for run in runs:
                    items = sorted(run.statistics_by_pb_loss_age.items(), key=lambda kv: kv[0])
                    if not items:
                        continue

                    ages_ma = np.asarray([a for a, _ in items], float) / 1e6
                    D_raw   = np.asarray([st.test_statistics[0] for _, st in items], float)
                    D_pen   = np.asarray([st.score for _, st in items], float)

                    S_raw = 1.0 - D_raw
                    S_pen = 1.0 - D_pen

                    pk_raw = per_run_peaks(
                        ages_ma, S_raw,
                        prom_frac=PER_RUN_PROM_FRAC,
                        min_dist=PER_RUN_MIN_DIST,
                        min_width_nodes=PER_RUN_MIN_WIDTH,
                        require_full_prom=False, max_keep=None, fallback_global_max=False
                    )
                    pk_pen = per_run_peaks(
                        ages_ma, S_pen,
                        prom_frac=PER_RUN_PROM_FRAC,
                        min_dist=PER_RUN_MIN_DIST,
                        min_width_nodes=PER_RUN_MIN_WIDTH,
                        require_full_prom=False, max_keep=None, fallback_global_max=False
                    )


                    w_raw = 1.0 / len(pk_raw) if len(pk_raw) else 0.0
                    w_pen = 1.0 / len(pk_pen) if len(pk_pen) else 0.0

                    mode_str = "WETHERILL" if mode == ConcordiaMode.WETHERILL else "TW"

                    for i, p in enumerate(pk_raw, 1):
                        rows.append([sample.name, mode_str, run.run_number, "RAW", i, float(p), w_raw])
                    for i, p in enumerate(pk_pen, 1):
                        rows.append([sample.name, mode_str, run.run_number, "PEN", i, float(p), w_pen])

        csvUtils.write_output(headers, rows, output_file)
        self.signals.taskComplete.emit(True, "Export Monte Carlo peak picks complete")

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
            inputFile = "/home/matthew/Code/concordia-applications/LeadLoss/tests/leadLossTest_with_errors.csv"
            Settings.setCurrentFile(inputFile)
            settings = Settings.get(SettingsType.IMPORT)
            self._importCSV(inputFile, settings)
        except:
            print(traceback.format_exc(), file=sys.stderr)


if __name__ == '__main__':
    app = LeadLossApplication()
