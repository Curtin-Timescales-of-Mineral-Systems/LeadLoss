from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from utils.ui.statusBar import StatusBarWidget


class AbstractView(QWidget):

    def __init__(self, controller, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controller = controller
        self.initUI()

        self.controller.signals.csvImported.connect(self.onCSVImportFinished)

    def initUI(self):
        self.graphPanel = self.createGraphWidget()
        self.dataPanel = self.createDataWidget()
        self.statusBar = StatusBarWidget(self.controller.signals)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.dataPanel)
        splitter.addWidget(self.graphPanel)
        splitter.setSizes([10000, 10000])
        splitter.setContentsMargins(1, 1, 1, 1)

        layout = QVBoxLayout()
        layout.addWidget(splitter, 1)
        layout.addWidget(self.statusBar, 0)
        self.setLayout(layout)


    ########
    ## IO ##
    ########

    def getInputFile(self):
        return QFileDialog.getOpenFileName(
            caption='Open CSV file',
            directory='/home/matthew/Dropbox/Academia/Code/Python/UnmixConcordia/tests'
        )[0]
        #return '/home/matthew/Dropbox/Academia/Code/Python/UnmixConcordia/data/unmixTest.csv'

    def getOutputFile(self):
        return QFileDialog.getSaveFileName(
            caption='Save CSV file',
            directory='/home/matthew/Dropbox/Academia/Code/Python/UnmixConcordia/tests'
        )[0]

    def getSettings(self, settingsType, callback):
        settingsPopup = self.getSettingsDialog(settingsType)

        def outerCallback(result):
            if result == QDialog.Rejected:
                return None
            callback(settingsPopup.settings)

        settingsPopup.finished.connect(outerCallback)
        settingsPopup.show()

    ############
    ## Events ##
    ############

    def onCSVImportFinished(self, result, inputFile):
        if not result:
            self.endTask(False, "Failed to import CSV file")
            return

        self.dataPanel.afterSuccessfulCSVImport(inputFile)

    def onProcessingProgress(self, progress, *args):
        self.updateTask(progress)

    def onProcessingCompleted(self):
        self.endTask(True, "Successfully processed data")

    def onProcessingCancelled(self):
        pass

    def onProcessingErrored(self, exception):
        self.endTask(False, "Error during processing of data")
        QMessageBox.critical(None, "Error", "An error occurred during processing: \n\n" + exception.__class__.__name__ + ": " + str(exception))

    ##############
    ## Clean-up ##
    ##############

    def closeEvent(self, event):
        self.haltEvent.set()
