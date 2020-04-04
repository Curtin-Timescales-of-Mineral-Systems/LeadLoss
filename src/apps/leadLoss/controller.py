import sys
import traceback

from apps.leadLoss.view.helpDialog import LeadLossHelpDialog
from apps.leadLoss.process.processing import ProgressType
from utils.settings import Settings
from apps.abstract.controller import AbstractTabController
from apps.leadLoss.model.model import LeadLossModel
from apps.leadLoss.view.view import LeadLossView
from apps.type import ApplicationType, SettingsType


class LeadLossTabController(AbstractTabController):

    def __init__(self):
        super().__init__(ApplicationType.LEAD_LOSS)
        self.name = "Pb loss"
        self.model = LeadLossModel(self.signals)
        self.view = LeadLossView(self)

        self.cheatLoad()

    ################
    ## Processing ##
    ################

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
        #self.model.updateRow(i, row)
        #self.view.onProcessingProgress(progress, i, row)

    def selectAgeToCompare(self, age):
        self.model.selectAgeToCompare(age)

    ###########
    ## Other ##
    ###########

    def showHelp(self):
        dialog = LeadLossHelpDialog()
        dialog.exec_()

    def cheatLoad(self):
        try:
            inputFile = "../tests/leadLossTest2.csv"
            self._importCSV(inputFile, Settings.get(ApplicationType.LEAD_LOSS, SettingsType.IMPORT))
        except:
            print(traceback.format_exc(), file=sys.stderr)