from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QLabel, QTabWidget, QVBoxLayout, QWidget

from utils import config, stringUtils
from process import calculations


class LeadLossHelpDialog(QDialog):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.getTitle() + " help")

        introductionLabel = QLabel(self._getIntroductionHelp())
        introductionLabel.title = "Introduction"

        inputsLabel = QLabel(self.getInputsHelpText())
        inputsLabel.title = "Inputs"

        processingLabel = QLabel(self.getProcessingHelpText())
        processingLabel.title = "Processing"

        outputsLabel = QLabel(self.getOutputsHelpText())
        outputsLabel.title = "Outputs"

        tabWidget = QTabWidget()
        for label in [introductionLabel, inputsLabel, processingLabel, outputsLabel]:
            label.setTextFormat(Qt.RichText)
            label.setWordWrap(True)
            layout = QVBoxLayout()
            layout.addWidget(label, 0, Qt.AlignTop)
            widget = QWidget()
            widget.setLayout(layout)
            tabWidget.addTab(widget, label.title)

        layout = QVBoxLayout()
        layout.addWidget(tabWidget)

        self.setLayout(layout)

    def _getIntroductionHelp(self):
        return "Hugo K.H. Olierook, Christopher L. Kirkland, ???, " \
               "Matthew L. Daggitt, ??? " \
               "<b>PAPER TITLE</b>, 2020"

    def _getStandardInputHelp(self):
        return \
            "Data can be parsed from a range of csv file layouts by specifying which columns the required values are " \
            "in. Columns can be referred to either by using:" \
            "<ul>" \
            "  <li> numbers (1, 2, 3, ..., 26, 27, ...)" \
            "  <li> letters (A, B, C, ..., Z, AA, ...)" \
            "</ul>" \
            "Different uncertainty formats are also supported:" \
            "<ul>" \
            "  <li> percentage vs absolute" \
            "  <li> 1σ vs 2σ" \
            "</ul>" \
            "If a row in the imported data is invalid then it will be highlighted in <font color='red'>RED</font>."

    def _getStandardProcessingHelp(self):
        return \
            "Constants used:" \
            "<ul>" \
            "<li> ²³⁸U/²³⁵U ratio " + "&nbsp;" * 10 + " = " + stringUtils.getConstantStr(calculations.U238U235_RATIO) + \
            "<li> ²³⁸U decay constant = " + stringUtils.getConstantStr(calculations.U238_DECAY_CONSTANT) + \
            "<li> ²³⁵U decay constant = " + stringUtils.getConstantStr(calculations.U235_DECAY_CONSTANT) + \
            "<ul>"

    def _getStandardOutputsHelp(self):
        return \
            "The plot may be fully customised (markers, colours, scale etc.) using the " \
            "button in the toolbar at the bottom. The plot can also be saved to various image formats." \
            "<br><br>" \
            "When the calculated values are exported back to a CSV file, the values are appended to the end of the " \
            "columns of the original CSV file."

    def getTitle(self):
        return config.LEAD_LOSS_TITLE

    def getInputsHelpText(self):
        return \
            "Input data required: <ul>" \
            "<li> measured ²³⁸U/²⁰⁶Pb and ²⁰⁷Pb/²⁰⁶Pb ratios" \
            "<li> uncertainties for all of the above" \
            "</ul>" + \
            self._getStandardInputHelp()

    def getProcessingHelpText(self):
        return \
            "Processing the data will attempt to reconstruct the expected distribution of ages " \
            "for a variety of different times of radiogenic-Pb. These distributions are then compared against the " \
            "distribution of concordant ages using the chosen statistic. The optimal time for radiogenic-Pb loss " \
            "is then chosen to maximise this statistic." \
            "<br><br>" \
            "Concordant rows will be highlighted in <font color='green'>GREEN</font>." \
            "<br><br>" \
            "Discordant rows will be highlighted in <font color='orange'>ORANGE</font>." \
            "<br><br>" + \
            self._getStandardProcessingHelp()

    def getOutputsHelpText(self):
        return \
            "The concordant and discordant points will be highlighted on the main concordia plot." \
            "The value of the chosen statistic as a function of the rim age will be plotted on the bottom left, " \
            "and the optimal age found will be highlighted in <font color='purple'>PURPLE</font>. " \
            "The distribution of the concordant and reconstructed distribution will plotted on the bottom right." \
            "<br><br>" \
            "Moving the mouse over the statistic plot will allow you see the distribution for the selected rim age." \
            "<br><br>"  + \
            self._getStandardOutputsHelp()
