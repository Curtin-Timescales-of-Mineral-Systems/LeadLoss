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
            "If a row in the imported data is invalid then it will be highlighted in <font color='red'>RED</font>." \
            "<br>" \
            "Symbols are not supported in column headings (e.g., ±, σ). Use only English alphabetic characters or numerals."

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
            "for a variety of different times of radiogenic-Pb loss. These distributions are then compared against the " \
            "distribution of concordant ages using the chosen statistic. A score for that age between 0.0 and 1.0 is " \
            "then calculated by combining the statistic with a penalty factor that is linear in the number of the " \
            "invalid reconstructed ages (i.e. when the line through the proposed Pb-loss age and the discordant measurement " \
            "does not intercept the concordia curve). " \
            "The optimal time for radiogenic-Pb loss is then chosen to maximise this score." \
            "<br><br>" + \
            self._getStandardProcessingHelp()

    def getOutputsHelpText(self):
        return \
            "The concordant and discordant points will be highlighted on the main concordia plot. " \
            "The score as a function of the Pb-loss age age will be plotted on the bottom left, " \
            "and the optimal age found will be highlighted in <font color='purple'>PURPLE</font>. " \
            "<br><br>" \
            "You can also explore each individual Monte Carlo sample. In this tab, the top plot contains the randomly" \
            "sampled points using the error associated with each point. The bottom left plot " \
            "shows the score as a function of Pb-loss age, and the bottom right plot shows the corresponding " \
            "CDFs (cumulative density function) for the concordant and reconstructed discordant distributions." \
            "<br><br>" \
            "Moving the mouse over the statistic plot will allow you see the distribution for the selected Pb-loss age." \
            "<br><br>"  + \
            self._getStandardOutputsHelp()
