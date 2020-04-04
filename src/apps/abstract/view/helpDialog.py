from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QTabWidget, QVBoxLayout, QWidget, QLabel

from utils import stringUtils, calculations


class AbstractHelpDialog(QDialog):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.getTitle() + " help")

        inputsLabel = QLabel(self.getInputsHelpText())
        inputsLabel.title = "Inputs"

        processingLabel = QLabel(self.getProcessingHelpText())
        processingLabel.title = "Processing"

        outputsLabel = QLabel(self.getOutputsHelpText())
        outputsLabel.title = "Outputs"

        tabWidget = QTabWidget()
        for label in [inputsLabel, processingLabel, outputsLabel]:
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