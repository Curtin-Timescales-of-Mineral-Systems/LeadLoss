from enum import Enum

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLineEdit, QWidget, QVBoxLayout, QHBoxLayout

from utils import stringUtils, csvUtils
from utils.csvUtils import ColumnReferenceType
from utils.ui import uiUtils


class ColumnReferenceInput(QWidget):
    width = 30

    def __init__(self, validation, referenceType, defaultValue):
        super().__init__()

        letterDefault = csvUtils.columnNumberToLetters(defaultValue, zeroIndexed=True)
        numberDefault = str(defaultValue + 1)

        self.numberWidget = QLineEdit(numberDefault)
        self.numberWidget.setFixedWidth(self.width)
        self.numberWidget.setAlignment(Qt.AlignCenter)
        self.numberWidget.textChanged.connect(validation)
        self.numberWidget.setVisible(referenceType is ColumnReferenceType.NUMBERS)
        uiUtils.attachValidator(self.numberWidget, csvUtils.COLUMN_REFERENCE_TYPE_REGEXES[ColumnReferenceType.NUMBERS])

        self.lettersWidget = QLineEdit(letterDefault)
        self.lettersWidget.setFixedWidth(self.width)
        self.lettersWidget.setAlignment(Qt.AlignCenter)
        self.lettersWidget.textChanged.connect(validation)
        self.lettersWidget.setVisible(referenceType is ColumnReferenceType.LETTERS)
        uiUtils.attachValidator(self.lettersWidget, csvUtils.COLUMN_REFERENCE_TYPE_REGEXES[ColumnReferenceType.LETTERS])

        layout = QHBoxLayout()
        layout.addWidget(self.numberWidget)
        layout.addWidget(self.lettersWidget)
        layout.addStretch()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.currentReferenceType = referenceType

    def changeColumnReferenceType(self, newReferenceType):
        if self.currentReferenceType == newReferenceType:
            return

        if self.currentReferenceType is ColumnReferenceType.LETTERS:
            oldText = self.lettersWidget.text()
            newText = str(csvUtils.columnLettersToNumber(oldText, zeroIndexed=False))
            self.numberWidget.setText(newText)
        else:
            oldText = self.numberWidget.text()
            newText = csvUtils.columnNumberToLetters(int(oldText), zeroIndexed=False)
            self.lettersWidget.setText(newText)

        self.lettersWidget.setVisible(newReferenceType is ColumnReferenceType.LETTERS)
        self.numberWidget.setVisible(newReferenceType is ColumnReferenceType.NUMBERS)

        self.currentReferenceType = newReferenceType

    def text(self):
        if self.currentReferenceType is ColumnReferenceType.LETTERS:
            text = self.lettersWidget.text()
            if not text:
                return None
            return csvUtils.columnLettersToNumber(text, zeroIndexed=True)

        if self.currentReferenceType is ColumnReferenceType.NUMBERS:
            text = self.numberWidget.text()
            if not text:
                return None
            return int(text) - 1

        raise Exception("Unknown ColumnReferenceType: " + str(self.currentReferenceType))
