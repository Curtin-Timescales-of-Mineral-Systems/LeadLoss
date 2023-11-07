from PyQt5.QtWidgets import QLineEdit, QWidget, QLabel, QHBoxLayout

from utils import stringUtils, config


class NumericInput(QWidget):

    def __init__(self, defaultValue, validation=None, unit=None, parseFn=None, stringifyFn=None):
        super().__init__()

        self.parseFn = parseFn if parseFn is not None else (lambda x: x)
        self.stringifyFn = stringifyFn if stringifyFn is not None else str

        self.lineEdit = QLineEdit()

        layout = QHBoxLayout()
        layout.addWidget(self.lineEdit)
        if unit:
            layout.addWidget(QLabel(unit))
        layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(layout)
        self.setValue(defaultValue)

        if validation:
            self.lineEdit.textChanged.connect(validation)

    def value(self):
        text = self.lineEdit.text()
        if text == "":
            return None
        return self.parseFn(text)

    def setValue(self, value):
        if value is None:
            text = ""
        else:
            text = self.stringifyFn(value)
        self.lineEdit.setText(text)

    def setReadOnly(self, readOnly):
        self.lineEdit.setReadOnly(readOnly)

class IntInput(NumericInput):
    def __init__(self, defaultValue=0, validation=None, unit=None, parseFn=lambda x: int(float(x)), stringifyFn=str):
        super().__init__(defaultValue, validation, unit, parseFn=parseFn, stringifyFn=stringifyFn)


class FloatInput(NumericInput):
    def __init__(self, defaultValue=0.0, validation=None, unit=None, parseFn=float, stringifyFn=str, sf=None):
        if sf is not None:
            newStringifyFn = lambda x : str(stringUtils.round_to_sf(x))
        else:
            newStringifyFn = stringifyFn
        super().__init__(defaultValue, validation, unit, parseFn=parseFn, stringifyFn=newStringifyFn)


class AgeInput(FloatInput):
    def __init__(self, defaultValue=0.0, validation=None, sf=config.DISPLAY_SF):
        def parseFn(x):
            if x == "":
                return None
            return (10 ** 6) * float(x)

        def stringifyFn(x):
            if x is None:
                return ""
            v = x / (10 ** 6)
            if sf is not None:
                v = stringUtils.round_to_sf(v, sf)
            return str(v)

        super().__init__(defaultValue, validation, unit="Ma", parseFn=parseFn, stringifyFn=stringifyFn, sf=None)


class PercentageInput(FloatInput):
    def __init__(self, defaultValue=0.0, validation=None):
        parseFn = lambda x: float(x) / 100
        stringifyFn = lambda x: str(x * 100)
        super().__init__(defaultValue, validation, unit="%", parseFn=parseFn, stringifyFn=stringifyFn)
