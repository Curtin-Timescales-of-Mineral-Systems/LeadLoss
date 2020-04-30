from PyQt5.QtWidgets import QLineEdit, QWidget, QLabel, QHBoxLayout

from utils import stringUtils


class NumericInput(QWidget):

    def __init__(self, defaultValue, validation=None, unit=None, parseFn=None, stringifyFn=None):
        super().__init__()

        self.parseFn = parseFn if parseFn is not None else (lambda x: x)
        self.stringifyFn = stringifyFn if stringifyFn is not None else str

        self.lineEdit = QLineEdit(self.stringifyFn(defaultValue))
        if validation:
            self.lineEdit.textChanged.connect(validation)

        layout = QHBoxLayout()
        layout.addWidget(self.lineEdit)
        if unit:
            layout.addWidget(QLabel(unit))
        layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(layout)

    def value(self):
        return self.parseFn(self.lineEdit.text())

    def setValue(self, value):
        self.lineEdit.setText(self.stringifyFn(value))

    def setReadOnly(self, readOnly):
        self.lineEdit.setReadOnly(readOnly)

class IntInput(NumericInput):
    def __init__(self, defaultValue=0, validation=None, unit=None, parseFn=lambda x: int(float(x)), stringifyFn=str):
        super().__init__(defaultValue, validation, unit, parseFn=parseFn, stringifyFn=stringifyFn)


class FloatInput(NumericInput):
    def __init__(self, defaultValue=0.0, validation=None, unit=None, parseFn=float, stringifyFn=str, sf=None):
        if sf is not None:
            newStringifyFn = lambda x : stringUtils.round_to_sf(x)
        else:
            newStringifyFn = stringifyFn
        super().__init__(defaultValue, validation, unit, parseFn=parseFn, stringifyFn=stringifyFn)


class AgeInput(FloatInput):
    def __init__(self, defaultValue=0.0, validation=None, sf=None):
        def parseFn(x):
            if x is "":
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
