from PyQt5.QtWidgets import QLineEdit, QWidget, QLabel, QHBoxLayout


class NumericInput(QWidget):

    def __init__(self, defaultValue, validation=None, unit=None, parseFn=None, stringifyFn=None):
        super().__init__()

        self.parseFn = parseFn if parseFn is not None else (lambda x : x)
        self.stringifyFn = stringifyFn if stringifyFn is not None else str

        self.lineEdit = QLineEdit(self.stringifyFn(defaultValue))
        self.lineEdit.textChanged.connect(validation)

        layout = QHBoxLayout()
        layout.addWidget(self.lineEdit)
        if unit:
            layout.addWidget(QLabel(unit))
        layout.setContentsMargins(0,0,0,0)

        self.setLayout(layout)

    def value(self):
        return self.parseFn(self.lineEdit.text())

class IntInput(NumericInput):
    def __init__(self, defaultValue=0, validation=None, unit=None, parseFn=lambda x: int(float(x)), stringifyFn=str):
        super().__init__(defaultValue, validation, unit, parseFn=parseFn, stringifyFn=stringifyFn)

class FloatInput(NumericInput):
    def __init__(self, defaultValue=0.0, validation=None, unit=None, parseFn=float, stringifyFn=str):
        super().__init__(defaultValue, validation, unit, parseFn=parseFn, stringifyFn=stringifyFn)

class AgeInput(FloatInput):
    def __init__(self, defaultValue=0.0, validation=None):
        parseFn = lambda x: (10**6)*float(x)
        stringifyFn = lambda x: str(x/(10**6))
        super().__init__(defaultValue, validation, unit="Ma", parseFn=parseFn, stringifyFn=stringifyFn)

class PercentageInput(FloatInput):
    def __init__(self, defaultValue=0.0, validation=None):
        parseFn = lambda x: float(x)/100
        stringifyFn = lambda x: str(x*100)
        super().__init__(defaultValue, validation, unit="%", parseFn=parseFn, stringifyFn=stringifyFn)