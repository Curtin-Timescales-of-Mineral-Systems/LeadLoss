from utils.csvUtils import ColumnReferenceType
from utils.ui.radioButtons import RadioButtons


class ColumnReferenceTypeInput(RadioButtons):

    def __init__(self, validation, default):
        super().__init__([e.value for e in ColumnReferenceType], validation, default.value)

    def selection(self):
        string = super().selection()
        return ColumnReferenceType(string)