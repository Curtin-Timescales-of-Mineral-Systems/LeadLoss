from PyQt5.QtWidgets import QWidget, QButtonGroup, QHBoxLayout, QRadioButton, QGridLayout


class RadioButtons(QWidget):

    def __init__(self, options, validation, default, rows=None, cols=None):
        super().__init__()

        self.options = options
        self.group = QButtonGroup()
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        for i, option in enumerate(options):
            button = QRadioButton(option)
            button.option = option
            button.setChecked(option == default)
            self.group.addButton(button, i)
            x, y = self._calculate_coordinates(i, rows, cols)
            layout.addWidget(button, x, y)

        self.group.buttonReleased.connect(validation)

        self.setLayout(layout)

    def _calculate_coordinates(self, i, rows, cols):
        if cols is None:
            return 0, i
        if rows is None:
            return i, 0
        return i // rows, i % cols

    def selection(self):
        return self.options[self.group.checkedId()]


class IntRadioButtonGroup(RadioButtons):

    def __init__(self, options, validation, default, *args, **kwargs):
        optionsStr = [str(v) for v in options]
        defaultStr = str(default)
        super().__init__(optionsStr, validation, defaultStr, *args, **kwargs)

    def selection(self):
        return int(super().selection())


class EnumRadioButtonGroup(RadioButtons):

    def __init__(self, options, validation, default, *args, **kwargs):
        self.enum_options = options
        optionsStr = [v.value for v in options]
        defaultStr = default.value
        super().__init__(optionsStr, validation, defaultStr, *args, **kwargs)

    def selection(self):
        selection = super().selection()
        return next(x for x in self.enum_options if x.value == selection)
