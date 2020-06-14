from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from utils.ui import uiUtils


class AbstractSettingsDialog(QDialog):

    def __init__(self, defaultSettings):
        super().__init__()
        self.__aligned_form_layouts = []
        self.setModal(True)
        self.defaultSettings = defaultSettings
        self.initUI()
        self.setSizeGripEnabled(False)
        self._validate()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.layout.setSizeConstraint(QLayout.SetFixedSize)
        self.layout.addWidget(self.initMainSettings())
        self.layout.addWidget(self.initErrorAndWarningLabels())
        self.layout.addWidget(self.initButtons())
        self.setLayout(self.layout)

    ###############
    ## UI layout ##
    ###############

    def initButtons(self):
        self.okButton = QPushButton("OK")
        self.okButton.clicked.connect(self.accept)
        self.okButton.setEnabled(False)

        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.reject)

        layout = QHBoxLayout()
        layout.setSpacing(uiUtils.FORM_HORIZONTAL_SPACING)
        layout.addWidget(self.okButton)
        layout.addWidget(self.cancelButton)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def initErrorAndWarningLabels(self):
        self.errorLabel = QLabel()
        self.errorLabel.setStyleSheet("QLabel { color : red; }")
        self.warningLabel = QLabel()
        self.warningLabel.setStyleSheet("QLabel { color : orange; }")

        for label in (self.errorLabel, self.warningLabel):
            label.setVisible(False)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.errorLabel)
        layout.addWidget(self.warningLabel)

        self.messageWidget = QWidget()
        self.messageWidget.setLayout(layout)
        return self.messageWidget

    def _registerFormLayoutForAlignment(self, formLayout):
        self.__aligned_form_layouts.append(formLayout)

    def show(self):
        super().show()
        self._alignLabels()

    def _alignLabels(self):
        if not self.__aligned_form_layouts:
            return

        labels = []
        for formLayout in self.__aligned_form_layouts:
            labels.extend(formLayout.itemAt(i, QFormLayout.LabelRole).widget() for i in range(formLayout.rowCount()))

        width = max(uiUtils.getTextWidth(label.text()) for label in labels)
        for label in labels:
            label.setMinimumWidth(width)

    ################
    ## Validation ##
    ################

    def _validate(self):
        settings = self._createSettings()
        error = settings.validate()
        warning = self.getWarning(settings)

        self.okButton.setEnabled(error is None)
        self.messageWidget.setVisible(error is not None or warning is not None)
        self.errorLabel.setVisible(error is not None)
        self.warningLabel.setVisible(error is None and warning is not None)
        if error is not None:
            self.errorLabel.setText(error)
        elif warning is not None:
            self.warningLabel.setText(warning)
        self.settings = settings

        self.updateGeometry()

    def createLabelWithHelp(self, labelText, helpText):

        label = QLabel(labelText)

        icon = self.style().standardIcon(getattr(QStyle, "SP_MessageBoxQuestion"))
        pixmap = icon.pixmap(QSize(20, 20))
        iconLabel = QLabel()
        iconLabel.setPixmap(pixmap)
        iconLabel.setToolTip(helpText)

        layout = QHBoxLayout()
        layout.addWidget(iconLabel)
        layout.addWidget(label)
        layout.setContentsMargins(0,0,0,0)

        widget = QWidget()
        widget.setLayout(layout)
        return widget