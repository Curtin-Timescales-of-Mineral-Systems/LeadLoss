from PyQt5.QtCore import Qt
from PyQt5.QtGui import QRegExpValidator, QPalette, QFont, QFontMetrics, QPixmap
from PyQt5.QtWidgets import QButtonGroup, QHBoxLayout, QRadioButton, QLabel, QWidget

FORM_HORIZONTAL_SPACING = 15


def attachValidator(widget, regex):
    widget.setValidator(None)
    validator = QRegExpValidator(regex)
    widget.setValidator(validator)


def colour(widget, color):
    palette = QPalette()
    palette.setColor(QPalette.Window, color)
    widget.setAutoFillBackground(True)
    widget.setPalette(palette)


def retainSizeWhenHidden(widget):
    policy = widget.sizePolicy()
    policy.setRetainSizeWhenHidden(True)
    widget.setSizePolicy(policy)


def clearChildren(layout):
    for i in reversed(range(layout.count())):
        layout.itemAt(i).widget().setParent(None)


def getTextWidth(text):
    fontMetrics = QFontMetrics(QFont())
    return fontMetrics.horizontalAdvance(text)


def createIconWithLabel(icon, text):
    iconLabel = QLabel()
    iconLabel.setPixmap(icon.pixmap(16,16))
    textLabel = QLabel(text)
    textLabel.setAlignment(Qt.AlignLeft)

    layout = QHBoxLayout()
    layout.addWidget(iconLabel)
    layout.addWidget(textLabel)
    layout.addStretch()

    widget = QWidget()
    widget.setLayout(layout)
    widget.setContentsMargins(0,0,0,0)
    return widget


def createNoDataWidget(sampleName, error_message=None):
    if error_message:
        label = QLabel(error_message)
    elif sampleName:
        label = QLabel("Sample '" + sampleName + "' has not yet been processed...")
    else:
        label = QLabel("The sample has not yet been processed...")
    label.setAlignment(Qt.AlignCenter)
    return label