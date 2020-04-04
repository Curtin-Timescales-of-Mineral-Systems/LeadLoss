from PyQt5.QtCore import Qt
from PyQt5.QtGui import QRegExpValidator, QPalette, QFont, QFontMetrics
from PyQt5.QtWidgets import QButtonGroup, QHBoxLayout, QRadioButton

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

def getTextWidth(text):
    fontMetrics = QFontMetrics(QFont())
    return fontMetrics.horizontalAdvance(text)