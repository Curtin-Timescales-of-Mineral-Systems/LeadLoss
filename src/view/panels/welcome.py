from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPixmap, QColor
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFrame

from utils import resourceUtils
from utils.ui.icons import Icons


class WelcomePanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.setObjectName("WelcomePanel")

        self._bg = QPixmap(resourceUtils.getResourcePath("welcome_bg_4.png"))

        self._overlay = QColor(220, 232, 255, 120)

        title = QLabel("Welcome to Curtin's Concordia Pb-loss application")
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        title.setStyleSheet("font-size: 14pt; font-weight: 800; color: #0B1F3A;")

        subtitle = QLabel("Import a CSV file to get started.")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #374151;")

        importButton = QPushButton("   Import CSV")
        importButton.setObjectName("PrimaryButton") 
        importButton.clicked.connect(controller.importCSV)
        importButton.setIcon(Icons.importCSV())
        importButton.setMinimumWidth(220)

        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(12)
        card_layout.setContentsMargins(18, 18, 18, 18)
        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)
        card_layout.addWidget(importButton, 0, Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(card, 0, Qt.AlignCenter)
        layout.addStretch(1)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if not self._bg.isNull():
            scaled = self._bg.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = (scaled.width() - self.width()) // 2
            y = (scaled.height() - self.height()) // 2
            painter.drawPixmap(-x, -y, scaled)

            painter.fillRect(self.rect(), self._overlay)
        else:
            painter.fillRect(self.rect(), QColor(220, 232, 255))
