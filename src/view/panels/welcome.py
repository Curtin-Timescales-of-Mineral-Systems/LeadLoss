from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QPushButton, QStyle, QLabel, QSizePolicy

from utils.ui.icons import Icons


class WelcomePanel(QGroupBox):

    def __init__(self, controller):
        super().__init__()

        text = QLabel("Welcome to Curtin's Concordia Pb-loss application. Import a CSV file to get started.")

        importButton = QPushButton("   Import CSV")
        importButton.clicked.connect(controller.importCSV)
        importButton.setIcon(Icons.importCSV())
        importButton.setMinimumWidth(200)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.addStretch(1)
        layout.addWidget(text)
        layout.addWidget(importButton, 0, Qt.AlignCenter)
        layout.addStretch(1)

        self.setLayout(layout)

