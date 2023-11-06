import unittest
from PyQt5.QtCore import QObject
from PyQt5.QtTest import QSignalSpy
from controller.signals import Signals

class TestProcessingSignals(unittest.TestCase):
    def test_exportAllAgesClicked(self):
        # Create a ProcessingSignals instance
        signals = Signals()

        # Create a QSignalSpy to track the exportAllAgesClicked signal
        spy = QSignalSpy(signals.exportAllAgesClicked)

        # Emit the exportAllAgesClicked signal
        signals.exportAllAgesClicked.emit()

        # Check that the signal was emitted once
        self.assertEqual(len(spy), 1)

if __name__ == '__main__':
    unittest.main()