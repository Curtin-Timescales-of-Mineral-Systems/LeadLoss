import unittest
from PyQt5.QtWidgets import QApplication
from unittest.mock import patch
from view.view import LeadLossView
from unittest.mock import Mock, MagicMock
from PyQt5.QtWidgets import QFileDialog

from PyQt5.QtCore import pyqtSignal

class MockController:
    def __init__(self):
        self.signals = Mock()
        self.signals.inputDataLoaded = pyqtSignal(str, list)

    def importCSV(self):
        pass

    def showHelp(self):
        pass

class TestLeadLossView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])  # Create a QApplication

    @patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName')
    def test_getAllAgesOutputFile(self, mock_getSaveFileName):
        # Arrange
        mock_getSaveFileName.return_value = ('/path/to/file.csv', 'All Files (*)')
        controller = MockController()  # Create a mock controller
        view = LeadLossView(controller, 'Test', '1.0')  # Use the mock controller

        # Act
        result = view.getAllAgesOutputFile()

        # Assert
        mock_getSaveFileName.assert_called_once_with(
            caption='Save All Ages CSV file',
            directory='/home/matthew/Dropbox/Academia/Code/Python/UnmixConcordia/tests',
            options=QFileDialog.DontUseNativeDialog
        )
        self.assertEqual(result, '/path/to/file.csv')

if __name__ == '__main__':
    unittest.main()