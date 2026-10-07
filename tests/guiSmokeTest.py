import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QPushButton

from controller.signals import Signals
from model.settings.imports import LeadLossImportSettings
from model.settings.ratio import ConcordiaRatioSpace, ConcordiaSpaceSelection
from view.dialogs.settings.imports import LeadLossImportSettingsDialog
from view.panels.summary.data import SummaryDataPanel
from view.view import LeadLossView


class _Controller:
    def __init__(self):
        self.signals = Signals()

    def importCSV(self):
        pass

    def showHelp(self):
        pass


class GUISmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_visual_theme_and_main_window_construct(self):
        theme = Path(__file__).resolve().parents[1] / "resources" / "theme.qss"
        mac_icon = Path(__file__).resolve().parents[1] / "resources" / "icon.icns"
        self.assertTrue(theme.is_file())
        self.assertTrue(mac_icon.is_file())
        self.assertGreater(mac_icon.stat().st_size, 1000)
        theme_text = theme.read_text(encoding="utf-8")
        self.assertIn("QGroupBox", theme_text)
        self.app.setStyleSheet(theme_text)

        view = LeadLossView(_Controller(), "Pb-loss", "test")
        self.assertEqual(view.bottomPanel.objectName(), "Footer")
        self.assertEqual(view.welcomePanel.objectName(), "WelcomePanel")
        primary_buttons = [
            button for button in view.welcomePanel.findChildren(QPushButton)
            if button.objectName() == "PrimaryButton"
        ]
        self.assertEqual(len(primary_buttons), 1)
        view.resize(1220, 700)
        view.show()
        self.app.processEvents()
        rendered = view.grab()
        self.assertFalse(rendered.isNull())
        self.assertGreater(rendered.width(), 1000)
        self.assertGreater(rendered.height(), 500)
        view.close()

    def test_native_wetherill_import_dialog_constructs(self):
        settings = LeadLossImportSettings()
        settings.inputRatioSpace = ConcordiaRatioSpace.WETHERILL
        settings.displayRatioSpace = ConcordiaSpaceSelection.WETHERILL
        settings.rhoColumn = 5

        dialog = LeadLossImportSettingsDialog(settings)
        self.assertEqual(
            dialog._ratioSpaceWidget.getInputRatioSpace(),
            ConcordiaRatioSpace.WETHERILL,
        )
        self.assertEqual(
            dialog._ratioSpaceWidget.getDisplayRatioSpace(),
            ConcordiaSpaceSelection.WETHERILL,
        )
        self.assertEqual(dialog._ratioSpaceWidget.getRhoColumn(), 5)
        self.assertFalse(dialog._ratioSpaceWidget._rhoColumn.isHidden())
        self.assertIn("²⁰⁷Pb/²³⁵U", dialog._uPbWidget.title())
        self.assertIn("²⁰⁶Pb/²³⁸U", dialog._pbPbWidget.title())
        dialog.close()

    def test_ensemble_table_uses_published_support_names(self):
        panel = SummaryDataPanel(_Controller(), [])
        headers = [
            panel.catalogueTable.horizontalHeaderItem(index).text().replace("\n", " ")
            for index in range(panel.catalogueTable.columnCount())
        ]
        self.assertIn("Direct support (%)", headers)
        self.assertIn("Winner support (%)", headers)
        panel.close()


if __name__ == "__main__":
    unittest.main()
