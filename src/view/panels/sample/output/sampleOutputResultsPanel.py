from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QWidget, QLabel, QFormLayout, QHBoxLayout, QAbstractItemView,
    QPushButton, QFileDialog, QApplication, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
import csv
import io

from utils import config
from utils.ui.numericInput import FloatInput, AgeInput, IntInput  # IntInput kept in case used elsewhere


class SampleOutputResultsPanel(QGroupBox):
    peakRowSelected = pyqtSignal(int)

    def __init__(self, sample):
        super().__init__("Result")
        self.sample = sample

        # Root layout
        self.rootLayout = QVBoxLayout()
        self.setLayout(self.rootLayout)

        # ----- Summary form -----
        self.optimalAge = AgeInput(defaultValue=None);         self.optimalAge.setReadOnly(True)
        self.optimalAgeLower = AgeInput(defaultValue=None);    self.optimalAgeLower.setReadOnly(True)
        self.optimalAgeUpper = AgeInput(defaultValue=None);    self.optimalAgeUpper.setReadOnly(True)

        boundsLayout = QHBoxLayout()
        boundsLayout.addWidget(self.optimalAgeLower)
        boundsLayout.addWidget(QLabel("-"))
        boundsLayout.addWidget(self.optimalAgeUpper)

        self.dValue = FloatInput(defaultValue=None, sf=config.DISPLAY_SF); self.dValue.setReadOnly(True)
        self.pValue = FloatInput(defaultValue=None, sf=config.DISPLAY_SF); self.pValue.setReadOnly(True)
        self.invalidAges = FloatInput(defaultValue=None);                  self.invalidAges.setReadOnly(True)
        self.score = FloatInput(defaultValue=None, sf=config.DISPLAY_SF);  self.score.setReadOnly(True)

        formHost = QWidget()
        form = QFormLayout(formHost)
        form.addRow("Optimal Pb-loss age", self.optimalAge)
        form.addRow("95% confidence interval", boundsLayout)
        form.addRow("Mean D value (KS test)", self.dValue)
        form.addRow("Mean p value (KS test)", self.pValue)
        form.addRow("Mean # of invalid ages", self.invalidAges)
        form.addRow("Mean score", self.score)
        self.rootLayout.addWidget(formHost)

        # ----- Peak catalogue group -----
        self.catBox = QGroupBox("Peak catalogue (winner votes)")
        catLayout = QVBoxLayout(self.catBox)

        self.catTable = QTableWidget(0, 4)
        self.catTable.setHorizontalHeaderLabels(["#", "Age (Ma)", "95% CI (Ma)", "Support"])
        self.catTable.verticalHeader().setVisible(False)
        self.catTable.setSelectionMode(QAbstractItemView.SingleSelection)
        self.catTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.catTable.itemSelectionChanged.connect(self._onCatalogueSelectionChanged)
        self.catTable.setEditTriggers(QTableWidget.NoEditTriggers)
        self.catTable.setAlternatingRowColors(True)

        catLayout.addWidget(self.catTable)
        self.rootLayout.addWidget(self.catBox)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.catBox.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Reserve some height so the group doesn’t collapse when empty
        self.catTable.setMinimumHeight(140)
        self._setCatalogueVisible(False)

        btnRow = QHBoxLayout()
        btnRow.addStretch(1)
        self.copyBtn = QPushButton("Copy")
        self.exportBtn = QPushButton("Export…")
        self.copyBtn.clicked.connect(self._copy_catalogue_to_clipboard)
        self.exportBtn.clicked.connect(self._export_catalogue_csv)
        btnRow.addWidget(self.copyBtn)
        btnRow.addWidget(self.exportBtn)
        catLayout.addLayout(btnRow)

        # Signals from the Sample
        sample.signals.processingCleared.connect(self._onProcessingCleared)
        sample.signals.optimalAgeCalculated.connect(self._onOptimalAgeCalculated)

    # ----- Helpers -----
    def _onCatalogueSelectionChanged(self):
        sel = self.catTable.selectionModel().selectedRows()
        self.peakRowSelected.emit(sel[0].row() if sel else -1)

    def _setCatalogueVisible(self, visible: bool) -> None:
        self.catBox.setVisible(bool(visible))

    def _should_show_catalogue(self) -> bool:
        rows = getattr(self.sample, "peak_catalogue", []) or []
        st = getattr(self.sample, "calculationSettings", None)
        enabled = getattr(st, "enable_ensemble_peak_picking", True)
        return bool(enabled) and len(rows) > 0

    # ----- Actions -----
    def update(self):
        # Safe attribute reads (None is fine; inputs accept None)
        self.optimalAge.setValue(getattr(self.sample, "optimalAge", None))
        self.optimalAgeLower.setValue(getattr(self.sample, "optimalAgeLowerBound", None))
        self.optimalAgeUpper.setValue(getattr(self.sample, "optimalAgeUpperBound", None))
        self.dValue.setValue(getattr(self.sample, "optimalAgeDValue", None))
        self.pValue.setValue(getattr(self.sample, "optimalAgePValue", None))
        self.invalidAges.setValue(getattr(self.sample, "optimalAgeNumberOfInvalidPoints", None))
        self.score.setValue(getattr(self.sample, "optimalAgeScore", None))

    def clear(self):
        self.catTable.setRowCount(0)
        self._setCatalogueVisible(False)
        self.optimalAge.setValue(None)
        self.optimalAgeLower.setValue(None)
        self.optimalAgeUpper.setValue(None)
        self.dValue.setValue(None)
        self.pValue.setValue(None)
        self.invalidAges.setValue(None)
        self.score.setValue(None)
        self.peakRowSelected.emit(-1)

    def _catalogue_rows_for_io(self):
        raw = getattr(self.sample, "peak_catalogue", []) or []
        if isinstance(raw, dict):
            return [raw]
        if isinstance(raw, list):
            out = []
            for r in raw:
                if isinstance(r, dict):
                    out.append(r)
                elif isinstance(r, (list, tuple)) and len(r) >= 3:
                    # tolerate tuples like (age, lo, hi, support)
                    age, lo, hi = r[:3]
                    sup = (r[3] if len(r) >= 4 else float("nan"))
                    out.append({
                        "age_ma": float(age),
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                        "support": float(sup),
                    })
            return out
        # anything else → empty
        return []

    def _sync_from_sample(self):
        rows = self._catalogue_rows_for_io()

        prev = self.catTable.signalsBlocked()
        self.catTable.blockSignals(True)

        self.catTable.clearSelection()
        self.catTable.setRowCount(len(rows))
        for i, r in enumerate(rows, 1):
            try:
                age = float(r.get("age_ma", float("nan")))
                lo  = float(r.get("ci_low", float("nan")))
                hi  = float(r.get("ci_high", float("nan")))
                sup = float(r.get("support", float("nan")))
                if sup == sup:  # not NaN
                    sup = max(0.0, min(1.0, sup))  # clip to [0,1]
            except Exception:
                continue  # skip malformed row safely

            self.catTable.setItem(i-1, 0, QTableWidgetItem(str(i)))
            self.catTable.setItem(i-1, 1, QTableWidgetItem(f"{age:,.2f}"))
            self.catTable.setItem(i-1, 2, QTableWidgetItem(f"{lo:,.2f} – {hi:,.2f}"))
            self.catTable.setItem(i-1, 3, QTableWidgetItem("" if sup != sup else f"{100*sup:.0f}%"))

        self.catTable.resizeColumnsToContents()
        self.catTable.resizeRowsToContents()

        show = len(rows) > 0 and self._should_show_catalogue()
        self._setCatalogueVisible(show)

        self.catTable.blockSignals(prev)
        self.catBox.updateGeometry()
        self.rootLayout.invalidate()
        self.rootLayout.activate()
        self.updateGeometry()

        if show and self.catTable.rowCount() > 0:
            self.catTable.selectRow(0)

    def _copy_catalogue_to_clipboard(self):
        rows = self._catalogue_rows_for_io()
        if not rows:
            return
        s = io.StringIO()
        s.write("rank,age_ma,ci_low,ci_high,support\n")
        for i, r in enumerate(rows, 1):
            s.write(
                f"{i},{float(r.get('age_ma', float('nan'))):.6f},"
                f"{float(r.get('ci_low', float('nan'))):.6f},"
                f"{float(r.get('ci_high', float('nan'))):.6f},"
                f"{float(r.get('support', float('nan'))):.6f}\n"
            )
        QApplication.clipboard().setText(s.getvalue())

    def _export_catalogue_csv(self):
        rows = self._catalogue_rows_for_io()
        if not rows:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export peak catalogue", "", "CSV files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "age_ma", "ci_low", "ci_high", "support"])
            for i, r in enumerate(rows, 1):
                w.writerow([
                    i,
                    float(r.get("age_ma", float("nan"))),
                    float(r.get("ci_low", float("nan"))),
                    float(r.get("ci_high", float("nan"))),
                    float(r.get("support", float("nan")))
                ])

    # ----- Events -----
    def _onProcessingCleared(self):
        self.clear()

    def _onOptimalAgeCalculated(self):
        self.update()
        self._sync_from_sample()
