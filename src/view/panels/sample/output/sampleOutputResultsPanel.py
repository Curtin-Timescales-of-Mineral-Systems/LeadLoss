from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QWidget, QLabel, QFormLayout, QHBoxLayout, QAbstractItemView,
    QPushButton, QFileDialog, QApplication, QSizePolicy, QMessageBox
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
        self.ensembleStatus = QLabel("—")
        self.ensembleStatus.setWordWrap(True)

        formHost = QWidget()
        form = QFormLayout(formHost)
        form.addRow("Optimal Pb-loss age", self.optimalAge)
        form.addRow("95% confidence interval", boundsLayout)
        form.addRow("Mean D value (KS test)", self.dValue)
        form.addRow("Mean p value (KS test)", self.pValue)
        form.addRow("Mean # of invalid ages", self.invalidAges)
        form.addRow("Mean score", self.score)
        form.addRow("Ensemble status", self.ensembleStatus)
        self.rootLayout.addWidget(formHost)

        # ----- Peak catalogue group -----
        self.catBox = QGroupBox("Peak catalogue (ensemble support)")
        catLayout = QVBoxLayout(self.catBox)

        self.catTable = QTableWidget(0, 5)
        self.catTable.setHorizontalHeaderLabels(
            ["#", "Age (Ma)", "95% CI (Ma)", "Direct support", "Winner support"]
        )
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

        self.catalogueNote = QLabel("")
        self.catalogueNote.setWordWrap(True)
        self.catalogueNote.setVisible(False)
        catLayout.addWidget(self.catalogueNote)

        btnRow = QHBoxLayout()
        btnRow.addStretch(1)
        self.copyBtn = QPushButton("Copy")
        self.exportBtn = QPushButton("Export…")
        self.copyBtn.clicked.connect(self._copy_catalogue_to_clipboard)
        self.exportBtn.clicked.connect(self._export_catalogue_csv)
        btnRow.addWidget(self.copyBtn)
        btnRow.addWidget(self.exportBtn)
        catLayout.addLayout(btnRow)

        # ----- Rejected-candidate table (for transparency) -----
        self.rejectedBox = QGroupBox("Rejected candidate peaks")
        rejLayout = QVBoxLayout(self.rejectedBox)
        self.rejectedTable = QTableWidget(0, 4)
        self.rejectedTable.setHorizontalHeaderLabels(
            ["Age (Ma)", "Direct support", "Winner support", "Reason"]
        )
        self.rejectedTable.verticalHeader().setVisible(False)
        self.rejectedTable.setEditTriggers(QTableWidget.NoEditTriggers)
        self.rejectedTable.setSelectionMode(QAbstractItemView.NoSelection)
        self.rejectedTable.setAlternatingRowColors(True)
        self.rejectedTable.setMinimumHeight(110)
        rejLayout.addWidget(self.rejectedTable)
        self.rejectedBox.setVisible(False)
        self.rootLayout.addWidget(self.rejectedBox)

        self.exportCurveButton = QPushButton("Export curve (CSV)")
        self.exportCurveButton.clicked.connect(self.exportGoodnessCurveCSV)

        # Signals from the Sample
        sample.signals.processingCleared.connect(self._onProcessingCleared)
        sample.signals.optimalAgeCalculated.connect(self._onOptimalAgeCalculated)

    # ----- Helpers -----

    def exportGoodnessCurveCSV(self):
        if self.sample is None:
            QMessageBox.warning(self, "No sample", "No sample is selected.")
            return

        ages = getattr(self.sample, "summedKS_ages_Ma", None)
        y    = getattr(self.sample, "summedKS_goodness", None)

        if ages is None or y is None or len(ages) == 0:
            QMessageBox.information(
                self,
                "No curve available",
                "No goodness curve values are available yet.\n\nRun processing first, then try again."
            )
            return

        default_name = "goodness_curve.csv"
        if getattr(self.sample, "name", ""):
            default_name = f"{self.sample.name}_goodness_curve.csv"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export goodness curve (CSV)",
            default_name,
            "CSV Files (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        import csv
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["age_Ma", "goodness"])
            for a, g in zip(ages, y):
                w.writerow([float(a), float(g)])

    def _onCatalogueSelectionChanged(self):
        sel = self.catTable.selectionModel().selectedRows()
        self.peakRowSelected.emit(sel[0].row() if sel else -1)

    def _setCatalogueVisible(self, visible: bool) -> None:
        self.catBox.setVisible(bool(visible))

    def _abstain_reason_text(self, reason: str) -> str:
        mapping = {
            "flat_or_monotonic_surface": "No ensemble peak reported: the ensemble surface is flat/monotonic in the tested window.",
            "boundary_dominated_surface": "No ensemble peak reported: run optima are boundary-dominated in the tested window.",
            "no_supported_peaks": "No ensemble peak reported: no candidate peak met the support/consistency filters.",
        }
        if reason in mapping:
            return mapping[reason]
        if reason:
            return f"No ensemble peak reported: {str(reason).replace('_', ' ')}."
        return ""

    def _ensemble_status_text(self) -> str:
        st = getattr(self.sample, "calculationSettings", None)
        if not bool(getattr(st, "enable_ensemble_peak_picking", True)):
            return "Disabled"

        rows = getattr(self.sample, "peak_catalogue", []) or []
        if len(rows) > 0:
            return f"Resolved ({len(rows)} peak{'s' if len(rows) != 1 else ''})"

        reason = getattr(self.sample, "ensemble_abstain_reason", None)
        if reason == "flat_or_monotonic_surface":
            return "Unresolved (flat/monotonic surface)"
        if reason == "boundary_dominated_surface":
            return "Unresolved (boundary-dominated)"
        if reason == "no_supported_peaks":
            return "Unresolved (no supported peaks)"
        if reason:
            return f"Unresolved ({str(reason).replace('_', ' ')})"
        return "Unresolved"

    def _rejected_reason_text(self, code: str) -> str:
        mapping = {
            "merged_overlapping_candidates": "Merged with nearby stronger peak",
            "plateau_duplicate": "Duplicate on same plateau/crest",
            "low_support": "Below support threshold",
            "suppressed_nearby_weaker_peak": "Nearby weaker shoulder",
            "coarse_surface_no_separate_mode": "Visible shoulder on same broad mode",
            "below_global_height_gate": "Below global height gate",
            "boundary_dominated_surface": "Boundary-dominated optima",
            "edge_degenerate_ci": "Edge-degenerate confidence interval",
            "wide_ci": "Confidence interval too broad",
        }
        return mapping.get(str(code), str(code).replace("_", " "))

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
        self.ensembleStatus.setText(self._ensemble_status_text())

    def clear(self):
        self.catTable.setRowCount(0)
        self._setCatalogueVisible(False)
        self.catalogueNote.setText("")
        self.catalogueNote.setVisible(False)
        self.rejectedTable.setRowCount(0)
        self.rejectedBox.setVisible(False)
        self.optimalAge.setValue(None)
        self.optimalAgeLower.setValue(None)
        self.optimalAgeUpper.setValue(None)
        self.dValue.setValue(None)
        self.pValue.setValue(None)
        self.invalidAges.setValue(None)
        self.score.setValue(None)
        self.ensembleStatus.setText("—")
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
                        "direct_support": float(sup),
                        "winner_support": float(sup),
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
                dir_sup = float(r.get("direct_support", r.get("support", float("nan"))))
                win_sup = float(r.get("winner_support", r.get("support", float("nan"))))
                if dir_sup == dir_sup:  # not NaN
                    dir_sup = max(0.0, min(1.0, dir_sup))
                if win_sup == win_sup:
                    win_sup = max(0.0, min(1.0, win_sup))
            except Exception:
                continue  # skip malformed row safely

            self.catTable.setItem(i-1, 0, QTableWidgetItem(str(i)))
            self.catTable.setItem(i-1, 1, QTableWidgetItem(f"{age:,.2f}"))
            self.catTable.setItem(i-1, 2, QTableWidgetItem(f"{lo:,.2f} – {hi:,.2f}"))
            self.catTable.setItem(i-1, 3, QTableWidgetItem("" if dir_sup != dir_sup else f"{100*dir_sup:.0f}%"))
            self.catTable.setItem(i-1, 4, QTableWidgetItem("" if win_sup != win_sup else f"{100*win_sup:.0f}%"))

        self.catTable.resizeColumnsToContents()
        self.catTable.resizeRowsToContents()

        show = len(rows) > 0 and self._should_show_catalogue()
        self._setCatalogueVisible(show)

        st = getattr(self.sample, "calculationSettings", None)
        ensemble_enabled = bool(getattr(st, "enable_ensemble_peak_picking", True))
        reason = getattr(self.sample, "ensemble_abstain_reason", None)
        note_text = self._abstain_reason_text(reason) if (ensemble_enabled and not show) else ""
        self.catalogueNote.setText(note_text)
        self.catalogueNote.setVisible(bool(note_text))
        self.ensembleStatus.setText(self._ensemble_status_text())

        # Rejected candidates (if any)
        rej = getattr(self.sample, "rejected_peak_candidates", None) or []
        self.rejectedTable.setRowCount(len(rej))
        for i, r in enumerate(rej):
            age = float(r.get("age_ma", float("nan")))
            ds = float(r.get("direct_support", float("nan")))
            ws = float(r.get("winner_support", float("nan")))
            if ds == ds:
                ds = max(0.0, min(1.0, ds))
            if ws == ws:
                ws = max(0.0, min(1.0, ws))
            reason = self._rejected_reason_text(r.get("reason", "filtered"))

            self.rejectedTable.setItem(i, 0, QTableWidgetItem("" if age != age else f"{age:,.2f}"))
            self.rejectedTable.setItem(i, 1, QTableWidgetItem("n/a" if ds != ds else f"{100*ds:.0f}%"))
            self.rejectedTable.setItem(i, 2, QTableWidgetItem("n/a" if ws != ws else f"{100*ws:.0f}%"))
            self.rejectedTable.setItem(i, 3, QTableWidgetItem(reason))
        self.rejectedTable.resizeColumnsToContents()
        self.rejectedTable.resizeRowsToContents()
        self.rejectedBox.setVisible(bool(rej))

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
        s.write("rank,age_ma,ci_low,ci_high,direct_support,winner_support\n")
        for i, r in enumerate(rows, 1):
            s.write(
                f"{i},{float(r.get('age_ma', float('nan'))):.6f},"
                f"{float(r.get('ci_low', float('nan'))):.6f},"
                f"{float(r.get('ci_high', float('nan'))):.6f},"
                f"{float(r.get('direct_support', r.get('support', float('nan')))):.6f},"
                f"{float(r.get('winner_support', r.get('support', float('nan')))):.6f}\n"
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
            w.writerow(["rank", "age_ma", "ci_low", "ci_high", "direct_support", "winner_support"])
            for i, r in enumerate(rows, 1):
                w.writerow([
                    i,
                    float(r.get("age_ma", float("nan"))),
                    float(r.get("ci_low", float("nan"))),
                    float(r.get("ci_high", float("nan"))),
                    float(r.get("direct_support", r.get("support", float("nan")))),
                    float(r.get("winner_support", r.get("support", float("nan")))),
                ])

    # ----- Events -----
    def _onProcessingCleared(self):
        self.clear()

    def _onOptimalAgeCalculated(self):
        self.update()
        self._sync_from_sample()
