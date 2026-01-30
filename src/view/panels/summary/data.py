from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,
    QHeaderView, QLabel, QFrame, QSplitter, QSizePolicy
)
from utils import stringUtils
from utils.ui.icons import Icons
from model.settings.calculation import ConcordiaMode


class SummaryDataPanel(QWidget):

    def __init__(self, controller, samples, mode: ConcordiaMode):
        super().__init__()
        self.controller = controller
        self.samples = samples
        self.mode = ConcordiaMode.coerce(mode)
        self._updating = False

        self._createUI()

        # wrapper that tolerates any signal args and binds the sample
        def app(f, x):
            return lambda *args, **kwargs: f(x)

        # connect signals for each sample
        for s in self.samples:
            s.signals.concordancyCalculated.connect(app(self._onSampleConcordancyCalculated, s))
            s.signals.optimalAgeCalculated.connect(app(self._onOptimalAgeCalculated, s))
            # clear only this sample's row when its processing is cleared/skipped
            s.signals.processingCleared.connect(app(self._onProcessingCleared, s))
            s.signals.skipped.connect(app(self._onProcessingCleared, s))

        # build the ensemble table now (it will be empty if no catalogues yet)
        self._refreshEnsembleTable()


    # ---------- UI ----------
    def _createUI(self):
        # --------- legacy optimal-age table ----------
        legacy_headers = [
            "Sample",
            "Concordant\npoints",
            "Discordant\npoints",
            "95%\nlower\nbound",
            "Pb-loss\nage (Ma)",
            "95%\nupper\nbound",
            "d-value",
            "p-value",
            "Score"
        ]
        self.dataTable = QTableWidget(len(self.samples), len(legacy_headers))
        self.dataTable.setHorizontalHeaderLabels(legacy_headers)
        for i, s in enumerate(self.samples):
            self.dataTable.setItem(i, 0, self._cell(s.name))
            for col in range(1, len(legacy_headers)):
                self.dataTable.setItem(i, col, self._cell(""))

        # make columns adjustable & show more by default
        h = self.dataTable.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.Interactive)  # user-draggable columns
        h.setDefaultSectionSize(120)                     # wider default so more fits
        h.setStretchLastSection(False)
        self.dataTable.setHorizontalScrollMode(self.dataTable.ScrollPerPixel)
        self.dataTable.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dataTable.setAlternatingRowColors(True)
        self.dataTable.verticalHeader().setVisible(False)

        self.exportButton = QPushButton("  Export optimal-age table")
        self.exportButton.setIcon(Icons.export())
        self.exportButton.clicked.connect(self._exportOptimalAge)
        self.exportButton.setObjectName("AccentButton")

        topBox = QWidget()
        topLay = QVBoxLayout(topBox)
        topLay.setContentsMargins(0, 0, 0, 0)
        topLay.addWidget(self.dataTable)
        topLay.addWidget(self.exportButton)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken)

        cat_headers = ["Sample", "Peak #", "95% low", "Age (Ma)", "95% high", "Support (%)"]
        self.catalogueTable = QTableWidget(0, len(cat_headers))
        self.catalogueTable.setHorizontalHeaderLabels(cat_headers)

        hc = self.catalogueTable.horizontalHeader()
        hc.setSectionResizeMode(QHeaderView.Interactive)
        hc.setDefaultSectionSize(120)
        hc.setStretchLastSection(False)
        self.catalogueTable.setSortingEnabled(True)  # optional: allow clicking headers to sort


        self.catalogueTable.setHorizontalScrollMode(self.catalogueTable.ScrollPerPixel)
        self.catalogueTable.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.catalogueTable.setAlternatingRowColors(True)
        self.catalogueTable.verticalHeader().setVisible(False)
        self.catalogueTable.setMinimumHeight(180)  # gives it some initial presence

        self.exportCatalogueButton = QPushButton("  Export ensemble peaks table")
        self.exportCatalogueButton.setIcon(Icons.export())
        self.exportCatalogueButton.setObjectName("AccentButton")
        self.exportCatalogueButton.clicked.connect(self._exportEnsemble)

        bottomBox = QWidget()
        bottomLay = QVBoxLayout(bottomBox)
        bottomLay.setContentsMargins(0, 0, 0, 0)
        bottomLay.addWidget(QLabel("Ensemble peak catalogue"))
        bottomLay.addWidget(sep)
        bottomLay.addWidget(self.catalogueTable)
        bottomLay.addWidget(self.exportCatalogueButton)

        # --------- stack both sections in a V-splitter so heights are adjustable ----------
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(topBox)
        splitter.addWidget(bottomBox)

        # sensible initial sizes: give more room to the top table; both resizeable by drag
        splitter.setSizes([600, 360])

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        outer.addWidget(splitter)
        self.setLayout(outer)

    # ---------- helpers ----------
    def _cell(self, text):
        item = QTableWidgetItem("" if text is None else str(text))
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        item.setFlags(item.flags() ^ Qt.ItemIsEditable)
        return item

    def _fmt(self, x):
        if x is None:
            return ""
        try:
            return str(stringUtils.round_to_sf(x))
        except Exception:
            return str(x)

    def _fmt_ma(self, x):
        if x is None:
            return ""
        try:
            return str(stringUtils.round_to_sf(float(x)))
        except Exception:
            return str(x)

    def _fmt_pct(self, x):
        if x is None:
            return ""
        try:
            return f"{100.0*float(x):.1f}"
        except Exception:
            return str(x)

    # ---------- legacy table updates ----------
    def _onSampleConcordancyCalculated(self, sample):
        row = getattr(sample, "id", None)
        if row is None or row < 0 or row >= self.dataTable.rowCount():
            try:
                row = self.samples.index(sample)
            except ValueError:
                return

        res = sample.mode_results(self.mode)
        if res is None:
            self.dataTable.setItem(row, 1, self._cell(""))
            self.dataTable.setItem(row, 2, self._cell(""))
            return

        conc = res.concordant or []
        rev  = res.reverse or []

        n_conc = sum(1 for v in conc if v is True)
        n_disc = sum(
            1 for i, v in enumerate(conc)
            if (v is False) and not (rev[i] if i < len(rev) else False)
        )

        self.dataTable.setItem(row, 1, self._cell(n_conc))
        self.dataTable.setItem(row, 2, self._cell(n_disc))
        self.dataTable.resizeColumnsToContents()

    def _onOptimalAgeCalculated(self, sample):
        # Re-entrancy guard (prevents infinite recursion / abort)
        if getattr(self, "_updating", False):
            return
        self._updating = True
        try:
            row = getattr(sample, "id", None)
            if row is None or row < 0 or row >= self.dataTable.rowCount():
                try:
                    row = self.samples.index(sample)
                except ValueError:
                    return

            res = sample.mode_results(self.mode)
            if res is None:
                for col in range(3, 9):
                    self.dataTable.setItem(row, col, self._cell(""))
                self._refreshEnsembleTable()
                return

            def put(col, val):
                self.dataTable.setItem(row, col, self._cell(self._fmt(val)))

            put(3, res.optimalAgeLowerBound/1e6 if res.optimalAgeLowerBound is not None else None)
            put(4, res.optimalAge/1e6          if res.optimalAge is not None else None)
            put(5, res.optimalAgeUpperBound/1e6 if res.optimalAgeUpperBound is not None else None)
            put(6, res.optimalAgeDValue)
            put(7, res.optimalAgePValue)
            put(8, res.optimalAgeScore)

            self.dataTable.resizeColumnsToContents()
            self.dataTable.resizeRowsToContents()

            self._refreshEnsembleTable()
            self.catalogueTable.resizeRowsToContents()

        finally:
            self._updating = False


    def _onProcessingCleared(self, sample):
        if sample.mode_results(self.mode) is not None:
            return

        row = getattr(sample, "id", None)
        if row is None or row < 0 or row >= self.dataTable.rowCount():
            try:
                row = self.samples.index(sample)
            except ValueError:
                return

        for col in range(1, self.dataTable.columnCount()):
            self.dataTable.setItem(row, col, self._cell(""))

        self._refreshEnsembleTable()


    # ---------- ensemble table population ----------
    def _normalize_peak(self, peak):
        """
        Accepts either a dict (preferred) or a tuple/list from legacy runs.
        Returns (age, ci_low, ci_high, support, peak_no) with Nones when unknown.
        Tuple/list is interpreted as: (age, ci_low, ci_high, support[, peak_no]).
        """
        # dict-like (new path)
        if isinstance(peak, dict):
            age = peak.get("age_ma", peak.get("age"))
            lo  = peak.get("ci_low", peak.get("ciLow"))
            hi  = peak.get("ci_high", peak.get("ciHigh"))
            sup = peak.get("support", peak.get("vote_fraction"))
            pno = peak.get("peak_no", peak.get("peakNo", peak.get("id")))
            return age, lo, hi, sup, pno

        # tuple/list (legacy path)
        if isinstance(peak, (tuple, list)):
            age = peak[0] if len(peak) > 0 else None
            lo  = peak[1] if len(peak) > 1 else None
            hi  = peak[2] if len(peak) > 2 else None
            sup = peak[3] if len(peak) > 3 else None
            pno = peak[4] if len(peak) > 4 else None
            return age, lo, hi, sup, pno

        # unknown type: ignore gracefully
        return None, None, None, None, None

    def _refreshEnsembleTable(self):
        rows = []

        for s in self.samples:
            res = s.mode_results(self.mode)
            raw = (res.peak_catalogue if res is not None else []) or []


            # Normalize entries to dicts with keys we care about
            norm = []
            for d in raw:
                if isinstance(d, dict):
                    norm.append(d)
                elif isinstance(d, (tuple, list)):  # very defensive: map tuple -> dict if order is (age, lo, hi, support[, peak_no])
                    tmp = {}
                    if len(d) >= 1: tmp["age_ma"]  = d[0]
                    if len(d) >= 2: tmp["ci_low"]  = d[1]
                    if len(d) >= 3: tmp["ci_high"] = d[2]
                    if len(d) >= 4: tmp["support"] = d[3]
                    if len(d) >= 5: tmp["peak_no"] = d[4]
                    norm.append(tmp)

            # Sort by age so peak numbering is stable if we need to synthesize it
            try:
                norm = sorted(norm, key=lambda z: z.get("age_ma", float("inf")))
            except Exception:
                pass

            # Build table rows
            for j, d in enumerate(norm, start=1):
                age = d.get("age_ma")
                lo  = d.get("ci_low")
                hi  = d.get("ci_high")
                sup = d.get("support")   # fraction 0..1
                pno = d.get("peak_no") or j  # fallback numbering per sample

                rows.append([s.name, pno, lo, age, hi, sup])

        # Repopulate table
        self.catalogueTable.setSortingEnabled(False)  # avoid re-sorts while filling
        self.catalogueTable.setRowCount(len(rows))

        for r, (sname, pkno, lo, age, hi, sup) in enumerate(rows):
            self.catalogueTable.setItem(r, 0, self._cell(sname))
            self.catalogueTable.setItem(r, 1, self._cell("" if pkno is None else pkno))
            self.catalogueTable.setItem(r, 2, self._cell(self._fmt_ma(lo)))   # 95% low
            self.catalogueTable.setItem(r, 3, self._cell(self._fmt_ma(age)))  # Age
            self.catalogueTable.setItem(r, 4, self._cell(self._fmt_ma(hi)))   # 95% high
            self.catalogueTable.setItem(r, 5, self._cell(self._fmt_pct(sup))) # Support (%)

        self.catalogueTable.resizeColumnsToContents()
        self.catalogueTable.resizeRowsToContents()
        self.catalogueTable.setSortingEnabled(True)



    # ---------- exports ----------
    def _exportOptimalAge(self):
        mode_txt = "TW" if self.mode == ConcordiaMode.TW else "WETHERILL"
        n = self.dataTable.columnCount()
        headers = ["Mode"] + [self.dataTable.horizontalHeaderItem(i).text().replace("\n", " ") for i in range(n)]
        data = []
        for r in range(self.dataTable.rowCount()):
            row = [mode_txt] + [
                self.dataTable.item(r, c).text() if self.dataTable.item(r, c) else ""
                for c in range(n)
            ]
            data.append(row)
        self.controller.exportCSV(headers, data)

    def _exportEnsemble(self):
        mode_txt = "TW" if self.mode == ConcordiaMode.TW else "WETHERILL"
        n = self.catalogueTable.columnCount()
        headers = ["Mode"] + [self.catalogueTable.horizontalHeaderItem(i).text().replace("\n", " ") for i in range(n)]
        data = []
        for r in range(self.catalogueTable.rowCount()):
            row = [mode_txt] + [
                self.catalogueTable.item(r, c).text() if self.catalogueTable.item(r, c) else ""
                for c in range(n)
            ]
            data.append(row)
        self.controller.exportCSV(headers, data)
