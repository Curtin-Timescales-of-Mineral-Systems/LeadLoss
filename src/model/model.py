import time
import csv
import numpy as np

from collections import defaultdict
from pathlib import Path

from model.sample import Sample
from model.spot import Spot
from process import processing
from process.cdc_population import concordant_ages_ma, discordant_reference_ages_ma

from utils import csvUtils

from PyQt5.QtWidgets import QFileDialog

class LeadLossModel:

    UPDATE_INTERVAL = 0.5

    def __init__(self, signals):
        self.signals = signals
        self.headers = []
        self.samples = []
        self.samplesByName = {}

        # legacy state used by getters/exports in tools
        self.rows = []
        self.concordantRows = []
        self.discordantRows = []

        self.dValuesByAge = {}
        self.pValuesByAge = {}
        self.reconstructedAges = {}
        self.optimalAge = None

        self.lastUpdateTime = 0

    ################
    ## Input data ##
    ################

    def loadInputData(self, inputFile, importSettings, rawHeaders, rawSpotData):
        self.headers = rawHeaders

        spotsBySampleName = defaultdict(list)
        for row in rawSpotData:
            spot = Spot(row, importSettings)
            spotsBySampleName[spot.sampleName].append(spot)

        self.samples = []
        self.samplesByName = {}
        for id, (sampleName, sampleRows) in enumerate(spotsBySampleName.items()):
            sample = Sample(id, sampleName, sampleRows)
            self.samples.append(sample)
            self.samplesByName[sampleName] = sample

        self.signals.inputDataLoaded.emit(inputFile, self.samples)
        self.signals.taskComplete.emit(True, "Successfully imported CSV file")

    def clearInputData(self):
        self.headers = []
        self.samples = []
        self.samplesByName = {}
        self.rows = []
        self.concordantRows = []
        self.discordantRows = []
        self.dValuesByAge = {}
        self.pValuesByAge = {}
        self.reconstructedAges = {}
        self.optimalAge = None

        self.signals.inputDataCleared.emit()
        
    def emitSummedKS(self, sampleName, payload):
        if sampleName in self.samplesByName:   # or self.samplesByName.keys() in your code
            sample = self.samplesByName[sampleName]

            # payload is: (ages_ma_list, y_curve_list, peaks_age, peaks_ci, support)
            try:
                ages_ma = payload[0]
                y_curve = payload[1]
                peaks_age = payload[2] if len(payload) > 2 else []
                peaks_ci = payload[3] if len(payload) > 3 else []
                sample.summedKS_ages_Ma = np.asarray(ages_ma, dtype=float)
                sample.summedKS_goodness = np.asarray(y_curve, dtype=float)
                sample.summedKS_peaks_Ma = np.asarray(peaks_age, dtype=float)
                if isinstance(peaks_ci, (list, tuple)) and len(peaks_ci):
                    sample.summedKS_ci_low_Ma = np.asarray([float(lo) for lo, _ in peaks_ci], dtype=float)
                    sample.summedKS_ci_high_Ma = np.asarray([float(hi) for _, hi in peaks_ci], dtype=float)
                else:
                    sample.summedKS_ci_low_Ma = np.asarray([], dtype=float)
                    sample.summedKS_ci_high_Ma = np.asarray([], dtype=float)
            except Exception:
                sample.summedKS_ages_Ma = None
                sample.summedKS_goodness = None
                sample.summedKS_peaks_Ma = None
                sample.summedKS_ci_low_Ma = None
                sample.summedKS_ci_high_Ma = None

            if sample.signals:
                sample.signals.summedKS.emit(payload)


    #################
    ## Calculation ##
    #################

    def clearCalculation(self):
        for sample in self.samples:
            sample.clearCalculation()

        self.lastUpdateTime = time.time()

    def getProcessingFunction(self):
        return processing.processSamples

    def getProcessingData(self):
        return [sample.createProcessingCopy() for sample in self.samples]

    def updateConcordance(self, sampleName, concordantAges, discordances, reverse_flags=None):
        sample = next((s for s in self.samples if s.name == sampleName), None)
        if not sample:
            return
        sample.updateConcordance(concordantAges, discordances, reverse_flags)


    def addMonteCarloRun(self, sampleName, run):
        sample = self.samplesByName[sampleName]
        sample.addMonteCarloRun(run)

    def setOptimalAge(self, sampleName, args):
        sample = self.samplesByName[sampleName]
        sample.setOptimalAge(args)

    #############
    ## Getters ##
    #############

    def addRimAgeStats(self, rimAge, discordantAges, dValue, pValue):
        self.dValuesByAge[rimAge] = dValue
        self.pValuesByAge[rimAge] = pValue
        self.reconstructedAges[rimAge] = discordantAges

        self.signals.statisticUpdated.emit(len(self.dValuesByAge)-1, dValue, pValue)
        now = time.time()
        if now - self.lastUpdateTime > self.UPDATE_INTERVAL:
            self.signals.allStatisticsUpdated.emit(self.dValuesByAge)
            self.lastUpdateTime = now

    def getAgeRange(self):
        concordantAges = [row.concordantAge for row in self.rows if getattr(row, "concordant", False)]
        recAges = [recAge for ages in self.reconstructedAges.values() for recAge in ages]
        discordantAges = [recAge.values[0] for recAge in recAges if recAge]
        allAges = concordantAges + discordantAges
        return (min(allAges), max(allAges)) if allAges else (None, None)

    def getNearestSampledAge(self, requestedAge):
        if not self.dValuesByAge:
            # Keep a stable 4-tuple shape for callers that unpack
            # (age, d, p, reconstructed_ages).
            return None, None, None, []

        if requestedAge is not None:
            actualAge = min(self.dValuesByAge, key=lambda a: abs(a-requestedAge))
        else:
            actualAge = self.optimalAge

        return actualAge, self.dValuesByAge[actualAge], self.pValuesByAge[actualAge], self.reconstructedAges[actualAge]
    
    ################
    ## Export data ##
    ################

    def exportMonteCarloRuns(self, append=False):
        filename = QFileDialog.getSaveFileName(
            caption='Save CSV file',
            directory='.',
            options=QFileDialog.DontUseNativeDialog
        )[0]
        if not filename:
            return
        mode = 'a' if append else 'w'
        with open(filename, mode, newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for sample in self.samples:
                for run in sample.getMonteCarloRuns():
                    run.calculateOptimalAge()
                    writer.writerow(run.toList())

    @staticmethod
    def _safe_float(value):
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if np.isfinite(out) else None

    @staticmethod
    def _summary_dict(sample):
        summary = getattr(sample, "disc_cluster_summary", None)
        return summary if isinstance(summary, dict) else {}

    def buildClusteringDiagnosticsTables(self, samples=None):
        samples = list(self.samples if samples is None else samples)

        summary_headers = [
            "sample",
            "clustering_requested",
            "split_accepted",
            "reporting_accepted",
            "reason",
            "n_anchors",
            "anchor_means_ma",
            "n_discordant",
            "n_assigned",
            "n_ambiguous",
            "assigned_fraction",
            "n_valid_proxies",
            "n_clustered_proxies",
            "n_unclustered_valid_proxies",
            "n_clusters",
            "n_reported_rows",
            "n_rejected_rows",
            "ensemble_status",
        ]
        anchor_headers = ["sample", "anchor_id", "anchor_age_ma", "n_concordant"]
        spot_headers = [
            "sample",
            "role",
            "spot_index",
            "u238u206pb",
            "pb207pb206",
            "u238u206pb_sigma_1s",
            "pb207pb206_sigma_1s",
            "discordance",
            "reference_age_ma",
            "assigned_anchor",
            "cluster_id",
            "ambiguous",
            "best_distance_ma",
            "second_distance_ma",
            "separation_ratio",
        ]
        cluster_headers = [
            "sample",
            "cluster_id",
            "cluster_n",
            "cluster_proxy_median_ma",
            "result_type",
            "age_ma",
            "ci_low_ma",
            "ci_high_ma",
            "upper_bound_ma",
            "direct_support",
            "winner_support",
            "reason",
        ]
        peak_headers = [
            "sample",
            "peak_no",
            "cluster_id",
            "mode",
            "age_ma",
            "ci_low_ma",
            "ci_high_ma",
            "direct_support",
            "winner_support",
            "selection",
            "label",
        ]
        rejected_headers = [
            "sample",
            "cluster_id",
            "age_ma",
            "direct_support",
            "winner_support",
            "reason",
        ]

        summary_rows = []
        anchor_rows = []
        spot_rows = []
        cluster_rows = []
        peak_rows = []
        rejected_rows = []

        for sample in samples:
            settings = getattr(sample, "calculationSettings", None)
            clustering_requested = bool(getattr(settings, "use_discordant_clustering", False))
            summary = self._summary_dict(sample)
            split_accepted = bool(
                summary.get("split_accepted", summary.get("accepted", getattr(sample, "_cdc_cluster_split_accepted", False)))
            )
            reporting_accepted = bool(
                summary.get("reporting_accepted", getattr(sample, "_cdc_cluster_reporting_accepted", False))
            )

            reported = list(getattr(sample, "peak_catalogue", []) or [])
            rejected = list(getattr(sample, "rejected_peak_candidates", []) or [])

            summary_rows.append([
                sample.name,
                clustering_requested,
                split_accepted,
                reporting_accepted,
                summary.get("reason", ""),
                summary.get("n_anchors", ""),
                "; ".join(
                    f"{float(age):.2f}" for age in (summary.get("anchor_means_ma") or [])
                    if self._safe_float(age) is not None
                ),
                summary.get("n_discordant", ""),
                summary.get("n_assigned", ""),
                summary.get("n_ambiguous", ""),
                self._safe_float(summary.get("assigned_fraction")),
                summary.get("n_valid_proxies", ""),
                summary.get("n_clustered_proxies", ""),
                summary.get("n_unclustered_valid_proxies", ""),
                len(summary.get("clusters", []) or []),
                len(reported),
                len(rejected),
                getattr(sample, "ensemble_abstain_reason", "") or ("resolved" if reported else ""),
            ])

            for anchor in summary.get("anchors", []) or []:
                anchor_rows.append([
                    sample.name,
                    anchor.get("anchor_id", ""),
                    self._safe_float(anchor.get("age_ma")),
                    anchor.get("n_concordant", ""),
                ])

            assignment_by_idx = {}
            for row in summary.get("assignments", []) or []:
                try:
                    assignment_by_idx[int(row.get("grain"))] = row
                except Exception:
                    continue

            conc_spots = list(sample.concordantSpots())
            conc_ages = concordant_ages_ma(conc_spots) if conc_spots else np.asarray([], float)
            for idx, (spot, age_ma) in enumerate(zip(conc_spots, conc_ages)):
                spot_rows.append([
                    sample.name,
                    "concordant",
                    idx,
                    self._safe_float(getattr(spot, "uPbValue", None)),
                    self._safe_float(getattr(spot, "pbPbValue", None)),
                    self._safe_float(getattr(spot, "uPbStDev", None)),
                    self._safe_float(getattr(spot, "pbPbStDev", None)),
                    self._safe_float(getattr(spot, "discordance", None)),
                    self._safe_float(age_ma),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ])

            disc_spots = list(sample.discordantSpots())
            disc_ages = discordant_reference_ages_ma(disc_spots) if disc_spots else np.asarray([], float)
            for idx, (spot, age_ma) in enumerate(zip(disc_spots, disc_ages)):
                assignment = assignment_by_idx.get(idx, {})
                spot_rows.append([
                    sample.name,
                    "discordant",
                    idx,
                    self._safe_float(getattr(spot, "uPbValue", None)),
                    self._safe_float(getattr(spot, "pbPbValue", None)),
                    self._safe_float(getattr(spot, "uPbStDev", None)),
                    self._safe_float(getattr(spot, "pbPbStDev", None)),
                    self._safe_float(getattr(spot, "discordance", None)),
                    self._safe_float(age_ma),
                    assignment.get("assigned_anchor", ""),
                    "" if getattr(spot, "cluster_id", None) is None else int(getattr(spot, "cluster_id")),
                    bool(assignment.get("ambiguous", False)) if assignment else "",
                    self._safe_float(assignment.get("best_distance_ma")),
                    self._safe_float(assignment.get("second_distance_ma")),
                    self._safe_float(assignment.get("separation_ratio")),
                ])

            for idx, spot in enumerate(sample.reverseDiscordantSpots()):
                spot_rows.append([
                    sample.name,
                    "reverse_discordant",
                    idx,
                    self._safe_float(getattr(spot, "uPbValue", None)),
                    self._safe_float(getattr(spot, "pbPbValue", None)),
                    self._safe_float(getattr(spot, "uPbStDev", None)),
                    self._safe_float(getattr(spot, "pbPbStDev", None)),
                    self._safe_float(getattr(spot, "discordance", None)),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ])

            for row in reported:
                peak_rows.append([
                    sample.name,
                    row.get("peak_no", ""),
                    row.get("cluster_id", ""),
                    row.get("mode", ""),
                    self._safe_float(row.get("age_ma")),
                    self._safe_float(row.get("ci_low")),
                    self._safe_float(row.get("ci_high")),
                    self._safe_float(row.get("direct_support", row.get("support"))),
                    self._safe_float(row.get("winner_support", row.get("support"))),
                    row.get("selection", ""),
                    row.get("label", ""),
                ])

            for row in rejected:
                rejected_rows.append([
                    sample.name,
                    row.get("cluster_id", ""),
                    self._safe_float(row.get("age_ma")),
                    self._safe_float(row.get("direct_support")),
                    self._safe_float(row.get("winner_support")),
                    row.get("reason", ""),
                ])

            reported_by_cluster = defaultdict(list)
            for row in reported:
                cid = row.get("cluster_id", None)
                if cid is None:
                    continue
                reported_by_cluster[int(cid)].append(row)

            rejected_by_cluster = defaultdict(list)
            for row in rejected:
                cid = row.get("cluster_id", None)
                if cid is None:
                    continue
                rejected_by_cluster[int(cid)].append(row)

            for cluster in summary.get("clusters", []) or []:
                cid = int(cluster.get("k", -1))
                base = [
                    sample.name,
                    cid,
                    cluster.get("n", ""),
                    self._safe_float(cluster.get("median_ma")),
                ]
                rows_for_cluster = reported_by_cluster.get(cid, [])
                if rows_for_cluster:
                    for row in rows_for_cluster:
                        mode = str(row.get("mode", ""))
                        cluster_rows.append(
                            base + [
                                "recent_boundary_mode" if mode == "recent_boundary" else "resolved_peak",
                                self._safe_float(row.get("age_ma")),
                                self._safe_float(row.get("ci_low")) if mode != "recent_boundary" else None,
                                self._safe_float(row.get("ci_high")) if mode != "recent_boundary" else None,
                                self._safe_float(row.get("ci_high")) if mode == "recent_boundary" else None,
                                self._safe_float(row.get("direct_support", row.get("support"))),
                                self._safe_float(row.get("winner_support", row.get("support"))),
                                "",
                            ]
                        )
                    continue

                rejected_for_cluster = rejected_by_cluster.get(cid, [])
                if rejected_for_cluster:
                    for row in rejected_for_cluster:
                        cluster_rows.append(
                            base + [
                                "rejected",
                                self._safe_float(row.get("age_ma")),
                                None,
                                None,
                                None,
                                self._safe_float(row.get("direct_support")),
                                self._safe_float(row.get("winner_support")),
                                row.get("reason", ""),
                            ]
                        )
                    continue

                cluster_rows.append(base + ["none", None, None, None, None, None, None, ""])

        return {
            "summary": (summary_headers, summary_rows),
            "anchors": (anchor_headers, anchor_rows),
            "spots": (spot_headers, spot_rows),
            "clusters": (cluster_headers, cluster_rows),
            "peaks": (peak_headers, peak_rows),
            "rejected": (rejected_headers, rejected_rows),
        }

    def exportClusteringDiagnostics(self, output_path, samples=None):
        base = Path(output_path)
        stem = base.with_suffix("") if base.suffix else base
        tables = self.buildClusteringDiagnosticsTables(samples=samples)
        written = []
        for suffix, (headers, rows) in tables.items():
            out_path = stem.parent / f"{stem.name}_{suffix}.csv"
            csvUtils.write_output(headers, rows, str(out_path))
            written.append(str(out_path))
        return written
