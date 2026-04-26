#!/usr/bin/env python3
"""
Replay CDC ensemble peak-picking from saved diagnostics NPZ files.

This tool avoids expensive Monte Carlo re-runs by reusing saved run-surfaces
(`*_runs_S.npz`) and rerunning only ensemble peak selection.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np

from process.cdcConfig import (
    ENS_DELTA_MIN,
    FD_DIST_FRAC,
    FP_PROM_FRAC,
    FR_RUN_REL,
    FS_SUPPORT,
    FW_WIN_FRAC,
    PER_RUN_MIN_DIST,
    PER_RUN_MIN_WIDTH,
    PER_RUN_PROM_FRAC,
    RMIN_RUNS,
    FV_VALLEY_FRAC,
)
from process.cdc.filtering import _collapse_ci_clusters
from process.cdc.surfaces import _smooth_frac_for_grid
from process.cdcUtils import infer_tier as _infer_tier
from process.ensemble import build_ensemble_catalogue


def _parse_float_list(text: str) -> List[float]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("No valid fp values supplied")
    return vals


def _fmt_list(vals: List[float], ndp: int = 4) -> str:
    if not vals:
        return ""
    return ";".join(f"{float(v):.{ndp}f}" for v in vals)


def _iter_runs_npz(source: Path) -> Iterator[Tuple[str, str, Dict[str, np.ndarray]]]:
    """
    Yield (sample_name, source_entry, arrays) for each *_runs_S.npz.
    arrays keys: age_Ma, S_runs_raw, S_runs_pen, optima_Ma(optional)
    """
    if source.is_dir():
        for p in sorted(source.rglob("*_runs_S.npz")):
            if p.name.startswith("._"):
                continue
            sample = p.name[: -len("_runs_S.npz")]
            # Manuscript archives are trusted local artifacts; allow pickle for older NPZ payloads.
            with np.load(p, allow_pickle=True) as z:
                out = {k: np.array(z[k]) for k in z.files}
            yield sample, str(p), out
        return

    if source.is_file() and source.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(source, "r:gz") as tf:
            members = [
                m
                for m in tf.getmembers()
                if m.isfile()
                and m.name.endswith("_runs_S.npz")
                and (not Path(m.name).name.startswith("._"))
            ]
            for m in sorted(members, key=lambda x: x.name):
                sample = Path(m.name).name[: -len("_runs_S.npz")]
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                buf = io.BytesIO(fh.read())
                # Manuscript archives are trusted local artifacts; allow pickle for older NPZ payloads.
                with np.load(buf, allow_pickle=True) as z:
                    out = {k: np.array(z[k]) for k in z.files}
                yield sample, m.name, out
        return

    raise ValueError("source must be a directory or a .tar.gz file")


def _postprocess_rows(
    rows: List[Dict],
    ages_ma: np.ndarray,
    *,
    support_min: float,
    collapse_overlap: bool,
    max_ci_frac: float,
) -> List[Dict]:
    """Mirror cdc_pipeline peak post-filters for replay consistency."""
    rows = [dict(r) for r in rows]
    if collapse_overlap:
        rows = _collapse_ci_clusters(rows)
    if not rows:
        return []

    step = float(np.median(np.diff(ages_ma))) if ages_ma.size >= 2 else 5.0
    min_age, max_age = float(ages_ma[0]), float(ages_ma[-1])

    fixed = []
    for r in rows:
        a = float(r["age_ma"])
        lo = float(r["ci_low"])
        hi = float(r["ci_high"])
        w = max(hi - lo, step)

        if (a < lo) or (a > hi):
            lo, hi = a - 0.5 * w, a + 0.5 * w
            lo, hi = max(lo, min_age), min(hi, max_age)
            if (hi - lo) < step:
                lo, hi = max(a - step, min_age), min(a + step, max_age)

        fixed.append(dict(r, ci_low=lo, ci_high=hi))
    rows = fixed

    cleaned = []
    for r in rows:
        a = float(r["age_ma"])
        lo = float(r["ci_low"])
        hi = float(r["ci_high"])
        if (hi - lo) < step:
            lo, hi = a - step, a + step
        lo, hi = max(lo, min_age), min(hi, max_age)
        near_edge = (a - min_age) <= step or (max_age - a) <= step
        degenerate = (hi - lo) <= 0.75 * step
        if near_edge and degenerate:
            if float(r.get("support", 0.0)) >= max(float(support_min), 0.12):
                lo, hi = a - step, a + step
                lo, hi = max(lo, min_age), min(hi, max_age)
            else:
                continue
        cleaned.append(dict(r, ci_low=lo, ci_high=hi))
    rows = cleaned

    for r in rows:
        a = float(r["age_ma"])
        if not (float(r["ci_low"]) <= a <= float(r["ci_high"])):
            r["ci_low"] = min(float(r["ci_low"]), a)
            r["ci_high"] = max(float(r["ci_high"]), a)

    total_span = float(ages_ma[-1] - ages_ma[0]) if ages_ma.size >= 2 else float("inf")
    if np.isfinite(total_span) and total_span > 0.0:
        rows = [r for r in rows if (float(r["ci_high"]) - float(r["ci_low"])) <= float(max_ci_frac) * total_span]

    rows = sorted(rows, key=lambda d: float(d["age_ma"]))
    for i, r in enumerate(rows, 1):
        r["peak_no"] = i
    return rows


def run(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_values = _parse_float_list(args.fp_values)
    sample_rx = re.compile(args.sample_regex) if args.sample_regex else None

    peaks_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    n_inputs = 0

    for sample_name, source_entry, arr in _iter_runs_npz(source):
        if sample_rx and not sample_rx.search(sample_name):
            continue

        n_inputs += 1
        ages_ma = np.asarray(arr["age_Ma"], float)
        s_raw = np.asarray(arr["S_runs_raw"], float)
        s_pen = np.asarray(arr["S_runs_pen"], float)
        optima_ma = np.asarray(arr["optima_Ma"], float) if "optima_Ma" in arr else None
        smf = _smooth_frac_for_grid(ages_ma)
        tier = _infer_tier(sample_name)

        for fp in fp_values:
            s_runs = s_pen if args.surface == "pen" else s_raw
            rows = build_ensemble_catalogue(
                sample_name,
                tier,
                ages_ma,
                s_runs,
                orientation="max",
                smooth_frac=smf,
                f_d=float(args.fd),
                f_p=float(fp),
                f_v=float(args.fv),
                f_w=float(args.fw),
                w_min_nodes=int(args.w_min_nodes),
                support_min=float(args.support_min),
                r_min=int(args.r_min),
                f_r=float(args.f_r),
                per_run_prom_frac=float(args.per_run_prom_frac),
                per_run_min_dist=int(args.per_run_min_dist),
                per_run_min_width=int(args.per_run_min_width),
                per_run_require_full_prom=bool(args.per_run_require_full_prom),
                delta_min=float(args.delta_min),
                height_frac=float(args.height_frac),
                optima_ma=optima_ma,
            ) or []

            rows = _postprocess_rows(
                rows,
                ages_ma,
                support_min=float(args.support_min),
                collapse_overlap=bool(args.collapse_overlap),
                max_ci_frac=float(args.max_ci_frac),
            )

            ages = [float(r["age_ma"]) for r in rows]
            cis = [f"{float(r['ci_low']):.4f}-{float(r['ci_high']):.4f}" for r in rows]
            sup = [float(r.get("support", float("nan"))) for r in rows]

            summary_rows.append(
                dict(
                    sample=sample_name,
                    source_entry=source_entry,
                    surface=args.surface,
                    fp=float(fp),
                    n_runs=int(s_runs.shape[0]),
                    n_grid=int(s_runs.shape[1]),
                    n_peaks=len(rows),
                    peak_ages_ma=_fmt_list(ages, 4),
                    peak_ci_ma=";".join(cis),
                    peak_support=_fmt_list(sup, 4),
                )
            )

            for r in rows:
                peaks_rows.append(
                    dict(
                        sample=sample_name,
                        source_entry=source_entry,
                        surface=args.surface,
                        fp=float(fp),
                        peak_no=int(r.get("peak_no", 0)),
                        age_ma=float(r["age_ma"]),
                        ci_low_ma=float(r["ci_low"]),
                        ci_high_ma=float(r["ci_high"]),
                        support=float(r.get("support", float("nan"))),
                    )
                )

    summary_csv = out_dir / "replay_summary.csv"
    peaks_csv = out_dir / "replay_peaks.csv"
    meta_json = out_dir / "replay_meta.json"

    with summary_csv.open("w", newline="") as fh:
        fieldnames = [
            "sample",
            "source_entry",
            "surface",
            "fp",
            "n_runs",
            "n_grid",
            "n_peaks",
            "peak_ages_ma",
            "peak_ci_ma",
            "peak_support",
        ]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    with peaks_csv.open("w", newline="") as fh:
        fieldnames = [
            "sample",
            "source_entry",
            "surface",
            "fp",
            "peak_no",
            "age_ma",
            "ci_low_ma",
            "ci_high_ma",
            "support",
        ]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(peaks_rows)

    meta = dict(
        source=str(source),
        n_samples_processed=n_inputs,
        n_summary_rows=len(summary_rows),
        n_peak_rows=len(peaks_rows),
        args=vars(args),
    )
    meta_json.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {summary_csv}")
    print(f"Wrote {peaks_csv}")
    print(f"Wrote {meta_json}")
    print(f"Processed {n_inputs} samples")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Replay CDC ensemble peak-picking from saved *_runs_S.npz surfaces."
    )
    ap.add_argument("--source", required=True, help="Directory or .tar.gz containing *_runs_S.npz files")
    ap.add_argument("--out-dir", required=True, help="Output directory for replay CSV/JSON files")
    ap.add_argument("--surface", choices=("pen", "raw"), default="pen", help="Surface to pick on")
    ap.add_argument("--fp-values", default=f"{FP_PROM_FRAC}", help="Comma-separated fp values (e.g. 0.10,0.07,0.05)")
    ap.add_argument("--sample-regex", default="", help="Optional regex to filter sample names")

    ap.add_argument("--delta-min", type=float, default=ENS_DELTA_MIN)
    ap.add_argument("--fd", type=float, default=FD_DIST_FRAC)
    ap.add_argument("--fv", type=float, default=FV_VALLEY_FRAC)
    ap.add_argument("--fw", type=float, default=FW_WIN_FRAC)
    ap.add_argument("--height-frac", type=float, default=0.0)
    ap.add_argument("--support-min", type=float, default=FS_SUPPORT)
    ap.add_argument("--r-min", type=int, default=RMIN_RUNS)
    ap.add_argument("--f-r", type=float, default=FR_RUN_REL)
    ap.add_argument("--w-min-nodes", type=int, default=3)
    ap.add_argument("--per-run-prom-frac", type=float, default=PER_RUN_PROM_FRAC)
    ap.add_argument("--per-run-min-dist", type=int, default=PER_RUN_MIN_DIST)
    ap.add_argument("--per-run-min-width", type=int, default=PER_RUN_MIN_WIDTH)
    ap.add_argument("--per-run-require-full-prom", action="store_true")

    ap.add_argument(
        "--collapse-overlap",
        dest="collapse_overlap",
        action="store_true",
        default=True,
        help="Apply CI-overlap collapse post-filter (default: on)",
    )
    ap.add_argument(
        "--no-collapse-overlap",
        dest="collapse_overlap",
        action="store_false",
        help="Disable CI-overlap collapse post-filter",
    )
    ap.add_argument("--max-ci-frac", type=float, default=0.50, help="Drop peaks with CI width above this fraction of grid span")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
