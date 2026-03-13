#!/usr/bin/env python3
"""Dedicated CDC goodness-grid figure for Case 8 fan-to-zero benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import fig03_fig05_cdc_goodness_grids as cdc_grid


def _default_paper_dir() -> Path:
    p = Path(__file__).resolve()
    cand = p.parents[2]
    if (cand / "data").is_dir() and (cand / "scripts").is_dir():
        return cand
    for parent in p.parents:
        if (parent / "data").is_dir() and (parent / "scripts").is_dir():
            return parent
    return cand


def main() -> None:
    paper_dir = _default_paper_dir()
    default_derived = paper_dir / "data" / "derived" / "case8_fan_to_zero"

    ap = argparse.ArgumentParser(description="Case 8 fan-to-zero CDC goodness grid.")
    ap.add_argument("--paper-dir", type=Path, default=paper_dir, help="Paper root directory.")
    ap.add_argument("--derived-dir", type=Path, default=default_derived, help="Derived-data directory for the isolated Case 8 run.")
    ap.add_argument("--fig-dir", type=Path, default=(paper_dir / "outputs" / "figures"), help="Output directory for figures.")
    ap.add_argument("--curve-surface", type=str, default="pen", choices=["pen", "raw"], help="Which CDC surface to draw.")
    ap.add_argument("--overlay-mode", type=str, default="curve", choices=["curve", "above"], help="How to draw catalogue intervals.")
    ap.add_argument("--formats", type=str, default="png,pdf,svg", help="Comma-separated output formats.")
    ap.add_argument("--no-save", action="store_true", help="Do not write figure files.")
    ap.add_argument("--no-show", action="store_true", help="Do not display interactively.")
    args = ap.parse_args()

    derived_dir = args.derived_dir.expanduser().resolve()
    ks_dir = derived_dir / "ks_diagnostics"
    catalogue_csv = derived_dir / "ensemble_catalogue.csv"
    fig_dir = args.fig_dir.expanduser().resolve()
    formats = [s.strip().lower() for s in str(args.formats).split(",") if s.strip()]

    # Case 8 is a falsification benchmark rather than a discrete-age recovery target.
    cdc_grid.TRUE["8"] = []
    catalogue_map = cdc_grid.load_catalogue_table(catalogue_csv)

    cdc_grid.make_grid(
        cases=["8"],
        ks_dir=ks_dir,
        curve_surface=str(args.curve_surface).strip().lower(),
        triangle_surface=str(args.curve_surface).strip().lower(),
        title="CDC goodness surfaces — Case 8 fan-to-zero",
        outfile_stub="fig_case8_fan_to_zero_cdc_goodness_grid",
        catalogue_map=catalogue_map,
        outdir=fig_dir,
        formats=formats,
        overlay_mode=args.overlay_mode,
        legend_compact=True,
        show_median_peaks=False,
        y_max=1.0,
        save=not args.no_save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
