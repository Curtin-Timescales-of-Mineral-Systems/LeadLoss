#!/usr/bin/env python3
"""
papers/2025-peak-picking/scripts/run_all.py

One-command reproduction runner for the 2025 peak-picking manuscript bundle.

What it does
------------
1) (Optional) Cleans paper outputs:
     papers/2025-peak-picking/outputs/{tables,figures,derived}
2) Ensures the K–S diagnostics NPZ surfaces are available by extracting:
     papers/2025-peak-picking/data/derived/ks_diagnostics_npz.tar.gz
   into:
     papers/2025-peak-picking/data/derived/ks_diagnostics/
3) Runs all table scripts (writes to outputs/tables)
4) Runs all figure scripts (writes to outputs/figures)

Run
---
From anywhere (repo-root not required):
  python papers/2025-peak-picking/scripts/run_all.py --clean

Outputs
-------
- papers/2025-peak-picking/outputs/tables/
- papers/2025-peak-picking/outputs/figures/
- papers/2025-peak-picking/outputs/PROVENANCE.txt

Notes
-----
- This script runs figure scripts headlessly by setting MPLBACKEND=Agg.
- It expects fig07_ks_goodness_case2a.py to support --no-show and to save files.
"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def paper_dir_from_here() -> Path:
    # <paper>/scripts/run_all.py -> paper dir is parents[1]
    return Path(__file__).resolve().parents[1]


def repo_root_from_paper(paper_dir: Path) -> Path:
    # <repo>/papers/2025-peak-picking -> repo root is parents[1] of papers/
    return paper_dir.parents[1]


def run(cmd: list[str], *, cwd: Path, env: dict, dry_run: bool = False) -> None:
    print(">>>", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def extract_ks_bundle(paper_dir: Path, *, dry_run: bool = False) -> None:
    tar_path = paper_dir / "data" / "derived" / "ks_diagnostics_npz.tar.gz"
    dest_dir = paper_dir / "data" / "derived"
    ks_dir   = dest_dir / "ks_diagnostics"

    if ks_dir.exists() and any(ks_dir.rglob("*.npz")):
        print(f"[ks] ok: {ks_dir} already present")
        return

    if not tar_path.exists():
        raise FileNotFoundError(f"Missing KS bundle: {tar_path}")

    print(f"[ks] extracting {tar_path} -> {dest_dir}")
    if dry_run:
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=dest_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all tables + figures for the 2025 peak-picking manuscript.")
    ap.add_argument("--clean", action="store_true", help="Delete papers/2025-peak-picking/outputs before running.")
    ap.add_argument("--skip-extract", action="store_true", help="Skip extracting the KS diagnostics tarball.")
    ap.add_argument("--sample-id", default="2A", help="Sample ID for Fig08 (default: 2A).")
    ap.add_argument("--dry-run", action="store_true", help="Print commands, do not execute.")
    args = ap.parse_args()

    paper_dir = paper_dir_from_here()
    repo_root = repo_root_from_paper(paper_dir)

    outputs_dir = paper_dir / "outputs"
    if args.clean and outputs_dir.exists():
        print(f"[clean] removing {outputs_dir}")
        if not args.dry_run:
            shutil.rmtree(outputs_dir, ignore_errors=True)

    # Headless / CI friendly environment
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONHASHSEED", "0")

    if not args.skip_extract:
        extract_ks_bundle(paper_dir, dry_run=args.dry_run)

    py = sys.executable
    ks_dir = paper_dir / "data" / "derived" / "ks_diagnostics"
    dd_dir = paper_dir / "data" / "derived" / "reimink_discordance_dating"
    out_fig_dir = paper_dir / "outputs" / "figures"

    # Tables
    table_cmds = [
        [py, str(paper_dir / "scripts" / "tables" / "tables_01_02_benchmark_definitions.py")],
        [py, str(paper_dir / "scripts" / "tables" / "tables03_to_08_benchmark_results.py")],
        [py, str(paper_dir / "scripts" / "tables" / "tables_09_concordant_fraction_sweep.py")],
        [py, str(paper_dir / "scripts" / "tables" / "tables_10_12_runtime_tables.py")],
    ]

    # Figures
    figure_cmds = [
        [py, str(paper_dir / "scripts" / "figures" / "fig01_synthetic_cases1to4.py")],
        [
            py, str(paper_dir / "scripts" / "figures" / "fig02_synthetic_cases5to7.py"),
            "--save-fig",
            "--fig-dir", str(out_fig_dir),
            "--formats", "svg,png,pdf",
        ],
        [
            py, str(paper_dir / "scripts" / "figures" / "fig03_fig05_cdc_goodness_grids.py"),
            "--ks-dir", str(ks_dir),
            "--no-show",
        ],
        [
            py, str(paper_dir / "scripts" / "figures" / "fig04_fig06_dd_likelihood_grids.py"),
            "--dd-dir", str(dd_dir),
            "--no-show",
        ],
        [
            py, str(paper_dir / "scripts" / "figures" / "fig07_ks_goodness_case2a.py"),
            "--no-show",
            "--fig-dir", str(out_fig_dir),
            "--outfile-stub", "ks_failure",
            "--formats", "png,pdf,svg",
        ],
        [
            py, str(paper_dir / "scripts" / "figures" / "fig08_cdc_upgrade.py"),
            "--sample-id", str(args.sample_id),
            "--no-show",
            "--fig-dir", str(out_fig_dir),
        ],
    ]

    print("[run_all] repo_root:", repo_root)
    print("[run_all] paper_dir:", paper_dir)
    print("[run_all] outputs :", outputs_dir)

    for cmd in table_cmds:
        run(cmd, cwd=repo_root, env=env, dry_run=args.dry_run)

    for cmd in figure_cmds:
        run(cmd, cwd=repo_root, env=env, dry_run=args.dry_run)

    print("[run_all] DONE")
    print("  tables :", paper_dir / "outputs" / "tables")
    print("  figures:", paper_dir / "outputs" / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
