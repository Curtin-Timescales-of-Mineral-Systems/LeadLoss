#!/usr/bin/env python3
"""
papers/2025-peak-picking/scripts/run_clustering_bundle.py

Regenerate the CDC-derived manuscript figures/tables from a fresh
ensemble-v2 anchor-clustered bundle without mixing them into the legacy
manuscript output folders.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def paper_dir_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root_from_paper(paper_dir: Path) -> Path:
    return paper_dir.parents[1]


def _latest_clustering_bundle(paper_dir: Path) -> Path | None:
    derived_root = paper_dir / "data" / "derived"
    candidates = [
        p for p in derived_root.glob("ensemble_v2_anchor_clustered_*") if p.is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run(cmd: list[str], *, cwd: Path, env: dict[str, str], dry_run: bool = False) -> None:
    print(">>>", " ".join(str(x) for x in cmd))
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), env=env, check=True)


def ensure_paper_dependencies() -> None:
    missing: list[str] = []
    for name in ("pandas", "numpy", "matplotlib"):
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    if missing:
        raise SystemExit(
            "The current Python interpreter does not have the paper-generation dependencies.\n"
            f"Missing modules: {', '.join(missing)}\n"
            "Run this script from the paper environment after installing:\n"
            "  pip install -r papers/2025-peak-picking/requirements.txt"
        )


def main() -> int:
    paper_dir = paper_dir_from_here()
    repo_root = repo_root_from_paper(paper_dir)
    default_bundle = _latest_clustering_bundle(paper_dir)

    ap = argparse.ArgumentParser(
        description="Rebuild CDC-derived figures/tables from a fresh ensemble-v2 anchor-clustered bundle."
    )
    ap.add_argument(
        "--derived-root",
        type=Path,
        default=default_bundle,
        help="Bundle root (default: most recent papers/2025-peak-picking/data/derived/ensemble_v2_anchor_clustered_* folder).",
    )
    ap.add_argument(
        "--rei-dir",
        type=Path,
        default=(paper_dir / "data" / "derived" / "reimink_discordance_dating"),
        help="Directory containing Reimink/DD outputs for comparison tables.",
    )
    ap.add_argument(
        "--dd-runtime-log",
        type=Path,
        default=(paper_dir / "data" / "derived" / "runtime_log_reimink.csv"),
        help="DD runtime log CSV used for Tables 11–12.",
    )
    ap.add_argument(
        "--upgrade-sample-id",
        type=str,
        default="2A",
        help="Sample ID for the Fig. 9 CDC upgrade diagnostic.",
    )
    ap.add_argument(
        "--out-subdir",
        type=str,
        default=None,
        help="Output subdirectory under papers/2025-peak-picking/outputs/ (default: clustering/<bundle_name>).",
    )
    ap.add_argument("--clean", action="store_true", help="Remove the chosen output directory before running.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = ap.parse_args()

    if args.derived_root is None:
        raise SystemExit(
            "No anchor-clustered bundle found automatically.\n"
            "Pass --derived-root papers/2025-peak-picking/data/derived/<bundle_name>"
        )

    ensure_paper_dependencies()

    derived_root = args.derived_root.expanduser().resolve()
    if not derived_root.exists():
        raise SystemExit(f"Derived bundle not found: {derived_root}")

    required = [
        derived_root / "ensemble_catalogue.csv",
        derived_root / "ks_diagnostics",
        derived_root / "ks_exports",
        derived_root / "runtime_log.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "Derived bundle is incomplete. Missing:\n- " + "\n- ".join(missing)
        )

    bundle_name = derived_root.name
    out_subdir = args.out_subdir or f"ensemble_v2_anchor_clustered/{bundle_name}"
    out_root = paper_dir / "outputs" / out_subdir
    out_fig_dir = out_root / "figures"
    out_tables_subdir = f"{out_subdir}/tables"

    if args.clean and out_root.exists():
        print(f"[clean] removing {out_root}")
        if not args.dry_run:
            shutil.rmtree(out_root, ignore_errors=True)

    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONHASHSEED", "0")

    py = sys.executable

    tables = [
        [
            py,
            paper_dir / "scripts" / "tables" / "tables03_to_08_benchmark_results.py",
            "--catalogue",
            derived_root / "ensemble_catalogue.csv",
            "--rei-dir",
            args.rei_dir.expanduser().resolve(),
            "--out-subdir",
            out_tables_subdir,
        ],
        [
            py,
            paper_dir / "scripts" / "tables" / "tables_10_12_runtime_tables.py",
            "--cdc-log",
            derived_root / "runtime_log.csv",
            "--dd-log",
            args.dd_runtime_log.expanduser().resolve(),
            "--out-subdir",
            out_tables_subdir,
        ],
    ]

    figures = [
        [
            py,
            paper_dir / "scripts" / "figures" / "fig03_fig05_cdc_goodness_grids.py",
            "--ks-dir",
            derived_root / "ks_diagnostics",
            "--catalogue-csv",
            derived_root / "ensemble_catalogue.csv",
            "--fig-dir",
            out_fig_dir,
            "--no-show",
        ],
        [
            py,
            paper_dir / "scripts" / "figures" / "fig07_ks_goodness_case2a.py",
            "--ks-dir",
            derived_root / "ks_exports",
            "--fig-dir",
            out_fig_dir,
            "--no-show",
        ],
        [
            py,
            paper_dir / "scripts" / "figures" / "fig09_cdc_upgrade.py",
            "--sample-id",
            args.upgrade_sample_id,
            "--runs-npz",
            derived_root / "ks_diagnostics" / f"{args.upgrade_sample_id}_runs_S.npz",
            "--catalogue-csv",
            derived_root / "ensemble_catalogue.csv",
            "--fig-dir",
            out_fig_dir,
            "--no-show",
        ],
        [
            py,
            paper_dir / "scripts" / "figures" / "fig_case8_fan_to_zero_cdc.py",
            "--derived-dir",
            derived_root,
            "--fig-dir",
            out_fig_dir,
            "--no-show",
        ],
    ]

    for cmd in tables:
        run(cmd, cwd=repo_root, env=env, dry_run=args.dry_run)
    for cmd in figures:
        run(cmd, cwd=repo_root, env=env, dry_run=args.dry_run)

    print(f"[run_clustering_bundle] DONE -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
