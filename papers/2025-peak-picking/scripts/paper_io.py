from pathlib import Path
import argparse
import sys
from typing import Union


def infer_paper_dir(script_file: Union[str, Path]) -> Path:
    p = Path(script_file).resolve()
    for parent in p.parents:
        if parent.name == "2025-peak-picking":
            return parent
    raise RuntimeError("Could not locate papers/2025-peak-picking from script path.")


def add_src_to_path(script_file: Union[str, Path]) -> Path:
    """
    Ensure the repo's ./src folder is on sys.path so paper scripts can import `process.*`.
    Returns the src_dir Path.
    """
    paper_dir = infer_paper_dir(script_file)     # .../papers/2025-peak-picking
    repo_root = paper_dir.parents[1]             # .../LeadLoss (repo root)
    src_dir = repo_root / "src"

    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir


def parse_paper_args(script_file: Union[str, Path]):
    paper_dir = infer_paper_dir(script_file)
    data_dir  = paper_dir / "data"
    out_dir   = paper_dir / "outputs"

    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", dest="paper_dir", type=Path, default=paper_dir)
    parser.add_argument("--data-dir",  dest="data_dir",  type=Path, default=data_dir)
    parser.add_argument("--out-dir",   dest="out_dir",   type=Path, default=out_dir)

    return parser.parse_args()
