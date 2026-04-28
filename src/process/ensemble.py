"""Compatibility shim for historical ensemble imports.

New code should import from ``process.ensemble_internal`` where practical.
This module is retained so older callers keep working.
"""

from __future__ import annotations

from process.ensemble_internal.catalogue import build_ensemble_catalogue
from process.ensemble_internal.curve import per_run_peaks, robust_ensemble_curve

__all__ = [
    "build_ensemble_catalogue",
    "per_run_peaks",
    "robust_ensemble_curve",
]
