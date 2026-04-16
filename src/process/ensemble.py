"""Re-exports from process.ensemble_internal for backwards compatibility."""

from __future__ import annotations

from process.ensemble_internal.catalogue import build_ensemble_catalogue
from process.ensemble_internal.curve import per_run_peaks, robust_ensemble_curve

__all__ = [
    "build_ensemble_catalogue",
    "per_run_peaks",
    "robust_ensemble_curve",
]
