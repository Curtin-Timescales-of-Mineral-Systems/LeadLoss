"""Public facade for ensemble CDC peak picking.

External callers should continue to import:
- `process.ensemble.per_run_peaks`
- `process.ensemble.robust_ensemble_curve`
- `process.ensemble.build_ensemble_catalogue`

The implementation now lives in the internal `process.ensemble_internal`
package so the public API stays stable while the code is split by role.
"""

from __future__ import annotations

from process.ensemble_internal.catalogue import build_ensemble_catalogue
from process.ensemble_internal.curve import per_run_peaks, robust_ensemble_curve

__all__ = [
    "build_ensemble_catalogue",
    "per_run_peaks",
    "robust_ensemble_curve",
]
