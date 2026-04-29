"""Small shared helpers for the CDC pipeline."""

from __future__ import annotations

import re
import zlib


def seed_from_name(name: str, base: int = 42) -> int:
    """Deterministic 32-bit seed derived from a sample name."""
    return (base ^ zlib.adler32((name or "").encode("utf-8"))) & 0xFFFFFFFF


def safe_prefix(name: str) -> str:
    """Sanitise a string for use in filenames."""
    s = str(name or "").strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_.")
    return s or "sample"


def infer_tier(sample_name: str) -> str:
    """Infer the scatter 'tier' (A/B/C/...) from a sample name suffix, if present."""
    s = (sample_name or "").strip()
    return s[-1].upper() if s and s[-1].isalpha() else ""
