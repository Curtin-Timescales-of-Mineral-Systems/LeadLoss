from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "LeadLoss"

def user_data_dir(app_name: str = APP_NAME) -> Path:
    """
    Cross-platform user-writable data directory.

    macOS:   ~/Library/Application Support/LeadLoss
    Windows: %APPDATA%\\LeadLoss
    Linux:   $XDG_DATA_HOME/LeadLoss or ~/.local/share/LeadLoss
    """
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / app_name
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))) / app_name
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / app_name

    base.mkdir(parents=True, exist_ok=True)
    return base

def save_data_path() -> str:
    return str(user_data_dir() / "leadloss_save_data.pkl")
