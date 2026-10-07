# -*- mode: python ; coding: utf-8 -*-

import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


PROJECT_ROOT = Path.cwd()
CONFIG_TEXT = (PROJECT_ROOT / "src" / "utils" / "config.py").read_text(encoding="utf-8")
VERSION_MATCH = re.search(r'^VERSION\s*=\s*["\']([^"\']+)["\']', CONFIG_TEXT, re.MULTILINE)
if VERSION_MATCH is None:
    raise RuntimeError("Could not read VERSION from src/utils/config.py")
RELEASE_LABEL = VERSION_MATCH.group(1)
MARKETING_VERSION = RELEASE_LABEL.split("-", 1)[0]

hiddenimports = ["pkg_resources.py2_warn"]
for package in ("utils", "controller", "model", "process", "view"):
    hiddenimports += collect_submodules(package)

analysis = Analysis(
    [str(PROJECT_ROOT / "src" / "application.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=[],
    datas=[(str(PROJECT_ROOT / "resources"), "resources")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="LeadLoss",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "resources" / "icon.icns"),
)

collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="LeadLoss",
)

app = BUNDLE(
    collection,
    name="LeadLoss.app",
    icon=str(PROJECT_ROOT / "resources" / "icon.icns"),
    bundle_identifier="au.edu.curtin.timescales.leadloss",
    version=MARKETING_VERSION,
    info_plist={
        "CFBundleVersion": MARKETING_VERSION,
        "LeadLossReleaseLabel": RELEASE_LABEL,
    },
)
