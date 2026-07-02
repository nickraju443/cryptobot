# PyInstaller spec for the single-file CryptoBot.exe build.
# Bundles Python + all bot deps + templates into one executable so a
# non-technical user can just double-click it.
#
# Build:
#   pyinstaller CryptoBot.spec --clean --noconfirm
# Output:
#   dist/CryptoBot.exe

# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Files the bot reads at runtime — must be bundled with the exe.
# We DON'T bundle .env / andx_credentials.json / portfolio_*.json — those
# are user data, the bot creates them next to the .exe at runtime.
datas = [
    ("templates/dashboard.html", "templates"),
    ("STRATEGY.pdf", "."),
    ("STRATEGY.md", "."),
]

# Hidden imports PyInstaller's static analysis misses. ML libs use dynamic
# import patterns that need to be enumerated explicitly.
hiddenimports = []
hiddenimports += collect_submodules("xgboost")
hiddenimports += collect_submodules("lightgbm")
hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("flask")
hiddenimports += [
    "pandas",
    "numpy",
    "requests",
    "json",
    "hmac",
    "hashlib",
]

# Excludes: keep the binary as small as possible. The bot's documented-API
# path doesn't need these heavy optional deps. If the user wants the
# website-session features later, they can use the .py source build.
excludes = [
    "playwright",        # ~200MB browser bundle, only needed for /p/v1/ scraping
    "ccxt",              # generic exchange lib, ~150MB of unused adapters
    "matplotlib",        # not used in headless mode
    "tkinter",           # we're a web app, no GUI
    "PyQt5", "PyQt6", "PySide2", "PySide6",
    "test", "tests", "unittest",
    "notebook", "ipython", "IPython",
    "jupyter",
]

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="CryptoBot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,              # keep console — Pat sees if anything goes wrong
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
