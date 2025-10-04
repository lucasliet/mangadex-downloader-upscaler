# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

# Detect basicsr installation path automatically
basicsr_path = None
for path in sys.path:
    candidate = Path(path) / 'basicsr'
    if candidate.exists():
        basicsr_path = candidate
        break

datas_list = [
    ('mangadex_downloader/fonts', 'mangadex_downloader/fonts'),
    ('mangadex_downloader/images', 'mangadex_downloader/images'),
    ('mangadex_downloader/tracker/sql_files', 'mangadex_downloader/tracker/sql_files'),
    ('mangadex_downloader/tracker/sql_migrations', 'mangadex_downloader/tracker/sql_migrations'),
]

if basicsr_path:
    datas_list.append((str(basicsr_path / 'archs'), 'basicsr/archs'))

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=datas_list,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='mangadex-dl_x64',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='mangadex-dl_x64',
)
