# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_data_files

collect = collect_submodules('asr_api_server')
print(collect)
hiddenimports_set = [
    'fastapi',
    'fastapi.staticfiles',
    'uvicorn',
    'numpy',
    'soundfile',
    'aiohttp',
    'torch',
    'librosa',
    'sklearn.preprocessing.label',
    'websockets',
    ]
hiddenimports = collect + hiddenimports_set


data_modules = ['librosa', 'asr_api_server']
datas = []
for module in data_modules:
    dd = collect_data_files(module)
    datas.extend(dd)
print(datas)

block_cipher = None


a = Analysis(
    ['sdc_asr_api.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
    name='sdc_asr_api',
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
    name='sdc_asr_api',
)
