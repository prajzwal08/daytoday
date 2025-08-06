#!/usr/bin/env python3
import zipfile
from pathlib import Path

SRC_DIR  = Path("/home/khanalp/Ecosystem_final quality_(L2)_product_in_ETC-Archive_format-release_2025-1")
DST_ROOT = Path("/home/khanalp/data/ICOS2025")

for zip_path in SRC_DIR.glob("*.zip"):
    # filename parts: ['ICOSETC', 'FR-FBn', 'METEOSENS', 'L2.zip']
    parts = zip_path.name.split("_")
    station = parts[1]                      # 'FR-FBn'

    target_dir = DST_ROOT / station
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    print(f"✓ {zip_path.name} → {target_dir}")
    
