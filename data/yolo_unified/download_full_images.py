#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga imagenes completas de Smoke24776 desde Google Drive.
Requiere: pip install gdown
"""
import gdown
import zipfile
from pathlib import Path

# Google Drive folder ID
FOLDER_URL = "https://drive.google.com/drive/folders/1FiZJ47TS-Ac0mBOfNGM2RChsNcGEXxVl"
# Resolve output directory relative to this script
OUTPUT_DIR = Path(__file__).parent

def main():
    print("Descargando imagenes completas de Smoke24776...")
    gdown.download_folder(FOLDER_URL, output=str(OUTPUT_DIR), quiet=False, use_cookies=False)

    # Descomprimir si vienen en zip
    for zip_file in OUTPUT_DIR.glob("*.zip"):
        print(f"Descomprimiendo {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(OUTPUT_DIR)
        zip_file.unlink()

    print("Listo! Imagenes en:", OUTPUT_DIR / "images")

if __name__ == "__main__":
    main()