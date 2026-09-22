#!/usr/bin/env python3
"""
Descarga, limpieza y preparación de los 3 datasets para AirGuard.

Datasets:
1. IoT Smoke Detection (HamzaMa96) - CSV en repo GitHub
2. Dalton Dataset (prasenjit52282) - Procesado en repo, raw en SharePoint
3. Smoke24776 (linjiefengFutureMediaSZU) - Anotaciones en repo, imágenes externas

Uso:
    python prepare_datasets.py --all
    python prepare_datasets.py --iot-smoke
    python prepare_datasets.py --dalton
    python prepare_datasets.py --smoke24776
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
YOLO_DIR = BASE_DIR / "data" / "yolo_unified"
HARD_NEG_DIR = BASE_DIR / "data" / "hard_negatives"

REPOS = {
    "iot_smoke": {
        "url": "https://github.com/HamzaMa96/Smoke-Detection-IOT.git",
        "dir": RAW_DIR / "Smoke-Detection-IOT",
        "csv_file": "smoke_detection_iot.csv",
    },
    "dalton": {
        "url": "https://github.com/prasenjit52282/dalton-dataset.git",
        "dir": RAW_DIR / "dalton-dataset",
        "processed_dir": "Processed",
        "metadata_dir": "Metadata",
    },
    "smoke24776": {
        "url": "https://github.com/linjiefengFutureMediaSZU/Smoke24776.git",
        "dir": RAW_DIR / "Smoke24776",
        "annotations": {
            "coco": "Smoke24776_coco_format/annotations",
            "voc": "Smoke24776_voc_format/Annotations",
            "yolo": "Smoke24776_yolo_format/labels",
        },
        "image_sources": {
            "baidu": "https://pan.baidu.com/s/19hBmTecDicXTmkRSCaET9Q?pwd=w3e4",
            "gdrive": "https://drive.google.com/drive/folders/1FiZJ47TS-Ac0mBOfNGM2RChsNcGEXxVl",
        },
    },
}

# URLs de descarga directa (si están disponibles)
DIRECT_DOWNLOADS = {
    "iot_smoke_csv": "https://raw.githubusercontent.com/HamzaMa96/Smoke-Detection-IOT/main/smoke_detection_iot.csv",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Ejecuta comando y muestra output en tiempo real."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Comando falló: {' '.join(cmd)}")
    return result


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Descarga archivo con barra de progreso."""
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ❌ Error descargando {url}: {e}")
        return False


def clone_repo(url: str, dest: Path) -> bool:
    """Clona repositorio Git (shallow para ahorrar espacio)."""
    if dest.exists():
        print(f"  📁 Repo ya existe en {dest}, actualizando...")
        run_cmd(["git", "pull"], cwd=dest, check=False)
        return True
    try:
        run_cmd(["git", "clone", "--depth", "1", url, str(dest)])
        return True
    except Exception as e:
        print(f"  ❌ Error clonando {url}: {e}")
        return False


def verify_checksum(filepath: Path, expected: Optional[str] = None) -> str:
    """Calcula SHA256 y verifica si se proporciona expected."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    if expected and digest != expected:
        raise ValueError(f"Checksum mismatch: {digest} != {expected}")
    return digest


# ---------------------------------------------------------------------------
# 1. IoT Smoke Detection Dataset
# ---------------------------------------------------------------------------

def prepare_iot_smoke() -> Path:
    """
    Descarga y limpia IoT Smoke Detection Dataset.
    Salida: data/processed/iot_smoke_clean.csv con features listas para Experto 2.
    """
    print("\n" + "="*60)
    print("[DOWNLOAD] PREPARANDO IoT SMOKE DETECTION DATASET")
    print("="*60)

    repo_info = REPOS["iot_smoke"]
    repo_dir = repo_info["dir"]

    # Clonar repo (solo CSV, es pequeño)
    if not clone_repo(repo_info["url"], repo_dir):
        # Fallback: descarga directa del CSV
        csv_url = DIRECT_DOWNLOADS["iot_smoke_csv"]
        csv_dest = RAW_DIR / "iot_smoke_detection.csv"
        if not download_file(csv_url, csv_dest):
            raise RuntimeError("No se pudo obtener el CSV de IoT Smoke Detection")
        csv_path = csv_dest
    else:
        csv_path = repo_dir / repo_info["csv_file"]

    print(f"  📄 Leyendo CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"     Shape original: {df.shape}")
    print(f"     Columnas: {list(df.columns)}")

    # Limpieza y validación
    print("  🧹 Limpiando...")

    # 1. Renombrar columnas a formato estándar
    rename_map = {
        "UTC": "timestamp",
        "Temperature[C]": "temperature",
        "Humidity[%]": "humidity",
        "TVOC[ppb]": "tvoc_ppb",
        "eCO2[ppm]": "eco2_ppm",
        "Raw H2": "raw_h2",
        "Raw Ethanol": "raw_ethanol",
        "Pressure[hPa]": "pressure_hpa",
        "PM1.0": "pm10",
        "PM2.5": "pm25",
        "NC0.5": "nc05",
        "NC1.0": "nc10",
        "NC2.5": "nc25",
        "CNT": "cnt",
        "Fire Alarm": "fire_alarm",
    }
    df = df.rename(columns=rename_map)

    # 2. Timestamp a datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["timestamp_unix"] = df["timestamp"].astype("int64") // 10**9

    # 3. Verificar nulos
    print(f"     Nulos por columna:\n{df.isnull().sum()}")
    df = df.dropna()
    print(f"     Shape tras dropna: {df.shape}")

    # 4. Crear etiquetas
    # Principal: binario fire/no-fire (para pre-entrenamiento)
    df["fire_binary"] = df["fire_alarm"].astype(int)
    
    # Secundaria: 4 clases para fine-tuning (heurística mejorada)
    # Usar percentiles para mejor distribución
    fire_mask = df["fire_alarm"] == 1
    fire_data = df[fire_mask]
    
    # Umbrales basados en percentiles de datos con fire=1
    eco2_p75 = fire_data["eco2_ppm"].quantile(0.75)
    tvoc_p75 = fire_data["tvoc_ppb"].quantile(0.75)
    pm25_p75 = fire_data["pm25"].quantile(0.75)
    
    def assign_label_4class(row):
        if row["fire_alarm"] == 0:
            return "normal"
        # Fire=1: clasificar por perfil químico relativo
        is_high_eco2 = row["eco2_ppm"] > eco2_p75
        is_high_tvoc = row["tvoc_ppb"] > tvoc_p75
        is_high_pm25 = row["pm25"] > pm25_p75
        
        # Humo tóxico: CO alto + VOC alto (combustión incompleta, plásticos)
        if is_high_eco2 and is_high_tvoc:
            return "humo_toxico"
        # Quema orgánica: PM2.5 muy alto (biomasa)
        elif is_high_pm25 and not is_high_eco2:
            return "quema_organica"
        # Polvo/niebla falso positivo: fire=1 pero perfil químico bajo
        else:
            return "polvo_niebla"

    df["label"] = df.apply(assign_label_4class, axis=1)
    print(f"     Distribución labels (4-clase):\n{df['label'].value_counts()}")
    print(f"     Distribución fire_binary:\n{df['fire_binary'].value_counts()}")

    # 5. Features derivadas para Experto 2
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Slopes (ventanas de 60s, 120s)
    df["pm25_delta_1min"] = df["pm25"].diff(60)
    df["co_slope_2min"] = df["eco2_ppm"].diff(120) / 2  # ppm/min aprox
    df["voc_slope_1min"] = df["tvoc_ppb"].diff(60)      # ppb/min

    # Ratios
    df["pm_ratio"] = df["pm25"] / df["pm10"].replace(0, pd.NA)
    df["co_pm_ratio"] = df["eco2_ppm"] / df["pm25"].replace(0, pd.NA)

    # Corrección humedad PM2.5 (Crilley et al. 2018)
    def correct_pm25_humidity(pm25, hr, threshold=80.0, max_corr=0.45):
        if hr < threshold:
            return pm25
        span = 100.0 - threshold
        factor = 1.0 - ((hr - threshold) / span) * (1.0 - max_corr)
        factor = max(max_corr, min(1.0, factor))
        return pm25 * factor

    df["pm25_corrected"] = df.apply(lambda r: correct_pm25_humidity(r["pm25"], r["humidity"]), axis=1)

    # Hora cíclica
    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # HR bucket
    hr_bins = [0, 30, 50, 70, 85, 95, 101]
    df["hr_bucket"] = pd.cut(df["humidity"], bins=hr_bins, labels=False, right=False)

    # Temp delta 5min
    df["temp_delta_5min"] = df["temperature"].diff(300)

    # 6. Seleccionar columnas finales para entrenamiento
    feature_cols = [
        "timestamp_unix", "pm25", "pm10", "pm25_corrected", "pm_ratio",
        "pm25_delta_1min", "co_slope_2min", "voc_slope_1min",
        "eco2_ppm", "tvoc_ppb", "co_pm_ratio",
        "temperature", "temp_delta_5min", "humidity", "hr_bucket",
        "hour_sin", "hour_cos", "label", "fire_binary"
    ]

    df_clean = df[feature_cols].dropna().reset_index(drop=True)

    # 7. Guardar
    out_path = PROCESSED_DIR / "iot_smoke_clean.csv"
    df_clean.to_csv(out_path, index=False)
    print(f"  ✅ Guardado: {out_path} ({len(df_clean)} filas)")

    # 8. Split temporal 70/15/15
    n = len(df_clean)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    splits = {
        "train": df_clean.iloc[:train_end],
        "val": df_clean.iloc[train_end:val_end],
        "test": df_clean.iloc[val_end:],
    }

    for name, split_df in splits.items():
        split_path = PROCESSED_DIR / f"iot_smoke_{name}.csv"
        split_df.to_csv(split_path, index=False)
        print(f"     {name}: {len(split_df)} filas -> {split_path}")

    # Guardar metadata
    meta = {
        "source": "HamzaMa96/Smoke-Detection-IOT",
        "original_rows": len(df),
        "clean_rows": len(df_clean),
        "features": feature_cols,
        "label_distribution": df_clean["label"].value_counts().to_dict(),
        "splits": {k: len(v) for k, v in splits.items()},
        "checksum": verify_checksum(out_path),
    }
    (PROCESSED_DIR / "iot_smoke_meta.json").write_text(json.dumps(meta, indent=2))

    return out_path


# ---------------------------------------------------------------------------
# 2. Dalton Dataset
# ---------------------------------------------------------------------------

def prepare_dalton() -> Path:
    """
    Prepara Dalton Dataset.
    El repo contiene datos procesados y metadata. Raw data en SharePoint.
    Salida: data/processed/dalton_merged.csv (muestreado para pre-entrenamiento)
    """
    print("\n" + "="*60)
    print("[DOWNLOAD] PREPARANDO DALTON DATASET")
    print("="*60)

    repo_info = REPOS["dalton"]
    repo_dir = repo_info["dir"]

    if not clone_repo(repo_info["url"], repo_dir):
        raise RuntimeError("No se pudo clonar Dalton dataset")

    # Leer metadata
    meta_dir = repo_dir / repo_info["metadata_dir"]
    annotations = pd.read_csv(meta_dir / "Annotations_cleaned.csv")
    sites = pd.read_csv(meta_dir / "Site_wise_details.csv")
    occupants = pd.read_csv(meta_dir / "Occupants.csv")

    print(f"  📊 Sitios: {len(sites)}")
    print(f"  📊 Anotaciones: {len(annotations)}")
    print(f"  📊 Ocupantes: {len(occupants)}")

    # Leer datos procesados (ya limpios, por sitio/fecha)
    proc_dir = repo_dir / repo_info["processed_dir"]
    site_dirs = list(proc_dir.iterdir())

    all_dfs = []
    for site_dir in site_dirs[:5]:  # Muestrear primeros 5 sitios para no explotar memoria
        date_dirs = list(site_dir.iterdir())
        for date_dir in date_dirs[:3]:  # 3 fechas por sitio
            csv_files = list(date_dir.glob("*.csv"))
            for csv_file in csv_files[:2]:  # 2 sensores por fecha
                try:
                    df = pd.read_csv(csv_file)
                    df["site_id"] = site_dir.name
                    df["date"] = date_dir.name
                    all_dfs.append(df)
                except Exception as e:
                    print(f"    ⚠️ Error leyendo {csv_file}: {e}")

    if not all_dfs:
        raise RuntimeError("No se pudieron leer datos procesados de Dalton")

    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"  📄 Muestras cargadas: {len(merged)}")

    # Normalizar columnas
    # Columnas típicas: ts, T, H, PMS1, PMS2_5, PMS10, CO2, NO2, CO, VoC, C2H5OH, ID, Loc, Customer, Ph, Valid, Valid_CO2, bkps
    rename_map = {
        "ts": "timestamp",
        "T": "temperature",
        "H": "humidity",
        "PMS1": "pm10",
        "PMS2_5": "pm25",
        "PMS10": "pm100",
        "CO2": "co2_ppm",
        "NO2": "no2_ppb",
        "CO": "co_ppm",
        "VoC": "voc_ppb",
        "C2H5OH": "ethanol_ppb",
    }
    merged = merged.rename(columns=rename_map)

    # Timestamp
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], format="mixed", utc=True)
    merged["timestamp_unix"] = merged["timestamp"].astype("int64") // 10**9

    # Filtrar solo válidos
    if "Valid" in merged.columns:
        merged = merged[merged["Valid"] == 1]

    # Features derivadas (similar a IoT)
    merged = merged.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    merged["pm_ratio"] = merged["pm25"] / merged["pm100"].replace(0, pd.NA)
    merged["co_pm_ratio"] = merged["co_ppm"] / merged["pm25"].replace(0, pd.NA)

    # Etiquetas: usar anotaciones de actividad
    # Mapear actividades a nuestras 4 clases
    activity_map = {
        "cooking": "humo_toxico",      # cocina -> posible tóxico
        "frying": "humo_toxico",
        "grilling": "quema_organica",
        "burning": "quema_organica",
        "incense": "humo_toxico",
        "candle": "humo_toxico",
        "cleaning": "polvo_niebla",
        "dusting": "polvo_niebla",
        "ventilation": "normal",
        "window_open": "normal",
        "ac_on": "normal",
        "heating": "normal",
        "sleeping": "normal",
        "away": "normal",
    }

    # Por simplicidad, asignar "normal" a todo y luego sobremuestrear eventos
    merged["label"] = "normal"

    # Guardar muestra para pre-entrenamiento (1M filas max)
    sample_size = min(1_000_000, len(merged))
    merged_sample = merged.sample(n=sample_size, random_state=42).reset_index(drop=True)

    out_path = PROCESSED_DIR / "dalton_sample.csv"
    merged_sample.to_csv(out_path, index=False)
    print(f"  ✅ Guardado muestra: {out_path} ({len(merged_sample)} filas)")

    meta = {
        "source": "prasenjit52282/dalton-dataset",
        "original_sites": len(sites),
        "sample_rows": len(merged_sample),
        "features": list(merged_sample.columns),
        "note": "Muestra aleatoria de datos procesados. Raw data requiere descarga de SharePoint.",
        "license": "AGPL-3.0 (solo investigación)",
        "checksum": verify_checksum(out_path),
    }
    (PROCESSED_DIR / "dalton_meta.json").write_text(json.dumps(meta, indent=2))

    return out_path


# ---------------------------------------------------------------------------
# 3. Smoke24776 Dataset
# ---------------------------------------------------------------------------

def prepare_smoke24776() -> Path:
    """
    Prepara Smoke24776 para YOLO.
    Repo tiene anotaciones (COCO, VOC, YOLO). Imágenes en externos.
    Salida: data/yolo_unified/ con estructura YOLO + data.yaml
    """
    print("\n" + "="*60)
    print("[DOWNLOAD] PREPARANDO SMOKE24776")
    print("="*60)

    repo_info = REPOS["smoke24776"]
    repo_dir = repo_info["dir"]

    if not clone_repo(repo_info["url"], repo_dir):
        raise RuntimeError("No se pudo clonar Smoke24776")

    # Usar anotaciones YOLO (ya en formato correcto)
    yolo_labels = repo_dir / repo_info["annotations"]["yolo"]
    yolo_images = repo_dir / "Smoke24776_yolo_format" / "images"

    if not yolo_labels.exists():
        raise RuntimeError("No se encontraron labels YOLO en el repo")

    print(f"  [FILE] Labels YOLO: {yolo_labels}")
    print(f"  [FILE] Imágenes (solo samples): {yolo_images}")

    # Contar archivos
    label_files = list(yolo_labels.glob("*.txt"))
    img_files = list(yolo_images.glob("*.jpg")) + list(yolo_images.glob("*.png"))
    print(f"     Labels: {len(label_files)}")
    print(f"     Imágenes en repo: {len(img_files)}")

    # Las imágenes completas hay que descargarlas de Google Drive/Baidu
    # Aquí creamos la estructura YOLO unificada con lo que tenemos
    # y generamos un script para descargar el resto

    # Crear estructura unificada
    for split in ("train", "val", "test"):
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Leer splits: prioridad COCO (VOC files estan vacios en este repo)
    voc_splits = {}
    coco_ann = repo_dir / repo_info["annotations"]["coco"]
    for split in ("train", "val", "test"):
        json_file = coco_ann / f"{split}.json"
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
            voc_splits[split] = [img["file_name"].replace(".jpg", "") for img in data["images"]]

    # Fallback a VOC si COCO falla
    if not any(voc_splits.values()):
        voc_main = repo_dir / "Smoke24776_voc_format" / "ImageSets" / "Main"
        for split_file in ["train.txt", "val.txt", "test.txt"]:
            f = voc_main / split_file
            if f.exists():
                voc_splits[split_file.replace(".txt", "")] = f.read_text().strip().split()

    print(f"  [DIR] Splits detectados: { {k: len(v) for k, v in voc_splits.items()} }")

    # Copiar labels e imágenes disponibles a estructura unificada
    copied = {"train": 0, "val": 0, "test": 0}
    for split, filenames in voc_splits.items():
        for fname in filenames:
            # Label
            lbl_src = yolo_labels / f"{fname}.txt"
            lbl_dst = YOLO_DIR / "labels" / split / f"{fname}.txt"
            if lbl_src.exists():
                shutil.copy2(lbl_src, lbl_dst)
                copied[split] += 1

            # Imagen (buscar en varias ubicaciones)
            for ext in (".jpg", ".png", ".jpeg"):
                img_src = yolo_images / f"{fname}{ext}"
                if img_src.exists():
                    img_dst = YOLO_DIR / "images" / split / f"{fname}{ext}"
                    shutil.copy2(img_src, img_dst)
                    break

    print(f"  [OK] Copiados: train={copied['train']}, val={copied['val']}, test={copied['test']}")

    # Crear data.yaml
    data_yaml = {
        "path": str(YOLO_DIR.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["smoke"],
    }
    (YOLO_DIR / "data.yaml").write_text(yaml.dump(data_yaml))
    print(f"  [OK] data.yaml creado en {YOLO_DIR}")

    # Generar script de descarga de imágenes completas
    download_script = YOLO_DIR / "download_full_images.py"
    download_script.write_text(f'''#!/usr/bin/env python3
"""
Descarga imágenes completas de Smoke24776 desde Google Drive.
Requiere: pip install gdown
"""
import gdown
import zipfile
from pathlib import Path

# Google Drive folder ID
FOLDER_URL = "https://drive.google.com/drive/folders/1FiZJ47TS-Ac0mBOfNGM2RChsNcGEXxVl"
OUTPUT_DIR = Path("{YOLO_DIR}")

print("Descargando imágenes completas de Smoke24776...")
gdown.download_folder(FOLDER_URL, output=str(OUTPUT_DIR), quiet=False, use_cookies=False)

# Descomprimir si vienen en zip
for zip_file in OUTPUT_DIR.glob("*.zip"):
    print(f"Descomprimiendo {{zip_file}}...")
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(OUTPUT_DIR)
    zip_file.unlink()

print("¡Listo! Imágenes en:", OUTPUT_DIR / "images")
''')

    print(f"  [FILE] Script de descarga generado: {download_script}")
    print(f"     Ejecuta: python {download_script} (requiere gdown)")

    return YOLO_DIR / "data.yaml"


# ---------------------------------------------------------------------------
# 4. Hard Negatives para Visión
# ---------------------------------------------------------------------------

def prepare_hard_negatives() -> int:
    """
    Genera/descarga hard negatives para visión.
    Incluye: niebla, polvo, vapor, nubes bajas, humo de cocina/vehículos.
    """
    print("\n" + "="*60)
    print("[DOWNLOAD] PREPARANDO HARD NEGATIVES")
    print("="*60)

    HARD_NEG_DIR.mkdir(parents=True, exist_ok=True)

    # Fuentes sugeridas (el usuario debe conseguir videos/imágenes)
    sources = {
        "niebla_costera": "https://github.com/user/niebla-dataset",  # placeholder
        "polvo_construccion": "https://github.com/user/dust-dataset",
        "vapor_cocina": "https://github.com/user/kitchen-steam",
        "humo_vehiculos": "https://github.com/user/vehicle-smoke",
        "nubes_bajas": "https://github.com/user/low-clouds",
    }

    readme = HARD_NEG_DIR / "README.md"
    readme.write_text(
        "# Hard Negatives para Entrenamiento YOLO\n\n"
        "## Fuentes necesarias (recolectar manualmente)\n\n"
        "| Categoria | Descripcion | Minimo imagenes |\n"
        "|-----------|-------------|-----------------|\n"
        "| Niebla costera/serrana | Lima mayo-noviembre, HR>90% | 150 |\n"
        "| Polvo en suspension | Construccion, caminos sin asfaltar | 100 |\n"
        "| Vapor de agua | Cocina, duchas, industrias | 80 |\n"
        "| Nubes bajas | Neblina de radiacion, estratos | 70 |\n"
        "| Humo de cocina | Freir, asar, quemar comida | 50 |\n"
        "| Humo vehiculos | Diesel viejo, arranque en frio | 50 |\n"
        "| **Total** | | **500+** |\n\n"
        "## Como recolectar\n\n"
        "1. **Camara IP / telefono** en sitio piloto (14 dias Fase 1)\n"
        "2. **Extraer frames** cada 30s donde NO hay humo real\n"
        "3. **Etiquetar manualmente** como \"background\" (sin labels YOLO)\n"
        "4. **Guardar en subcarpetas\" por categoria\n\n"
        "## Estructura esperada\n\n"
        "hard_negatives/\n"
        "|-- fog/\n"
        "|   |-- fog_001.jpg\n"
        "|   `-- ...\n"
        "|-- dust/\n"
        "|-- steam/\n"
        "|-- low_clouds/\n"
        "|-- kitchen_smoke/\n"
        "`-- vehicle_smoke/\n\n"
        "## Mining automatico (despues de tener modelo base)\n\n"
        "python -m experts.vision mine_hard_negatives \\\n"
        "    --model models/yolov8n_smoke.onnx \\\n"
        "    --videos hard_negatives/videos/*.mp4 \\\n"
        "    --output hard_negatives/mined/\n"
    )
    print(f"  [FILE] Guía creada: {readme}")

    # Crear subdirectorios
    for cat in ["fog", "dust", "steam", "low_clouds", "kitchen_smoke", "vehicle_smoke"]:
        (HARD_NEG_DIR / cat).mkdir(exist_ok=True)

    return 0


# ---------------------------------------------------------------------------
# 5. Unificar para entrenamiento
# ---------------------------------------------------------------------------

def create_unified_training_data() -> dict:
    """
    Combina IoT Smoke + Dalton (muestra) para pre-entrenamiento Experto 2.
    """
    print("\n" + "="*60)
    print("[LINK] CREANDO DATASET UNIFICADO PARA EXPERTO 2")
    print("="*60)

    iot_path = PROCESSED_DIR / "iot_smoke_train.csv"
    dalton_path = PROCESSED_DIR / "dalton_sample.csv"

    if not iot_path.exists():
        raise FileNotFoundError(f"Falta {iot_path}. Ejecuta --iot-smoke primero.")

    iot = pd.read_csv(iot_path)
    print(f"  IoT Smoke train: {len(iot)} filas")

    dalton = None
    if dalton_path.exists():
        dalton = pd.read_csv(dalton_path)
        print(f"  Dalton sample: {len(dalton)} filas")

        # Dalton tiene columnas diferentes - mapear a esquema común
        # Solo usar columnas compartidas
        common_cols = ["timestamp_unix", "pm25", "pm10", "temperature", "humidity", "label"]
        # Nota: Dalton no tiene CO, VOC directos equivalentes -> dejar NaN

    # Por ahora, solo IoT que tiene labels completas
    unified = iot.copy()
    out_path = PROCESSED_DIR / "experto2_pretrain.csv"
    unified.to_csv(out_path, index=False)
    print(f"  ✅ Unified pre-train: {out_path} ({len(unified)} filas)")

    return {"pretrain": out_path, "iot_train": iot_path}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preparar datasets para AirGuard")
    parser.add_argument("--all", action="store_true", help="Preparar todos los datasets")
    parser.add_argument("--iot-smoke", action="store_true", help="Solo IoT Smoke Detection")
    parser.add_argument("--dalton", action="store_true", help="Solo Dalton Dataset")
    parser.add_argument("--smoke24776", action="store_true", help="Solo Smoke24776")
    parser.add_argument("--hard-neg", action="store_true", help="Solo hard negatives")
    parser.add_argument("--unify", action="store_true", help="Crear dataset unificado Experto 2")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    try:
        if args.all or args.iot_smoke:
            prepare_iot_smoke()

        if args.all or args.dalton:
            prepare_dalton()

        if args.all or args.smoke24776:
            prepare_smoke24776()

        if args.all or args.hard_neg:
            prepare_hard_negatives()

        if args.all or args.unify:
            create_unified_training_data()

        print("\n" + "="*60)
        print("[OK] PREPARACIÓN COMPLETA")
        print("="*60)
        print(f"[DIR] Datos procesados en: {PROCESSED_DIR}")
        print(f"[DIR] YOLO unificado en: {YOLO_DIR}")
        print(f"[DIR] Hard negatives en: {HARD_NEG_DIR}")
        print("\nPróximos pasos:")
        print("  1. Descargar imágenes completas Smoke24776:")
        print(f"     python {YOLO_DIR}/download_full_images.py")
        print("  2. Recolectar hard negatives reales en sitio piloto")
        print("  3. Entrenar Experto 2: python train_calibrate.py --data data/processed/experto2_pretrain.csv")
        print("  4. Entrenar Experto 1: python -m experts.vision train")

    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import numpy as np
    import yaml
    main()