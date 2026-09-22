#!/usr/bin/env python3
"""
Entrenamiento + calibración isotónica del Experto 2 (Gases y Partículas).

Optimizado para CPU (no usa GPU). Entrena XGBoost multiclase (4 clases) 
+ calibración isotónica con TimeSeriesSplit + export ONNX.

Uso:
    python train_calibrate.py --data data/processed/iot_smoke_train.csv --out models/experto2_calibrated.onnx
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

REQUIRED_COLS = [
    "timestamp_unix", "pm25", "pm10", "eco2_ppm", "tvoc_ppb",
    "temperature", "humidity", "label", "fire_binary"
]

LABEL_MAP = {
    "normal": 0,
    "quema_organica": 1,
    "humo_toxico": 2,
    "polvo_niebla": 3,
}

LABEL_NAMES = ["normal", "quema_organica", "humo_toxico", "polvo_niebla"]

FEATURE_NAMES = [
    "pm25_raw", "pm25_corrected", "pm10_raw", "pm_ratio", "pm25_delta_1min",
    "pm_slope_2min", "co_ppm", "co_slope_2min", "voc_index", "voc_slope_1min",
    "co_pm_ratio", "temperature", "temp_delta_5min", "humidity", "hr_bucket",
    "hour_sin", "hour_cos"
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def load_and_validate(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        # Intentar mapear nombres alternativos
        alt_map = {
            "pm25": "pm25", "pm10": "pm10", "eco2_ppm": "co_ppm", 
            "tvoc_ppb": "voc_index", "co_slope_2min": "co_slope_2min",
            "voc_slope_1min": "voc_slope_1min", "pm25_delta_1min": "pm25_delta_1min",
            "pm_slope_2min": "pm_slope_2min", "pm25_corrected": "pm25_corrected",
            "pm_ratio": "pm_ratio", "co_pm_ratio": "co_pm_ratio",
            "temperature": "temperature", "temp_delta_5min": "temp_delta_5min",
            "humidity": "humidity", "hr_bucket": "hr_bucket",
            "hour_sin": "hour_sin", "hour_cos": "hour_cos",
            "label": "label", "fire_binary": "fire_binary"
        }
        for req, alt in alt_map.items():
            if req not in df.columns and alt in df.columns:
                df[req] = df[alt]
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
    
    df = df.sort_values("timestamp_unix").reset_index(drop=True)
    
    # Mapear labels string a int
    if df["label"].dtype == object:
        df["label_id"] = df["label"].map(LABEL_MAP)
        if df["label_id"].isna().any():
            unknown = df.loc[df["label_id"].isna(), "label"].unique()
            raise ValueError(f"Labels desconocidos: {unknown}")
    else:
        df["label_id"] = df["label"].astype(int)
    
    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye matriz de features X, labels y (4-clase) y y_binary."""
    # Mapear features del CSV a nombres estándar del extractor
    feature_map = {
        "pm25": "pm25_raw",
        "pm25_corrected": "pm25_corrected", 
        "pm10": "pm10_raw",
        "pm_ratio": "pm_ratio",
        "pm25_delta_1min": "pm25_delta_1min",
        "pm_slope_2min": "pm_slope_2min",
        "eco2_ppm": "co_ppm",
        "co_slope_2min": "co_slope_2min",
        "tvoc_ppb": "voc_index",
        "voc_slope_1min": "voc_slope_1min",
        "co_pm_ratio": "co_pm_ratio",
        "temperature": "temperature",
        "temp_delta_5min": "temp_delta_5min",
        "humidity": "humidity",
        "hr_bucket": "hr_bucket",
        "hour_sin": "hour_sin",
        "hour_cos": "hour_cos",
    }
    
    # Verificar qué features están disponibles
    available = [c for c in feature_map.keys() if c in df.columns]
    missing = [c for c in feature_map.keys() if c not in df.columns]
    if missing:
        print(f"   Features faltantes (se imputarán): {missing}")
    
    X = df[available].copy()
    X.columns = [feature_map[c] for c in available]
    
    # Añadir features faltantes como NaN (se imputarán)
    for fn in FEATURE_NAMES:
        if fn not in X.columns:
            X[fn] = np.nan
    
    # Reordenar según FEATURE_NAMES
    X = X[FEATURE_NAMES]
    
    y = df["label_id"].values.astype(np.int64)
    y_binary = df["fire_binary"].values.astype(np.int64)
    
    return X.values.astype(np.float32), y, y_binary


def impute_nan(X: np.ndarray, strategy: str = "median") -> Tuple[np.ndarray, SimpleImputer]:
    imp = SimpleImputer(strategy=strategy)
    return imp.fit_transform(X), imp


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict | None = None,
    use_early_stopping: bool = True,
) -> xgb.XGBClassifier:
    default_params = {
        "objective": "multi:softprob",
        "num_class": 4,
        "eval_metric": "mlogloss",
        "n_estimators": 800,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "enable_categorical": False,
    }
    if use_early_stopping:
        default_params["early_stopping_rounds"] = 50
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    if use_early_stopping:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, verbose=False)
    return model


def calibrate_isotonic(
    base_model: xgb.XGBClassifier,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
) -> Tuple[CalibratedClassifierCV, np.ndarray]:
    """
    Calibración simple usando prefit: entrena modelo base, 
    luego calibra con IsotonicRegression en conjunto de calibración separado.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import _CalibratedClassifier
    
    # Obtener probabilidades del modelo base en conjunto de calibración
    proba_base = base_model.predict_proba(X_calib)  # (n_calib, 4)
    
    # Ajustar IsotonicRegression por clase
    calibrators = []
    proba_calibrated = np.zeros_like(proba_base)
    
    for c in range(4):
        iso_reg = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        y_binary = (y_calib == c).astype(int)
        iso_reg.fit(proba_base[:, c], y_binary)
        proba_calibrated[:, c] = iso_reg.predict(proba_base[:, c])
        calibrators.append(iso_reg)
    
    # Renormalizar
    proba_calibrated = np.clip(proba_calibrated, 1e-8, 1.0)
    proba_calibrated /= proba_calibrated.sum(axis=1, keepdims=True)
    
    # Crear objeto compatible con CalibratedClassifierCV para exportación
    class SimpleCalibratedClassifier:
        def __init__(self, base_model, calibrators, classes):
            self.base_model = base_model
            self.calibrators = calibrators
            self.classes_ = classes
        
        def predict_proba(self, X):
            proba = self.base_model.predict_proba(X)
            proba_cal = np.zeros_like(proba)
            for c, iso_reg in enumerate(self.calibrators):
                proba_cal[:, c] = iso_reg.predict(proba[:, c])
            proba_cal = np.clip(proba_cal, 1e-8, 1.0)
            proba_cal /= proba_cal.sum(axis=1, keepdims=True)
            return proba_cal
        
        def predict(self, X):
            return self.predict_proba(X).argmax(axis=1)
    
    calibrated = SimpleCalibratedClassifier(base_model, calibrators, base_model.classes_)
    return calibrated, proba_calibrated


def compute_metrics(
    calibrated,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict:
    y_pred = calibrated.predict(X)
    y_proba = calibrated.predict_proba(X)

    # Debug
    print(f"  Debug: y_proba shape={y_proba.shape}, y unique={np.unique(y)}")
    print(f"  Debug: calibrated.classes_={getattr(calibrated, 'classes_', 'N/A')}")

    # Asegurar que y_proba tenga 4 columnas (una por clase)
    if y_proba.shape[1] != 4:
        print(f"   y_proba shape: {y_proba.shape}, esperado (n, 4)")
        if y_proba.shape[1] < 4:
            pad = np.zeros((y_proba.shape[0], 4 - y_proba.shape[1]))
            y_proba = np.hstack([y_proba, pad])
        y_proba = y_proba[:, :4]

    # Métricas que no requieren todas las clases en test
    f1_macro = float(f1_score(y, y_pred, average="macro"))
    f1_per_class = f1_score(y, y_pred, average=None).tolist()
    
    # ROC AUC solo si hay al menos 2 clases en test
    y_classes = np.unique(y)
    if len(y_classes) >= 2:
        try:
            roc_auc = float(roc_auc_score(y, y_proba, multi_class="ovr"))
        except ValueError:
            roc_auc = float("nan")
    else:
        roc_auc = float("nan")

    # Brier score multiclase: promedio de Brier por clase (one-vs-rest)
    brier_scores = []
    for c in range(4):
        y_binary = (y == c).astype(int)
        brier_scores.append(float(brier_score_loss(y_binary, y_proba[:, c])))

    # Métricas que no requieren todas las clases en test
    f1_macro = float(f1_score(y, y_pred, average="macro"))
    f1_per_class = f1_score(y, y_pred, average=None).tolist()

    metrics = {
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "roc_auc_ovr": roc_auc,
        "brier_score": float(np.mean(brier_scores)),
        "brier_per_class": brier_scores,
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0,1,2,3]).tolist(),
        "classification_report": classification_report(y, y_pred, target_names=LABEL_NAMES, labels=[0,1,2,3], output_dict=True, zero_division=0),
    }

    return metrics


def reliability_diagram_data(
    calibrated,
    X: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> List[Dict]:
    y_proba = calibrated.predict_proba(X)
    results = []

    for c in range(4):
        y_binary = (y == c).astype(int)
        prob_true, prob_pred = calibration_curve(
            y_binary, y_proba[:, c], n_bins=n_bins, strategy="quantile"
        )
        bins = np.quantile(y_proba[:, c], np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)
        bin_indices = np.digitize(y_proba[:, c], bins) - 1

        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            results.append({
                "class": LABEL_NAMES[c],
                "bin_left": float(bins[i]),
                "bin_right": float(bins[i + 1]),
                "mean_predicted": float(y_proba[mask, c].mean()),
                "fraction_positive": float(y_binary[mask].mean()),
                "count": int(mask.sum()),
            })
    return results


def export_onnx(calibrated, output_path: Path, feature_names: List[str]) -> None:
    # Usar exportación nativa de XGBoost a ONNX (requiere xgboost >= 1.6)
    base_estimator = calibrated.base_model
    
    try:
        import onnx
        from xgboost import XGBClassifier
        # XGBoost native ONNX export
        base_estimator.save_model(str(output_path.with_suffix(".xgb.json")))
        # Convertir JSON a ONNX usando xgboost's built-in
        base_estimator.get_booster().save_model(str(output_path))
        # Para compatibilidad, también guardar como JSON
        base_estimator.save_model(str(output_path.with_suffix(".json")))
        
        print(f"   Modelo XGBoost guardado: {output_path}")
        print(f"   Modelo JSON: {output_path.with_suffix('.json')}")
        
    except Exception as e:
        print(f"   Exportación nativa falló: {e}")
        print("   Intentando con onnxmltools...")
        try:
            import onnxmltools
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
            
            initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
            onnx_model = convert_xgboost(base_estimator, initial_types=initial_type)
            onnxmltools.utils.save_model(onnx_model, str(output_path))
            print(f"   ONNX via onnxmltools: {output_path}")
        except Exception as e2:
            print(f"   onnxmltools también falló: {e2}")
            print("   Guardando solo modelo XGBoost nativo (.json) para inferencia con xgboost runtime")
            base_estimator.save_model(str(output_path.with_suffix(".json")))

    # Guardar calibradores para runtime edge
    calib_path = output_path.with_suffix(".calib.joblib")
    joblib.dump({
        "calibrators": calibrated.calibrators,
        "classes": calibrated.classes_,
        "feature_names": feature_names,
    }, calib_path)

    print(f"   Calibradores: {calib_path}")


class SimpleBinCalibrated:
    """Wrapper simple para modelo binario calibrado con IsotonicRegression."""
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator
    
    def predict_proba(self, X):
        proba = self.base_model.predict_proba(X)[:, 1]
        cal_proba = self.calibrator.predict(proba)
        cal_proba = np.clip(cal_proba, 1e-8, 1 - 1e-8)
        return np.vstack([1 - cal_proba, cal_proba]).T
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def save_extractor_config(path: Path) -> None:
    """Guarda config del feature extractor para inferencia edge."""
    # FeatureConfig inline para evitar dependencia circular
    config_dict = {
        "co_slope_window": 120,
        "pm_slope_window": 120,
        "voc_slope_window": 60,
        "temp_delta_window": 300,
        "pm25_delta_window": 60,
        "min_samples_slope": 5,
        "min_dt_seconds": 30.0,
        "hr_threshold": 80.0,
        "hr_max_correction": 0.45,
        "max_history": 600,
        "hr_bins": [0, 30, 50, 70, 85, 95, 101],
    }
    joblib.dump({"config": config_dict}, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train + calibrate Experto 2")
    parser.add_argument("--data", type=Path, required=True, help="CSV train")
    parser.add_argument("--val", type=Path, help="CSV validation (opcional, si no se usa split interno)")
    parser.add_argument("--out", type=Path, required=True, help="Ruta ONNX salida")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("="*60)
    print(" ENTRENANDO EXPERTO 2: GASES Y PARTÍCULAS")
    print("="*60)

    # 1. Cargar datos
    print(f" Cargando: {args.data}")
    df = load_and_validate(args.data)
    print(f"   {len(df)} muestras")
    print(f"   Label dist: {df['label'].value_counts().to_dict()}")
    print(f"   Fire binary: {df['fire_binary'].value_counts().to_dict()}")

    # 2. Features
    print(" Construyendo features...")
    X, y, y_binary = build_feature_matrix(df)
    print(f"   X shape: {X.shape}")

    # 3. Imputar
    print(" Imputando NaN...")
    X, imputer = impute_nan(X, strategy="median")

    # 4. Split temporal: train/val/calib/test
    n = len(X)
    test_start = int(n * (1 - args.test_size))
    calib_end = int(test_start * 0.85)
    val_end = int(calib_end * 0.85)

    X_train, X_val, X_calib, X_test = X[:val_end], X[val_end:calib_end], X[calib_end:test_start], X[test_start:]
    y_train, y_val, y_calib, y_test = y[:val_end], y[val_end:calib_end], y[calib_end:test_start], y[test_start:]
    y_binary_train, y_binary_val, y_binary_calib, y_binary_test = y_binary[:val_end], y_binary[val_end:calib_end], y_binary[calib_end:test_start], y_binary[test_start:]

    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Calib: {len(X_calib)}, Test: {len(X_test)}")

    # 5. Entrenar XGBoost base (multiclase 4)
    print(" Entrenando XGBoost (4 clases)...")
    base_model = train_xgboost(X_train, y_train, X_val, y_val, use_early_stopping=True)

    # 6. Calibrar en conjunto de calibración separado
    print(" Calibrando isotónicamente (conjunto dedicado)...")
    calibrated, proba_calib = calibrate_isotonic(base_model, X_calib, y_calib)

    # 7. Evaluar en test
    print(" Evaluando en test hold-out...")
    metrics = compute_metrics(calibrated, X_test, y_test)
    reliability = reliability_diagram_data(calibrated, X_test, y_test)

    print(f"\n   F1-macro: {metrics['f1_macro']:.4f}")
    print(f"   ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
    print(f"   Brier score: {metrics['brier_score']:.4f}")
    print(f"   Brier per class: {[f'{b:.4f}' for b in metrics['brier_per_class']]}")
    print(f"   F1 per class: {dict(zip(LABEL_NAMES, [f'{f:.4f}' for f in metrics['f1_per_class']]))}")

    # 8. Guardar artefactos
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "reliability.json", "w") as f:
        json.dump(reliability, f, indent=2)

    # 9. Exportar ONNX + calibradores
    print(" Exportando a ONNX...")
    export_onnx(calibrated, args.out, FEATURE_NAMES)

    # 10. Guardar imputer y config
    joblib.dump(imputer, out_dir / "imputer.joblib")
    save_extractor_config(out_dir / "extractor.joblib")

    # 11. Guardar también modelo binario fire/no-fire (para umbral rápido)
    print(" Entrenando modelo binario fire/no-fire...")
    bin_model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    bin_model.fit(X_train, y_binary_train, eval_set=[(X_val, y_binary_val)], verbose=False)
    
    # Calibración isotónica simple (prefit)
    from sklearn.isotonic import IsotonicRegression
    proba_val = bin_model.predict_proba(X_val)[:, 1]
    iso_reg = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso_reg.fit(proba_val, y_binary_val)
    
    bin_calibrated = SimpleBinCalibrated(bin_model, iso_reg)
    
    bin_metrics = {
        "brier": float(brier_score_loss(y_binary_test, bin_calibrated.predict_proba(X_test)[:, 1])),
        "auc": float(roc_auc_score(y_binary_test, bin_calibrated.predict_proba(X_test)[:, 1])),
    }
    print(f"   Binario - Brier: {bin_metrics['brier']:.4f}, AUC: {bin_metrics['auc']:.4f}")
    
    joblib.dump(bin_calibrated, out_dir / "experto2_binary_calibrated.joblib")

    print("\n" + "="*60)
    print(" EXPERTO 2 COMPLETADO")
    print("="*60)
    print(f" Artefactos en: {out_dir}")
    print(f"   - {args.out.name} (ONNX base 4-clase)")
    print(f"   - {args.out.stem}.calib.joblib (calibradores isotónicos)")
    print(f"   - imputer.joblib (imputador NaN)")
    print(f"   - extractor.joblib (config feature extractor)")
    print(f"   - experto2_binary_calibrated.joblib (modelo binario fire/no-fire)")
    print(f"   - metrics.json, reliability.json")

    return metrics


if __name__ == "__main__":
    main()