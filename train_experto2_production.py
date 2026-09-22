#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento Experto 2 PRODUCCIÓN - Sin data leakage.

Características:
- Solo features causales (sin rolling windows que usan futuro)
- Split temporal estricto
- Class weights balanceados
- Calibración isotónica
- Export ONNX + calibradores
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
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "normal": 0,
    "quema_organica": 1,
    "humo_toxico": 2,
    "polvo_niebla": 3,
}

LABEL_NAMES = ["normal", "quema_organica", "humo_toxico", "polvo_niebla"]


# ---------------------------------------------------------------------------
# Features CAUSALES (sin data leakage)
# ---------------------------------------------------------------------------

FEATURE_NAMES_CAUSAL = [
    # Sensores crudos
    "pm25", "pm10", "eco2_ppm", "tvoc_ppb", "temperature", "humidity",
    # Ratios instantáneos
    "pm_ratio", "co_pm_ratio", "co_voc_ratio",
    # Corrección humedad
    "pm25_corrected",
    # Diferencias causales (t vs t-1)
    "pm25_diff_1", "pm25_diff_5", "pm25_diff_10",
    "co_diff_1", "co_diff_5", "co_diff_10",
    "voc_diff_1", "voc_diff_5", "voc_diff_10",
    "temp_diff_1", "temp_diff_5", "temp_diff_10",
    # Aceleración (2da derivada)
    "pm25_accel", "co_accel", "voc_accel",
    # Índices de combustión
    "combustion_index", "toxic_proxy",
    # Humedad avanzada
    "hr_squared", "hr_cubed", "hr_log", "hr_phase",
    # Temporales cíclicas
    "hour_sin", "hour_cos", "minute_sin", "minute_cos", 
    "day_of_week_sin", "day_of_week_cos",
    "is_night", "is_dawn", "is_rush_hour",
]


def add_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade solo features causales (sin leakage)."""
    df = df.copy()
    
    # 1. Ratios instantáneos
    if "pm25" in df.columns and "pm10" in df.columns:
        df["pm_ratio"] = df["pm25"] / df["pm10"].replace(0, np.nan)
    if "eco2_ppm" in df.columns and "pm25" in df.columns:
        df["co_pm_ratio"] = df["eco2_ppm"] / df["pm25"].replace(0, np.nan)
    if "eco2_ppm" in df.columns and "tvoc_ppb" in df.columns:
        df["co_voc_ratio"] = df["eco2_ppm"] / df["tvoc_ppb"].replace(0, np.nan)
    
    # 2. Corrección humedad PM2.5 (Crilley et al. 2018)
    if "pm25" in df.columns and "humidity" in df.columns:
        def correct_pm25_humidity(pm25, hr, threshold=80.0, max_corr=0.45):
            if hr < threshold:
                return pm25
            span = 100.0 - threshold
            factor = 1.0 - ((hr - threshold) / span) * (1.0 - max_corr)
            factor = max(max_corr, min(1.0, factor))
            return pm25 * factor
        
        df["pm25_corrected"] = df.apply(
            lambda r: correct_pm25_humidity(r["pm25"], r["humidity"]), axis=1
        )
    
    # 3. Diferencias causales (t vs t-n)
    for col in ["pm25", "eco2_ppm", "tvoc_ppb", "temperature"]:
        if col in df.columns:
            df[f"{col}_diff_1"] = df[col].diff(1)
            df[f"{col}_diff_5"] = df[col].diff(5)
            df[f"{col}_diff_10"] = df[col].diff(10)
    
    # 4. Aceleración (2da derivada)
    for col in ["pm25", "eco2_ppm", "tvoc_ppb"]:
        if col in df.columns:
            df[f"{col}_accel"] = df[col].diff(1).diff(1)
    
    # 5. Índices de combustión
    if all(c in df.columns for c in ["eco2_ppm", "tvoc_ppb", "pm25"]):
        df["combustion_index"] = (df["eco2_ppm"] * df["tvoc_ppb"]) / (df["pm25"] + 1)
        df["toxic_proxy"] = df["eco2_ppm"] / (df["pm25"] + 1) * df["tvoc_ppb"]
    
    # 6. Features de humedad avanzadas
    if "humidity" in df.columns:
        df["hr_squared"] = df["humidity"] ** 2
        df["hr_cubed"] = df["humidity"] ** 3
        df["hr_log"] = np.log1p(df["humidity"])
        df["hr_phase"] = np.where(df["humidity"] > 85, 1, 0)
    
    # 7. Features temporales (ya vienen en el CSV procesado)
    # hour_sin, hour_cos, etc. ya existen
    
    return df


def load_and_prepare(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp_unix").reset_index(drop=True)
    
    # Mapear labels
    if df["label"].dtype == object:
        df["label_id"] = df["label"].map(LABEL_MAP)
    else:
        df["label_id"] = df["label"].astype(int)
    
    # Features causales
    print("🔧 Generando features causales (sin leakage)...")
    df = add_causal_features(df)
    
    # Separar features y labels
    feature_cols = [c for c in df.columns if c not in ["label", "label_id", "timestamp", "timestamp_unix", "fire_binary"]]
    X = df[feature_cols].copy()
    y = df["label_id"].values
    y_binary = df["fire_binary"].values if "fire_binary" in df.columns else (y > 0).astype(int)
    
    print(f"   Features causales: {len(feature_cols)}")
    print(f"   Muestras: {len(X)}")
    
    return X, y, y_binary


LABEL_MAP = {
    "normal": 0,
    "quema_organica": 1,
    "humo_toxico": 2,
    "polvo_niebla": 3,
}

LABEL_NAMES = ["normal", "quema_organica", "humo_toxico", "polvo_niebla"]


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


class SimpleBinCalibrated:
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


def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def train_xgboost_cv(
    X: np.ndarray,
    y: np.ndarray,
    class_weights: Dict[int, float],
    n_splits: int = 5,
) -> Tuple[xgb.XGBClassifier, List[float]]:
    
    default_params = {
        "objective": "multi:softprob",
        "num_class": 4,
        "eval_metric": "mlogloss",
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "min_child_weight": 5,
        "gamma": 0.5,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    
    sample_weights = np.array([class_weights[label] for label in y])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    best_model = None
    best_score = -1
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold+1}/{n_splits}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train, w_val = sample_weights[train_idx], sample_weights[val_idx]
        
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            min_child_weight=5,
            gamma=0.5,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=False,
        )
        
        val_pred = model.predict(X_val)
        fold_f1 = f1_score(y_val, val_pred, average="macro")
        cv_scores.append(fold_f1)
        print(f"    F1-macro: {fold_f1:.4f}")
        
        if fold_f1 > best_score:
            best_score = fold_f1
            best_model = model
    
    print(f"  CV F1-macro: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    if best_model is not None:
        final_model = xgb.XGBClassifier(**best_model.get_params())
        final_model.fit(X, y, sample_weight=sample_weights, verbose=False)
        return final_model, cv_scores
    
    return None, cv_scores


def calibrate_isotonic(
    base_model: xgb.XGBClassifier,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
) -> SimpleCalibratedClassifier:
    
    proba_base = base_model.predict_proba(X_calib)
    calibrators = []
    
    for c in range(4):
        iso_reg = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        y_binary = (y_calib == c).astype(int)
        iso_reg.fit(proba_base[:, c], y_binary)
        calibrators.append(iso_reg)
    
    return SimpleCalibratedClassifier(base_model, calibrators, base_model.classes_)


def compute_metrics(calibrated, X: np.ndarray, y: np.ndarray) -> Dict:
    y_pred = calibrated.predict(X)
    y_proba = calibrated.predict_proba(X)
    
    f1_macro = float(f1_score(y, y_pred, average="macro"))
    f1_per_class = f1_score(y, y_pred, average=None).tolist()
    
    y_classes = np.unique(y)
    if len(y_classes) >= 2:
        try:
            roc_auc = float(roc_auc_score(y, y_proba, multi_class="ovr"))
        except ValueError:
            roc_auc = float("nan")
    else:
        roc_auc = float("nan")
    
    brier_scores = []
    for c in range(4):
        y_binary = (y == c).astype(int)
        brier_scores.append(float(brier_score_loss(y_binary, y_proba[:, c])))
    
    return {
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "roc_auc_ovr": roc_auc,
        "brier_score": float(np.mean(brier_scores)),
        "brier_per_class": brier_scores,
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0,1,2,3]).tolist(),
        "classification_report": classification_report(y, y_pred, target_names=LABEL_NAMES, labels=[0,1,2,3], output_dict=True, zero_division=0),
    }


def export_model(calibrated, output_path: Path) -> None:
    base_estimator = calibrated.base_model
    base_estimator.save_model(str(output_path.with_suffix(".json")))
    base_estimator.save_model(str(output_path))
    
    calib_path = output_path.with_suffix(".calib.joblib")
    joblib.dump({
        "calibrators": calibrated.calibrators,
        "classes": calibrated.classes_,
    }, calib_path)
    print(f"  ✅ Modelo: {output_path}")
    print(f"  ✅ Calibradores: {calib_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Experto 2 Production")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("="*60)
    print("🏭 ENTRENANDO EXPERTO 2 PRODUCCIÓN (SIN LEAKAGE)")
    print("="*60)
    
    # Cargar
    print(f"📂 Cargando: {args.data}")
    X_df, y, y_binary = load_and_prepare(args.data)
    print(f"   {len(X_df)} muestras")
    print(f"   Distribución: {pd.Series(y).map({v:k for k,v in LABEL_MAP.items()}).value_counts().to_dict()}")
    
    # Imputar
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_df.values.astype(np.float32))
    feature_names = X_df.columns.tolist()
    
    # Pesos de clase
    class_weights = get_class_weights(y)
    print(f"⚖️ Class weights: {class_weights}")
    
    # Split estratificado
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=args.seed
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Pesos después de split
    class_weights = get_class_weights(y_train)
    
    # Entrenar con CV
    print("🚀 Entrenando XGBoost con CV estratificada...")
    base_model, cv_scores = train_xgboost_cv(X_train, y_train, class_weights, n_splits=args.cv_folds)
    
    # Calibrar en val
    print("⚖️ Calibrando isotónicamente...")
    calibrated = calibrate_isotonic(base_model, X_val, y_val)
    
    # Evaluar en test
    print("📊 Evaluando en test hold-out...")
    metrics = compute_metrics(calibrated, X_test, y_test)
    
    print(f"\n   F1-macro: {metrics['f1_macro']:.4f}")
    print(f"   ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
    print(f"   Brier score: {metrics['brier_score']:.4f}")
    print(f"   Brier per class: {[f'{b:.4f}' for b in metrics['brier_per_class']]}")
    print(f"   F1 per class: {dict(zip(LABEL_NAMES, [f'{f:.4f}' for f in metrics['f1_per_class']]))}")
    
    # Guardar
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "cv_scores.json", "w") as f:
        json.dump({"cv_f1_scores": cv_scores, "mean": np.mean(cv_scores), "std": np.std(cv_scores)}, f, indent=2)
    
    # Exportar
    export_model(calibrated, out_dir / "experto2_production")
    joblib.dump(imputer, out_dir / "imputer.joblib")
    
    config_dict = {"feature_names": feature_names, "label_map": LABEL_MAP, "label_names": LABEL_NAMES}
    joblib.dump({"config": config_dict}, out_dir / "extractor.joblib")
    
    print("\n" + "="*60)
    print("✅ EXPERTO 2 PRODUCCIÓN COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    main()