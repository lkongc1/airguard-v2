#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento mejorado Experto 2: XGBoost con balanceo de clases, SMOTE temporal,
optimización de hiperparámetros y features mejoradas.
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
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print(" imbalanced-learn no instalado. SMOTE no disponible.")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print(" Optuna no instalado. Optimización Bayesiana no disponible.")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Clases auxiliares (a nivel de módulo para pickling)
# ---------------------------------------------------------------------------

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

FEATURE_NAMES = [
    "pm25_raw", "pm25_corrected", "pm10_raw", "pm_ratio", "pm25_delta_1min",
    "pm_slope_2min", "co_ppm", "co_slope_2min", "voc_index", "voc_slope_1min",
    "co_pm_ratio", "temperature", "temp_delta_5min", "humidity", "hr_bucket",
    "hour_sin", "hour_cos"
]


# ---------------------------------------------------------------------------
# Features avanzadas
# ---------------------------------------------------------------------------

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features avanzadas para mejor discriminación."""
    df = df.copy()
    
    # 1. Ratios adicionales
    if "pm25" in df.columns and "pm10" in df.columns:
        df["pm25_pm10_ratio"] = df["pm25"] / df["pm10"].replace(0, np.nan)
        df["pm10_pm25_ratio"] = df["pm10"] / df["pm25"].replace(0, np.nan)
    
    if "eco2_ppm" in df.columns and "tvoc_ppb" in df.columns:
        df["co_voc_ratio"] = df["eco2_ppm"] / df["tvoc_ppb"].replace(0, np.nan)
    
    # 2. Ventanas móviles (rolling stats)
    for col in ["pm25", "eco2_ppm", "tvoc_ppb", "temperature", "humidity"]:
        if col in df.columns:
            for window in [5, 15, 60]:  # 5s, 15s, 60s
                df[f"{col}_roll_mean_{window}"] = df[col].rolling(window, min_periods=1).mean()
                df[f"{col}_roll_std_{window}"] = df[col].rolling(window, min_periods=1).std()
                df[f"{col}_roll_max_{window}"] = df[col].rolling(window, min_periods=1).max()
                df[f"{col}_roll_min_{window}"] = df[col].rolling(window, min_periods=1).min()
    
    # 3. Diferencias (momentum)
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
        # Transición de fase (condensación)
        df["hr_phase"] = np.where(df["humidity"] > 85, 1, 0)
    
    # 7. Features temporales expandidas
    if "timestamp_unix" in df.columns:
        dt = pd.to_datetime(df["timestamp_unix"], unit="s", utc=True)
        df["hour"] = dt.dt.hour
        df["minute"] = dt.dt.minute
        df["day_of_week"] = dt.dt.dayofweek
        df["is_night"] = ((dt.dt.hour >= 22) | (dt.dt.hour <= 5)).astype(int)
        df["is_dawn"] = ((dt.dt.hour >= 5) & (dt.dt.hour <= 8)).astype(int)
        df["is_rush_hour"] = (((dt.dt.hour >= 7) & (dt.dt.hour <= 9)) | 
                               ((dt.dt.hour >= 17) & (dt.dt.hour <= 19))).astype(int)
        
        # Cíclicas expandidas
        for period, name in [(24, "hour"), (60, "minute"), (7, "day_of_week")]:
            df[f"{name}_sin"] = np.sin(2 * np.pi * df[name] / period)
            df[f"{name}_cos"] = np.cos(2 * np.pi * df[name] / period)
    
    return df


def load_and_prepare(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Carga datos y prepara features + labels."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp_unix").reset_index(drop=True)
    
    # Mapear labels
    if df["label"].dtype == object:
        df["label_id"] = df["label"].map(LABEL_MAP)
    else:
        df["label_id"] = df["label"].astype(int)
    
    # Features avanzadas
    print(" Generando features avanzadas...")
    df = add_advanced_features(df)
    
    # Separar features y labels
    feature_cols = [c for c in df.columns if c not in ["label", "label_id", "timestamp", "timestamp_unix", "fire_binary"]]
    X = df[feature_cols].copy()
    y = df["label_id"].values
    y_binary = df["fire_binary"].values if "fire_binary" in df.columns else (y > 0).astype(int)
    
    print(f"   Features totales: {len(feature_cols)}")
    print(f"   Muestras: {len(X)}")
    print(f"   Distribución labels: {pd.Series(y).map({v:k for k,v in LABEL_MAP.items()}).value_counts().to_dict()}")
    
    return X, y, y_binary


def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Calcula pesos de clase balanceados."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


# ---------------------------------------------------------------------------
# Entrenamiento con validación cruzada estratificada
# ---------------------------------------------------------------------------

def train_xgboost_cv(
    X: np.ndarray,
    y: np.ndarray,
    class_weights: Dict[int, float],
    params: Dict | None = None,
    n_splits: int = 5,
) -> Tuple[xgb.XGBClassifier, List[float]]:
    """Entrena XGBoost con validación cruzada estratificada."""
    
    default_params = {
        "objective": "multi:softprob",
        "num_class": 4,
        "eval_metric": "mlogloss",
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "min_child_weight": 5,
        "gamma": 0.5,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "enable_categorical": False,
    }
    if params:
        default_params.update(params)
    
    # Aplicar class weights
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
        
        model = xgb.XGBClassifier(**default_params)
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
    print(f"  Best fold: {best_score:.4f}")
    
    # Reentrenar en todos los datos con el mejor modelo (SIN early stopping)
    if best_model is not None:
        final_params = best_model.get_params()
        final_params.pop("early_stopping_rounds", None)
        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(X, y, sample_weight=sample_weights, verbose=False)
        return final_model, cv_scores
    
    return None, cv_scores


def calibrate_isotonic(
    base_model: xgb.XGBClassifier,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
) -> Tuple[object, np.ndarray]:
    """Calibración isotónica simple (prefit)."""
    from sklearn.isotonic import IsotonicRegression
    
    proba_base = base_model.predict_proba(X_calib)
    
    calibrators = []
    proba_calibrated = np.zeros_like(proba_base)
    
    for c in range(4):
        iso_reg = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        y_binary = (y_calib == c).astype(int)
        iso_reg.fit(proba_base[:, c], y_binary)
        proba_calibrated[:, c] = iso_reg.predict(proba_base[:, c])
        calibrators.append(iso_reg)
    
    proba_calibrated = np.clip(proba_calibrated, 1e-8, 1.0)
    proba_calibrated /= proba_calibrated.sum(axis=1, keepdims=True)
    
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


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

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


def reliability_diagram_data(calibrated, X: np.ndarray, y: np.ndarray, n_bins: int = 10) -> List[Dict]:
    y_proba = calibrated.predict_proba(X)
    results = []
    
    for c in range(4):
        y_binary = (y == c).astype(int)
        prob_true, prob_pred = calibration_curve(y_binary, y_proba[:, c], n_bins=n_bins, strategy="quantile")
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


def export_model(calibrated, output_path: Path, feature_names: List[str]) -> None:
    """Exporta modelo XGBoost nativo + calibradores."""
    base_estimator = calibrated.base_model
    
    # Guardar nativo
    base_estimator.save_model(str(output_path.with_suffix(".json")))
    base_estimator.save_model(str(output_path))
    
    # Guardar calibradores
    calib_path = output_path.with_suffix(".calib.joblib")
    joblib.dump({
        "calibrators": calibrated.calibrators,
        "classes": calibrated.classes_,
        "feature_names": feature_names,
    }, calib_path)
    
    print(f"   Modelo: {output_path}")
    print(f"   JSON: {output_path.with_suffix('.json')}")
    print(f"   Calibradores: {calib_path}")


def save_artifacts(calibrated, imputer, feature_names, out_dir: Path) -> None:
    """Guarda todos los artefactos."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    export_model(calibrated, out_dir / "experto2_improved", FEATURE_NAMES)
    joblib.dump(imputer, out_dir / "imputer.joblib")
    
    config_dict = {
        "feature_names": feature_names,
        "label_map": LABEL_MAP,
        "label_names": LABEL_NAMES,
    }
    joblib.dump({"config": config_dict}, out_dir / "extractor.joblib")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train improved Experto 2")
    parser.add_argument("--data", type=Path, required=True, help="CSV train")
    parser.add_argument("--out", type=Path, default=Path("models/experto2_improved"))
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-smote", action="store_true", help="Usar SMOTE para balanceo")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("="*60)
    print(" ENTRENANDO EXPERTO 2 MEJORADO: GASES Y PARTÍCULAS")
    print("="*60)
    
    # 1. Cargar y preparar
    print(f" Cargando: {args.data}")
    X_df, y, y_binary = load_and_prepare(args.data)
    
    # 2. Imputar NaN
    print(" Imputando NaN...")
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_df.values.astype(np.float32))
    feature_names = X_df.columns.tolist()
    
    # 3. Pesos de clase
    class_weights = get_class_weights(y)
    print(f" Class weights: {class_weights}")
    
    # 4. Split estratificado (para 4-clases) + temporal hold-out
    X_trainval, X_test, y_trainval, y_test, y_bin_trainval, y_bin_test = train_test_split(
        X, y, y_binary, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    
    # Split train/val dentro de trainval
    X_train, X_val, y_train, y_val, y_bin_train, y_bin_val = train_test_split(
        X_trainval, y_trainval, y_bin_trainval, test_size=0.15, stratify=y_trainval, random_state=args.seed
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"   Train dist: {pd.Series(y_train).map({v:k for k,v in LABEL_MAP.items()}).value_counts().to_dict()}")
    
    # 5. SMOTE opcional (solo en train, DESPUÉS del split)
    if args.use_smote and HAS_IMBLEARN:
        print(" Aplicando SMOTE solo en train...")
        smote = SMOTE(random_state=args.seed, k_neighbors=3)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"   Después SMOTE: {pd.Series(y_train).map({v:k for k,v in LABEL_MAP.items()}).value_counts().to_dict()}")
        
        # Actualizar y_bin_train para que coincida con SMOTE
        y_bin_train = (y_train > 0).astype(int)
    
    # 6. Pesos de clase (después de SMOTE)
    class_weights = get_class_weights(y_train)
    
    # 7. Entrenar con CV
    print(" Entrenando XGBoost con CV estratificada...")
    base_model, cv_scores = train_xgboost_cv(X_train, y_train, class_weights, n_splits=args.cv_folds)
    
    # 7b. Entrenar modelo binario fire/no-fire
    print(" Entrenando modelo binario fire/no-fire...")
    bin_class_weights = get_class_weights(y_bin_train)
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
        scale_pos_weight=class_weights.get(1, 1) / class_weights.get(0, 1) if 0 in class_weights else 1,
    )
    bin_model.fit(
        X_train, y_bin_train,
        verbose=False,
    )
    
    # Calibración binaria
    from sklearn.isotonic import IsotonicRegression
    proba_val = bin_model.predict_proba(X_val)[:, 1]
    iso_reg = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso_reg.fit(proba_val, y_bin_val)
    
    bin_calibrated = SimpleBinCalibrated(bin_model, iso_reg)
    
    # 8. Calibrar multiclase en conjunto de validación
    print(" Calibrando isotónicamente...")
    calibrated, _ = calibrate_isotonic(base_model, X_val, y_val)
    
    # 9. Evaluar en test
    print(" Evaluando en test hold-out...")
    metrics = compute_metrics(calibrated, X_test, y_test)
    reliability = reliability_diagram_data(calibrated, X_test, y_test)
    
    print(f"\n   F1-macro: {metrics['f1_macro']:.4f}")
    print(f"   ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
    print(f"   Brier score: {metrics['brier_score']:.4f}")
    print(f"   Brier per class: {[f'{b:.4f}' for b in metrics['brier_per_class']]}")
    print(f"   F1 per class: {dict(zip(LABEL_NAMES, [f'{f:.4f}' for f in metrics['f1_per_class']]))}")
    
    # Binario
    bin_metrics = {
        "brier": float(brier_score_loss(y_bin_test, bin_calibrated.predict_proba(X_test)[:, 1])),
        "auc": float(roc_auc_score(y_bin_test, bin_calibrated.predict_proba(X_test)[:, 1])),
    }
    print(f"   Binario - Brier: {bin_metrics['brier']:.4f}, AUC: {bin_metrics['auc']:.4f}")
    
    # 10. Guardar artefactos
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "reliability.json", "w") as f:
        json.dump(reliability, f, indent=2)
    with open(out_dir / "cv_scores.json", "w") as f:
        json.dump({"cv_f1_scores": cv_scores, "mean": np.mean(cv_scores), "std": np.std(cv_scores)}, f, indent=2)
    with open(out_dir / "bin_metrics.json", "w") as f:
        json.dump(bin_metrics, f, indent=2)
    
    save_artifacts(calibrated, None, feature_names, out_dir)
    joblib.dump(bin_calibrated, out_dir / "experto2_binary_calibrated.joblib")
    
    print("\n" + "="*60)
    print(" EXPERTO 2 MEJORADO COMPLETADO")
    print("="*60)
    print(f" Artefactos en: {out_dir}")


if __name__ == "__main__":
    import pandas as pd
    main()