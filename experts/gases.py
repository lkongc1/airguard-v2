"""
Experto 2: Gases y Partículas (Clasificación Química)

Modelo: XGBoost + Calibración Isotónica
Features: 38 causales (sin data leakage)
Clases: normal, quema_organica, humo_toxico, polvo_niebla
"""

from __future__ import annotations

import joblib
import numpy as np
import onnxruntime as ort

from features import SensorFeatureExtractor, SensorReading, load_extractor, FEATURE_NAMES


class SimpleCalibratedClassifier:
    """Wrapper para modelo calibrado con IsotonicRegression por clase."""
    def __init__(self, base_model, calibrators, classes):
        self.base_model = base_model
        self.calibrators = calibrators
        self.classes_ = classes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.base_model.predict_proba(X)
        proba_cal = np.zeros_like(proba)
        for c, iso_reg in enumerate(self.calibrators):
            proba_cal[:, c] = iso_reg.predict(proba[:, c])
        proba_cal = np.clip(proba_cal, 1e-8, 1.0)
        proba_cal /= proba_cal.sum(axis=1, keepdims=True)
        return proba_cal

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


class SimpleBinCalibrated:
    """Wrapper para modelo binario calibrado."""
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.base_model.predict_proba(X)[:, 1]
        cal_proba = self.calibrator.predict(proba)
        cal_proba = np.clip(cal_proba, 1e-8, 1 - 1e-8)
        return np.vstack([1 - cal_proba, cal_proba]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class Experto2Inference:
    """
    Inferencia Experto 2: ONNX Runtime + calibración isotónica via lookup tables.

    Uso:
        infer = Experto2Inference(
            "models/experto2_production.onnx",
            "models/experto2_production.calib.joblib",
            "models/extractor.joblib"
        )
        p_gas = infer.predict(reading_dict)
    """

    LABEL_NAMES = ["normal", "quema_organica", "humo_toxico", "polvo_niebla"]

    def __init__(
        self,
        onnx_path: str,
        calib_path: str,
        extractor_path: str,
    ):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        calib_data = joblib.load(calib_path)
        self.calibrators = calib_data["calibrators"]
        self.classes = calib_data["classes"]

        self.extractor = load_extractor(extractor_path)

    def predict(self, reading_dict: dict) -> np.ndarray:
        """
        Retorna array de 4 probabilidades calibradas:
        [normal, quema_organica, humo_toxico, polvo_niebla]
        """
        from features import SensorReading
        reading = SensorReading(**reading_dict)
        vec = self.extractor.feature_vector(reading)
        X = vec[None, :]

        proba_raw = self.session.run([self.output_name], {self.input_name: X})[0][0]
        proba_cal = np.zeros(4, dtype=np.float32)

        for c in range(4):
            iso_reg = self.calibrators[c]
            proba_cal[c] = iso_reg.predict([proba_raw[c]])[0]

        proba_cal = np.clip(proba_cal, 1e-8, 1.0)
        proba_cal /= proba_cal.sum()
        return proba_cal

    def predict_class(self, reading_dict: dict) -> Tuple[int, float]:
        """Retorna (clase_predicha, confianza)."""
        proba = self.predict(reading_dict)
        return int(proba.argmax()), float(proba.max())

    def predict_binary(self, reading_dict: dict) -> Tuple[int, float]:
        """Retorna (fire/no-fire, confianza) usando modelo binario."""
        from features import SensorReading
        reading = SensorReading(**reading_dict)
        vec = self.extractor.feature_vector(reading)
        X = vec[None, :]
        proba = self.session.run([self.output_name], {self.input_name: X})[0][0]
        return int(proba[1] > 0.5), float(proba[1])


class Experto2BinaryInference:
    """Inferencia solo modelo binario fire/no-fire."""
    def __init__(self, bin_model_path: str):
        import joblib
        self.model = joblib.load(bin_model_path)

    def predict_proba(self, reading_dict: dict) -> np.ndarray:
        from features import SensorReading, load_extractor
        extractor = load_extractor("models/extractor.joblib")
        reading = SensorReading(**reading_dict)
        vec = extractor.feature_vector(reading)
        X = vec[None, :]
        return self.model.predict_proba(X)[0]

    def predict(self, reading_dict: dict) -> Tuple[int, float]:
        proba = self.predict_proba(reading_dict)
        return int(proba[1] > 0.5), float(proba[1])