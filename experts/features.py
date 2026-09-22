"""
Feature engineering para Experto 2 (Gases y Partículas).

Diseñado para:
- Training offline con pandas
- Inference online en edge (stateful, sin pandas)
- Serialización via joblib
- Tests de invariancia

Autor: AirGuard Team
Licencia: MIT
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Parámetros del extractor. Inmutables para serialización."""

    # Ventanas (en segundos)
    co_slope_window: int = 120
    pm_slope_window: int = 120
    voc_slope_window: int = 60
    temp_delta_window: int = 300
    pm25_delta_window: int = 60

    # Mínimo de muestras para calcular slope
    min_samples_slope: int = 5
    min_dt_seconds: float = 30.0

    # Corrección de humedad
    hr_threshold: float = 80.0
    hr_max_correction: float = 0.45

    # Historial máximo (para deque)
    max_history: int = 600

    # Bins de humedad
    hr_bins: Tuple[float, ...] = (0, 30, 50, 70, 85, 95, 101)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _safe_ratio(num: Optional[float], den: Optional[float], eps: float = 1e-6) -> float:
    """Ratio seguro que evita división por cero y propaga NaN."""
    if num is None or den is None:
        return float("nan")
    if abs(den) < eps:
        return float("nan")
    return num / den


def _slope(samples: List[Tuple[float, float]], min_dt: float = 30.0, min_n: int = 5) -> float:
    """
    Slope robusto via regresión lineal (no diferencia de extremos).

    samples: lista de (timestamp, valor) ordenada por timestamp.
    Retorna unidades por minuto. NaN si no hay datos suficientes.
    """
    if len(samples) < min_n:
        return float("nan")

    ts = np.array([s[0] for s in samples], dtype=np.float64)
    vs = np.array([s[1] for s in samples], dtype=np.float64)

    dt = ts[-1] - ts[0]
    if dt < min_dt:
        return float("nan")

    # Regresión lineal: slope = cov(t,v) / var(t)
    t_mean = ts.mean()
    v_mean = vs.mean()
    denom = ((ts - t_mean) ** 2).sum()
    if denom < 1e-9:
        return float("nan")

    slope_per_sec = ((ts - t_mean) * (vs - v_mean)).sum() / denom
    return float(slope_per_sec * 60.0)  # por minuto


def _correct_pm25_humidity(pm25: Optional[float], hr: Optional[float], cfg: FeatureConfig) -> float:
    """
    Corrección de higroscopocidad. Curva empírica para PMS5003.

    Referencias:
    - Crilley et al. (2018): PMS5003 sobreestima PM2.5 >80% HR
    - Corrección tipo factor lineal decreciente
    """
    if pm25 is None or hr is None:
        return float("nan")
    if hr < cfg.hr_threshold:
        return float(pm25)

    span = 100.0 - cfg.hr_threshold
    factor = 1.0 - ((hr - cfg.hr_threshold) / span) * (1.0 - cfg.hr_max_correction)
    factor = max(cfg.hr_max_correction, min(1.0, factor))
    return float(pm25 * factor)


def _hour_cyclical(hour: int) -> Tuple[float, float]:
    """Codificación cíclica de hora. hour en [0, 23]."""
    rad = 2.0 * math.pi * (hour % 24) / 24.0
    return math.sin(rad), math.cos(rad)


# ---------------------------------------------------------------------------
# Extractor stateful (edge-friendly)
# ---------------------------------------------------------------------------

FEATURE_NAMES: Tuple[str, ...] = (
    "pm25_raw",
    "pm25_corrected",
    "pm10_raw",
    "pm_ratio",
    "pm25_delta_1min",
    "pm_slope_2min",
    "co_ppm",
    "co_slope_2min",
    "voc_index",
    "voc_slope_1min",
    "co_pm_ratio",
    "temperature",
    "temp_delta_5min",
    "humidity",
    "hr_bucket",
    "hour_sin",
    "hour_cos",
)


@dataclass
class SensorReading:
    """Lectura cruda de un instante. Campos None si el sensor no reportó."""
    timestamp: float
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    co_ppm: Optional[float] = None
    voc_index: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None


class SensorFeatureExtractor:
    """
    Extractor stateful. Uso:

        extractor = SensorFeatureExtractor()
        for reading in stream:
            features = extractor.transform_one(reading)
            # features: dict[str, float] con FEATURE_NAMES
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._history: Deque[SensorReading] = deque(maxlen=self.config.max_history)

    # --- API pública ---

    def reset(self) -> None:
        """Limpia el historial. Útil entre tests o reinicios."""
        self._history.clear()

    def transform_one(self, reading: SensorReading) -> Dict[str, float]:
        """Procesa una lectura y devuelve dict de features."""
        self._history.append(reading)
        return self._compute(reading)

    def transform_batch(self, readings: List[SensorReading]) -> List[Dict[str, float]]:
        """Procesa un batch secuencialmente (mantiene estado temporal)."""
        out = []
        for r in readings:
            out.append(self.transform_one(r))
        return out

    def feature_vector(self, reading: SensorReading) -> np.ndarray:
        """Devuelve vector numpy en orden FEATURE_NAMES (para modelo)."""
        feats = self.transform_one(reading)
        return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)

    # --- Cálculo interno ---

    def _window(self, current_ts: float, window_sec: float) -> List[SensorReading]:
        """Devuelve lecturas dentro de la ventana temporal."""
        cutoff = current_ts - window_sec
        return [r for r in self._history if r.timestamp >= cutoff]

    def _compute(self, r: SensorReading) -> Dict[str, float]:
        cfg = self.config
        out: Dict[str, float] = {}

        # --- PM ---
        out["pm25_raw"] = float(r.pm25) if r.pm25 is not None else float("nan")
        out["pm10_raw"] = float(r.pm10) if r.pm10 is not None else float("nan")
        out["pm25_corrected"] = _correct_pm25_humidity(r.pm25, r.humidity, cfg)
        out["pm_ratio"] = _safe_ratio(r.pm25, r.pm10)

        # pm25_delta_1min: diferencia vs lectura ~60s atrás
        w = self._window(r.timestamp, cfg.pm25_delta_window)
        if len(w) >= 2 and r.pm25 is not None and w[0].pm25 is not None:
            out["pm25_delta_1min"] = float(r.pm25 - w[0].pm25)
        else:
            out["pm25_delta_1min"] = float("nan")

        # pm_slope_2min sobre pm25_corrected
        pm_samples = [
            (s.timestamp, _correct_pm25_humidity(s.pm25, s.humidity, cfg))
            for s in self._window(r.timestamp, cfg.pm_slope_window)
            if s.pm25 is not None
        ]
        pm_samples = [(t, v) for t, v in pm_samples if not math.isnan(v)]
        out["pm_slope_2min"] = _slope(pm_samples, cfg.min_dt_seconds, cfg.min_samples_slope)

        # --- CO ---
        out["co_ppm"] = float(r.co_ppm) if r.co_ppm is not None else float("nan")
        co_samples = [
            (s.timestamp, s.co_ppm)
            for s in self._window(r.timestamp, cfg.co_slope_window)
            if s.co_ppm is not None
        ]
        out["co_slope_2min"] = _slope(co_samples, cfg.min_dt_seconds, cfg.min_samples_slope)

        # --- VOC ---
        out["voc_index"] = float(r.voc_index) if r.voc_index is not None else float("nan")
        voc_samples = [
            (s.timestamp, s.voc_index)
            for s in self._window(r.timestamp, cfg.voc_slope_window)
            if s.voc_index is not None
        ]
        out["voc_slope_1min"] = _slope(voc_samples, cfg.min_dt_seconds, cfg.min_samples_slope)

        # --- Ratios cruzados ---
        pm_corr = out["pm25_corrected"]
        if r.co_ppm is not None and not math.isnan(pm_corr) and pm_corr > 1.0:
            out["co_pm_ratio"] = float(r.co_ppm / pm_corr)
        else:
            out["co_pm_ratio"] = float("nan")

        # --- Temperatura ---
        out["temperature"] = float(r.temperature) if r.temperature is not None else float("nan")
        t_samples = [
            (s.timestamp, s.temperature)
            for s in self._window(r.timestamp, cfg.temp_delta_window)
            if s.temperature is not None
        ]
        if len(t_samples) >= 2:
            out["temp_delta_5min"] = float(t_samples[-1][1] - t_samples[0][1])
        else:
            out["temp_delta_5min"] = float("nan")

        # --- Humedad ---
        out["humidity"] = float(r.humidity) if r.humidity is not None else float("nan")
        if r.humidity is not None:
            bucket = 0
            for i in range(len(cfg.hr_bins) - 1):
                if cfg.hr_bins[i] <= r.humidity < cfg.hr_bins[i + 1]:
                    bucket = i
                    break
            out["hr_bucket"] = float(bucket)
        else:
            out["hr_bucket"] = float("nan")

        # --- Hora cíclica ---
        import datetime as _dt
        hour = _dt.datetime.utcfromtimestamp(r.timestamp).hour
        sin_h, cos_h = _hour_cyclical(hour)
        out["hour_sin"] = sin_h
        out["hour_cos"] = cos_h

        return out


# ---------------------------------------------------------------------------
# Serialización
# ---------------------------------------------------------------------------

def save_extractor(extractor: SensorFeatureExtractor, path: str) -> None:
    """Guarda configuración. El historial NO se serializa (se reconstruye)."""
    import joblib
    joblib.dump({"config": extractor.config}, path)


def load_extractor(path: str) -> SensorFeatureExtractor:
    import joblib
    payload = joblib.load(path)
    return SensorFeatureExtractor(config=payload["config"])