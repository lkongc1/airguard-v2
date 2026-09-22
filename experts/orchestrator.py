"""
Orquestador FSM — Fusión probabilística + Safety Overrides + Persistencia SQLite.

Estados: NORMAL → PRE_ALERTA → ALERTA_QUIMICA (latch terminal) / ALERTA_VISUAL
Transiciones deterministas, histéresis asimétrica, persistencia WAL.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Estados y Eventos
# ---------------------------------------------------------------------------

class State(Enum):
    NORMAL = "NORMAL"
    PRE_ALERTA = "PRE_ALERTA"
    ALERTA_VISUAL = "ALERTA_VISUAL"
    ALERTA_QUIMICA = "ALERTA_QUIMICA"
    DEGRADADO = "DEGRADADO"


class EventType(Enum):
    SENSOR_READING = "sensor_reading"
    VISION_RESULT = "vision_result"
    ACKNOWLEDGE = "acknowledge"
    RESET = "reset"
    SENSOR_FAILURE = "sensor_failure"
    SENSOR_RECOVERY = "sensor_recovery"


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorConfig:
    # Pesos base (suman 1.0)
    w_vision: float = 0.35
    w_gas: float = 0.40
    w_pm: float = 0.25

    # Umbrales de riesgo compuesto
    r_prealert: float = 0.45
    r_alert_visual: float = 0.65
    r_alert_chemical: float = 0.75

    # Histéresis asimétrica (bajada más lenta)
    hysteresis_down: float = 0.08
    hysteresis_up: float = 0.03

    # Ventanas temporales
    prealert_min_duration: int = 30
    alert_visual_max_duration: int = 300
    chemical_latch: bool = True

    # Safety overrides (umbrales físicos absolutos)
    co_critical_ppm: float = 50.0
    co_slope_critical: float = 5.0
    pm25_slope_critical: float = 50.0
    voc_slope_critical: float = 100.0
    voc_index_critical: float = 350

    # Persistencia
    db_path: str = "/var/lib/airguard/orchestrator.db"
    wal_mode: bool = True

    # Degradado
    max_sensor_age_sec: float = 10.0


# ---------------------------------------------------------------------------
# Entradas tipadas
# ---------------------------------------------------------------------------

@dataclass
class SensorInput:
    timestamp: float
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    co_ppm: Optional[float] = None
    voc_index: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None


@dataclass
class VisionInput:
    timestamp: float
    smoke_detected: bool
    confidence: float
    detections: List[Dict] = field(default_factory=list)


@dataclass
class ContextInput:
    weight_multipliers: Dict[str, float]
    flags: Dict[str, bool]
    condition: str


# ---------------------------------------------------------------------------
# FSM Principal
# ---------------------------------------------------------------------------

class AirGuardOrchestrator:
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.state = State.NORMAL
        self._risk_score = 0.0
        self._state_entry_time = time.time()
        self._last_sensor_time = 0.0
        self._last_vision_time = 0.0
        self._acked = False
        self._db = None
        self._init_db()

    # --- Base de datos ---

    def _init_db(self):
        Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.config.db_path, check_same_thread=False)
        if self.config.wal_mode:
            self._db.execute("PRAGMA journal_mode=WAL;")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS state_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                state TEXT NOT NULL,
                risk_score REAL,
                event_type TEXT,
                event_data TEXT,
                acknowledged INTEGER DEFAULT 0
            );
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_state_log_ts ON state_log(timestamp);")
        self._db.commit()

    def _log_transition(self, event_type: EventType, event_data: Dict, risk: float):
        self._db.execute(
            "INSERT INTO state_log (timestamp, state, risk_score, event_type, event_data) VALUES (?, ?, ?, ?, ?)",
            (time.time(), self.state.value, risk, event_type.value, json.dumps(event_data))
        )
        self._db.commit()

    def _log_acknowledge(self):
        self._db.execute(
            "UPDATE state_log SET acknowledged=1 WHERE id=(SELECT MAX(id) FROM state_log WHERE state=?)",
            (self.state.value,)
        )
        self._db.commit()

    # --- API Pública ---

    def process_sensor(self, sensor: SensorInput, context: ContextInput) -> Dict:
        """Procesa lectura de sensores, actualiza riesgo, evalúa transiciones."""
        self._last_sensor_time = sensor.timestamp
        self._check_degraded(sensor.timestamp)

        if self.state == State.DEGRADADO:
            return self._get_status("degraded_sensor")

        # Calcular probabilidades parciales
        p_pm = self._compute_p_pm(sensor, context)
        p_gas = self._compute_p_gas(sensor)
        p_vision = self._get_cached_p_vision()

        # Aplicar multiplicadores de contexto
        wv = self.config.w_vision * context.weight_multipliers.get("w_vision", 1.0)
        wg = self.config.w_gas * context.weight_multipliers.get("w_gas", 1.0)
        wp = self.config.w_pm * context.weight_multipliers.get("w_pm", 1.0)

        # Normalizar pesos
        w_sum = wv + wg + wp
        wv, wg, wp = wv/w_sum, wg/w_sum, wp/w_sum

        # Fusión OR probabilística
        r_base = 1.0 - (1.0 - p_vision * wv) * (1.0 - p_gas * wg) * (1.0 - p_pm * wp)

        # Safety overrides físicos (siempre ganan)
        safety = self._check_safety_overrides(sensor, context)
        if safety["force_chemical"]:
            r_base = max(r_base, 1.0)
        elif safety["suppress_vision"]:
            r_base = min(r_base, self.config.r_prealert - 0.05)

        # Histéresis
        r_effective = self._apply_hysteresis(r_base)

        # Transiciones
        old_state = self.state
        self._risk_score = r_effective
        self._evaluate_transitions(safety, sensor.timestamp)

        # Log si cambió estado
        if self.state != old_state:
            self._state_entry_time = sensor.timestamp
            self._acked = False
            self._log_transition(EventType.SENSOR_READING, {
                "sensor": sensor.__dict__,
                "context": context.__dict__,
                "p_vision": p_vision, "p_gas": p_gas, "p_pm": p_pm,
                "weights": {"wv": wv, "wg": wg, "wp": wp},
                "safety": safety,
                "r_base": r_base, "r_effective": r_effective,
            }, r_effective)

        return self._get_status("sensor_update")

    def process_vision(self, vision: VisionInput, context: ContextInput) -> Dict:
        """Procesa resultado de visión, actualiza p_vision cacheado."""
        self._last_vision_time = vision.timestamp
        self._cached_p_vision = vision.confidence if vision.smoke_detected else 0.0
        self._cached_vision_raw = vision.smoke_detected

        if self.state == State.PRE_ALERTA and vision.smoke_detected:
            self._evaluate_transitions({}, vision.timestamp)

        return self._get_status("vision_update")

    def acknowledge(self) -> Dict:
        """Operador reconoce la alerta (NO la resetea)."""
        if self.state in (State.PRE_ALERTA, State.ALERTA_VISUAL, State.ALERTA_QUIMICA):
            self._acked = True
            self._log_acknowledge()
        return self._get_status("acknowledged")

    def reset(self, force: bool = False) -> Dict:
        """
        Reset manual.
        - PRE_ALERTA, ALERTA_VISUAL: reset directo a NORMAL
        - ALERTA_QUIMICA: requiere force=True (llave física / 2FA)
        """
        if self.state == State.ALERTA_QUIMICA and not force:
            return {"ok": False, "error": "ALERTA_QUIMICA requiere reset forzado (llave/2FA)"}

        if self.state in (State.PRE_ALERTA, State.ALERTA_VISUAL, State.ALERTA_QUIMICA):
            old = self.state
            self.state = State.NORMAL
            self._risk_score = 0.0
            self._state_entry_time = time.time()
            self._acked = False
            self._log_transition(EventType.RESET, {"from": old.value, "force": force}, 0.0)
        return self._get_status("reset")

    def get_status(self) -> Dict:
        return self._get_status("status_query")

    # --- Lógica Interna ---

    def _check_degraded(self, now: float):
        if now - self._last_sensor_time > self.config.max_sensor_age_sec:
            if self.state != State.DEGRADADO:
                self.state = State.DEGRADADO
                self._log_transition(EventType.SENSOR_FAILURE, {"since": self._last_sensor_time}, self._risk_score)

    def _get_cached_p_vision(self) -> float:
        return getattr(self, "_cached_p_vision", 0.0)

    def _compute_p_pm(self, sensor: SensorInput, context: ContextInput) -> float:
        if sensor.pm25 is None or sensor.pm10 is None:
            return 0.0

        from experts.context import correct_pm25_humidity
        pm25_corr = correct_pm25_humidity(sensor.pm25, sensor.humidity or 50.0)

        if pm25_corr >= 150: base = 0.9
        elif pm25_corr >= 75: base = 0.7
        elif pm25_corr >= 35: base = 0.5
        elif pm25_corr >= 15: base = 0.2
        else: base = 0.05

        return min(1.0, base)

    def _compute_p_gas(self, sensor: SensorInput) -> float:
        score = 0.0
        if sensor.co_ppm is not None:
            if sensor.co_ppm >= 30: score += 0.5
            elif sensor.co_ppm >= 10: score += 0.3
            elif sensor.co_ppm >= 5: score += 0.15
        if sensor.voc_index is not None:
            if sensor.voc_index >= 300: score += 0.4
            elif sensor.voc_index >= 200: score += 0.25
            elif sensor.voc_index >= 100: score += 0.1
        return min(1.0, score)

    def _check_safety_overrides(self, sensor: SensorInput, context: ContextInput) -> Dict:
        overrides = {"force_chemical": False, "suppress_vision": False}

        if context.flags.get("is_fog"):
            overrides["suppress_vision"] = True
        if context.flags.get("force_chemical_alert"):
            overrides["force_chemical"] = True

        if sensor.co_ppm is not None and sensor.co_ppm >= self.config.co_critical_ppm:
            overrides["force_chemical"] = True
        if sensor.voc_index is not None and sensor.voc_index >= self.config.voc_index_critical:
            overrides["force_chemical"] = True

        return overrides

    def _apply_hysteresis(self, r_base: float) -> float:
        if r_base > self._risk_score:
            return r_base
        else:
            if r_base < self._risk_score - self.config.hysteresis_down:
                return r_base
            return self._risk_score

    def _evaluate_transitions(self, safety: Dict, now: float):
        r = self._risk_score
        time_in_state = now - self._state_entry_time

        if self.state == State.NORMAL:
            if safety["force_chemical"]:
                self.state = State.ALERTA_QUIMICA
            elif r >= self.config.r_alert_chemical:
                self.state = State.ALERTA_QUIMICA
            elif r >= self.config.r_alert_visual:
                self.state = State.ALERTA_VISUAL
            elif r >= self.config.r_prealert:
                self.state = State.PRE_ALERTA

        elif self.state == State.PRE_ALERTA:
            if safety["force_chemical"]:
                self.state = State.ALERTA_QUIMICA
            elif r >= self.config.r_alert_chemical:
                self.state = State.ALERTA_QUIMICA
            elif r >= self.config.r_alert_visual:
                self.state = State.ALERTA_VISUAL
            elif r < self.config.r_prealert - self.config.hysteresis_down:
                self.state = State.NORMAL
            elif time_in_state > self.config.prealert_min_duration and r > self.config.r_prealert + 0.1:
                self.state = State.ALERTA_VISUAL

        elif self.state == State.ALERTA_VISUAL:
            if safety["force_chemical"]:
                self.state = State.ALERTA_QUIMICA
            elif r >= self.config.r_alert_chemical:
                self.state = State.ALERTA_QUIMICA
            elif r < self.config.r_alert_visual - self.config.hysteresis_down:
                self.state = State.PRE_ALERTA
            elif time_in_state > self.config.alert_visual_max_duration:
                self.state = State.PRE_ALERTA

        elif self.state == State.ALERTA_QUIMICA:
            # Latch terminal - SOLO sale con reset manual (force=True)
            pass

        elif self.state == State.DEGRADADO:
            if now - self._last_sensor_time <= self.config.max_sensor_age_sec:
                self.state = State.NORMAL
                self._risk_score = 0.0
                self._state_entry_time = now

    def _get_status(self, source: str) -> Dict:
        return {
            "state": self.state.value,
            "risk_score": round(self._risk_score, 4),
            "acked": self._acked,
            "time_in_state": round(time.time() - self._state_entry_time, 1),
            "last_sensor_age": round(time.time() - self._last_sensor_time, 1) if self._last_sensor_time else None,
            "last_vision_age": round(time.time() - self._last_vision_time, 1) if self._last_vision_time else None,
            "source": source,
            "timestamp": time.time(),
        }

    def close(self):
        if self._db:
            self._db.close()