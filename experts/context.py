"""
Experto 3: Contexto Ambiental — Corrección de falsos positivos por condiciones meteorológicas.

NO es ML. Es física + reglas interpretables + modelo Bayesiano ligero.
Objetivo: modular los pesos del orquestador según condiciones (niebla, polvo, condensación).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class ContextConfig:
    # Umbrales de niebla
    fog_hr_threshold: float = 90.0
    fog_temp_threshold: float = 15.0
    fog_pm_ratio_max: float = 0.3

    # Umbrales de polvo
    dust_hr_threshold: float = 40.0
    dust_pm_ratio_min: float = 0.7

    # Condensación nocturna
    night_hours: Tuple[int, ...] = (22, 23, 0, 1, 2, 3, 4, 5)
    condensation_hr: float = 85.0

    # Multiplicadores de peso para el orquestador
    fog_vision_weight_mult: float = 0.4
    dust_pm_weight_mult: float = 0.6
    condensation_vision_mult: float = 0.7

    # Modelo Bayesiano simple (opcional)
    use_bayesian: bool = True
    prior_fog: float = 0.15
    prior_dust: float = 0.10
    prior_condensation: float = 0.08


# ---------------------------------------------------------------------------
# Motor de Contexto
# ---------------------------------------------------------------------------

class EnvironmentalContext:
    """
    Evalúa condiciones ambientales y devuelve factores de corrección
    para los pesos del orquestador.
    """

    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()

    def evaluate(
        self,
        temperature: Optional[float],
        humidity: Optional[float],
        pm25: Optional[float],
        pm10: Optional[float],
        hour_utc: int,
        wind_speed: Optional[float] = None,
    ) -> Dict:
        """
        Retorna dict con:
        - condition: "fog" | "dust" | "condensation" | "normal"
        - probabilities: dict con probabilidad de cada condición
        - weight_multipliers: dict para ajustar w_vision, w_pm, w_gas
        - flags: dict de booleanos para safety overrides
        """
        result = {
            "condition": "normal",
            "probabilities": {"fog": 0.0, "dust": 0.0, "condensation": 0.0, "normal": 1.0},
            "weight_multipliers": {"w_vision": 1.0, "w_pm": 1.0, "w_gas": 1.0},
            "flags": {"is_fog": False, "is_dust": False, "is_condensation": False},
        }

        if humidity is None or temperature is None:
            return result

        pm_ratio = pm25 / pm10 if (pm25 and pm10 and pm10 > 0) else None

        # --- 1. Niebla ---
        fog_score = 0.0
        if humidity >= self.config.fog_hr_threshold:
            fog_score += 0.5
        if temperature <= self.config.fog_temp_threshold:
            fog_score += 0.3
        if pm_ratio is not None and pm_ratio <= self.config.fog_pm_ratio_max:
            fog_score += 0.2

        # --- 2. Polvo ---
        dust_score = 0.0
        if humidity <= self.config.dust_hr_threshold:
            dust_score += 0.4
        if pm_ratio is not None and pm_ratio >= self.config.dust_pm_ratio_min:
            dust_score += 0.4
        if wind_speed is not None and wind_speed > 5.0:
            dust_score += 0.2

        # --- 3. Condensación nocturna ---
        cond_score = 0.0
        if hour_utc in self.config.night_hours:
            cond_score += 0.4
        if humidity >= self.config.condensation_hr:
            cond_score += 0.4
        if temperature is not None and temperature < 18:
            cond_score += 0.2

        # Normalizar a probabilidades (softmax simple)
        scores = {
            "fog": fog_score,
            "dust": dust_score,
            "condensation": cond_score,
            "normal": 1.0,
        }

        if self.config.use_bayesian:
            scores["fog"] *= self.config.prior_fog / 0.15
            scores["dust"] *= self.config.prior_dust / 0.10
            scores["condensation"] *= self.config.prior_condensation / 0.08

        total = sum(scores.values())
        probs = {k: v / total for k, v in scores.items()}

        # Condición dominante
        dominant = max(probs, key=probs.get)
        result["condition"] = dominant
        result["probabilities"] = probs

        # Multiplicadores de peso
        mult = {"w_vision": 1.0, "w_pm": 1.0, "w_gas": 1.0}
        flags = {"is_fog": False, "is_dust": False, "is_condensation": False}

        if dominant == "fog" and probs["fog"] > 0.5:
            mult["w_vision"] *= self.config.fog_vision_weight_mult
            mult["w_pm"] *= 1.2
            flags["is_fog"] = True

        if dominant == "dust" and probs["dust"] > 0.5:
            mult["w_pm"] *= self.config.dust_pm_weight_mult
            mult["w_gas"] *= 1.1
            flags["is_dust"] = True

        if dominant == "condensation" and probs["condensation"] > 0.5:
            mult["w_vision"] *= self.config.condensation_vision_mult
            flags["is_condensation"] = True

        result["weight_multipliers"] = mult
        result["flags"] = flags

        return result


# ---------------------------------------------------------------------------
# Corrección de PM2.5 por higroscopocidad (reutilizable)
# ---------------------------------------------------------------------------

def correct_pm25_humidity(pm25: float, humidity: float, hr_threshold: float = 80.0, max_correction: float = 0.45) -> float:
    """
    Corrección empírica para PMS5003.
    Factor lineal de 1.0 (HR=threshold) a max_correction (HR=100).
    """
    if humidity < hr_threshold:
        return pm25
    span = 100.0 - hr_threshold
    factor = 1.0 - ((humidity - hr_threshold) / span) * (1.0 - max_correction)
    factor = max(max_correction, min(1.0, factor))
    return pm25 * factor


# ---------------------------------------------------------------------------
# Safety Overrides basados en contexto
# ---------------------------------------------------------------------------

def context_safety_overrides(ctx_result: Dict, gas_reading: Dict) -> Dict[str, bool]:
    """
    Genera flags de safety override que el orquestador puede usar
    para forzar alertas o suprimir falsos positivos.
    """
    flags = ctx_result["flags"]
    overrides = {
        "suppress_vision_alert": False,
        "require_gas_confirmation": False,
        "elevate_pm_weight": False,
    }

    if flags["is_fog"]:
        overrides["suppress_vision_alert"] = True
        overrides["require_gas_confirmation"] = True

    if flags["is_dust"]:
        overrides["elevate_pm_weight"] = True
        overrides["require_gas_confirmation"] = True

    if flags["is_condensation"]:
        overrides["suppress_vision_alert"] = True

    # Overrides por gases tóxicos (independiente de contexto)
    co = gas_reading.get("co_ppm")
    voc = gas_reading.get("voc_index")
    if co is not None and co >= 50.0:
        overrides["force_chemical_alert"] = True
    if voc is not None and voc >= 350:
        overrides["force_chemical_alert"] = True

    return overrides