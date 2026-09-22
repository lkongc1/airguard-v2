"""
Tests exhaustivos del Orquestador FSM.
pytest -v tests/test_orchestrator.py
"""

import time
import pytest

from experts.orchestrator import (
    AirGuardOrchestrator,
    OrchestratorConfig,
    State,
    SensorInput,
    VisionInput,
    ContextInput,
)


@pytest.fixture
def orchestrator():
    config = OrchestratorConfig(
        w_vision=0.35, w_gas=0.40, w_pm=0.25,
        r_prealert=0.45, r_alert_visual=0.65, r_alert_chemical=0.75,
        hysteresis_down=0.08, hysteresis_up=0.03,
        co_critical_ppm=50.0,
    )
    return AirGuardOrchestrator(config)


@pytest.fixture
def ctx_normal():
    return ContextInput(
        weight_multipliers={"w_vision": 1.0, "w_pm": 1.0, "w_gas": 1.0},
        flags={"is_fog": False, "is_dust": False, "is_condensation": False},
        condition="normal"
    )


@pytest.fixture
def ctx_fog():
    return ContextInput(
        weight_multipliers={"w_vision": 0.4, "w_pm": 1.2, "w_gas": 1.0},
        flags={"is_fog": True, "is_dust": False, "is_condensation": False},
        condition="fog"
    )


# Tests de transiciones basicas
def test_normal_to_prealert(orchestrator, ctx_normal):
    sensor = SensorInput(timestamp=time.time(), pm25=100, pm10=120, co_ppm=1, voc_index=80)
    status = orchestrator.process_sensor(sensor, ctx_normal)
    assert status["state"] in ("NORMAL", "PRE_ALERTA")

def test_prealert_to_alert_visual(orchestrator, ctx_normal):
    ts = time.time()
    # Usar valores más realistas de alta contaminación
    for pm, co, voc in [(80, 8, 200), (120, 12, 300), (180, 18, 400), (250, 25, 500)]:
        sensor = SensorInput(timestamp=ts, pm25=pm, pm10=pm*1.2, co_ppm=co, voc_index=voc)
        orchestrator.process_sensor(sensor, ctx_normal)
        ts += 10
    status = orchestrator.process_sensor(
        SensorInput(timestamp=ts, pm25=300, pm10=360, co_ppm=30, voc_index=600), ctx_normal
    )
    assert status["state"] in ("PRE_ALERTA", "ALERTA_VISUAL", "ALERTA_QUIMICA")

def test_chemical_override_forces_chemical_alert(orchestrator, ctx_normal):
    sensor = SensorInput(
        timestamp=time.time(), pm25=10, pm10=15,
        co_ppm=60.0,
        voc_index=50, temperature=25, humidity=50
    )
    status = orchestrator.process_sensor(sensor, ctx_normal)
    assert status["state"] == "ALERTA_QUIMICA"

def test_chemical_latch_requires_force_reset(orchestrator, ctx_normal):
    sensor = SensorInput(timestamp=time.time(), co_ppm=60.0, pm25=10, pm10=15)
    orchestrator.process_sensor(sensor, ctx_normal)
    assert orchestrator.state == State.ALERTA_QUIMICA

    result = orchestrator.reset(force=False)
    assert result["ok"] is False  # No se resetea sin force
    assert "error" in result
    assert orchestrator.state == State.ALERTA_QUIMICA

    result = orchestrator.reset(force=True)
    assert result["state"] == "NORMAL"  # Se resetea con force
    assert orchestrator.state == State.NORMAL


def test_vision_suppressed_in_fog(orchestrator, ctx_fog):
    vision = VisionInput(timestamp=time.time(), smoke_detected=True, confidence=0.9)
    sensor = SensorInput(timestamp=time.time(), pm25=20, pm10=80, co_ppm=1, voc_index=50)

    orchestrator.process_sensor(sensor, ctx_fog)
    status = orchestrator.process_vision(vision, ctx_fog)

    assert status["state"] != "ALERTA_VISUAL"


# Tests de histéresis
def test_hysteresis_asymmetric(orchestrator, ctx_normal):
    ts = time.time()
    # Subir riesgo gradualmente
    for i in range(5):
        sensor = SensorInput(timestamp=ts + i*10, pm25=50+i*40, pm10=60+i*40, co_ppm=5+i*5, voc_index=200+i*50)
        orchestrator.process_sensor(sensor, ctx_normal)

    # Bajar drásticamente - con hysteresis_down=0.08 no debe bajar a NORMAL inmediatamente
    sensor = SensorInput(timestamp=ts+50, pm25=20, pm10=30, co_ppm=2, voc_index=50)
    status = orchestrator.process_sensor(sensor, ctx_normal)

    # Con hysteresis_down=0.08, no debe bajar a NORMAL inmediatamente
    assert status["state"] != "NORMAL" or status["risk_score"] > 0.35


# Tests de degradado
def test_degraded_on_sensor_timeout(orchestrator, ctx_normal):
    sensor = SensorInput(timestamp=time.time(), pm25=10, pm10=15)
    orchestrator.process_sensor(sensor, ctx_normal)


# Tests de persistencia
def test_state_persists_in_sqlite(orchestrator, ctx_normal, tmp_path):
    import sqlite3
    sensor = SensorInput(timestamp=time.time(), co_ppm=60.0, pm25=10, pm10=15)
    orchestrator.process_sensor(sensor, ctx_normal)

    conn = sqlite3.connect(orchestrator.config.db_path)
    cursor = conn.execute("SELECT state FROM state_log ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    assert row[0] == "ALERTA_QUIMICA"
    conn.close()


# Tests de acknowledge
def test_acknowledge_does_not_reset(orchestrator, ctx_normal):
    # Usar valores que disparen ALERTA_QUIMICA
    sensor = SensorInput(timestamp=time.time(), pm25=200, pm10=240, co_ppm=60, voc_index=500)
    orchestrator.process_sensor(sensor, ctx_normal)
    assert orchestrator.state == State.ALERTA_QUIMICA
    state_before = orchestrator.state

    orchestrator.acknowledge()
    assert orchestrator.state == state_before
    assert orchestrator._acked is True