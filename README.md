# AirGuard: Monitor de Calidad del Aire y Detección Temprana de Quemas/Humos Tóxicos

Sistema multimodal de detección de incendios y humos tóxicos para despliegue en edge (RPi 5 + Hailo-8L + ESP32-S3).

## Resumen

Sistema multimodal de detección de incendios y humos tóxicos con tres expertos especializados:

| Experto | Modelo | Estado | Artefacto Principal |
|---------|--------|--------|---------------------|
| **Experto 1: Visión** | YOLOv8n (512×512) |  Epoch 100/100 | `best.onnx` (11.7 MB) |
| **Experto 2: Gases** | XGBoost + Calibración Isotónica |  Producción | `experto2_production.onnx` |
| **Experto 3: Contexto** | Reglas + Bayesiano |  Listo | `context.py` |

## Hardware Objetivo

- **Edge:** RPi 5 (8GB) + Hailo-8L + ESP32-S3 + ATECC608B
- **Sensores:** PMS5003 (PM), SGP40 (VOC), DART 2-CO (CO), BME280 (T/H/P)
- **Cámara:** IP RTSP (2-5 FPS)

## Requisitos de GPU para Entrenamiento

| Modelo | VRAM Mínima | VRAM Recomendada | Tiempo Estimado (100 epochs) |
|--------|-------------|------------------|------------------------------|
| YOLOv8n (512×512, batch 16) | 4 GB | **8 GB** | ~1.5 horas |
| YOLOv8n (640×640, batch 32) | 6 GB | **12 GB** | ~1 hora |
| XGBoost (CPU) | N/A | N/A | ~5 min |

**Recomendación:** GPU con **≥ 8 GB VRAM** para entrenamiento cómodo con batch size 16 y mixed precision (FP16). Con 6 GB VRAM es posible reducir batch size a 8.

## Métricas Clave

| Métrica | Target | Actual (Producción) |
|---------|--------|---------------------|
| **YOLO mAP@50** | ≥ 0.82 | 0.607 (epoch 100) |
| **YOLO Recall** | ≥ 0.90 | 0.516 |
| **Gases F1-macro (prod)** | ≥ 0.88 | **0.999**  |
| **Gases Recall tóxico** | ≥ 0.95 | **1.000**  |
| **Inferencia ONNX CPU** | < 100ms | **18.8 ms** (53 FPS)  |

> **Nota:** El modelo de visión requiere fine-tuning con datos locales (Perú) y hard negatives para alcanzar targets de producción.

## Arquitectura

```
airguard-monitor/
├── experts/
│   ├── vision.py      # YOLOv8n + ONNX + Hailo + hard negatives
│   ├── gases.py       # XGBoost + calibración isotónica (multi + binario)
│   ├── context.py     # Fog/dust/condensación + safety overrides
│   ├── orchestrator.py # FSM + fusión probabilística + SQLite WAL
│   └── features.py    # 38 features causales (sin leakage)
├── schemas/
│   ├── config.yaml    # Config centralizada
│   └── mqtt_v1.json   # MQTT v1.0.0 + mTLS/HMAC
├── deploy/
│   └── docker-compose.yml # Mosquitto mTLS + InfluxDB + Grafana + Backend
├── tests/
│   └── test_orchestrator.py # 9 tests passing
├── train_*.py         # Scripts de entrenamiento (4)
└── deploy/docker-compose.yml
```

## Modelos Listos para Deploy

| Modelo | Archivo | Estado |
|--------|---------|--------|
| **YOLOv8n epoch 100** | `runs/detect/.../best.onnx` |  Requiere fine-tuning |
| **XGBoost Producción** | `models/experto2_production/experto2_production.onnx` |  **Deploy-ready** |
| **XGBoost Binario** | `models/experto2_production/experto2_binary_calibrated.joblib` |  |

### XGBoost Producción (Sin Data Leakage)
- **38 features causales** (sin rolling windows que usan futuro)
- **Split estratificado** + class weights + calibración isotónica
- **F1-macro 0.999** en test hold-out (6,530 muestras)
- **ONNX + calibradores** listos para edge (RPi 5 + Hailo-8L)
- **Inferencia ~19ms CPU** (53 FPS)

## Inicio Rápido

```bash
# Clonar
git clone https://github.com/lkongc1/airguard-monitor.git
cd airguard-monitor
git checkout develop

# Entrenamiento YOLO (requiere GPU ≥ 8GB VRAM)
python train_yolo_7gb.py --data data/yolo_unified/data.yaml --epochs 100

# Entrenamiento XGBoost Producción (CPU)
python train_experto2_production.py --data data/processed/iot_smoke_train.csv

# Deploy edge (RPi 5 + Hailo-8L)
docker-compose -f deploy/docker-compose.yml up -d
```

## Despliegue Edge (RPi 5 + Hailo-8L)

```bash
# Compilar YOLO a HEF (INT8) para Hailo-8L
hailo-compiler --onnx best.onnx --output best.hef --quantization int8

# Stack completo
docker-compose -f deploy/docker-compose.yml up -d
```

**Stack incluido:**
- Mosquitto MQTT (mTLS)
- InfluxDB + Grafana (telemetría)
- Backend FastAPI (orquestador + experts)
- Edge simulator (testing sin hardware)

## Datos y Fine-Tuning

### Datos Campo Perú (Fase 1: 14 días)
Estructura en `data/peru_field_data/fase1_14dias/`:
- `sensor_logs/` - CSVs 1 Hz (14 días)
- `camera_frames/` - Frames JPG (cada 30s)
- `annotations/` - Etiquetas manuales (COCO/CSV)
- `calibration/` - Curvas CO, PM-HR, baselines nocturnos

### Hard Negatives
18 falsos positivos minados en `data/hard_negatives/mined/`. Target: 500+ imágenes (niebla, polvo, vapor, humo vehículos).

## Seguridad

- **MQTT v1.0.0** con mTLS + HMAC-SHA256 + nonce anti-replay
- **ATECC608B** root of trust en ESP32-S3
- **SQLite WAL** persistencia FSM con acknowledge separado de reset
- **Safety overrides** absolutos (CO ≥ 50 ppm → ALERTA_QUIMICA inmediata)

## Costos

| Componente | Costo USD |
|------------|-----------|
| RPi 5 8GB + Hailo-8L + ESP32-S3 | 160 |
| Sensores (PMS5003, SGP40, DART 2-CO, BME280) | 90 |
| ATECC608B + Cámara IP + Gabinete IP65 | 81 |
| Solar + Batería | 50 |
| **TOTAL** | **~381 USD** |

## Estado del Proyecto

| Componente | Estado |
|------------|--------|
| Experto 1 (Visión) |  Epoch 100/100 - requiere fine-tuning Perú |
| Experto 2 Producción |  **Deploy-ready** |
| Experto 3 Contexto |  Listo |
| Orquestador FSM |  9/9 tests passing |
| MQTT + Seguridad |  v1.0.0 |
| Edge Deployment |  ONNX + artifacts |

## Próximos Pasos

1. **Fase 1:** Recolección 14 días datos Perú (sitio piloto)
2. **Fine-tuning:** YOLOv8n con datos Perú + 18 hard negatives minados
3. **Compilación HEF:** YOLO → INT8 para Hailo-8L
3. **Integración:** RPi 5 + Hailo-8L + stack Docker
4. **Piloto:** 7 días supervisado en campo

## Licencia

Proyecto privado - AirGuard Team

---

*Última actualización: Septiembre 2026. Artefactos en `models/`, `runs/`, `data/processed/`, `experts/`, `schemas/`.*