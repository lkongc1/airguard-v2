"""
Experto 1: Visión - Detección visual de humo/incendio con YOLOv8n.

Pipeline:
1. Entrenamiento: D-Fire + Smoke24776 + negativos duros locales
2. Validación: mAP50, recall, FP/día en replay
3. Export: ONNX → Hailo Compiler (INT8)
4. Inferencia edge: 2-5 FPS, ventana temporal 5/10 frames

Clases: smoke (0), fire (1), person (2-opcional)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class VisionConfig:
    # Modelo
    model_name: str = "yolov8n"
    img_size: int = 640
    classes: Tuple[str, ...] = ("smoke", "fire")
    class_map: Dict[str, int] = field(default_factory=lambda: {"smoke": 0, "fire": 1})

    # Entrenamiento
    epochs: int = 100
    batch_size: int = 16
    lr0: float = 0.01
    patience: int = 20
    device: str = "auto"

    # Augmentación (conservadora para humo)
    mosaic: float = 1.0
    mixup: float = 0.1
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5

    # Inferencia
    conf_thres: float = 0.35
    iou_thres: float = 0.45
    max_det: int = 50
    temporal_window: int = 10
    temporal_threshold: int = 5

    # ROI (opcional): [x1, y1, x2, y2] normalizado 0-1
    roi_norm: Optional[List[float]] = None

    # Hard negative mining
    hard_negative_dir: str = "data/hard_negatives"
    min_hard_negatives: int = 500


# ---------------------------------------------------------------------------
# Dataset YOLO
# ---------------------------------------------------------------------------

def prepare_yolo_dataset(
    data_root: Path,
    dfire_dir: Path,
    smoke24776_dir: Path,
    hard_neg_dir: Path,
    output_dir: Path,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Path:
    """
    Prepara dataset YOLO unificado:
    - D-Fire: imágenes + labels YOLO
    - Smoke24776: imágenes + labels YOLO (ya vienen en formato YOLO)
    - Hard negatives: imágenes SIN labels (background)
    Estructura salida:
        output_dir/
            images/{train,val,test}/
            labels/{train,val,test}/
            data.yaml
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_samples: List[Tuple[Path, Path]] = []

    # 1. D-Fire
    dfire_imgs = list((dfire_dir / "images").glob("*.jpg")) + list((dfire_dir / "images").glob("*.png"))
    dfire_lbls = {p.stem: p for p in (dfire_dir / "labels").glob("*.txt")}
    for img in dfire_imgs:
        if img.stem in dfire_lbls:
            all_samples.append((img, dfire_lbls[img.stem]))

    # 2. Smoke24776 (ya en formato YOLO)
    s24776_imgs = list((smoke24776_dir / "images").glob("*.jpg"))
    s24776_lbls = {p.stem: p for p in (smoke24776_dir / "labels").glob("*.txt")}
    for img in s24776_imgs:
        if img.stem in s24776_lbls:
            all_samples.append((img, s24776_lbls[img.stem]))

    # 3. Hard negatives (solo imágenes, labels vacíos)
    hard_imgs = list(Path(hard_neg_dir).glob("*.jpg")) + list(Path(hard_neg_dir).glob("*.png"))
    for img in hard_imgs:
        all_samples.append((img, None))

    print(f"Total muestras: {len(all_samples)} (con labels: {sum(1 for _, l in all_samples if l)}, negativos: {sum(1 for _, l in all_samples if not l)})")

    # Split
    random.shuffle(all_samples)
    n = len(all_samples)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:],
    }

    # Copiar archivos
    for split_name, samples in splits.items():
        for img_path, lbl_path in samples:
            dst_img = output_dir / "images" / split_name / img_path.name
            shutil.copy2(img_path, dst_img)

            dst_lbl = output_dir / "labels" / split_name / (img_path.stem + ".txt")
            if lbl_path and lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl)
            else:
                dst_lbl.write_text("")

    # data.yaml
    data_yaml = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(VisionConfig().classes),
        "names": list(VisionConfig().classes),
    }
    (output_dir / "data.yaml").write_text(yaml.dump(data_yaml))

    print(f"✅ Dataset YOLO listo en {output_dir}")
    return output_dir / "data.yaml"


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train_yolo(data_yaml: Path, config: VisionConfig, run_name: str = "exp") -> Path:
    """Entrena YOLOv8n y retorna ruta al best.pt"""
    model = YOLO(f"{config.model_name}.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=config.epochs,
        imgsz=config.img_size,
        batch=config.batch_size,
        lr0=config.lr0,
        patience=config.patience,
        device=config.device,
        name=run_name,
        exist_ok=True,
        mosaic=config.mosaic,
        mixup=config.mixup,
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        degrees=config.degrees,
        translate=config.translate,
        scale=config.scale,
        shear=config.shear,
        perspective=config.perspective,
        flipud=config.flipud,
        fliplr=config.fliplr,
        classes=list(range(len(config.classes))),
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"✅ Entrenamiento completado. Mejor modelo: {best_pt}")
    return best_pt


# ---------------------------------------------------------------------------
# Export ONNX + Hailo
# ---------------------------------------------------------------------------

def export_onnx(best_pt: Path, config: VisionConfig, output_path: Path) -> Path:
    """Exporta a ONNX con dynamic batch."""
    model = YOLO(str(best_pt))
    onnx_path = model.export(
        format="onnx",
        imgsz=config.img_size,
        opset=15,
        simplify=True,
        dynamic=True,
    )
    src = Path(onnx_path)
    shutil.move(str(src), str(output_path))
    print(f"✅ ONNX exportado a {output_path}")
    return output_path


def compile_hailo(onnx_path: Path, output_dir: Path, config: VisionConfig) -> Path:
    """Compila ONNX a HEF (Hailo Executable Format) para Hailo-8L."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hef_path = output_dir / f"{config.model_name}_smoke.hef"

    compile_script = f"""
import numpy as np
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch="hailo8l")
runner.translate_onnx_model(
    "{onnx_path}",
    model_name="{config.model_name}_smoke",
    start_node_names=["images"],
    end_node_names=["output0"],
    net_input_shapes={{"images": [1, 3, {config.img_size}, {config.img_size}]}}
)
runner.optimize()
runner.save_har("{output_dir}/{config.model_name}_smoke.har")

hef = runner.compile()
with open("{hef_path}", "wb") as f:
    f.write(hef)
print("HEF guardado en {hef_path}")
"""
    script_path = output_dir / "compile_hailo.py"
    script_path.write_text(compile_script)

    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            raise RuntimeError(f"Compilación Hailo falló: {result.stderr}")
        print(f"✅ HEF compilado: {hef_path}")
    except FileNotFoundError:
        print("⚠️ Hailo SDK no instalado. Saltando compilación HEF.")
        print(f"   Instala hailo-dataflow-compiler y ejecuta manualmente:")
        print(f"   python {script_path}")

    return hef_path


# ---------------------------------------------------------------------------
# Inferencia Edge (ONNX Runtime - fallback si no hay Hailo)
# ---------------------------------------------------------------------------

class VisionONNXInference:
    """Inferencia YOLOv8n ONNX en CPU/GPU (fallback si no hay Hailo)."""

    def __init__(
        self,
        onnx_path: Path,
        config: VisionConfig,
        providers: Optional[List[str]] = None,
    ):
        import onnxruntime as ort
        self.config = config
        self.session = ort.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.class_names = list(config.classes)
        self._frame_buffer: List[bool] = []

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, float]]:
        h, w = frame.shape[:2]
        target = self.config.img_size

        scale = min(target / w, target / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        dx, dy = (target - nw) // 2, (target - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized

        inp = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return inp[None, :], (dx, dy, scale)

    def _postprocess(self, output: np.ndarray, pad_info: Tuple[int, int, float]) -> List[Dict]:
        dx, dy, scale = pad_info
        preds = output[0].T
        boxes = preds[:, :4]
        scores = preds[:, 4:]

        max_scores = scores.max(axis=1)
        mask = max_scores >= self.config.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = scores.argmax(axis=1)
        confs = scores.max(axis=1)

        xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (xc - w/2 - dx) / scale
        y1 = (yc - h/2 - dy) / scale
        x2 = (xc + w/2 - dx) / scale
        y2 = (yc + h/2 - dy) / scale

        x1 = np.clip(x1, 0, 1)
        y1 = np.clip(y1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        y2 = np.clip(y2, 0, 1)

        keep = self._nms_per_class(x1, y1, x2, y2, confs, class_ids)

        detections = []
        for i in keep:
            detections.append({
                "class": self.class_names[class_ids[i]],
                "class_id": int(class_ids[i]),
                "confidence": float(confs[i]),
                "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
            })
        return detections

    def _nms_per_class(self, x1, y1, x2, y2, confs, class_ids) -> np.ndarray:
        keep_all = []
        for cid in np.unique(class_ids):
            mask = class_ids == cid
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            boxes_xywh = np.stack([x1[idx], y1[idx], x2[idx]-x1[idx], y2[idx]-y1[idx]], axis=1)
            scores = confs[idx]
            nms_idx = cv2.dnn.NMSBoxes(
                boxes_xywh.tolist(), scores.tolist(),
                self.config.conf_thres, self.config.iou_thres
            )
            if len(nms_idx) > 0:
                keep_all.extend(idx[nms_idx.flatten()])
        return np.array(keep_all, dtype=int)

    def _apply_roi(self, detections: List[Dict]) -> List[Dict]:
        if not self.config.roi_norm:
            return detections
        rx1, ry1, rx2, ry2 = self.config.roi_norm
        filtered = []
        for d in detections:
            bx1, by1, bx2, by2 = d["bbox"]
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                filtered.append(d)
        return filtered

    def _temporal_filter(self, has_smoke: bool) -> bool:
        self._frame_buffer.append(has_smoke)
        if len(self._frame_buffer) > self.config.temporal_window:
            self._frame_buffer.pop(0)
        return sum(self._frame_buffer) >= self.config.temporal_threshold

    def infer(self, frame: np.ndarray) -> Dict:
        t0 = time.perf_counter()

        inp, pad_info = self._preprocess(frame)
        output = self.session.run([self.output_name], {self.input_name: inp})[0]
        detections = self._postprocess(output, pad_info)
        detections = self._apply_roi(detections)

        raw_smoke = any(d["class"] == "smoke" for d in detections)
        smoke_detected = self._temporal_filter(raw_smoke)

        latency = (time.perf_counter() - t0) * 1000

        return {
            "detections": detections,
            "smoke_detected": smoke_detected,
            "raw_smoke": raw_smoke,
            "latency_ms": latency,
        }

    def reset_temporal(self):
        self._frame_buffer.clear()


# ---------------------------------------------------------------------------
# Hard Negative Mining (automatizado)
# ---------------------------------------------------------------------------

def mine_hard_negatives(
    model_path: Path,
    video_sources: List[str],
    output_dir: Path,
    config: VisionConfig,
    max_per_source: int = 200,
) -> int:
    """
    Ejecuta modelo actual en videos de negativos conocidos (niebla, polvo, vapor, cocina)
    y guarda frames donde el modelo predice 'smoke' falsamente.
    """
    import onnxruntime as ort
    output_dir.mkdir(parents=True, exist_ok=True)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    saved = 0
    for src in video_sources:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"⚠️ No se pudo abrir: {src}")
            continue

        frame_count = 0
        while saved < max_per_source:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 30 != 0:
                continue

            h, w = frame.shape[:2]
            target = config.img_size
            scale = min(target / w, target / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (nw, nh))
            canvas = np.full((target, target, 3), 114, dtype=np.uint8)
            dx, dy = (target - nw) // 2, (target - nh) // 2
            canvas[dy:dy+nh, dx:dx+nw] = resized
            inp = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            inp = inp[None, :]

            out = session.run(None, {input_name: inp})[0]
            preds = out[0].T
            scores = preds[:, 4:].max(axis=1)
            class_ids = preds[:, 4:].argmax(axis=1)
            smoke_mask = (class_ids == 0) & (scores > config.conf_thres)

            if smoke_mask.any():
                fname = output_dir / f"hardneg_{src.replace('/', '_')}_{saved:04d}.jpg"
                cv2.imwrite(str(fname), frame)
                saved += 1
                print(f"  🎯 Hard negative #{saved}: conf={scores[smoke_mask].max():.3f}")

        cap.release()

    print(f"✅ Minados {saved} hard negatives en {output_dir}")
    return saved