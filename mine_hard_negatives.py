#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mina hard negatives usando el modelo YOLO entrenado.
Busca falsos positivos en imágenes de niebla, polvo, vapor, nubes bajas.
"""

from pathlib import Path
import glob
import cv2
import onnxruntime as ort
import numpy as np
from dataclasses import dataclass


@dataclass
class VisionConfig:
    model_name: str = "yolov8n"
    img_size: int = 640
    conf_thres: float = 0.35
    iou_thres: float = 0.45
    classes: tuple = ("smoke", "fire")


def main():
    config = VisionConfig()
    model_path = Path(r"runs/detect/runs/detect/yolov8n_smoke_7gb/weights/best.onnx")
    
    video_sources = [
        Path(r"data/raw/Smoke24776/Smoke24776_images_by_types/Aerial"),
        Path(r"data/raw/Smoke24776/Smoke24776_images_by_types/Complex"),
        Path(r"data/raw/Smoke24776/Smoke24776_images_by_types/Dense"),
        Path(r"data/raw/Smoke24776/Smoke24776_images_by_types/Indoor"),
        Path(r"data/raw/Smoke24776/Smoke24776_images_by_types/Outdoor"),
        Path(r"data/raw/Smoke24776/Smoke24776_images_by_types/Simulated"),
    ]
    
    all_images = []
    for src in video_sources:
        all_images.extend(glob.glob(str(src / "*.jpg")))
        all_images.extend(glob.glob(str(src / "*.png")))
        all_images.extend(glob.glob(str(src / "*.jpeg")))
    
    print(f"Total images to check: {len(all_images)}")
    
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    output_dir = Path("data/hard_negatives/mined")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    for img_path in all_images:
        if saved >= 500:
            break
        frame = cv2.imread(img_path)
        if frame is None:
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
        
        out = session.run([output_name], {input_name: inp})[0]
        preds = out[0].T
        scores = preds[:, 4:].max(axis=1)
        class_ids = preds[:, 4:].argmax(axis=1)
        smoke_mask = (class_ids == 0) & (scores > config.conf_thres)
        
        if smoke_mask.any():
            fname = output_dir / f"hardneg_{Path(img_path).stem}_{saved:04d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved += 1
            if saved % 50 == 0:
                print(f"  Hard negative #{saved}: {Path(img_path).name}")
    
    print(f"Total hard negatives mined: {saved}")


if __name__ == "__main__":
    import glob
    import cv2
    import onnxruntime as ort
    import numpy as np
    from pathlib import Path
    main()