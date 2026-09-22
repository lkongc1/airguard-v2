#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento YOLOv8n optimizado para 7GB VRAM.

Optimizaciones para GPU pequeña:
- batch_size reducido (8-16)
- gradient accumulation
- mixed precision (FP16)
- image size 512 en lugar de 640 si necesario
- workers reducidos
"""

from ultralytics import YOLO
import torch
import argparse
from pathlib import Path


def train_yolo_optimized(
    data_yaml: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 512,
    device: str = "0",
    workers: int = 4,
    project: str = "runs/detect",
    name: str = "yolov8n_smoke_7gb",
):
    """
    Entrena YOLOv8n con optimizaciones para 7GB VRAM.
    """
    # Verificar VRAM disponible
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU VRAM: {vram_gb:.1f} GB")
        if vram_gb < 8:
            print("⚠️ VRAM < 8GB - Aplicando optimizaciones agresivas")
            if batch_size > 16:
                batch_size = 16
            if img_size > 512:
                img_size = 512
    
    # Configurar mixed precision
    torch.backends.cudnn.benchmark = True
    
    # Cargar modelo
    model = YOLO("yolov8n.pt")
    
    # Entrenar con optimizaciones
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=workers,
        project=project,
        name=name,
        exist_ok=True,
        # Optimizaciones memoria
        amp=True,              # Mixed precision (FP16)
        cache=False,           # No cachear imágenes en RAM/VRAM
        rect=False,            # No rectangular training (usa más memoria)
        # Hiperparámetros conservadores
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Augmentación conservadora para humo
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        # Early stopping
        patience=30,
        save_period=10,
        # Logging
        verbose=True,
        plots=True,
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n for smoke detection (7GB VRAM optimized)")
    parser.add_argument("--data", type=str, default="data/yolo_unified/data.yaml", help="data.yaml path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="yolov8n_smoke_7gb")
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 ENTRENANDO YOLOv8n - DETECCIÓN DE HUMO (7GB VRAM)")
    print("="*60)
    
    results = train_yolo_optimized(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )
    
    # Exportar a ONNX
    print("\n📦 Exportando a ONNX...")
    model = YOLO(f"{args.project}/{args.name}/weights/best.pt")
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, opset=15, simplify=True, dynamic=True)
    print(f"✅ ONNX exportado: {onnx_path}")
    
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"📁 Modelo: {args.project}/{args.name}/weights/best.pt")
    print(f"📁 ONNX: {onnx_path}")


if __name__ == "__main__":
    main()