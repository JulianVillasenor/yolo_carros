"""
train_yolo_placas.py
--------------------

Entrena un modelo YOLO (ultralytics) para detección de placas
usando el dataset convertido a formato YOLO:

    datasets/license_plates_yolo
y el archivo de configuración:

    data_license_plates.yaml
"""

from pathlib import Path
from ultralytics import YOLO


def main():
    # Raíz del proyecto (asumiendo que este archivo está en src/)
    project_root = Path(__file__).resolve().parent.parent

    # Archivo de configuración del dataset
    data_yaml = project_root / "data_license_plates.yaml"

    # Modelo base (puede ser yolov8n.pt, yolov8s.pt, etc.)
    model = YOLO("yolov8n.pt")

    # Carpeta donde se guardarán los runs de este proyecto de placas
    runs_dir = project_root / "runs_plates"

    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(runs_dir),
        name="yolov8n-plates",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
