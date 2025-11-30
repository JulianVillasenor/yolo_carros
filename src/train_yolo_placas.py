"""
train_yolo_placas.py
--------------------

Entrena un modelo YOLO (ultralytics) para detección de placas
usando el dataset convertido a formato YOLO.

Estructura esperada del dataset:

    datasets/license_plates_yolo/
        images/train
        images/val
        labels/train
        labels/val

Y uno de estos archivos de configuración, según la máquina:

    data_license_plates.yaml         (Windows / laptop)
    data_license_plates_choya.yaml   (Linux / choya)
"""

from pathlib import Path
import platform
import argparse
from ultralytics import YOLO


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_default_data_yaml(project_root: Path) -> Path:
    """
    Devuelve el YAML de dataset por defecto según el sistema operativo.
    - En Windows:  data_license_plates.yaml
    - En Linux:    data_license_plates_choya.yaml (si existe), si no, el genérico.
    """
    system = platform.system()

    win_yaml = project_root / "data_license_plates.yaml"
    linux_yaml = project_root / "data_license_plates_choya.yaml"

    if system == "Windows":
        return win_yaml

    # En Linux (choya), si existe el YAML específico, úsalo
    if linux_yaml.exists():
        return linux_yaml

    # Fallback
    return win_yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenar modelo YOLO para detección de placas."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Ruta al archivo .yaml del dataset (por defecto se elige según el sistema).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Modelo base de Ultralytics (yolov8n.pt, yolov8s.pt, etc.).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número de épocas de entrenamiento.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Tamaño de batch.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamaño de imagen (imgsz).",
    )
    return parser.parse_args()


def main():
    project_root = get_project_root()
    runs_dir = project_root / "runs_plates"

    args = parse_args()

    # Elegir YAML
    if args.data is not None:
        data_yaml = Path(args.data)
    else:
        data_yaml = get_default_data_yaml(project_root)

    if not data_yaml.exists():
        raise FileNotFoundError(f"No se encontró el archivo de dataset: {data_yaml}")

    print(f"📁 Proyecto: {project_root}")
    print(f"📄 Dataset YAML: {data_yaml}")
    print(f"📂 Runs dir: {runs_dir}")
    print(f"🔧 Modelo base: {args.model}")
    print(f"🔁 Épocas: {args.epochs} | Batch: {args.batch} | imgsz: {args.imgsz}")

    # Cargar modelo base
    model = YOLO(args.model)

    # Entrenamiento
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(runs_dir),
        name="yolov8n-plates",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
