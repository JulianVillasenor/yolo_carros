"""
train_yolo_carros.py
--------------------

Entrena un modelo YOLO (ultralytics) para detección de CARROS
usando el dataset en formato YOLO generado por:

    src/convert_car_csv_to_yolo.py

Estructura esperada del dataset:

    datasets/car_detection_yolo/
        images/train
        images/val
        labels/train
        labels/val

Y el archivo de configuración:

    data_cars.yaml
"""

from pathlib import Path
import argparse
from ultralytics import YOLO


def get_project_root() -> Path:
    # Raíz del proyecto (asumiendo que este archivo está en src/)
    return Path(__file__).resolve().parent.parent


def parse_args():
    """
    Docstring for parse_args
    
    Configura y procesa los argumentos de línea de comandos para el entrenamiento.

    Permite al usuario sobrescribir parámetros clave como la ruta del dataset,
    el modelo base, las épocas, el tamaño del batch y el tamaño de la imagen.

    Returns:
        argparse.Namespace: Objeto con los argumentos procesados.
    """
    parser = argparse.ArgumentParser(
        description="Entrenar modelo YOLO para detección de CARROS."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Ruta al archivo .yaml del dataset (por defecto: data_cars.yaml en la raíz del proyecto).",
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
    runs_dir = project_root / "runs_cars"

    args = parse_args()

    # Elegir YAML de dataset
    if args.data is not None:
        data_yaml = Path(args.data)
    else:
        data_yaml = project_root / "data_cars.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"No se encontró el archivo de dataset: {data_yaml}")

    print(f"📁 Proyecto:       {project_root}")
    print(f"📄 Dataset YAML:   {data_yaml}")
    print(f"📂 Runs dir:       {runs_dir}")
    print(f"🔧 Modelo base:    {args.model}")
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
        name="yolov8n-cars",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
