"""
convert_car_csv_to_yolo.py
--------------------------

Convierte el dataset de Kaggle:
    sshikamaru/car-object-detection

desde:
    datasets/car_detection_kaggle/
        training_images/
        train_solution_bounding_boxes (1).csv

a formato YOLOv8:

    datasets/car_detection_yolo/
        images/train
        images/val
        labels/train
        labels/val

y crea un archivo de config:

    data_cars.yaml
"""

from pathlib import Path
import csv
import cv2
import random
import shutil


def yolo_from_xyxy(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convierte bbox en formato VOC (xmin, ymin, xmax, ymax) a YOLO normalizado."""
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x_center, y_center, w, h


def main():
    project_root = Path(__file__).resolve().parent.parent
    kaggle_dir = project_root / "datasets" / "car_detection_kaggle"

    # Buscar el CSV que empieza con "train_solution_bounding_boxes"
    csv_candidates = list(kaggle_dir.glob("train_solution_bounding_boxes*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"No se encontró 'train_solution_bounding_boxes*.csv' en {kaggle_dir}"
        )
    csv_path = csv_candidates[0]

    train_images_dir = kaggle_dir / "training_images"
    if not train_images_dir.exists():
        raise FileNotFoundError(f"No se encontró carpeta training_images en {train_images_dir}")

    print("📄 CSV de anotaciones:", csv_path)
    print("📁 Carpeta de imágenes:", train_images_dir)

    # Directorio destino en formato YOLO
    yolo_root = project_root / "datasets" / "car_detection_yolo"
    if yolo_root.exists():
        shutil.rmtree(yolo_root)
    (yolo_root / "images" / "train").mkdir(parents=True)
    (yolo_root / "images" / "val").mkdir(parents=True)
    (yolo_root / "labels" / "train").mkdir(parents=True)
    (yolo_root / "labels" / "val").mkdir(parents=True)

    # Leer el CSV y agrupar por imagen
    annotations = {}  # image_name -> list of bboxes (xmin, ymin, xmax, ymax)
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["image"]
            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])
            annotations.setdefault(image_name, []).append((xmin, ymin, xmax, ymax))

    print(f"🔢 Imágenes anotadas en CSV: {len(annotations)}")

    # Lista de imágenes con anotaciones
    image_names = list(annotations.keys())
    random.seed(42)
    random.shuffle(image_names)

    # Split 80/20
    n_total = len(image_names)
    n_train = int(0.8 * n_total)
    train_imgs = set(image_names[:n_train])
    val_imgs = set(image_names[n_train:])

    print(f"📊 Split -> train: {len(train_imgs)} | val: {len(val_imgs)}")

    def process_split(split_names, split: str):
        img_out_dir = yolo_root / "images" / split
        label_out_dir = yolo_root / "labels" / split

        for img_name in split_names:
            src_img_path = train_images_dir / img_name
            if not src_img_path.exists():
                print(f"⚠ Imagen {img_name} no encontrada en {train_images_dir}, se omite.")
                continue

            # Leer imagen para obtener tamaño
            img = cv2.imread(str(src_img_path))
            if img is None:
                print(f"⚠ No se pudo leer la imagen {src_img_path}, se omite.")
                continue
            img_h, img_w = img.shape[:2]

            # Copiar imagen al destino
            dst_img_path = img_out_dir / img_name
            shutil.copy(src_img_path, dst_img_path)

            # Crear archivo de etiquetas YOLO
            label_name = Path(img_name).with_suffix(".txt").name
            dst_label_path = label_out_dir / label_name

            with dst_label_path.open("w") as lf:
                for (xmin, ymin, xmax, ymax) in annotations[img_name]:
                    x_c, y_c, w, h = yolo_from_xyxy(xmin, ymin, xmax, ymax, img_w, img_h)
                    # Clase única: 0 = car
                    lf.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    print("🚧 Generando split de entrenamiento...")
    process_split(train_imgs, "train")
    print("🚧 Generando split de validación...")
    process_split(val_imgs, "val")

    # Crear archivo data_cars.yaml en la raíz del proyecto
    data_yaml = project_root / "data_cars.yaml"
    data_yaml.write_text(
        "train: " + str(yolo_root / "images" / "train") + "\n"
        "val: " + str(yolo_root / "images" / "val") + "\n\n"
        "names:\n"
        "  0: car\n"
    )

    print("\n✅ Conversión COMPLETADA.")
    print("📂 Dataset YOLO de carros creado en:", yolo_root)
    print("📄 Config de dataset creada en:", data_yaml)
    print("\nAhora puedes entrenar con, por ejemplo:")
    print("  python src/train_yolo_carros.py  (cuando lo tengamos)  o")
    print("  yolo detect/train data=data_cars.yaml model=yolov8n.pt imgsz=640 epochs=50")


if __name__ == "__main__":
    main()
