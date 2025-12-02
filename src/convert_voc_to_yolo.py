"""
convert_voc_to_yolo.py (FIXED VERSION)
--------------------------------------

Convierte VOC XML del dataset:

    datasets/car_plate_detection_voc/annotations
    datasets/car_plate_detection_voc/images

a formato YOLO:

    datasets/license_plates_yolo/
"""

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Proporción de datos destinados al conjunto de entrenamiento.
TRAIN_RATIO = 0.8

# Mapeo de nombres de clase a ID (todas son clase 0 = "plate")
CLASS_NAME_TO_ID = {
    "license-plate": 0,
    "licence-plate": 0,
    "plate": 0,
    "car-plate": 0,
    "car_plate": 0,
    "licence": 0,   # <-- el que aparece en tu XML
    "license": 0,
}


def get_project_root() -> Path:
    """
    Docstring for get_project_root
    
    :return: Description
    :rtype: Path
    Calcula y devuelve la ruta absoluta del directorio raíz del proyecto.
    """
    return Path(__file__).resolve().parent.parent


def parse_voc_xml(xml_path: Path):
    """
    Docstring for parse_voc_xml
    
    :param xml_path: Description
    :type xml_path: Path

    Parsea un archivo de anotación XML en formato Pascal VOC y extrae las
    bounding boxes, convirtiéndolas al formato normalizado de YOLO.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    #Obtener dimensiones de la imagen
    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)

    #Iterar sobre los objetos detecciones
    boxes = []
    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is None or not name_tag.text:
            continue

        class_name = name_tag.text.strip()

        # Mapear nombre de clase a ID
        if class_name in CLASS_NAME_TO_ID:
            class_id = CLASS_NAME_TO_ID[class_name]
        else:
            # Cualquier clase rara la tratamos como "plate" (clase 0)
            print(f"⚠ Clase desconocida '{class_name}' en {xml_path.name}, se asigna como 'plate' (0).")
            class_id = 0

        b = obj.find("bndbox")
        if b is None:
            continue

        # Coordenadas de la Bounding Box en formato pixel
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)

        # Convertir a YOLO normalizado
        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        boxes.append((class_id, x_center, y_center, bw, bh))

    return boxes


def main():
    root = get_project_root()
    voc_root = root / "datasets" / "car_plate_detection_voc"
    ann_dir = voc_root / "annotations"
    img_dir = voc_root / "images"

    yolo_root = root / "datasets" / "license_plates_yolo"
    (yolo_root / "images/train").mkdir(parents=True, exist_ok=True)
    (yolo_root / "images/val").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels/train").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels/val").mkdir(parents=True, exist_ok=True)

    xml_files = sorted(ann_dir.glob("*.xml"))
    print(f"🔎 XML encontrados: {len(xml_files)}")

    samples = []

    for xml_file in xml_files:
        img_file = img_dir / (xml_file.stem + ".png")
        if not img_file.exists():
            img_file = img_dir / (xml_file.stem + ".jpg")
        if not img_file.exists():
            print(f"⚠ No se encontró imagen para {xml_file.stem}")
            continue

        boxes = parse_voc_xml(xml_file)
        if not boxes:
            print(f"⚠ Sin boxes válidos: {xml_file.name}")
            continue

        samples.append((img_file, boxes))

    print(f"✅ Muestras válidas: {len(samples)}")

    # Shuffle + split
    random.shuffle(samples)
    split = int(len(samples) * TRAIN_RATIO)
    train = samples[:split]
    val = samples[split:]

    print(f"📊 Train: {len(train)}  |  Val: {len(val)}")

    # Save images + labels
    def save_split(samples, split_name):
        img_out = yolo_root / f"images/{split_name}"
        lbl_out = yolo_root / f"labels/{split_name}"

        for img_path, boxes in samples:
            shutil.copy2(img_path, img_out / img_path.name)

            label_path = lbl_out / (img_path.stem + ".txt")
            lines = [
                f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                for cid, x, y, w, h in boxes
            ]
            label_path.write_text("\n".join(lines), encoding="utf-8")

    save_split(train, "train")
    save_split(val, "val")

    print("\n🎉 Conversión COMPLETADA correctamente.")
    print(" Verifica ahora:")
    print("  datasets/license_plates_yolo/images/train")
    print("  datasets/license_plates_yolo/labels/train")


if __name__ == "__main__":
    main()
