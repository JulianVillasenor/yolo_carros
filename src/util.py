"""
util.py
-------

Funciones auxiliares para el pipeline de detección:

- get_car: asignar una placa detectada al carro más probable (usando IoU).
- read_license_plate: leer el texto de la placa con OCR (EasyOCR si está instalado).
- write_csv: guardar los resultados en un CSV compatible con el script
  de interpolación que ya tienes.

Formato del CSV de salida (columnas):

    frame_nmr
    car_id
    car_bbox
    license_plate_bbox
    license_plate_bbox_score
    license_number
    license_number_score
"""

from __future__ import annotations

from pathlib import Path
import csv
import numpy as np

# OCR (opcional, pero recomendado)
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    _EASYOCR_AVAILABLE = False
    print(
        "[util.py] ⚠ EasyOCR no está instalado. "
        "Instálalo con: pip install easyocr\n"
        "           read_license_plate() devolverá (None, 0)."
    )


def _iou(boxA, boxB) -> float:
    """
    Calcula IoU entre dos cajas [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    boxA_area = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    if boxA_area <= 0 or boxB_area <= 0:
        return 0.0

    return inter_area / float(boxA_area + boxB_area - inter_area)


def get_car(license_plate_det, track_ids, iou_threshold: float = 0.1):
    """
    Asigna una detección de placa a un carro usando IoU con las cajas trackeadas.

    Params
    ------
    license_plate_det : list/tuple
        Detección de placa: [x1, y1, x2, y2, score, class_id]
    track_ids : np.ndarray
        Salida de Sort: Nx5 con filas [x1, y1, x2, y2, track_id]
    iou_threshold : float
        IoU mínimo para aceptar la asignación.

    Returns
    -------
    (xcar1, ycar1, xcar2, ycar2, car_id)
        Si no se encuentra carro, car_id = -1 y las coords son 0.
    """
    if track_ids is None or len(track_ids) == 0:
        return 0, 0, 0, 0, -1

    x1, y1, x2, y2, score, class_id = license_plate_det
    lp_box = [float(x1), float(y1), float(x2), float(y2)]

    best_iou = 0.0
    best_car = None

    for track in track_ids:
        # track: [x1, y1, x2, y2, track_id]
        cx1, cy1, cx2, cy2, car_id = track
        car_box = [float(cx1), float(cy1), float(cx2), float(cy2)]
        iou = _iou(lp_box, car_box)
        if iou > best_iou:
            best_iou = iou
            best_car = track

    if best_car is None or best_iou < iou_threshold:
        return 0, 0, 0, 0, -1

    xcar1, ycar1, xcar2, ycar2, car_id = best_car
    return float(xcar1), float(ycar1), float(xcar2), float(ycar2), int(car_id)


def read_license_plate(plate_img):
    """
    Lee el texto de la placa usando EasyOCR, si está disponible.

    Params
    ------
    plate_img : np.ndarray
        Imagen de la placa (idealmente ya preprocesada: gris + threshold).

    Returns
    -------
    (text, score)
        text : str o None
        score: float (0 si no se pudo leer)
    """
    if not _EASYOCR_AVAILABLE:
        return None, 0.0

    # Lazy init del reader para no cargarlo múltiples veces
    reader = getattr(read_license_plate, "_reader", None)
    if reader is None:
        # Idiomas típicos: solo inglés (mayúsculas/números) está bien
        reader = easyocr.Reader(["en"], gpu=False)
        read_license_plate._reader = reader

    # EasyOCR espera BGR o gris; si es binaria igual se la damos
    result = reader.readtext(plate_img, detail=1)

    if not result:
        return None, 0.0

    # Tomar el resultado con mayor confiabilidad
    best = max(result, key=lambda x: x[2])  # (bbox, text, conf)
    text = best[1]
    conf = float(best[2])

    # Normalizar texto: mayúsculas, sin espacios
    if text is not None:
        text = text.strip().upper().replace(" ", "")

    # Pequeño filtro: si el texto queda muy corto o nada, lo descartamos
    if not text or len(text) < 3:
        return None, 0.0

    return text, conf


def write_csv(results: dict, output_path: str | Path):
    """
    Guarda el diccionario `results` en un CSV.

    Estructura esperada de `results`:

    results[frame_nmr][car_id] = {
        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
        'license_plate': {
            'bbox': [x1, y1, x2, y2],
            'text': license_plate_text,
            'bbox_score': score,
            'text_score': license_plate_text_score
        }
    }

    El CSV se escribe con columnas compatibles con tu script de interpolación.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "frame_nmr",
        "car_id",
        "car_bbox",
        "license_plate_bbox",
        "license_plate_bbox_score",
        "license_number",
        "license_number_score",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for frame_nmr in sorted(results.keys()):
            cars_dict = results[frame_nmr]
            for car_id in sorted(cars_dict.keys()):
                data = cars_dict[car_id]
                car_bbox = data["car"]["bbox"]
                lp_data = data["license_plate"]

                lp_bbox = lp_data.get("bbox", [0, 0, 0, 0])
                lp_text = lp_data.get("text", "")
                lp_bbox_score = lp_data.get("bbox_score", 0.0)
                lp_text_score = lp_data.get("text_score", 0.0)

                # Formato tipo "[x1 x2 y1 y2]" para ser compatible con tu process_interpolation.py
                car_bbox_str = "[" + " ".join(f"{v:.2f}" for v in car_bbox) + "]"
                lp_bbox_str = "[" + " ".join(f"{v:.2f}" for v in lp_bbox) + "]"

                writer.writerow(
                    [
                        int(frame_nmr),
                        int(car_id),
                        car_bbox_str,
                        lp_bbox_str,
                        float(lp_bbox_score),
                        lp_text,
                        float(lp_text_score),
                    ]
                )

    print(f"[util.py] ✅ CSV guardado en: {output_path}")
