"""
main_pipeline.py
----------------

Pipeline principal:

- Detecta CARROS con tu modelo entrenado (yolov8n-cars best.pt).
- Trackea carros entre frames usando SORT.
- Detecta PLACAS con tu modelo entrenado (yolov8n-plates best.pt).
- Asocia cada placa con el carro más probable (IoU).
- Hace OCR sobre la placa.
- Guarda resultados en un CSV: results/test.csv

Ajusta la ruta del video en VIDEO_PATH.
"""

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from sort.sort import Sort  # Asegúrate de tener src/sort/sort.py
from util import get_car, read_license_plate, write_csv


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main():
    project_root = get_project_root()

    # Rutas de modelos entrenados
    car_weights = project_root / "runs_cars" / "yolov8n-cars" / "weights" / "best.pt"
    plate_weights = project_root / "runs_plates" / "yolov8n-plates" / "weights" / "best.pt"

    if not car_weights.exists():
        raise FileNotFoundError(f"No se encontró el modelo de CARROS en: {car_weights}")

    if not plate_weights.exists():
        raise FileNotFoundError(f"No se encontró el modelo de PLACAS en: {plate_weights}")

    print(f"🚗 Modelo carros:   {car_weights}")
    print(f"📛 Modelo placas:   {plate_weights}")

    # Ruta del video de entrada (ajusta a lo que vas a usar)
    # Por ejemplo puedes crear B:\unison\redes_neuronales\yolo_carros\videos\sample.mp4
    VIDEO_PATH = project_root / "sample.mp4"
    # VIDEO_PATH = project_root / "videos" / "sample.mp4"

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el video de entrada en: {VIDEO_PATH}\n"
            f"Pon tu video ahí o ajusta la ruta en main_pipeline.py"
        )

    print(f"🎞 Video de entrada: {VIDEO_PATH}")

    # Cargar modelos
    car_model = YOLO(str(car_weights))
    plate_model = YOLO(str(plate_weights))

    # Inicializar tracker SORT
    mot_tracker = Sort()

    # Diccionario donde guardaremos todos los resultados
    results = {}

    # Abrir video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {VIDEO_PATH}")

    frame_nmr = -1
    ret = True

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Inicializar entrada para este frame
        results[frame_nmr] = {}

        # ==============
        # 1) DETECCIÓN DE CARROS
        # ==============
        car_dets = car_model(frame)[0]  # predicción sobre el frame

        detections_for_sort = []
        for det in car_dets.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            # Tu modelo de carros solo tiene 1 clase, no filtramos por class_id.
            # Podrías filtrar por 'score > umbral' si quieres:
            if score < 0.25:
                continue
            detections_for_sort.append([x1, y1, x2, y2, score])

        if len(detections_for_sort) > 0:
            dets_np = np.asarray(detections_for_sort)
        else:
            dets_np = np.empty((0, 5))

        # ==============
        # 2) TRACKING DE CARROS (SORT)
        # ==============
        track_ids = mot_tracker.update(dets_np)  # Nx5: [x1, y1, x2, y2, track_id]

        # ==============
        # 3) DETECCIÓN DE PLACAS
        # ==============
        plate_dets = plate_model(frame)[0]

        for plate in plate_dets.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = plate

            # Asignar placa al carro más cercano
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(plate, track_ids)

            if car_id == -1:
                # No se pudo asociar la placa a ningún carro
                continue

            # Recortar placa del frame (cuidando los límites)
            h, w = frame.shape[:2]
            x1_i = max(0, min(w - 1, int(x1)))
            x2_i = max(0, min(w - 1, int(x2)))
            y1_i = max(0, min(h - 1, int(y1)))
            y2_i = max(0, min(h - 1, int(y2)))

            if x2_i <= x1_i or y2_i <= y1_i:
                continue

            plate_crop = frame[y1_i:y2_i, x1_i:x2_i, :]

            # Preprocesado básico como en el ejemplo
            plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            _, plate_thresh = cv2.threshold(
                plate_gray, 64, 255, cv2.THRESH_BINARY_INV
            )

            # OCR
            lp_text, lp_text_score = read_license_plate(plate_thresh)

            if lp_text is None:
                continue

            # Guardar en results
            results[frame_nmr][car_id] = {
                "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                "license_plate": {
                    "bbox": [x1, y1, x2, y2],
                    "text": lp_text,
                    "bbox_score": float(score),
                    "text_score": float(lp_text_score),
                },
            }

        if frame_nmr % 50 == 0:
            print(f"🧠 Procesado frame {frame_nmr}")

    cap.release()

    # =================
    # 4) GUARDAR CSV
    # =================
    results_dir = project_root / "results"
    csv_path = results_dir / "test.csv"

    write_csv(results, csv_path)

    print("\n✅ Pipeline completado.")
    print(f"   Resultados guardados en: {csv_path}")


if __name__ == "__main__":
    main()
