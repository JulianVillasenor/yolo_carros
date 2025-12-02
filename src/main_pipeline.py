"""
main_pipeline.py
----------------

Pipeline completo:

- Detecta CARROS con tu modelo entrenado (runs_cars/.../best.pt).
- Trackea carros entre frames usando SORT.
- Detecta PLACAS con tu modelo entrenado (runs_plates/.../best.pt).
- Asocia cada placa con el carro más probable usando IoU.
- Hace OCR sobre la placa (util.read_license_plate).
- Guarda resultados en un CSV: results/test.csv
- Genera video anotado: results/<nombre_video>_out.mp4

Uso recomendado:

    python -m src.main_pipeline --video videos/trafico_corto.mp4

o desde la raíz:

    python src/main_pipeline.py --video videos/trafico_corto.mp4
"""

from __future__ import annotations

from pathlib import Path
import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from sort.sort import Sort
from util import get_car, read_license_plate, write_csv


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args():
    """
    Docstring for parse_args
    Calcula y devuelve la ruta absoluta del directorio raíz del proyecto,
    asumiendo que este script está anidado en una subcarpeta
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de detección de carros + placas + OCR + tracking."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="videos/trafico_corto.mp4",
        help="Ruta al video de entrada relativa a la raíz del proyecto.",
    )
    parser.add_argument(
        "--score_car",
        type=float,
        default=0.25,
        help="Umbral mínimo de score para detección de carros.",
    )
    parser.add_argument(
        "--score_plate",
        type=float,
        default=0.25,
        help="Umbral mínimo de score para detección de placas.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
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

    # Ruta del video de entrada
    video_path = project_root / args.video
    if not video_path.exists():
        raise FileNotFoundError(
            f"No se encontró el video de entrada en: {video_path}\n"
            f"Pon tu video ahí o ajusta la ruta con --video."
        )

    print(f"🎞 Video de entrada: {video_path}")

    # Cargar modelos YOLO
    car_model = YOLO(str(car_weights))
    plate_model = YOLO(str(plate_weights))

    # Inicializar tracker SORT
    mot_tracker = Sort()

    # Diccionario donde guardaremos todos los resultados
    results: dict[int, dict[int, dict]] = {}

    # Abrir video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Preparar carpeta de resultados y writer de video
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    out_video_path = results_dir / f"{video_path.stem}_out.mp4"

    # Codec H.264 o MP4V, según tu sistema; aquí probamos con mp4v
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    print(f"📂 Video anotado se guardará en: {out_video_path}")
    print(f"🎬 FPS: {fps:.2f} | Tamaño: {width}x{height} | Frames: {total_frames}")

    frame_nmr = -1
    ret = True

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Inicializar entrada para este frame
        results[frame_nmr] = {}

        # ==========================
        # 1) DETECCIÓN DE CARROS
        # ==========================
        car_preds = car_model(frame)[0]  # predicción sobre el frame

        detections_for_sort = []
        for det in car_preds.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            if score < args.score_car:
                continue
            detections_for_sort.append([x1, y1, x2, y2, score])

        if len(detections_for_sort) > 0:
            dets_np = np.asarray(detections_for_sort)
        else:
            dets_np = np.empty((0, 5))

        # ==========================
        # 2) TRACKING DE CARROS (SORT)
        # ==========================
        track_ids = mot_tracker.update(dets_np)  # Nx5: [x1, y1, x2, y2, track_id]

        # Dibujar carros (sin placas todavía)
        for track in track_ids:
            x1, y1, x2, y2, car_id = track
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {int(car_id)}",
                (p1[0], p1[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # ==========================
        # 3) DETECCIÓN DE PLACAS
        # ==========================
        plate_preds = plate_model(frame)[0]

        for plate in plate_preds.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = plate
            if score < args.score_plate:
                continue

            # Asignar placa al carro más cercano
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(plate, track_ids)

            if car_id == -1:
                # No se pudo asociar la placa a ningún carro
                continue

            # Recortar placa del frame
            h, w = frame.shape[:2]
            x1_i = max(0, min(w - 1, int(x1)))
            x2_i = max(0, min(w - 1, int(x2)))
            y1_i = max(0, min(h - 1, int(y1)))
            y2_i = max(0, min(h - 1, int(y2)))

            if x2_i <= x1_i or y2_i <= y1_i:
                continue

            plate_crop = frame[y1_i:y2_i, x1_i:x2_i, :]

            # Preprocesado básico: gris + threshold
            plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            _, plate_thresh = cv2.threshold(
                plate_gray, 64, 255, cv2.THRESH_BINARY_INV
            )

            # OCR
            lp_text, lp_text_score = read_license_plate(plate_thresh)

            if lp_text is None:
                # Si no se pudo leer texto, igual podemos guardar la bbox si quisieras
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

            # Dibujar placa y texto en el frame
            p1 = (x1_i, y1_i)
            p2 = (x2_i, y2_i)
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{lp_text}",
                (p1[0], p2[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Escribir frame anotado al video de salida
        out_writer.write(frame)

        if frame_nmr % 10 == 0:
            print(f"🧠 Procesado frame {frame_nmr}/{total_frames}")

    cap.release()
    out_writer.release()

    # =================
    # 4) GUARDAR CSV
    # =================
    csv_path = results_dir / "test.csv"
    write_csv(results, csv_path)

    print("\n✅ Pipeline completado.")
    print(f"   📄 CSV guardado en:   {csv_path}")
    print(f"   🎥 Video anotado en: {out_video_path}")


if __name__ == "__main__":
    main()
