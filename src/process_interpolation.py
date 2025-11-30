import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


def parse_bbox_str(s: str):
    """
    Convierte un string de bbox a lista de floats.
    Soporta formatos tipo:
      "[x1 x2 x3 x4]"  o  "x1 x2 x3 x4"
    """
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = s.replace(",", " ").split()
    return [float(p) for p in parts]


def interpolate_track(track_rows):
    """
    Recibe las filas de un solo car_id ya filtradas.
    Devuelve una lista de filas (dict) con frames interpolados.
    """

    # Ordenar por frame_nmr por si acaso
    track_rows = sorted(track_rows, key=lambda r: int(r["frame_nmr"]))

    frame_numbers = np.array([int(r["frame_nmr"]) for r in track_rows])
    car_bboxes = np.array([parse_bbox_str(r["car_bbox"]) for r in track_rows])
    lp_bboxes = np.array([parse_bbox_str(r["license_plate_bbox"]) for r in track_rows])

    interpolated = []

    first_frame = frame_numbers[0]
    last_frame = frame_numbers[-1]

    # Vamos frame por frame dentro del rango [first_frame, last_frame]
    all_frames = np.arange(first_frame, last_frame + 1)

    # Interpoladores por componente
    # shape car_bboxes: (N, 4)
    x = frame_numbers.astype(float)

    car_interp = interp1d(x, car_bboxes, axis=0, kind="linear")
    lp_interp = interp1d(x, lp_bboxes, axis=0, kind="linear")

    # Set para saber cuáles frames eran originales
    original_frames_set = set(frame_numbers.tolist())

    for f in all_frames:
        row_out = {}
        row_out["frame_nmr"] = str(f)
        row_out["car_id"] = track_rows[0]["car_id"]

        # Bboxes interpoladas (incluye originales, es continuo)
        bbox_car = car_interp(float(f)).tolist()
        bbox_lp = lp_interp(float(f)).tolist()

        row_out["car_bbox"] = " ".join(map(str, bbox_car))
        row_out["license_plate_bbox"] = " ".join(map(str, bbox_lp))

        if f in original_frames_set:
            # Buscar la fila original para recuperar campos extra
            orig = next(r for r in track_rows if int(r["frame_nmr"]) == f)
            row_out["license_plate_bbox_score"] = orig.get("license_plate_bbox_score", "0")
            row_out["license_number"] = orig.get("license_number", "0")
            row_out["license_number_score"] = orig.get("license_number_score", "0")
        else:
            # Fila imputada
            row_out["license_plate_bbox_score"] = "0"
            row_out["license_number"] = "0"
            row_out["license_number_score"] = "0"

        interpolated.append(row_out)

    return interpolated


def interpolate_bounding_boxes(data):
    # Agrupar filas por car_id
    by_car = defaultdict(list)
    for row in data:
        by_car[row["car_id"]].append(row)

    final_rows = []
    for car_id, rows in by_car.items():
        track_interp = interpolate_track(rows)
        final_rows.extend(track_interp)

    # Opcional: ordenar todo por frame y luego por car_id
    final_rows = sorted(final_rows, key=lambda r: (int(r["frame_nmr"]), float(r["car_id"])))
    return final_rows


def main():
    input_csv = Path("test.csv")
    output_csv = Path("test_interpolated.csv")

    with input_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    interpolated_data = interpolate_bounding_boxes(data)

    header = [
        "frame_nmr",
        "car_id",
        "car_bbox",
        "license_plate_bbox",
        "license_plate_bbox_score",
        "license_number",
        "license_number_score",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(interpolated_data)

    print(f"✅ Archivo interpolado guardado en: {output_csv}")


if __name__ == "__main__":
    main()
