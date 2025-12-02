# server.py
"""
Servidor HTTP que:
 - Sirve la carpeta web/ (index.html + assets).
 - Expone POST /procesar para recibir frames desde el navegador.
 - Carga dos modelos YOLO (carros y placas).
 - Realiza OCR con EasyOCR si está disponible.
 - Mantiene tracking simple por centroides (con historial prev_centroid + prev_time)
   para calcular velocidad por vehículo.
 - Guarda infracciones en infracciones.csv
"""

import os
import io
import json
import time
import csv
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote, urlparse
import threading
import traceback

import numpy as np
import cv2

# Intentar cargar ultralytics + easyocr
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Instala 'ultralytics' (pip install ultralytics). Error: " + str(e))

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

BASE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE, "web")

# Rutas a tus modelos (ajustadas a tu estructura)
RUTA_MODELO_CARROS = os.path.join(BASE, "runs_cars", "yolov8n-cars", "weights", "best.pt")
RUTA_MODELO_PLACAS = os.path.join(BASE, "runs_plates", "yolov8n-plates", "weights", "best.pt")

# Parámetros (ajusta PIXELS_TO_METERS con calibración real)
PIXELS_TO_METERS = 0.04   # ejemplo: 1 px = 0.04 m  (¡calibra en campo!)
VELOCITY_LIMIT = 60.0     # km/h
FRAME_INTERVAL = 0.3      # s — intervalo aproximado entre requests del cliente

# Tracking simple por centroides
# Estructura: tracks[track_id] = {"centroid": (x,y), "last_time": t, "prev_centroid": (x,y) or None, "prev_time": t or None}
tracks = {}
_next_track_id = 1
tracks_lock = threading.Lock()

def next_track_id():
    global _next_track_id
    with tracks_lock:
        tid = _next_track_id
        _next_track_id += 1
    return tid

# Cargar modelos (lento)
print("Cargando modelos YOLO. Esto puede tardar...")
if not os.path.exists(RUTA_MODELO_CARROS):
    raise FileNotFoundError(f"No se encontró modelo carros: {RUTA_MODELO_CARROS}")
if not os.path.exists(RUTA_MODELO_PLACAS):
    raise FileNotFoundError(f"No se encontró modelo placas: {RUTA_MODELO_PLACAS}")

model_car = YOLO(RUTA_MODELO_CARROS)
model_plate = YOLO(RUTA_MODELO_PLACAS)
print("Modelos cargados correctamente.")

# OCR (easyocr)
if EASYOCR_AVAILABLE:
    # Ajusta gpu=True si tienes GPU y quieres acelerarlo
    reader = easyocr.Reader(["en"], gpu=False)
    print("EasyOCR listo.")
else:
    reader = None
    print("Warning: easyocr no instalado. OCR devolverá None. Instala easyocr para habilitar OCR.")

# Helpers
def save_infraccion(placa, velocidad):
    header_needed = not os.path.exists(os.path.join(BASE, "infracciones.csv"))
    with open(os.path.join(BASE, "infracciones.csv"), "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["timestamp", "placa", "vel_kmh"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), placa, f"{velocidad:.2f}"])

def decode_b64_image(data_url: str):
    # data:image/jpeg;base64,...
    if "," in data_url:
        _, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    b = base64.b64decode(b64)
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def run_ocr_on_crop(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0
    if not EASYOCR_AVAILABLE:
        return None, 0.0
    # convert BGR -> RGB for easyocr
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    try:
        results = reader.readtext(img_rgb)
    except Exception:
        return None, 0.0
    if not results:
        return None, 0.0
    best = max(results, key=lambda r: r[2])  # (bbox, text, conf)
    text = best[1].strip().upper().replace(" ", "")
    conf = float(best[2])
    if len(text) < 3:
        return None, 0.0
    return text, conf

def iou(boxA, boxB):
    # boxes: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, (boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
    areaB = max(0, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    if areaA + areaB - interArea == 0:
        return 0.0
    return interArea / float(areaA + areaB - interArea)

def assign_tracks(detections):
    """
    detections: list of boxes [x1,y1,x2,y2]
    Returns list of (track_id, box)
    Uses centroid nearest association with timeout.
    IMPORTANT: preserves prev_centroid and prev_time before updating centroid/last_time
    so speed calculation later can use the previous position.
    """
    assigned = []
    now = time.time()
    used_tracks = set()

    # compute centroids
    det_centroids = [((b[0]+b[2])/2.0, (b[1]+b[3])/2.0) for b in detections]

    with tracks_lock:
        # For each detection, find nearest active track (within threshold)
        for i, centroid in enumerate(det_centroids):
            cx, cy = centroid
            best_tid = None
            best_dist = None
            for tid, info in tracks.items():
                # skip too-old tracks
                if now - info["last_time"] > 2.0:
                    continue
                tx, ty = info["centroid"]
                dist = (tx - cx)**2 + (ty - cy)**2
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None and best_tid not in used_tracks and best_dist is not None and best_dist < (200**2):
                # preserve previous centroid/time for speed calc
                info = tracks[best_tid]
                # set prev only if there's an existing centroid (not first assign)
                info["prev_centroid"] = info.get("centroid")
                info["prev_time"] = info.get("last_time")
                # now update centroid and last_time to the new detection
                info["centroid"] = det_centroids[i]
                info["last_time"] = now
                assigned.append((best_tid, detections[i]))
                used_tracks.add(best_tid)
            else:
                # create new track (no prev)
                tid_new = next_track_id()
                tracks[tid_new] = {
                    "centroid": det_centroids[i],
                    "last_time": now,
                    "prev_centroid": None,
                    "prev_time": None
                }
                assigned.append((tid_new, detections[i]))
                used_tracks.add(tid_new)

        # cleanup very old tracks
        for tid in list(tracks.keys()):
            if time.time() - tracks[tid]["last_time"] > 5.0:
                del tracks[tid]

    return assigned

class Handler(BaseHTTPRequestHandler):
    # Serve static files + API
    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        # Serve web files from WEB_DIR
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/" or path == "/index.html":
            path = "/index.html"
        file_path = os.path.join(WEB_DIR, path.lstrip("/"))
        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, "index.html")
        if not os.path.exists(file_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")
            return
        # content type
        content_type = "application/octet-stream"
        if file_path.endswith(".html"): content_type = "text/html"
        if file_path.endswith(".js"): content_type = "application/javascript"
        if file_path.endswith(".css"): content_type = "text/css"
        if file_path.endswith(".png"): content_type = "image/png"
        if file_path.endswith(".jpg") or file_path.endswith(".jpeg"): content_type = "image/jpeg"

        with open(file_path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path != "/procesar":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return

        # Read request
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_response(400); self.end_headers(); return

        img_data = data.get("imagen")
        if not img_data:
            self.send_response(400); self.end_headers(); return

        frame = decode_b64_image(img_data)
        if frame is None:
            self.send_response(400); self.end_headers(); return

        # Run car detection
        try:
            res_car = model_car(frame)[0]
        except Exception as e:
            print("Error infer car:", e)
            traceback.print_exc()
            self.send_response(500); self.end_headers(); return

        # extract boxes as list of float [x1,y1,x2,y2]
        car_boxes = []
        try:
            if hasattr(res_car.boxes, "xyxy"):
                arr = res_car.boxes.xyxy.cpu().numpy() if hasattr(res_car.boxes.xyxy, "cpu") else np.array(res_car.boxes.xyxy)
                for row in arr:
                    x1,y1,x2,y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    car_boxes.append([x1,y1,x2,y2])
            else:
                arr = res_car.boxes.data.cpu().numpy()
                for row in arr:
                    x1,y1,x2,y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    car_boxes.append([x1,y1,x2,y2])
        except Exception as e:
            print("Error reading car boxes:", e)
            traceback.print_exc()

        # Assign tracks (this updates tracks dict and preserves prev_centroid/prev_time)
        assigned = assign_tracks(car_boxes)  # list of (track_id, box)
        response_boxes = []  # to return to client
        max_speed_here = 0.0
        infraccion_flag = False

        for tid, box in assigned:
            try:
                x1,y1,x2,y2 = box
                # crop car region safely
                h_frame, w_frame = frame.shape[:2]
                xa, xb = max(0,int(x1)), min(w_frame-1,int(x2))
                ya, yb = max(0,int(y1)), min(h_frame-1,int(y2))
                car_crop = frame[ya:yb, xa:xb].copy() if ya<yb and xa<xb else None

                # compute centroid x,y for speed
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # compute speed using prev_centroid/prev_time stored in tracks
                speed_kmh = 0.0
                with tracks_lock:
                    info = tracks.get(tid)
                    if info is not None and info.get("prev_centroid") is not None and info.get("prev_time") is not None:
                        px, py = info["prev_centroid"]
                        dt = info["last_time"] - info["prev_time"]
                        if dt <= 0:
                            dt = FRAME_INTERVAL
                        dx = ((cx - px)**2 + (cy - py)**2)**0.5
                        meters = dx * PIXELS_TO_METERS
                        speed_kmh = (meters / dt) * 3.6
                    else:
                        speed_kmh = 0.0

                # plate detection + OCR inside car crop (per vehicle)
                plate_text_best = None
                plate_conf_best = 0.0
                if car_crop is not None and car_crop.size > 0:
                    try:
                        res_plate = model_plate(car_crop)[0]
                        # extract plate boxes in car_crop coords
                        if hasattr(res_plate.boxes, "xyxy"):
                            arrp = res_plate.boxes.xyxy.cpu().numpy() if hasattr(res_plate.boxes.xyxy, "cpu") else np.array(res_plate.boxes.xyxy)
                            for prow in arrp:
                                px1,py1,px2,py2 = [int(v) for v in prow[:4]]
                                pxa = max(0, px1); pxb = min(car_crop.shape[1]-1, px2)
                                pya = max(0, py1); pyb = min(car_crop.shape[0]-1, py2)
                                plate_crop = car_crop[pya:pyb, pxa:pxb] if pya<pyb and pxa<pxb else None
                                if plate_crop is not None:
                                    text, conf = run_ocr_on_crop(plate_crop)
                                    if text and conf > plate_conf_best:
                                        plate_conf_best = conf
                                        plate_text_best = text
                        else:
                            arrp = res_plate.boxes.data.cpu().numpy()
                            for prow in arrp:
                                px1,py1,px2,py2 = [int(v) for v in prow[:4]]
                                pxa = max(0, px1); pxb = min(car_crop.shape[1]-1, px2)
                                pya = max(0, py1); pyb = min(car_crop.shape[0]-1, py2)
                                plate_crop = car_crop[pya:pyb, pxa:pxb] if pya<pyb and pxa<pxb else None
                                if plate_crop is not None:
                                    text, conf = run_ocr_on_crop(plate_crop)
                                    if text and conf > plate_conf_best:
                                        plate_conf_best = conf
                                        plate_text_best = text
                    except Exception as e:
                        print("Error plate model:", e)
                        traceback.print_exc()

                # If violation for THIS vehicle and we have plate_text_best, save it.
                if speed_kmh > VELOCITY_LIMIT and plate_text_best:
                    try:
                        save_infraccion(plate_text_best, speed_kmh)
                        infraccion_flag = True
                    except Exception as e:
                        print("Error saving infraccion:", e)

                # Update overall max speed variable
                if speed_kmh > max_speed_here:
                    max_speed_here = speed_kmh

                # Build response box (in full frame coordinates)
                response_boxes.append({
                    "track_id": int(tid),
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "speed_kmh": round(speed_kmh, 2),
                    "plate": plate_text_best if plate_text_best else "",
                })
            except Exception as e:
                print("Error processing assigned box:", e)
                traceback.print_exc()

        # Prepare JSON response
        out = {
            "timestamp": time.time(),
            "velocidad_max": round(max_speed_here, 2),
            "limite": VELOCITY_LIMIT,
            "infraccion": infraccion_flag,
            "boxes": response_boxes
        }

        resp_bytes = json.dumps(out).encode("utf-8")
        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

def run_server(port=8000):
    httpd = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Servidor corriendo en http://0.0.0.0:{port}  — sirviendo carpeta {WEB_DIR}")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server(8000)

