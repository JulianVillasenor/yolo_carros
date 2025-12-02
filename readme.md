# 🚗🔍 YOLO Carros y Placas

Repositorio del proyecto de **Redes Neuronales** para entrenar dos modelos YOLOv8 y montar un pipeline de:

1. **Detección de placas** (modelo YOLO entrenado con dataset tipo VOC convertido a YOLO).
2. **Detección de carros** (modelo YOLO entrenado con dataset de Kaggle).
3. **Tracking de vehículos** (tracker tipo SORT).
4. **Lectura de placas (OCR)** y generación de un CSV + video anotado.

La parte de **estimación de velocidad** y **demo web con ngrok** está planeada como fase siguiente.

---

## 🗂 Estructura del proyecto

Estructura principal del repo:

```text
yolo_carros/
│  .dvcignore
│  .gitignore
│  data_cars.yaml              # config YOLO para autos
│  data_license_plates.yaml    # config YOLO para placas
│  dvc.yaml                    # configuración de DVC
│  import.py                   # descarga dataset de placas
│  import2.py                  # descarga dataset de autos
│  readme.md
│  requirements.txt
│  yolov8n.pt                  # pesos base YOLOv8n (opcional si ya descargó Ultralytics)
│
├─.dvc/                        # metadatos de DVC
├─datasets/                    # datasets locales (ignorados en Git)
├─dvclive/                     # logs de entrenamiento (DVC / dvclive)
├─notebooks/                   # notebooks de pruebas y exploración
├─remote/                      # configs para correr en máquina remota
├─results/                     # salidas del pipeline (CSV + videos anotados)
├─runs/                        # runs genéricos de Ultralytics
├─runs_cars/                   # runs de entrenamiento YOLO para autos
├─runs_plates/                 # runs de entrenamiento YOLO para placas
├─src/                         # código fuente del proyecto
├─videos/                      # videos de entrada para pruebas
└─web/                         # base para futura demo web/ngrok
