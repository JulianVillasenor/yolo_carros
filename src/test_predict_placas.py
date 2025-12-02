"""
test_predict_placas.py
----------------------

Usa el modelo entrenado de placas (best.pt) para hacer predicciones
sobre una imagen de validación y guardar el resultado con las bounding boxes.
"""

from pathlib import Path
from ultralytics import YOLO


def main():
    # Localización del Directorio Raíz del Proyecto
    # Asume que este archivo está anidado en una subcarpeta
    project_root = Path(__file__).resolve().parent.parent

    # Ruta del modelo entrenado
    # Localización del Modelo Entrenado
    # La ruta estándar donde Ultralytics guarda el mejor modelo entrenado (`best.pt`).
    weights_path = project_root / "runs_plates" / "yolov8n-plates" / "weights" / "best.pt"

    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo entrenado en: {weights_path}")

    # Carpeta de imágenes de validación
    # Localización de la Imagen de Validación
    # Directorio donde se encuentran las imágenes para probar.
    val_images_dir = project_root / "datasets" / "license_plates_yolo" / "images" / "val"
    if not val_images_dir.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de imágenes de validación: {val_images_dir}")

    # Tomamos la primera imagen de val (png/jpg/jpeg)
    candidates = list(val_images_dir.glob("*.png")) + \
                 list(val_images_dir.glob("*.jpg")) + \
                 list(val_images_dir.glob("*.jpeg"))

    if not candidates:
        raise RuntimeError(f"No se encontraron imágenes en: {val_images_dir}")

    img_path = candidates[0]
    print(f"🔍 Usando imagen de prueba: {img_path}")

    # Carpeta donde guardaremos las predicciones
    # Configuración del Directorio de Salida
    # Directorio donde se guardará la imagen con las predicciones dibujadas.
    preds_dir = project_root / "runs_plates" / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Cargar modelo
    model = YOLO(str(weights_path))

    # Hacer predicción
    # Realizar Predicción (Inferencia)
    # El método `predict` ejecuta el modelo sobre la fuente proporcionada.
    results = model.predict(
        source=str(img_path),
        save=True,
        project=str(preds_dir),
        name="placas_test",
        exist_ok=True,
        conf=0.25  # umbral de confianza
    )

    print("\n✅ Predicción completada.")
    print(f"   Resultados guardados en: {preds_dir / 'placas_test'}")
    print(f"   Imagen de entrada: {img_path}")
    print("   Revisa la imagen con la placa detectada.")


if __name__ == "__main__":
    main()
