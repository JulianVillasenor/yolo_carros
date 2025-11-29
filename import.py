import kagglehub
import shutil
from pathlib import Path

def main():
    # ------------------------------------------------------------
    # 1. Descargar dataset desde Kaggle
    # ------------------------------------------------------------
    print("⏳ Descargando dataset 'andrewmvd/car-plate-detection' desde Kaggle...")

    path = kagglehub.dataset_download("andrewmvd/car-plate-detection")
    src_path = Path(path)
    print("✔️  Dataset descargado en (cache kagglehub):", src_path)

    # ------------------------------------------------------------
    # 2. Copiar dataset a la carpeta del proyecto
    #    yolo_carros/datasets/car_plate_detection_voc
    # ------------------------------------------------------------
    project_root = Path(__file__).resolve().parent
    target_root = project_root / "datasets"
    target_dir = target_root / "car_plate_detection_voc"

    # Crear carpeta raíz de datasets si no existe
    target_root.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        print(f"ℹ️ La carpeta destino ya existe: {target_dir}")
        print("   (Si quieres forzar una copia limpia, borra esa carpeta y vuelve a correr el script.)")
    else:
        print(f"📁 Copiando dataset a: {target_dir}")
        # Copia TODO el contenido del dataset (imágenes + xml)
        shutil.copytree(src_path, target_dir)
        print("✅ Dataset copiado correctamente.")

    print("\n📌 Dataset listo para usar en:", target_dir)
    print("   (Recuerda que 'datasets/' está en .gitignore, así que no se subirá a Git.)")

if __name__ == "__main__":
    main()