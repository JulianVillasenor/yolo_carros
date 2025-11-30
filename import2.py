import kagglehub
import shutil
from pathlib import Path


def copytree(src, dst):
    """Copiar directorios ignorando caché y archivos innecesarios."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    print("⏳ Descargando dataset 'sshikamaru/car-object-detection' desde Kaggle...")

    # Descargar dataset via KaggleHub
    path = kagglehub.dataset_download("sshikamaru/car-object-detection")

    print(f"📁 Dataset descargado en cache: {path}")

    # Ruta del proyecto (dos niveles arriba si este archivo está en raíz)
    project_root = Path(__file__).resolve().parent
    datasets_dir = project_root / "datasets"

    # Asegurar directorio datasets/
    datasets_dir.mkdir(exist_ok=True)

    # Carpeta destino organizada
    target_dir = datasets_dir / "car_detection_kaggle"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir()

    print(f"📦 Copiando dataset a: {target_dir}")

    # Copiar carpetas principales
    src_path = Path(path)

    folders_to_copy = [
        "training_images",
        "testing_images"
    ]

    for folder in folders_to_copy:
        src_folder = src_path / folder
        dst_folder = target_dir / folder

        if src_folder.exists():
            print(f"   ➜ Copiando {folder} ...")
            shutil.copytree(src_folder, dst_folder)
        else:
            print(f"   ⚠ No se encontró {folder} en {src_path}")

    # Copiar CSVs
    csv_files = [
        "train_solution_bounding_boxes.csv",
        "sample_submission.csv"
    ]

    for csv_file in csv_files:
        src_csv = src_path / csv_file
        if src_csv.exists():
            print(f"   ➜ Copiando {csv_file}")
            shutil.copy(src_csv, target_dir / csv_file)
        else:
            print(f"   ⚠ No se encontró {csv_file}")

    print("\n✅ Dataset copiado correctamente.")
    print("📌 Estructura final:")
    print(target_dir)
    for item in target_dir.iterdir():
        print("   -", item.name)

    print("\n🚀 Ahora puedes ejecutar tu convertidor (convert_csv_to_yolo.py)")
    print("o crear uno nuevo para procesar el archivo train_solution_bounding_boxes.csv")


if __name__ == "__main__":
    main()
