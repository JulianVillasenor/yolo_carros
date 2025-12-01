import kagglehub
import shutil
from pathlib import Path


def safe_rmtree(path: Path):
    if path.exists():
        shutil.rmtree(path)


def find_dir(base: Path, name: str) -> Path | None:
    """
    Busca recursivamente un directorio con nombre exacto `name`.
    """
    for p in base.rglob("*"):
        if p.is_dir() and p.name == name:
            return p
    return None


def find_file(base: Path, name_startswith: str) -> Path | None:
    """
    Busca recursivamente un archivo que empiece con `name_startswith`.
    Ej: "train_solution_bounding_boxes" encontrará también
    "train_solution_bounding_boxes (1).csv".
    """
    for p in base.rglob("*.csv"):
        if p.name.startswith(name_startswith):
            return p
    return None


def main():
    print("⏳ Descargando dataset 'sshikamaru/car-object-detection' desde Kaggle...")

    # 1) Descargar dataset via KaggleHub
    path_str = kagglehub.dataset_download("sshikamaru/car-object-detection")
    cache_path = Path(path_str)

    print(f"📁 Dataset descargado en cache: {cache_path}")

    # 2) Directorio destino en tu proyecto
    project_root = Path(__file__).resolve().parent
    datasets_dir = project_root / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    target_dir = datasets_dir / "car_detection_kaggle"
    safe_rmtree(target_dir)
    target_dir.mkdir()

    print(f"📦 Copiando dataset a: {target_dir}")

    # 3) Buscar carpetas de imágenes
    train_dir_src = find_dir(cache_path, "training_images")
    test_dir_src = find_dir(cache_path, "testing_images")

    if train_dir_src is not None:
        dst = target_dir / "training_images"
        print(f"   ➜ Copiando training_images desde: {train_dir_src}")
        shutil.copytree(train_dir_src, dst)
    else:
        print("   ⚠ No se encontró carpeta 'training_images' en el cache (se buscaron recursivamente).")

    if test_dir_src is not None:
        dst = target_dir / "testing_images"
        print(f"   ➜ Copiando testing_images desde: {test_dir_src}")
        shutil.copytree(test_dir_src, dst)
    else:
        print("   ⚠ No se encontró carpeta 'testing_images' en el cache (se buscaron recursivamente).")

    # 4) Buscar CSVs
    train_csv_src = find_file(cache_path, "train_solution_bounding_boxes")
    sample_csv_src = find_file(cache_path, "sample_submission")

    if train_csv_src is not None:
        print(f"   ➜ Copiando {train_csv_src.name}")
        shutil.copy(train_csv_src, target_dir / train_csv_src.name)
    else:
        print("   ⚠ No se encontró 'train_solution_bounding_boxes*.csv' en el cache.")

    if sample_csv_src is not None:
        print(f"   ➜ Copiando {sample_csv_src.name}")
        shutil.copy(sample_csv_src, target_dir / sample_csv_src.name)
    else:
        print("   ⚠ No se encontró 'sample_submission*.csv' en el cache.")

    print("\n✅ Dataset copiado (hasta donde se encontró).")
    print("📌 Estructura final en tu proyecto:")
    print(f"   {target_dir}")
    for p in target_dir.iterdir():
        print("   -", p.name)

    print("\n🚀 Siguiente paso: convertir 'train_solution_bounding_boxes*.csv' a formato YOLO.")
    print("   (podemos crear convert_car_csv_to_yolo.py para eso).")


if __name__ == "__main__":
    main()
