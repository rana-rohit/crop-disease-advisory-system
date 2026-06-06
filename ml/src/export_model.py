"""Export a trained crop disease model and class-name metadata."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import tensorflow as tf

try:
    from .data_loader import load_datasets
except ImportError:  # Allows `python ml/src/export_model.py` in Colab or VS Code.
    from data_loader import load_datasets  # type: ignore[no-redef]


# Export defaults match the backend artifact folder used by the Flask service.
DEFAULT_OUTPUT_DIR = Path("ml") / "outputs"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "models" / "crop_disease_mobilenetv2.keras"
DEFAULT_CLASS_NAMES_PATH = DEFAULT_OUTPUT_DIR / "models" / "class_names.json"
DEFAULT_EXPORT_DIR = Path("backend") / "model" / "artifacts"
DEFAULT_EXPORTED_MODEL_FILENAME = "crop_disease_mobilenetv2.keras"
DEFAULT_EXPORTED_CLASS_NAMES_FILENAME = "class_names.json"
DEFAULT_SAVED_MODEL_DIRNAME = "saved_model"


def read_class_names(class_names_path: Path) -> list[str]:
    """Read class names from a JSON file."""

    with class_names_path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list) or not all(
        isinstance(name, str) for name in class_names
    ):
        raise ValueError("Class names JSON must contain a list of strings.")

    return class_names


def write_class_names(class_names: list[str], output_path: Path) -> None:
    """Write class names in the format expected by the Flask backend."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)


def get_class_names(args: argparse.Namespace) -> list[str]:
    """Resolve class names from an existing JSON file or the dataset folders."""

    if args.class_names_path:
        class_names_path = Path(args.class_names_path).expanduser().resolve()
        if class_names_path.is_file():
            return read_class_names(class_names_path)

    if args.data_dir:
        dataset_bundle = load_datasets(dataset_root=args.data_dir)
        return dataset_bundle.class_names

    raise ValueError("Provide --class-names-path or --data-dir to export class names.")


def export_saved_model(model: tf.keras.Model, saved_model_dir: Path) -> None:
    """Export a TensorFlow SavedModel for serving workflows."""

    saved_model_dir.parent.mkdir(parents=True, exist_ok=True)
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    if hasattr(model, "export"):
        model.export(str(saved_model_dir))
    else:
        tf.saved_model.save(model, str(saved_model_dir))


def export_model(args: argparse.Namespace) -> dict[str, Path]:
    """Export the trained model and class-name metadata."""

    model_path = Path(args.model_path).expanduser().resolve()
    export_dir = Path(args.export_dir).expanduser().resolve()
    exported_model_path = export_dir / args.exported_model_filename
    exported_class_names_path = export_dir / args.exported_class_names_filename
    saved_model_dir = export_dir / args.saved_model_dirname

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(str(model_path))
    class_names = get_class_names(args)

    export_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(exported_model_path))
    write_class_names(class_names, exported_class_names_path)

    if args.export_saved_model:
        export_saved_model(model, saved_model_dir)

    return {
        "model_path": exported_model_path,
        "class_names_path": exported_class_names_path,
        "saved_model_dir": saved_model_dir,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for export."""

    parser = argparse.ArgumentParser(
        description="Export the trained crop disease model for backend reuse."
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("CROP_DISEASE_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        help="Path to the trained Keras model.",
    )
    parser.add_argument(
        "--class-names-path",
        default=os.getenv("CROP_DISEASE_CLASS_NAMES_PATH", str(DEFAULT_CLASS_NAMES_PATH)),
        help="Path to class_names.json generated during training.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("PLANTVILLAGE_DATA_DIR"),
        help="Optional dataset root used when class names JSON is unavailable.",
    )
    parser.add_argument(
        "--export-dir",
        default=os.getenv("BACKEND_ARTIFACT_DIR", str(DEFAULT_EXPORT_DIR)),
        help="Directory where backend-ready artifacts will be written.",
    )
    parser.add_argument(
        "--exported-model-filename",
        default=DEFAULT_EXPORTED_MODEL_FILENAME,
    )
    parser.add_argument(
        "--exported-class-names-filename",
        default=DEFAULT_EXPORTED_CLASS_NAMES_FILENAME,
    )
    parser.add_argument("--saved-model-dirname", default=DEFAULT_SAVED_MODEL_DIRNAME)
    parser.add_argument("--export-saved-model", action="store_true")

    return parser.parse_args()


def main() -> None:
    """Export model artifacts from command-line arguments."""

    artifacts = export_model(parse_args())
    print("Export complete.")
    print(f"Backend model: {artifacts['model_path']}")
    print(f"Class names: {artifacts['class_names_path']}")
    if artifacts["saved_model_dir"].exists():
        print(f"SavedModel: {artifacts['saved_model_dir']}")


if __name__ == "__main__":
    main()
