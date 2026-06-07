"""
Google Colab setup utility.

Responsibilities:
1. Verify dataset ZIP exists
2. Extract dataset if needed
3. Verify dataset structure
4. Print dataset statistics
"""

from pathlib import Path
from zipfile import ZipFile
import os

from ml.config import (
    DATASET_ZIP_PATH,
    DATASET_ROOT,
    TRAIN_DIR,
    VAL_DIR,
)


def verify_zip() -> None:
    """
    Verify dataset ZIP exists.
    """

    zip_path = Path(DATASET_ZIP_PATH)

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset ZIP not found:\n{zip_path}"
        )

    print("✓ Dataset ZIP found")


def extract_dataset() -> None:
    """
    Extract dataset only if not already extracted.
    """

    if Path(DATASET_ROOT).exists():
        print("✓ Dataset already extracted")
        return

    print("Extracting dataset...")

    with ZipFile(DATASET_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("/content")

    print("✓ Dataset extracted")


def verify_dataset_structure() -> None:
    """
    Verify train and validation folders exist.
    """

    if not Path(TRAIN_DIR).exists():
        raise FileNotFoundError(
            f"Train directory missing:\n{TRAIN_DIR}"
        )

    if not Path(VAL_DIR).exists():
        raise FileNotFoundError(
            f"Validation directory missing:\n{VAL_DIR}"
        )

    print("✓ Dataset structure verified")


def print_dataset_statistics() -> None:
    """
    Print useful dataset information.
    """

    train_classes = sorted(
        [
            folder
            for folder in os.listdir(TRAIN_DIR)
            if os.path.isdir(os.path.join(TRAIN_DIR, folder))
        ]
    )

    val_classes = sorted(
        [
            folder
            for folder in os.listdir(VAL_DIR)
            if os.path.isdir(os.path.join(VAL_DIR, folder))
        ]
    )

    print("\nDataset Summary")
    print("-" * 40)

    print(f"Train Classes: {len(train_classes)}")
    print(f"Validation Classes: {len(val_classes)}")

    print("\nFirst 10 Classes:")

    for class_name in train_classes[:10]:
        print(f"  • {class_name}")

    print("-" * 40)


def main() -> None:

    print("\nCrop Disease Advisory System")
    print("Colab Setup")
    print("=" * 40)

    verify_zip()

    extract_dataset()

    verify_dataset_structure()

    print_dataset_statistics()

    print("\n✓ Setup complete")
    print("=" * 40)


if __name__ == "__main__":
    main()