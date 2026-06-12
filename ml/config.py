"""
Central ML configuration.
"""

from pathlib import Path

# Dataset

DATASET_ZIP_PATH = (
    "/content/drive/MyDrive/SmartCropDisease/dataset/archive.zip"
)

DATASET_ROOT = "/content/PlantVillage"

TRAIN_DIR = f"{DATASET_ROOT}/train"
VAL_DIR = f"{DATASET_ROOT}/val"

# Images

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

# Reproducibility

SEED = 42

# Training

NUM_EPOCHS = 20

# Outputs

OUTPUT_DIR = Path("ml") / "outputs"

MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
METRICS_OUTPUT_DIR = OUTPUT_DIR / "metrics"
PLOTS_OUTPUT_DIR = OUTPUT_DIR / "plots"