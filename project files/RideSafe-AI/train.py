from ultralytics import YOLO
import torch


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = r"C:\AI Projects\HelmetViolationSystem\datasets\helmet_final\data.yaml"

MODEL = r"C:\AI Projects\HelmetViolationSystem\yolo11n.pt"

PROJECT = r"C:\AI Projects\HelmetViolationSystem\runs"

RUN_NAME = "helmet_yolo11n"


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 60)
print("HELMET VIOLATION DETECTION - YOLO11 TRAINING")
print("=" * 60)

print()

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    DEVICE = 0
else:
    print("WARNING: CUDA not available. Using CPU.")
    DEVICE = "cpu"

print()


# ============================================================
# LOAD YOLO11
# ============================================================

print("Loading YOLO11 model...")

model = YOLO(MODEL)

print("Model loaded successfully.")
print()


# ============================================================
# TRAIN
# ============================================================

results = model.train(
    data=DATASET,

    # Training duration
    epochs=100,

    # Image resolution
    imgsz=640,

    # RTX 4060 8GB
    batch=16,

    # GPU
    device=DEVICE,

    # Number of dataloader workers
    workers=0,

    # Output
    project=PROJECT,
    name=RUN_NAME,

    # Save checkpoints
    save=True,
    save_period=10,

    # Early stopping
    patience=20,

    # Pretrained model
    pretrained=True,

    # Validation during training
    val=True,

    # Cache images in RAM if possible
    cache=False,

    # Automatically close mosaic augmentation near end
    close_mosaic=10,

    # Reproducibility
    seed=42,

    # Training output
    verbose=True,
)


# ============================================================
# TRAINING COMPLETE
# ============================================================

from pathlib import Path

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

run_directory = Path(PROJECT) / RUN_NAME
best_model = run_directory / "weights" / "best.pt"

print()
print("Training results:")
print(run_directory)

print()
print("Best model:")
print(best_model)