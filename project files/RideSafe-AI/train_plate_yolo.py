from pathlib import Path
import torch
from ultralytics import YOLO


ROOT = Path(r"C:\AI Projects\HelmetViolationSystem")

DATASET = ROOT / "datasets" / "plate_raw" / "data.yaml"
MODEL = ROOT / "yolo11n.pt"

PROJECT = ROOT / "runs"
RUN_NAME = "plate_yolo11n"

DEVICE = 0 if torch.cuda.is_available() else "cpu"


print("=" * 70)
print("INDIAN LICENSE PLATE - YOLO11 TRAINING")
print("=" * 70)

print("Dataset :", DATASET)
print("Model   :", MODEL)
print("Device  :", DEVICE)

if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))

print("=" * 70)


if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found:\n{DATASET}"
    )

if not MODEL.exists():
    raise FileNotFoundError(
        f"YOLO11 model not found:\n{MODEL}"
    )


model = YOLO(str(MODEL))


model.train(
    data=str(DATASET),

    epochs=80,

    imgsz=640,

    batch=16,

    device=DEVICE,

    workers=0,

    project=str(PROJECT),

    name=RUN_NAME,

    pretrained=True,

    save=True,

    save_period=10,

    val=True,

    patience=15,

    cache=False,

    close_mosaic=10,

    seed=42,

    verbose=True,
)


best_model = (
    PROJECT
    / RUN_NAME
    / "weights"
    / "best.pt"
)


print()
print("=" * 70)
print("PLATE TRAINING COMPLETE")
print("=" * 70)

print("Best model:")
print(best_model)

print("=" * 70)