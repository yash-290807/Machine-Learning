import os
import cv2
import torch
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\AI Projects\HelmetViolationSystem"

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "runs",
    "helmet_yolo11n-3",
    "weights",
    "best.pt"
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "cctv_training",
    "images"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "yolo_scan"
)


# ============================================================
# SETTINGS
# ============================================================

# We deliberately use a lower confidence for this diagnostic.
CONFIDENCE = 0.15

IMAGE_SIZE = 640

MAX_SAVED = 10


# ============================================================
# START
# ============================================================

print("=" * 60)
print("YOLO11n CCTV FRAME SCANNER")
print("=" * 60)
print()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:")
print(torch.cuda.get_device_name(0))
print()

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO11n...")

model = YOLO(MODEL_PATH)

print("YOLO11n loaded successfully.")
print()

print("Classes:")
print(model.names)
print()


# ============================================================
# FIND IMAGES
# ============================================================

image_files = [
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

image_files.sort()

print(
    f"CCTV images found: {len(image_files)}"
)

print()


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SCAN
# ============================================================

best_frames = []

total_detections = 0

print("Scanning CCTV frames...")
print()

for index, filename in enumerate(image_files):

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:
        print(
            f"Could not read: {filename}"
        )
        continue

    results = model.predict(
        source=image,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device=0,
        verbose=False
    )

    result = results[0]

    detection_count = len(
        result.boxes
    )

    total_detections += detection_count

    # --------------------------------------------------------
    # Keep frames with detections
    # --------------------------------------------------------

    if detection_count > 0:

        highest_confidence = float(
            result.boxes.conf.max().item()
        )

        best_frames.append(
            (
                detection_count,
                highest_confidence,
                filename
            )
        )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        (index + 1) % 25 == 0
        or index == len(image_files) - 1
    ):

        print(
            f"Processed "
            f"{index + 1}/{len(image_files)}"
        )


# ============================================================
# SORT RESULTS
# ============================================================

best_frames.sort(
    key=lambda x: x[1],
    reverse=True
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("SCAN COMPLETE")
print("=" * 60)

print()

print(
    f"Total images scanned: "
    f"{len(image_files)}"
)

print(
    f"Images with detections: "
    f"{len(best_frames)}"
)

print(
    f"Total detections: "
    f"{total_detections}"
)

print()


# ============================================================
# SAVE TOP DETECTION FRAMES
# ============================================================

if len(best_frames) == 0:

    print(
        "NO YOLO DETECTIONS FOUND."
    )

    print()
    print(
        "The current helmet/no_helmet model "
        "does not detect anything in these CCTV frames "
        "at confidence 0.15."
    )

else:

    print("Top detection frames:")
    print()

    for rank, (
        detection_count,
        highest_confidence,
        filename
    ) in enumerate(
        best_frames[:MAX_SAVED],
        start=1
    ):

        print(
            f"{rank}. "
            f"{filename} | "
            f"detections={detection_count} | "
            f"highest_conf={highest_confidence:.3f}"
        )

        image_path = os.path.join(
            IMAGE_DIR,
            filename
        )

        image = cv2.imread(
            image_path
        )

        results = model.predict(
            source=image,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=0,
            verbose=False
        )

        annotated = results[0].plot()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"rank_{rank:02d}_{filename}"
        )

        cv2.imwrite(
            output_path,
            annotated
        )


# ============================================================
# FINISH
# ============================================================

print()

print(
    "Saved detection previews to:"
)

print(
    OUTPUT_DIR
)

print()

print("=" * 60)