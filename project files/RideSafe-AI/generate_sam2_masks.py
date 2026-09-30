from pathlib import Path
import cv2
import numpy as np
import torch
import json

from ultralytics import YOLO

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"C:\AI Projects\HelmetViolationSystem")

DATASET_DIR = PROJECT_DIR / "datasets" / "helmet_final"

IMAGE_DIR = DATASET_DIR / "train" / "images"
LABEL_DIR = DATASET_DIR / "train" / "labels"

OUTPUT_DIR = PROJECT_DIR / "sam2_instance_training"

MASK_DIR = OUTPUT_DIR / "masks"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
METADATA_DIR = OUTPUT_DIR / "metadata"


# ============================================================
# PROCESSING LIMIT
# ============================================================

# Process up to 10,000 images.
# Your dataset currently contains 5,520 training images.
MAX_IMAGES = 10000


# ============================================================
# YOLO MODEL
# ============================================================

YOLO_MODEL = (
    PROJECT_DIR
    / "runs"
    / "helmet_yolo11n-3"
    / "weights"
    / "best.pt"
)


# ============================================================
# SAM 2.1 SMALL
# ============================================================

SAM2_CHECKPOINT = (
    PROJECT_DIR
    / "sam2_repo"
    / "checkpoints"
    / "sam2.1_hiera_small.pt"
)

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "helmet",
    1: "no_helmet"
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("=" * 70)
print("YOLO11n + SAM2.1 SMALL INSTANCE MASK GENERATION")
print("=" * 70)
print()

print("Device:", DEVICE)
print("Dataset:", DATASET_DIR)
print("Images:", IMAGE_DIR)
print("Labels:", LABEL_DIR)
print("Output:", OUTPUT_DIR)
print()


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

MASK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO11n...")

yolo_model = YOLO(str(YOLO_MODEL))

print("YOLO11n loaded successfully.")
print("Classes:", yolo_model.names)
print()


# ============================================================
# LOAD SAM2
# ============================================================

print("Loading SAM 2.1 Small...")

sam2_model = build_sam2(
    SAM2_CONFIG,
    str(SAM2_CHECKPOINT),
    device=DEVICE
)

sam2_predictor = SAM2ImagePredictor(sam2_model)

print("SAM 2.1 Small loaded successfully.")
print()


# ============================================================
# FIND IMAGES
# ============================================================

image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg"))
    + list(IMAGE_DIR.glob("*.jpeg"))
    + list(IMAGE_DIR.glob("*.png"))
)

print("Images found:", len(image_files))
print(
    "Images to process:",
    min(MAX_IMAGES, len(image_files))
)
print()


# ============================================================
# PROCESSING COUNTERS
# ============================================================

processed = 0
total_objects = 0


# ============================================================
# PROCESS IMAGES
# ============================================================

for image_path in image_files[:MAX_IMAGES]:

    print("=" * 70)
    print("PROCESSING:", image_path.name)
    print("=" * 70)

    # --------------------------------------------------------
    # LABEL PATH
    # --------------------------------------------------------

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():

        print("WARNING: Label not found")
        print(label_path)
        print()

        continue


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:

        print("WARNING: Could not load image")
        print()

        continue


    height, width = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    print(
        f"Image size: {width} x {height}"
    )


    # --------------------------------------------------------
    # READ YOLO LABELS
    # --------------------------------------------------------

    detections = []

    with open(label_path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) < 5:
                continue

            class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])

            box_width = float(values[3])
            box_height = float(values[4])


            # ------------------------------------------------
            # YOLO NORMALIZED COORDINATES -> PIXELS
            # ------------------------------------------------

            x1 = int(
                (x_center - box_width / 2)
                * width
            )

            y1 = int(
                (y_center - box_height / 2)
                * height
            )

            x2 = int(
                (x_center + box_width / 2)
                * width
            )

            y2 = int(
                (y_center + box_height / 2)
                * height
            )


            # ------------------------------------------------
            # CLAMP COORDINATES
            # ------------------------------------------------

            x1 = max(
                0,
                min(x1, width - 1)
            )

            y1 = max(
                0,
                min(y1, height - 1)
            )

            x2 = max(
                0,
                min(x2, width - 1)
            )

            y2 = max(
                0,
                min(y2, height - 1)
            )


            # ------------------------------------------------
            # VALID BOX
            # ------------------------------------------------

            if x2 <= x1 or y2 <= y1:
                continue


            detections.append(
                (
                    class_id,
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


    print(
        "YOLO ground-truth boxes:",
        len(detections)
    )


    if not detections:

        print("No valid labels.")
        print()

        continue


    # ========================================================
    # GIVE IMAGE TO SAM2
    # ========================================================

    sam2_predictor.set_image(
        image_rgb
    )


    # ========================================================
    # CREATE INSTANCE MASK
    # ========================================================

    # IMPORTANT:
    #
    # 0 = background
    # 1 = object 1
    # 2 = object 2
    # 3 = object 3
    # ...
    #
    # Each object gets a UNIQUE ID.
    #
    # uint16 allows many object IDs.

    combined_mask = np.zeros(
        (height, width),
        dtype=np.uint16
    )


    # ========================================================
    # OBJECT METADATA
    # ========================================================

    object_metadata = []

    object_count = 0


    # ========================================================
    # PROCESS EACH YOLO OBJECT WITH SAM2
    # ========================================================

    for class_id, x1, y1, x2, y2 in detections:

        object_id = object_count + 1

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}"
        )


        print(
            f"SAM2 object {object_id}: "
            f"class={class_id} ({class_name}) "
            f"box=({x1},{y1},{x2},{y2})"
        )


        # ----------------------------------------------------
        # SAM2 BOX
        # ----------------------------------------------------

        sam_box = np.array(
            [x1, y1, x2, y2],
            dtype=np.float32
        )


        # ----------------------------------------------------
        # SAM2 PREDICTION
        # ----------------------------------------------------

        with torch.inference_mode():

            if DEVICE == "cuda":

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    masks, scores, _ = (
                        sam2_predictor.predict(
                            box=sam_box,
                            multimask_output=False
                        )
                    )

            else:

                masks, scores, _ = (
                    sam2_predictor.predict(
                        box=sam_box,
                        multimask_output=False
                    )
                )


        # ----------------------------------------------------
        # GET MASK
        # ----------------------------------------------------

        mask = masks[0]

        if mask.ndim == 3:
            mask = mask[0]

        mask = mask.astype(bool)


        # ----------------------------------------------------
        # STORE UNIQUE OBJECT ID
        # ----------------------------------------------------

        combined_mask[mask] = object_id


        # ----------------------------------------------------
        # SAM2 SCORE
        # ----------------------------------------------------

        sam2_score = float(
            scores[0]
        )


        print(
            f"    SAM2 score: {sam2_score:.4f}"
        )


        # ----------------------------------------------------
        # SAVE OBJECT METADATA
        # ----------------------------------------------------

        object_metadata.append(
            {
                "object_id": object_id,
                "class_id": int(class_id),
                "class_name": class_name,
                "sam2_score": sam2_score,
                "box": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ]
            }
        )


        object_count += 1


    # ========================================================
    # SAVE INSTANCE MASK
    # ========================================================

    mask_output = (
        MASK_DIR
        / f"{image_path.stem}.png"
    )


    success = cv2.imwrite(
        str(mask_output),
        combined_mask
    )


    if not success:

        print(
            "ERROR: Failed to save mask:"
        )

        print(mask_output)

        continue


    # ========================================================
    # SAVE CORRESPONDING IMAGE
    # ========================================================

    image_output = (
        IMAGE_OUTPUT_DIR
        / image_path.name
    )


    cv2.imwrite(
        str(image_output),
        image_bgr
    )


    # ========================================================
    # SAVE METADATA JSON
    # ========================================================

    metadata_output = (
        METADATA_DIR
        / f"{image_path.stem}.json"
    )


    metadata = {
        "image": image_path.name,
        "width": int(width),
        "height": int(height),
        "objects": object_metadata
    }


    with open(
        metadata_output,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )


    # ========================================================
    # UPDATE COUNTERS
    # ========================================================

    processed += 1

    total_objects += object_count


    print()
    print(
        "Objects segmented:",
        object_count
    )

    print(
        "Instance mask saved:",
        mask_output
    )

    print(
        "Metadata saved:",
        metadata_output
    )

    print()


# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("SAM2 INSTANCE MASK GENERATION COMPLETE")
print("=" * 70)

print(
    "Images processed:",
    processed
)

print(
    "Objects segmented:",
    total_objects
)

print()

print("Masks:")
print(MASK_DIR)

print()

print("Images:")
print(IMAGE_OUTPUT_DIR)

print()

print("Metadata:")
print(METADATA_DIR)

print("=" * 70)