import os
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\AI Projects\HelmetViolationSystem"

YOLO_MODEL = os.path.join(
    PROJECT_ROOT,
    "runs",
    "helmet_yolo11n-3",
    "weights",
    "best.pt"
)

SAM2_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "sam2",
    "checkpoints",
    "sam2.1_hiera_small.pt"
)

# IMPORTANT:
# SAM 2 expects this as a Hydra config name,
# NOT as a Windows filesystem path.
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

INPUT_IMAGE = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "cctv_training",
    "images",
    "cctv_0276.jpg"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

OUTPUT_IMAGE = os.path.join(
    OUTPUT_DIR,
    "hybrid_test_cctv_0000.jpg"
)


# ============================================================
# SETTINGS
# ============================================================

YOLO_CONFIDENCE = 0.15
MAX_DETECTIONS = 10
DEVICE = "cuda"


# ============================================================
# START
# ============================================================

print("=" * 60)
print("YOLO11n + SAM 2.1 SMALL HYBRID TEST")
print("=" * 60)
print()


# ============================================================
# CHECK CUDA
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. GPU cannot be used."
    )

print("GPU:")
print(torch.cuda.get_device_name(0))
print()


# ============================================================
# LOAD YOLO11n
# ============================================================

print("Loading YOLO11n...")

yolo_model = YOLO(YOLO_MODEL)

print("YOLO11n loaded successfully.")
print()


# ============================================================
# LOAD SAM 2.1 SMALL
# ============================================================

print("Loading SAM 2.1 Small...")

sam2_model = build_sam2(
    SAM2_CONFIG,
    SAM2_CHECKPOINT,
    device=DEVICE
)

sam2_predictor = SAM2ImagePredictor(
    sam2_model
)

print("SAM 2.1 Small loaded successfully.")
print()


# ============================================================
# LOAD CCTV IMAGE
# ============================================================

print("Loading CCTV image:")
print(INPUT_IMAGE)

image_bgr = cv2.imread(INPUT_IMAGE)

if image_bgr is None:
    raise FileNotFoundError(
        f"Could not load image:\n{INPUT_IMAGE}"
    )

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)

print(
    f"Image size: "
    f"{image_bgr.shape[1]} x {image_bgr.shape[0]}"
)

print()


# ============================================================
# YOLO DETECTION
# ============================================================

print("Running YOLO11n detection...")

with torch.inference_mode():

    yolo_results = yolo_model.predict(
        source=image_bgr,
        conf=YOLO_CONFIDENCE,
        device=0,
        verbose=False
    )

result = yolo_results[0]

boxes = result.boxes

print(
    f"YOLO detections: {len(boxes)}"
)

print()


# ============================================================
# PREPARE IMAGE FOR SAM 2
# ============================================================

print("Preparing image for SAM 2...")

with torch.inference_mode():

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):

        sam2_predictor.set_image(
            image_rgb
        )

print(
    "SAM 2 image embedding created."
)

print()


# ============================================================
# PROCESS YOLO DETECTIONS
# ============================================================

annotated = image_bgr.copy()

class_names = yolo_model.names

processed = 0


for i, box in enumerate(boxes):

    if processed >= MAX_DETECTIONS:
        break

    # --------------------------------------------------------
    # YOLO BOX
    # --------------------------------------------------------

    xyxy = (
        box.xyxy[0]
        .detach()
        .cpu()
        .numpy()
    )

    x1, y1, x2, y2 = xyxy.astype(int)

    # Keep coordinates inside image

    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        image_bgr.shape[1] - 1,
        x2
    )

    y2 = min(
        image_bgr.shape[0] - 1,
        y2
    )

    if x2 <= x1 or y2 <= y1:
        continue


    # --------------------------------------------------------
    # YOLO CLASS
    # --------------------------------------------------------

    class_id = int(
        box.cls[0].item()
    )

    confidence = float(
        box.conf[0].item()
    )

    class_name = class_names[
        class_id
    ]

    print(
        f"Detection {processed + 1}: "
        f"{class_name} "
        f"confidence={confidence:.3f} "
        f"box=({x1},{y1},{x2},{y2})"
    )


    # --------------------------------------------------------
    # SAM 2 BOX PROMPT
    # --------------------------------------------------------

    sam_box = np.array(
        [
            x1,
            y1,
            x2,
            y2
        ],
        dtype=np.float32
    )

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):

            masks, scores, _ = (
                sam2_predictor.predict(
                    box=sam_box,
                    multimask_output=False
                )
            )


    # --------------------------------------------------------
    # GET MASK
    # --------------------------------------------------------

    mask = masks[0]

    if mask.ndim == 3:
        mask = mask[0]

    mask = mask.astype(bool)


    # --------------------------------------------------------
    # SEGMENTATION OVERLAY
    # --------------------------------------------------------

    overlay = annotated.copy()

    if class_name == "helmet":

        color = (
            0,
            255,
            0
        )

    else:

        color = (
            0,
            0,
            255
        )


    overlay[mask] = color


    annotated = cv2.addWeighted(
        overlay,
        0.35,
        annotated,
        0.65,
        0
    )


    # --------------------------------------------------------
    # DRAW YOLO BOX
    # --------------------------------------------------------

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        color,
        2
    )


    # --------------------------------------------------------
    # DRAW LABEL
    # --------------------------------------------------------

    label = (
        f"SAM2 {class_name} "
        f"{confidence:.2f}"
    )

    cv2.putText(
        annotated,
        label,
        (
            x1,
            max(
                25,
                y1 - 8
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )

    processed += 1


# ============================================================
# SAVE RESULT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

success = cv2.imwrite(
    OUTPUT_IMAGE,
    annotated
)

if not success:

    raise RuntimeError(
        "Failed to save output image."
    )


# ============================================================
# VRAM INFORMATION
# ============================================================

allocated = (
    torch.cuda.memory_allocated(0)
    / (1024 ** 3)
)

reserved = (
    torch.cuda.memory_reserved(0)
    / (1024 ** 3)
)


# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 60)

print(
    "HYBRID TEST COMPLETE"
)

print("=" * 60)

print(
    f"YOLO detections processed by SAM 2: "
    f"{processed}"
)

print(
    f"VRAM allocated: "
    f"{allocated:.2f} GB"
)

print(
    f"VRAM reserved: "
    f"{reserved:.2f} GB"
)

print()

print(
    "Output:"
)

print(
    OUTPUT_IMAGE
)

print()

print("=" * 60)