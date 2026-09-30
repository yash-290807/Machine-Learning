from pathlib import Path
import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


PROJECT_ROOT = Path(r"C:\AI Projects\HelmetViolationSystem")

DATASET = PROJECT_ROOT / "datasets" / "helmet_combined"

IMAGE_DIR = DATASET / "train" / "images"
LABEL_DIR = DATASET / "train" / "labels"

OUTPUT_DIR = PROJECT_ROOT / "sam2_mask_test_helmet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = PROJECT_ROOT / "sam2_repo" / "checkpoints" / "sam2.1_hiera_small.pt"

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"


print("=" * 70)
print("SAM2 HELMET MASK TEST")
print("=" * 70)

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))


# ------------------------------------------------------------
# Load SAM2
# ------------------------------------------------------------

print("\nLoading SAM2.1 Hiera Small...")

model = build_sam2(
    MODEL_CFG,
    str(CHECKPOINT),
    device="cuda",
)

predictor = SAM2ImagePredictor(model)

print("SAM2 MODEL LOADED")


# ------------------------------------------------------------
# Find an image containing class 0 = helmet
# ------------------------------------------------------------

print("\nSearching for a helmet image...")

helmet_image = None
helmet_boxes = []

for image_path in sorted(IMAGE_DIR.iterdir()):

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    boxes = []

    with label_path.open("r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls = int(parts[0])

            # We specifically want helmet = class 0
            if cls != 0:
                continue

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            boxes.append(
                [
                    x_center,
                    y_center,
                    box_width,
                    box_height,
                ]
            )

    if boxes:
        helmet_image = image_path
        helmet_boxes = boxes
        break


if helmet_image is None:
    raise RuntimeError("Could not find any helmet (class 0) image.")


print("\nFOUND HELMET IMAGE:")
print(helmet_image.name)

print("Helmet objects:", len(helmet_boxes))


# ------------------------------------------------------------
# Read image
# ------------------------------------------------------------

image_bgr = cv2.imread(str(helmet_image))

if image_bgr is None:
    raise RuntimeError("Could not read image.")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

height, width = image_rgb.shape[:2]


# ------------------------------------------------------------
# Convert YOLO boxes to XYXY
# ------------------------------------------------------------

sam_boxes = []

for box in helmet_boxes:

    x_center, y_center, box_width, box_height = box

    x1 = (x_center - box_width / 2) * width
    y1 = (y_center - box_height / 2) * height

    x2 = (x_center + box_width / 2) * width
    y2 = (y_center + box_height / 2) * height

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    sam_boxes.append(
        np.array(
            [x1, y1, x2, y2],
            dtype=np.float32,
        )
    )


# ------------------------------------------------------------
# Run SAM2
# ------------------------------------------------------------

with torch.inference_mode():

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):

        predictor.set_image(image_rgb)

        overlay = image_bgr.copy()

        for index, box in enumerate(sam_boxes):

            print(
                f"\nObject {index + 1}"
            )

            print(
                "Box:",
                box.astype(int).tolist()
            )

            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=True,
                return_logits=False,
            )

            best_index = int(np.argmax(scores))

            mask = masks[best_index]

            print(
                "SAM2 scores:",
                [float(s) for s in scores]
            )

            print(
                "Best score:",
                float(scores[best_index])
            )

            # ------------------------------------------------
            # Save binary mask
            # ------------------------------------------------

            mask_uint8 = (
                mask.astype(np.uint8) * 255
            )

            mask_path = (
                OUTPUT_DIR /
                f"{helmet_image.stem}_helmet_mask.png"
            )

            cv2.imwrite(
                str(mask_path),
                mask_uint8,
            )

            # ------------------------------------------------
            # Bright green visualization
            # ------------------------------------------------

            green = np.zeros_like(image_bgr)
            green[:, :, 1] = 255

            colored_mask = cv2.bitwise_and(
                green,
                green,
                mask=mask_uint8,
            )

            overlay = cv2.addWeighted(
                overlay,
                1.0,
                colored_mask,
                0.60,
                0,
            )

            # Bounding box

            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            cv2.putText(
                overlay,
                "HELMET",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )


# ------------------------------------------------------------
# Save result
# ------------------------------------------------------------

overlay_path = (
    OUTPUT_DIR /
    f"{helmet_image.stem}_helmet_overlay.jpg"
)

cv2.imwrite(
    str(overlay_path),
    overlay,
)


print("\n" + "=" * 70)
print("HELMET MASK TEST COMPLETE")
print("=" * 70)

print("\nOriginal image:")
print(helmet_image)

print("\nMask:")
print(mask_path)

print("\nOverlay:")
print(overlay_path)

print("\nOpen the overlay image and inspect the GREEN mask.")
print("=" * 70)