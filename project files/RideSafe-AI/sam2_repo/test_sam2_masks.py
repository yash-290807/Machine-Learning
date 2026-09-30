from pathlib import Path
import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\AI Projects\HelmetViolationSystem")

DATASET = PROJECT_ROOT / "datasets" / "helmet_combined"

IMAGE_DIR = DATASET / "train" / "images"
LABEL_DIR = DATASET / "train" / "labels"

OUTPUT_DIR = PROJECT_ROOT / "sam2_mask_test"

CHECKPOINT = PROJECT_ROOT / "sam2_repo" / "checkpoints" / "sam2.1_hiera_small.pt"

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"


# ============================================================
# SETTINGS
# ============================================================

NUM_IMAGES = 10

# We use the highest-quality mask returned by SAM2.
MULTIMASK_OUTPUT = True


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASK_DIR = OUTPUT_DIR / "masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK CUDA
# ============================================================

print("=" * 70)
print("SAM2 MASK GENERATION TEST")
print("=" * 70)

print("\nCUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD SAM2
# ============================================================

print("\nLoading SAM2.1 Hiera Small...")

model = build_sam2(
    MODEL_CFG,
    str(CHECKPOINT),
    device="cuda",
)

predictor = SAM2ImagePredictor(model)

print("SAM2 MODEL LOADED")


# ============================================================
# FIND 10 IMAGES
# ============================================================

images = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
])[:NUM_IMAGES]

if len(images) == 0:
    raise RuntimeError("No images found.")

print(f"\nTesting {len(images)} images...")


# ============================================================
# PROCESS IMAGES
# ============================================================

for index, image_path in enumerate(images, start=1):

    print("\n" + "-" * 70)
    print(f"[{index}/{len(images)}] {image_path.name}")

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        print("WARNING: label file missing. Skipping.")
        continue

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        print("WARNING: could not read image. Skipping.")
        continue

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    height, width = image_rgb.shape[:2]

    # --------------------------------------------------------
    # Read YOLO boxes
    # --------------------------------------------------------

    boxes = []

    with label_path.open("r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # YOLO normalized coordinates -> pixel coordinates

            x1 = (x_center - box_width / 2) * width
            y1 = (y_center - box_height / 2) * height

            x2 = (x_center + box_width / 2) * width
            y2 = (y_center + box_height / 2) * height

            # Clamp to image boundaries

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append(
                (
                    cls,
                    np.array(
                        [x1, y1, x2, y2],
                        dtype=np.float32,
                    ),
                )
            )

    if not boxes:
        print("WARNING: no valid boxes. Skipping.")
        continue

    print("Objects:", len(boxes))

    # --------------------------------------------------------
    # Set image
    # --------------------------------------------------------

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            predictor.set_image(image_rgb)

            overlay = image_bgr.copy()

            for object_index, (cls, box) in enumerate(boxes):

                print(
                    f"  Object {object_index + 1}: "
                    f"class={cls}, "
                    f"box={box.astype(int).tolist()}"
                )

                # ------------------------------------------------
                # SAM2 box prompt
                # ------------------------------------------------

                masks, scores, _ = predictor.predict(
                    box=box,
                    multimask_output=MULTIMASK_OUTPUT,
                    return_logits=False,
                )

                # Select highest-scoring mask

                best_index = int(np.argmax(scores))

                mask = masks[best_index]

                # ------------------------------------------------
                # Save binary mask
                # ------------------------------------------------

                mask_uint8 = (
                    mask.astype(np.uint8) * 255
                )

                # Use a unique mask filename for every object

                mask_name = (
                    f"{image_path.stem}"
                    f"_obj{object_index + 1}"
                    f"_class{cls}.png"
                )

                cv2.imwrite(
                    str(MASK_DIR / mask_name),
                    mask_uint8,
                )

                # ------------------------------------------------
                # Create visualization overlay
                # ------------------------------------------------

                overlay_color = np.zeros_like(image_bgr)

                # Helmet = green
                # No helmet = red

                if cls == 0:
                    overlay_color[:, :, 1] = 255
                else:
                    overlay_color[:, :, 2] = 255

                colored_mask = cv2.bitwise_and(
                    overlay_color,
                    overlay_color,
                    mask=mask_uint8,
                )

                overlay = cv2.addWeighted(
                    overlay,
                    1.0,
                    colored_mask,
                    0.45,
                    0,
                )

                # Draw bounding box

                x1, y1, x2, y2 = box.astype(int)

                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2,
                )

                label_text = (
                    "helmet"
                    if cls == 0
                    else "no_helmet"
                )

                cv2.putText(
                    overlay,
                    label_text,
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                print(
                    f"    Best SAM2 score: "
                    f"{float(scores[best_index]):.4f}"
                )

            # ----------------------------------------------------
            # Save overlay
            # ----------------------------------------------------

            overlay_path = (
                OVERLAY_DIR /
                f"{image_path.stem}_overlay.jpg"
            )

            cv2.imwrite(
                str(overlay_path),
                overlay,
            )

            print(
                "  Saved:",
                overlay_path.name,
            )


print("\n" + "=" * 70)
print("MASK TEST COMPLETE")
print("=" * 70)

print("\nMasks:")
print(MASK_DIR)

print("\nOverlays:")
print(OVERLAY_DIR)

print("\nOpen the overlay images and inspect them.")
print("=" * 70)