import os

# Disable the OneDNN/MKLDNN path that is causing the
# PaddlePaddle 3.3.1 + PaddleOCR Windows runtime error.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

from pathlib import Path

import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\AI Projects\HelmetViolationSystem")

MODEL = (
    ROOT
    / "runs"
    / "plate_yolo11n"
    / "weights"
    / "best.pt"
)

IMAGES = (
    ROOT
    / "datasets"
    / "cctv_training"
    / "images"
)

OUT = (
    ROOT
    / "outputs"
    / "plate_debug_crops"
)


# ============================================================
# SETTINGS
# ============================================================

PLATE_CONFIDENCE = 0.20
UPSCALE_FACTOR = 5
MAX_SAVED = 5


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL.exists():
    raise FileNotFoundError(
        f"Plate model not found:\n{MODEL}"
    )

if not IMAGES.exists():
    raise FileNotFoundError(
        f"CCTV image directory not found:\n{IMAGES}"
    )


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("PLATE YOLO + PADDLEOCR DEBUG TEST")
print("=" * 70)

print()
print("Plate model:")
print(MODEL)

print()
print("CCTV images:")
print(IMAGES)

print()
print("Output directory:")
print(OUT)

print()
print("Loading plate YOLO model...")

model = YOLO(
    str(MODEL)
)

print("Plate YOLO loaded successfully.")

print()
print("Loading PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    device="cpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False
)

print("PaddleOCR loaded successfully.")
print()


# ============================================================
# FIND IMAGES
# ============================================================

images = sorted(
    IMAGES.glob("*.jpg")
)

if not images:
    raise RuntimeError(
        f"No JPG images found in:\n{IMAGES}"
    )

print(
    f"CCTV images found: {len(images)}"
)

print()


# ============================================================
# PROCESS
# ============================================================

saved = 0

for image_path in images:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"Could not read image: {image_path.name}"
        )

        continue


    # --------------------------------------------------------
    # PLATE YOLO DETECTION
    # --------------------------------------------------------

    result = model.predict(
        source=image,
        conf=PLATE_CONFIDENCE,
        imgsz=640,
        device=0,
        verbose=False
    )[0]


    if result.boxes is None:
        continue


    # --------------------------------------------------------
    # PROCESS EACH PLATE
    # --------------------------------------------------------

    for j, box in enumerate(
        result.boxes.xyxy.cpu().tolist()
    ):

        x1, y1, x2, y2 = map(
            int,
            box
        )


        # ----------------------------------------------------
        # CLAMP COORDINATES
        # ----------------------------------------------------

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            image.shape[1],
            x2
        )

        y2 = min(
            image.shape[0],
            y2
        )


        # ----------------------------------------------------
        # VALID BOX
        # ----------------------------------------------------

        if x2 <= x1 or y2 <= y1:
            continue


        # ----------------------------------------------------
        # CROP PLATE
        # ----------------------------------------------------

        crop = image[
            y1:y2,
            x1:x2
        ]


        if crop.size == 0:
            continue


        # ----------------------------------------------------
        # UPSCALE
        # ----------------------------------------------------

        crop = cv2.resize(
            crop,
            None,
            fx=UPSCALE_FACTOR,
            fy=UPSCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC
        )


        # ----------------------------------------------------
        # SAVE CROP
        # ----------------------------------------------------

        filename = (
            OUT
            / f"{image_path.stem}_plate_{j + 1}.jpg"
        )

        cv2.imwrite(
            str(filename),
            crop
        )


        print("=" * 70)
        print("IMAGE :", image_path.name)
        print("CROP  :", filename)
        print("SIZE  :", crop.shape)
        print()


        # ----------------------------------------------------
        # PADDLEOCR TEST
        # ----------------------------------------------------

        try:

            ocr_results = ocr.predict(
                input=crop
            )


            got_text = False


            for res in ocr_results:

                print("RAW OCR RESULT:")
                print(res)

                got_text = True


            if not got_text:
                print(
                    "OCR completed but returned no text."
                )


        except Exception as e:

            print(
                "OCR ERROR:"
            )

            print(
                repr(e)
            )


        print()


        saved += 1


        # ----------------------------------------------------
        # STOP AFTER MAXIMUM NUMBER OF CROPS
        # ----------------------------------------------------

        if saved >= MAX_SAVED:
            break


    if saved >= MAX_SAVED:
        break


# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)

print(
    "Saved crops:",
    saved
)

print(
    "Folder:",
    OUT
)

print("=" * 70)