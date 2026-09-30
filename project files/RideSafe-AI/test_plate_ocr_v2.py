import os

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

from pathlib import Path
import cv2

from ultralytics import YOLO
from paddleocr import PaddleOCR


ROOT = Path(r"C:\AI Projects\HelmetViolationSystem")

MODEL = (
    ROOT
    / "runs"
    / "plate_yolo11n"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = (
    ROOT
    / "datasets"
    / "cctv_training"
    / "images"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "plate_ocr_v2"
)

PLATE_CONF = 0.40
TOP_PLATES = 10


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("PLATE YOLO + PADDLEOCR V2")
print("=" * 70)

print("Loading plate detector...")

plate_model = YOLO(str(MODEL))

print("Plate detector loaded.")

print("Loading PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    device="cpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)

print("PaddleOCR loaded.")


# ============================================================
# FIND BEST PLATE DETECTIONS
# ============================================================

images = sorted(
    IMAGE_DIR.glob("*.jpg")
)

detections = []

print()
print("Scanning CCTV frames for strong plate detections...")
print()

for index, image_path in enumerate(images):

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    result = plate_model.predict(
        source=image,
        conf=PLATE_CONF,
        imgsz=640,
        device=0,
        verbose=False,
    )[0]

    if result.boxes is None:
        continue

    boxes = result.boxes.xyxy.cpu().tolist()
    confs = result.boxes.conf.cpu().tolist()

    for box, conf in zip(boxes, confs):

        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        w = x2 - x1
        h = y2 - y1

        # Reject extremely tiny detections.
        if w < 20 or h < 10:
            continue

        # License plates are normally wider than tall.
        aspect = w / max(h, 1)

        # Keep reasonable plate-like proportions.
        if aspect < 1.0 or aspect > 8.0:
            continue

        detections.append(
            (
                float(conf),
                image_path,
                (x1, y1, x2, y2)
            )
        )

    if (index + 1) % 50 == 0:
        print(
            f"Scanned {index + 1}/{len(images)}"
        )


# ============================================================
# SORT BY CONFIDENCE
# ============================================================

detections.sort(
    key=lambda x: x[0],
    reverse=True
)

best = detections[:TOP_PLATES]


print()
print("=" * 70)
print("TOP PLATE DETECTIONS")
print("=" * 70)

print(
    f"Candidates found: {len(detections)}"
)

print(
    f"Testing: {len(best)}"
)

print()


# ============================================================
# OCR
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

for rank, (
    det_conf,
    image_path,
    box
) in enumerate(
    best,
    start=1
):

    image = cv2.imread(
        str(image_path)
    )

    x1, y1, x2, y2 = box

    crop = image[
        y1:y2,
        x1:x2
    ]

    # Add a small border around the plate.
    pad_x = max(3, int(crop.shape[1] * 0.05))
    pad_y = max(3, int(crop.shape[0] * 0.15))

    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(image.shape[1], x2 + pad_x)
    py2 = min(image.shape[0], y2 + pad_y)

    crop = image[
        py1:py2,
        px1:px2
    ]

    # Upscale.
    crop = cv2.resize(
        crop,
        None,
        fx=6,
        fy=6,
        interpolation=cv2.INTER_CUBIC
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    # CLAHE.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    # Light sharpening.
    blur = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        3
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.5,
        blur,
        -0.5,
        0
    )

    variants = {
        "color": crop,
        "gray": gray,
        "enhanced": enhanced,
        "sharpened": sharpened,
    }

    print("=" * 70)
    print(
        f"#{rank} {image_path.name}"
    )

    print(
        f"YOLO confidence: {det_conf:.3f}"
    )

    print(
        f"Original plate box: "
        f"{x1},{y1},{x2},{y2}"
    )

    print(
        f"Crop size: {crop.shape}"
    )

    # Save all variants.
    for name, variant in variants.items():

        out = (
            OUTPUT_DIR
            / f"{rank:02d}_{name}.jpg"
        )

        cv2.imwrite(
            str(out),
            variant
        )

    # ========================================================
    # OCR EACH VARIANT
    # ========================================================

    for name, variant in variants.items():

        print()
        print(
            f"OCR variant: {name}"
        )

        try:

            results = ocr.predict(
                input=variant
            )

            found = False

            for res in results:

                print(res)

                # Print direct recognized text if available.
                try:
                    data = res.json

                    if callable(data):
                        data = data()

                    if isinstance(data, dict):

                        texts = data.get(
                            "rec_texts",
                            []
                        )

                        scores = data.get(
                            "rec_scores",
                            []
                        )

                        if texts:
                            print(
                                "TEXT:",
                                texts
                            )

                            print(
                                "SCORES:",
                                scores
                            )

                            found = True

                except Exception:
                    pass

            if not found:
                print(
                    "No recognized text."
                )

        except Exception as e:

            print(
                "OCR ERROR:",
                repr(e)
            )

    print()


print("=" * 70)
print("V2 TEST COMPLETE")
print("=" * 70)

print()
print(
    "Saved crops and preprocessing variants:"
)

print(
    OUTPUT_DIR
)

print("=" * 70)