import os

# Prevent the PaddlePaddle OneDNN/PIR runtime problem we encountered.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

from pathlib import Path
import csv
import re
import cv2
from ultralytics import YOLO
from paddleocr import TextRecognition


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(r"C:\AI Projects\HelmetViolationSystem")

HELMET_MODEL = (
    ROOT
    / "runs"
    / "helmet_yolo11n-3"
    / "weights"
    / "best.pt"
)

PLATE_MODEL = (
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
    / "final_pipeline"
)

EVIDENCE_DIR = OUTPUT_DIR / "evidence"
PLATE_CROP_DIR = OUTPUT_DIR / "plate_crops"

CSV_PATH = OUTPUT_DIR / "violation_records.csv"


# ============================================================
# DETECTION SETTINGS
# ============================================================

HELMET_CONF = 0.25
PLATE_CONF = 0.35

# OCR confidence above this is considered readable.
OCR_MIN_CONF = 0.55

# Minimum distance between saved evidence frames.
EVIDENCE_FRAME_GAP = 8


# ============================================================
# KNOWN CLASS MAPPINGS
# ============================================================

# Your helmet model was verified as:
# 0 = helmet
# 1 = not_helmet

HELMET_CLASS_HELMET = 0
HELMET_CLASS_NO_HELMET = 1


# ============================================================
# DIRECTORY SETUP
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVIDENCE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLATE_CROP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_plate_text(text):
    """
    Remove spaces and non-alphanumeric characters.
    """
    text = str(text).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


def looks_like_indian_plate(text):
    """
    Loose plausibility check for an Indian registration number.

    This is NOT a legal validation system.
    It is only used to reject obvious OCR garbage.
    """

    text = clean_plate_text(text)

    if not text:
        return False

    patterns = [
        r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$",
        r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{1,4}$",
        r"^[A-Z]{2}\d{1,2}\d{4,6}$",
    ]

    for pattern in patterns:

        if re.match(
            pattern,
            text
        ):
            return True

    return False


# ============================================================
# BOX HELPERS
# ============================================================

def box_center(box):
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def box_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    intersection = iw * ih

    area_a = max(
        0.0,
        ax2 - ax1
    ) * max(
        0.0,
        ay2 - ay1
    )

    area_b = max(
        0.0,
        bx2 - bx1
    ) * max(
        0.0,
        by2 - by1
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# OCR RESULT EXTRACTION
# ============================================================

def extract_ocr_result(result):

    try:

        data = result.json

        if callable(data):
            data = data()

        if isinstance(
            data,
            str
        ):

            import json

            data = json.loads(
                data
            )

        if (
            isinstance(data, dict)
            and "res" in data
        ):

            data = data["res"]

        if not isinstance(
            data,
            dict
        ):

            return "", 0.0

        text = data.get(
            "rec_text",
            ""
        )

        score = data.get(
            "rec_score",
            0.0
        )

        text = clean_plate_text(
            text
        )

        return (
            text,
            float(score)
        )

    except Exception:

        return "", 0.0


# ============================================================
# PLATE OCR
# ============================================================

def recognize_plate(
    ocr,
    crop
):

    if crop is None:
        return "", 0.0, ""

    if crop.size == 0:
        return "", 0.0, ""

    height, width = crop.shape[:2]

    if width < 25 or height < 10:
        return "", 0.0, ""

    # --------------------------------------------
    # SMALL BORDER CROP
    # --------------------------------------------

    pad_x = max(
        1,
        int(width * 0.02)
    )

    pad_y = max(
        1,
        int(height * 0.05)
    )

    x1 = pad_x
    y1 = pad_y

    x2 = max(
        x1 + 1,
        width - pad_x
    )

    y2 = max(
        y1 + 1,
        height - pad_y
    )

    crop = crop[
        y1:y2,
        x1:x2
    ]

    # --------------------------------------------
    # UPSCALE
    # --------------------------------------------

    upscaled = cv2.resize(
        crop,
        None,
        fx=6,
        fy=6,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------
    # GRAYSCALE
    # --------------------------------------------

    gray = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------
    # CLAHE
    # --------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    variants = {
        "color": upscaled,
        "gray": gray,
        "enhanced": enhanced,
    }

    candidates = []

    temp_dir = OUTPUT_DIR / "_ocr_temp"

    temp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for variant_name, variant_image in variants.items():

        temp_path = (
            temp_dir
            / f"{variant_name}.jpg"
        )

        cv2.imwrite(
            str(temp_path),
            variant_image
        )

        try:

            results = ocr.predict(
                input=str(temp_path)
            )

            for result in results:

                text, score = extract_ocr_result(
                    result
                )

                if not text:
                    continue

                # Give plausible Indian plate strings
                # a small preference.
                bonus = (
                    0.12
                    if looks_like_indian_plate(text)
                    else 0.0
                )

                candidates.append(
                    (
                        score + bonus,
                        text,
                        score,
                        variant_name
                    )
                )

        except Exception:

            continue

    if not candidates:

        return "", 0.0, ""

    candidates.sort(
        reverse=True
    )

    (
        _,
        best_text,
        best_score,
        best_variant
    ) = candidates[0]

    return (
        best_text,
        best_score,
        best_variant
    )


# ============================================================
# FIND PLATE NEAREST TO NO-HELMET DETECTION
# ============================================================

def find_related_plate(
    nohelmet_box,
    plate_detections,
    image_shape
):

    if not plate_detections:
        return None

    hx, hy = box_center(
        nohelmet_box
    )

    image_height, image_width = (
        image_shape[:2]
    )

    best_plate = None
    best_score = float("inf")

    for plate in plate_detections:

        px, py = box_center(
            plate["box"]
        )

        dx = px - hx
        dy = py - hy

        distance = (
            dx * dx
            +
            dy * dy
        ) ** 0.5

        # Don't associate completely unrelated
        # objects from far away.
        max_distance = (
            0.45
            * (
                image_width
                +
                image_height
            )
            / 2
        )

        if distance > max_distance:
            continue

        # Prefer plates below the person's head.
        vertical_penalty = 0.0

        if py < hy:
            vertical_penalty = 150.0

        # Penalize extremely far horizontal positions.
        horizontal_penalty = (
            abs(dx) * 0.15
        )

        # IoU is unlikely for head vs plate,
        # but if regions overlap, it's useful.
        overlap = box_iou(
            nohelmet_box,
            plate["box"]
        )

        score = (
            distance
            + vertical_penalty
            + horizontal_penalty
            - overlap * 250
        )

        if score < best_score:

            best_score = score
            best_plate = plate

    return best_plate


# ============================================================
# DRAW TEXT
# ============================================================

def put_label(
    image,
    text,
    position,
    color,
    scale=0.55
):

    x, y = position

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("FINAL HELMET + LICENSE PLATE + OCR PIPELINE")
    print("=" * 72)

    # --------------------------------------------
    # CHECK FILES
    # --------------------------------------------

    if not HELMET_MODEL.exists():

        raise FileNotFoundError(
            f"Helmet model not found:\n{HELMET_MODEL}"
        )

    if not PLATE_MODEL.exists():

        raise FileNotFoundError(
            f"Plate model not found:\n{PLATE_MODEL}"
        )

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"CCTV image directory not found:\n{IMAGE_DIR}"
        )

    # --------------------------------------------
    # FIND CCTV FRAMES
    # --------------------------------------------

    images = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    if not images:

        raise RuntimeError(
            f"No JPG images found in:\n{IMAGE_DIR}"
        )

    print()
    print(
        "CCTV frames:",
        len(images)
    )

    print()
    print(
        "Helmet model:",
        HELMET_MODEL
    )

    print(
        "Plate model:",
        PLATE_MODEL
    )

    # --------------------------------------------
    # LOAD HELMET MODEL
    # --------------------------------------------

    print()
    print("Loading helmet YOLO...")

    helmet_model = YOLO(
        str(HELMET_MODEL)
    )

    print(
        "Helmet classes:",
        helmet_model.names
    )

    # --------------------------------------------
    # LOAD PLATE MODEL
    # --------------------------------------------

    print()
    print("Loading plate YOLO...")

    plate_model = YOLO(
        str(PLATE_MODEL)
    )

    print(
        "Plate classes:",
        plate_model.names
    )

    # --------------------------------------------
    # LOAD OCR
    # --------------------------------------------

    print()
    print(
        "Loading PaddleOCR recognition model..."
    )

    ocr = TextRecognition(
        engine="paddle"
    )

    print(
        "PaddleOCR loaded."
    )

    print()
    print("=" * 72)

    # --------------------------------------------
    # CSV STORAGE
    # --------------------------------------------

    records = []

    total_helmet = 0
    total_nohelmet = 0
    total_plates = 0
    total_readable_plates = 0

    evidence_saved = 0

    last_evidence_frame = -999999

    # ========================================================
    # PROCESS CCTV
    # ========================================================

    for frame_index, image_path in enumerate(
        images
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        annotated = image.copy()

        # ====================================================
        # HELMET DETECTION
        # ====================================================

        helmet_result = helmet_model.predict(
            source=image,
            conf=HELMET_CONF,
            imgsz=640,
            device=0,
            verbose=False
        )[0]

        helmet_detections = []
        nohelmet_detections = []

        if helmet_result.boxes is not None:

            boxes = (
                helmet_result
                .boxes
                .xyxy
                .cpu()
                .tolist()
            )

            classes = (
                helmet_result
                .boxes
                .cls
                .cpu()
                .tolist()
            )

            confs = (
                helmet_result
                .boxes
                .conf
                .cpu()
                .tolist()
            )

            for box, class_id, confidence in zip(
                boxes,
                classes,
                confs
            ):

                box = [
                    float(value)
                    for value in box
                ]

                class_id = int(
                    class_id
                )

                confidence = float(
                    confidence
                )

                detection = {
                    "box": box,
                    "confidence": confidence
                }

                if class_id == HELMET_CLASS_HELMET:

                    helmet_detections.append(
                        detection
                    )

                elif class_id == HELMET_CLASS_NO_HELMET:

                    nohelmet_detections.append(
                        detection
                    )

        total_helmet += len(
            helmet_detections
        )

        total_nohelmet += len(
            nohelmet_detections
        )

        # ====================================================
        # PLATE DETECTION
        # ====================================================

        plate_result = plate_model.predict(
            source=image,
            conf=PLATE_CONF,
            imgsz=640,
            device=0,
            verbose=False
        )[0]

        plate_detections = []

        if plate_result.boxes is not None:

            boxes = (
                plate_result
                .boxes
                .xyxy
                .cpu()
                .tolist()
            )

            confs = (
                plate_result
                .boxes
                .conf
                .cpu()
                .tolist()
            )

            for box, confidence in zip(
                boxes,
                confs
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

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

                if x2 <= x1 or y2 <= y1:
                    continue

                width = x2 - x1
                height = y2 - y1

                # Reject tiny detections.
                if width < 20 or height < 10:
                    continue

                aspect = (
                    width
                    /
                    max(1, height)
                )

                # Basic plate-shape filter.
                if aspect < 1.0 or aspect > 8.0:
                    continue

                crop = image[
                    y1:y2,
                    x1:x2
                ]

                # Save a raw crop for inspection.
                crop_name = (
                    f"{image_path.stem}"
                    f"_plate_{len(plate_detections)+1}.jpg"
                )

                crop_path = (
                    PLATE_CROP_DIR
                    / crop_name
                )

                cv2.imwrite(
                    str(crop_path),
                    crop
                )

                plate_text = ""
                ocr_score = 0.0
                ocr_variant = ""

                # OCR only when there is enough
                # visual information to attempt it.
                plate_text, ocr_score, ocr_variant = (
                    recognize_plate(
                        ocr,
                        crop
                    )
                )

                readable = (
                    bool(plate_text)
                    and
                    ocr_score >= OCR_MIN_CONF
                )

                if readable:
                    total_readable_plates += 1

                plate_detections.append({
                    "box": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],
                    "confidence": float(confidence),
                    "text": plate_text,
                    "ocr_score": float(ocr_score),
                    "ocr_variant": ocr_variant,
                    "readable": readable,
                    "crop_path": str(crop_path),
                })

        total_plates += len(
            plate_detections
        )

        # ====================================================
        # DRAW NORMAL HELMET DETECTIONS
        # ====================================================

        for detection in helmet_detections:

            x1, y1, x2, y2 = map(
                int,
                detection["box"]
            )

            confidence = detection[
                "confidence"
            ]

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 200, 0),
                2
            )

            put_label(
                annotated,
                f"HELMET {confidence:.2f}",
                (
                    x1,
                    max(
                        22,
                        y1 - 6
                    )
                ),
                (0, 255, 0)
            )

        # ====================================================
        # DRAW PLATES
        # ====================================================

        for plate in plate_detections:

            x1, y1, x2, y2 = map(
                int,
                plate["box"]
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            if plate["readable"]:

                plate_label = (
                    f"{plate['text']} "
                    f"({plate['ocr_score']:.2f})"
                )

            else:

                plate_label = (
                    "PLATE UNCLEAR"
                )

            put_label(
                annotated,
                plate_label,
                (
                    x1,
                    min(
                        image.shape[0] - 5,
                        max(
                            22,
                            y1 - 6
                        )
                    )
                ),
                (255, 255, 0)
            )

        # ====================================================
        # PROCESS NO-HELMET VIOLATIONS
        # ====================================================

        frame_violation = False

        for violation_index, detection in enumerate(
            nohelmet_detections,
            start=1
        ):

            frame_violation = True

            nohelmet_box = detection[
                "box"
            ]

            nohelmet_conf = detection[
                "confidence"
            ]

            x1, y1, x2, y2 = map(
                int,
                nohelmet_box
            )

            # -----------------------------------------------
            # DRAW NO-HELMET BOX
            # -----------------------------------------------

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            put_label(
                annotated,
                f"NO HELMET {nohelmet_conf:.2f}",
                (
                    x1,
                    max(
                        22,
                        y1 - 8
                    )
                ),
                (0, 0, 255),
                0.60
            )

            # -----------------------------------------------
            # FIND RELATED PLATE
            # -----------------------------------------------

            related_plate = (
                find_related_plate(
                    nohelmet_box,
                    plate_detections,
                    image.shape
                )
            )

            if related_plate is not None:

                px1, py1, px2, py2 = map(
                    int,
                    related_plate["box"]
                )

                # Draw connection.
                hx, hy = box_center(
                    nohelmet_box
                )

                px, py = box_center(
                    related_plate["box"]
                )

                cv2.line(
                    annotated,
                    (
                        int(hx),
                        int(hy)
                    ),
                    (
                        int(px),
                        int(py)
                    ),
                    (0, 255, 255),
                    2
                )

                if related_plate["readable"]:

                    plate_value = (
                        related_plate["text"]
                    )

                    plate_ocr_conf = (
                        related_plate["ocr_score"]
                    )

                else:

                    plate_value = (
                        "PLATE UNCLEAR"
                    )

                    plate_ocr_conf = (
                        related_plate["ocr_score"]
                    )

                # -------------------------------------------
                # DRAW RELATED PLATE
                # -------------------------------------------

                cv2.rectangle(
                    annotated,
                    (px1, py1),
                    (px2, py2),
                    (255, 0, 0),
                    3
                )

                put_label(
                    annotated,
                    plate_value,
                    (
                        px1,
                        max(
                            22,
                            py1 - 8
                        )
                    ),
                    (255, 255, 0),
                    0.60
                )

            else:

                plate_value = (
                    "NO PLATE DETECTED"
                )

                plate_ocr_conf = 0.0

            # =================================================
            # RECORD VIOLATION
            # =================================================

            records.append([
                frame_index,
                image_path.name,
                "NO_HELMET",
                f"{nohelmet_conf:.3f}",
                plate_value,
                f"{plate_ocr_conf:.3f}",
            ])

        # ====================================================
        # HEADER
        # ====================================================

        cv2.rectangle(
            annotated,
            (0, 0),
            (
                min(
                    annotated.shape[1],
                    900
                ),
                50
            ),
            (0, 0, 0),
            -1
        )

        header = (
            f"NO HELMET: "
            f"{len(nohelmet_detections)}"
            f"  |  PLATES: "
            f"{len(plate_detections)}"
        )

        put_label(
            annotated,
            header,
            (10, 34),
            (255, 255, 255),
            0.75
        )

        # ====================================================
        # SAVE EVIDENCE
        # ====================================================

        if (
            frame_violation
            and
            (
                frame_index
                -
                last_evidence_frame
                >= EVIDENCE_FRAME_GAP
            )
        ):

            evidence_path = (
                EVIDENCE_DIR
                / f"violation_frame_{frame_index:04d}.jpg"
            )

            cv2.imwrite(
                str(evidence_path),
                annotated
            )

            evidence_saved += 1

            last_evidence_frame = (
                frame_index
            )

        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            (frame_index + 1) % 25 == 0
            or
            frame_index
            ==
            len(images) - 1
        ):

            print(
                f"Processed "
                f"{frame_index + 1}/"
                f"{len(images)} | "
                f"no_helmet="
                f"{total_nohelmet} | "
                f"plates="
                f"{total_plates} | "
                f"ocr_ok="
                f"{total_readable_plates}"
            )

    # ========================================================
    # WRITE CSV
    # ========================================================

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "frame_index",
            "image",
            "violation",
            "no_helmet_confidence",
            "plate",
            "ocr_confidence",
        ])

        writer.writerows(
            records
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 72)
    print("FINAL PIPELINE COMPLETE")
    print("=" * 72)

    print(
        "Frames processed        :",
        len(images)
    )

    print(
        "Helmet detections       :",
        total_helmet
    )

    print(
        "No-helmet detections    :",
        total_nohelmet
    )

    print(
        "Plate detections        :",
        total_plates
    )

    print(
        "Readable OCR plates     :",
        total_readable_plates
    )

    print(
        "Violation records        :",
        len(records)
    )

    print(
        "Evidence images          :",
        evidence_saved
    )

    print()
    print(
        "Evidence folder:"
    )

    print(
        EVIDENCE_DIR
    )

    print()
    print(
        "Plate crops:"
    )

    print(
        PLATE_CROP_DIR
    )

    print()
    print(
        "Violation CSV:"
    )

    print(
        CSV_PATH
    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()