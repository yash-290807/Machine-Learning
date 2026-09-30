import os

# ============================================================
# PADDLEOCR WINDOWS FIX
# ============================================================

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"


from pathlib import Path
import csv
import cv2

from ultralytics import YOLO
from paddleocr import TextRecognition


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    r"C:\AI Projects\HelmetViolationSystem"
)


# ============================================================
# PLATE MODEL
# ============================================================

PLATE_MODEL = (
    ROOT
    / "runs"
    / "plate_yolo11n"
    / "weights"
    / "best.pt"
)


# ============================================================
# DEMO VIDEO
# ============================================================

VIDEO_DIR = Path(
    r"C:\Users\chyas\Downloads\Telegram Desktop"
)

VIDEO_NAME = (
    "Motorcycles_moving_in_urban_traffic_20260912134417"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "demo_ocr_test"
)


# ============================================================
# SETTINGS
# ============================================================

# Process every 6th frame.
FRAME_STEP = 6

# License plate detector confidence.
PLATE_CONF = 0.25

# Maximum number of plate crops to save/test.
MAX_PLATE_CROPS = 20


# ============================================================
# FIND VIDEO
# ============================================================

def find_video():

    # --------------------------------------------------------
    # Try exact MP4 path first.
    # --------------------------------------------------------

    exact_path = (
        VIDEO_DIR
        / f"{VIDEO_NAME}.mp4"
    )

    if exact_path.exists():
        return exact_path


    # --------------------------------------------------------
    # If Windows is hiding the extension, look for the
    # filename with any extension.
    # --------------------------------------------------------

    matches = list(
        VIDEO_DIR.glob(
            f"{VIDEO_NAME}.*"
        )
    )

    for path in matches:

        if path.is_file():

            return path


    return None


# ============================================================
# EXTRACT OCR TEXT
# ============================================================

def extract_text(result):

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
            and
            "res" in data
        ):

            data = data["res"]


        if not isinstance(
            data,
            dict
        ):

            return "", 0.0


        text = str(
            data.get(
                "rec_text",
                ""
            )
        ).strip()


        score = float(
            data.get(
                "rec_score",
                0.0
            )
        )


        return (
            text,
            score
        )


    except Exception:

        return (
            "",
            0.0
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("DEMO VIDEO - PLATE YOLO + PADDLEOCR")
    print("=" * 72)


    # ========================================================
    # CHECK PLATE MODEL
    # ========================================================

    if not PLATE_MODEL.exists():

        raise FileNotFoundError(
            "Plate model not found:\n"
            f"{PLATE_MODEL}"
        )


    # ========================================================
    # FIND VIDEO
    # ========================================================

    video_path = find_video()


    if video_path is None:

        print()
        print("ERROR: Demo video not found.")
        print()
        print(
            "Expected folder:"
        )
        print(
            VIDEO_DIR
        )
        print()
        print(
            "Expected filename:"
        )
        print(
            VIDEO_NAME
        )
        print()
        print(
            "Run this command to verify the filename:"
        )
        print()
        print(
            f'dir "{VIDEO_DIR}\\{VIDEO_NAME}*"'
        )
        print()

        return


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # DISPLAY CONFIGURATION
    # ========================================================

    print()
    print(
        "Video :",
        video_path
    )

    print(
        "Model :",
        PLATE_MODEL
    )

    print(
        "Output:",
        OUTPUT_DIR
    )


    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video:\n"
            f"{video_path}"
        )


    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )


    )


    print()
    print(
        f"Video: "
        f"{width}x{height} | "
        f"{fps:.2f} FPS | "
        f"{total_frames} frames"
    )


    # ========================================================
    # LOAD YOLO
    # ========================================================

    print()
    print(
        "Loading plate YOLO..."
    )


    plate_model = YOLO(
        str(PLATE_MODEL)
    )


    print(
        "Plate YOLO loaded."
    )


    print(
        "Classes:",
        plate_model.names
    )


    # ========================================================
    # LOAD PADDLEOCR
    # ========================================================

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
    print("STARTING OCR TEST")
    print("=" * 72)


    # ========================================================
    # COUNTERS
    # ========================================================

    frame_index = 0

    saved_crops = 0

    plate_detections = 0

    ocr_successes = 0

    results_rows = []


    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

        ok, frame = cap.read()


        if not ok:

            break


        # ----------------------------------------------------
        # Process every Nth frame.
        # ----------------------------------------------------

        if frame_index % FRAME_STEP != 0:

            frame_index += 1

            continue


        # ====================================================
        # PLATE DETECTION
        # ====================================================

        result = plate_model.predict(
            source=frame,
            conf=PLATE_CONF,
            imgsz=640,
            device=0,
            verbose=False
        )[0]


        # ----------------------------------------------------
        # No detections.
        # ----------------------------------------------------

        if result.boxes is None:

            frame_index += 1

            continue


        boxes = (
            result
            .boxes
            .xyxy
            .cpu()
            .tolist()
        )


        confs = (
            result
            .boxes
            .conf
            .cpu()
            .tolist()
        )


        # ====================================================
        # PROCESS DETECTED PLATES
        # ====================================================

        for detection_index, (
            box,
            detection_conf
        ) in enumerate(
            zip(
                boxes,
                confs
            ),
            start=1
        ):


            # ------------------------------------------------
            # BOX COORDINATES
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ------------------------------------------------
            # CLAMP COORDINATES
            # ------------------------------------------------

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )


            # ------------------------------------------------
            # INVALID BOX
            # ------------------------------------------------

            if (
                x2 <= x1
                or
                y2 <= y1
            ):

                continue


            # ------------------------------------------------
            # CROP PLATE
            # ------------------------------------------------

            crop = frame[
                y1:y2,
                x1:x2
            ]


            if crop.size == 0:

                continue


            plate_detections += 1


            # =================================================
            # UPSCALE
            # =================================================

            crop_upscaled = cv2.resize(
                crop,
                None,
                fx=6,
                fy=6,
                interpolation=cv2.INTER_CUBIC
            )


            # =================================================
            # SAVE PLATE CROP
            # =================================================

            if (
                saved_crops
                <
                MAX_PLATE_CROPS
            ):

                crop_path = (
                    OUTPUT_DIR
                    /
                    (
                        f"frame_"
                        f"{frame_index:05d}"
                        f"_plate_"
                        f"{saved_crops + 1:02d}"
                        f".jpg"
                    )
                )


                cv2.imwrite(
                    str(crop_path),
                    crop_upscaled
                )

            else:

                crop_path = None


            # =================================================
            # OCR
            # =================================================

            best_text = ""

            best_score = 0.0


            # -------------------------------------------------
            # Convert to grayscale.
            # -------------------------------------------------

            gray = cv2.cvtColor(
                crop_upscaled,
                cv2.COLOR_BGR2GRAY
            )


            # -------------------------------------------------
            # OCR variants.
            # -------------------------------------------------

            variants = [

                (
                    "color",
                    crop_upscaled
                ),

                (
                    "gray",
                    gray
                ),

            ]


            # =================================================
            # RUN OCR
            # =================================================

            for (
                variant_name,
                variant_image
            ) in variants:


                temp_path = (
                    OUTPUT_DIR
                    /
                    f"_ocr_{variant_name}.jpg"
                )


                cv2.imwrite(
                    str(temp_path),
                    variant_image
                )


                try:

                    ocr_results = ocr.predict(
                        input=str(temp_path)
                    )


                    for ocr_result in ocr_results:

                        text, score = (
                            extract_text(
                                ocr_result
                            )
                        )


                        if (
                            text
                            and
                            score > best_score
                        ):

                            best_text = text

                            best_score = score


                except Exception as exc:

                    print(
                        f"Frame "
                        f"{frame_index:04d} | "
                        f"OCR ERROR: "
                        f"{exc!r}"
                    )


            # =================================================
            # PRINT RESULT
            # =================================================

            if best_text:

                ocr_successes += 1


                print(
                    f"Frame "
                    f"{frame_index:04d} | "
                    f"Plate "
                    f"{detection_index} | "
                    f"det="
                    f"{float(detection_conf):.3f} | "
                    f"OCR="
                    f"{best_text!r} | "
                    f"score="
                    f"{best_score:.3f}"
                )


            else:

                print(
                    f"Frame "
                    f"{frame_index:04d} | "
                    f"Plate "
                    f"{detection_index} | "
                    f"det="
                    f"{float(detection_conf):.3f} | "
                    f"OCR=NO TEXT"
                )


            # =================================================
            # STORE RESULT
            # =================================================

            results_rows.append([
                frame_index,
                detection_index,
                f"{float(detection_conf):.3f}",
                best_text,
                f"{best_score:.3f}",
            ])


            # =================================================
            # CROP COUNT
            # =================================================

            if (
                saved_crops
                <
                MAX_PLATE_CROPS
            ):

                saved_crops += 1


            # -------------------------------------------------
            # Stop after enough crops.
            # -------------------------------------------------

            if (
                saved_crops
                >=
                MAX_PLATE_CROPS
            ):

                break


        # -----------------------------------------------------
        # Stop once max crops have been tested.
        # -----------------------------------------------------

        if (
            saved_crops
            >=
            MAX_PLATE_CROPS
        ):

            break


        frame_index += 1


    # ========================================================
    # RELEASE VIDEO
    # ========================================================

    cap.release()


    # ========================================================
    # SAVE OCR RESULTS CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        /
        "ocr_results.csv"
    )


    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow([
            "frame",
            "plate_index",
            "plate_detection_confidence",
            "ocr_text",
            "ocr_confidence",
        ])


        writer.writerows(
            results_rows
        )


    # ========================================================
    # REMOVE TEMPORARY FILES
    # ========================================================

    for temp_name in [
        "_ocr_color.jpg",
        "_ocr_gray.jpg",
    ]:

        temp_path = (
            OUTPUT_DIR
            /
            temp_name
        )


        if temp_path.exists():

            try:

                temp_path.unlink()

            except OSError:

                pass


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("DEMO OCR TEST COMPLETE")
    print("=" * 72)


    print(
        "Plate detections tested :",
        plate_detections
    )


    print(
        "Non-empty OCR results   :",
        ocr_successes
    )


    if plate_detections > 0:

        rate = (
            100.0
            *
            ocr_successes
            /
            plate_detections
        )


        print(
            "OCR success rate        :",
            f"{rate:.1f}%"
        )

    else:

        print(
            "OCR success rate        :",
            "0.0%"
        )


    print()
    print(
        "Plate crops:"
    )

    print(
        OUTPUT_DIR
    )


    print()
    print(
        "OCR results CSV:"
    )

    print(
        csv_path
    )


    print()
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()