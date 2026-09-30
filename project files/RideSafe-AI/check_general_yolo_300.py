from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"C:\AI Projects\HelmetViolationSystem")
MODEL_PATH = ROOT / "yolo11n.pt"
IMAGE_DIR = ROOT / "datasets" / "cctv_training" / "images"

CONF = 0.25


def main():
    image_files = sorted(IMAGE_DIR.glob("*.jpg"))

    print("=" * 70)
    print("GENERAL YOLO 300-FRAME FEASIBILITY TEST")
    print("=" * 70)
    print(f"Model : {MODEL_PATH}")
    print(f"Images: {IMAGE_DIR}")
    print(f"Frames: {len(image_files)}")
    print(f"Conf  : {CONF}")
    print("=" * 70)

    if len(image_files) != 300:
        raise RuntimeError(
            f"Expected 300 JPG images, found {len(image_files)}"
        )

    model = YOLO(str(MODEL_PATH))

    total_person = 0
    total_motorcycle = 0

    frames_with_motorcycle = 0
    frames_with_motorcycle_and_people = 0

    for i, image_path in enumerate(image_files, start=1):

        result = model.predict(
            source=str(image_path),
            conf=CONF,
            verbose=False
        )[0]

        person_count = 0
        motorcycle_count = 0

        if result.boxes is not None:
            for cls_id in result.boxes.cls.tolist():

                cls_id = int(cls_id)

                # COCO classes:
                # 0 = person
                # 3 = motorcycle

                if cls_id == 0:
                    person_count += 1

                elif cls_id == 3:
                    motorcycle_count += 1

        total_person += person_count
        total_motorcycle += motorcycle_count

        if motorcycle_count > 0:
            frames_with_motorcycle += 1

            if person_count > 0:
                frames_with_motorcycle_and_people += 1

        if i % 25 == 0 or i == 300:
            print(
                f"Processed {i:3d}/300 | "
                f"persons={total_person:4d} | "
                f"motorcycles={total_motorcycle:3d}"
            )

    avg_person = total_person / 300
    avg_motorcycle = total_motorcycle / 300

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"Total frames                     : 300")
    print(
        f"Frames with motorcycles         : "
        f"{frames_with_motorcycle}"
    )
    print(
        f"Frames with motorcycle + people : "
        f"{frames_with_motorcycle_and_people}"
    )
    print(
        f"Total person detections         : "
        f"{total_person}"
    )
    print(
        f"Total motorcycle detections     : "
        f"{total_motorcycle}"
    )
    print(
        f"Average people / frame          : "
        f"{avg_person:.2f}"
    )
    print(
        f"Average motorcycles / frame     : "
        f"{avg_motorcycle:.2f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()