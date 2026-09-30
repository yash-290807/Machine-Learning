from ultralytics import YOLO
from pathlib import Path
import cv2

MODEL = r"C:\AI Projects\HelmetViolationSystem\runs\helmet_yolo11n-3\weights\best.pt"
INPUT_DIR = Path(r"C:\AI Projects\HelmetViolationSystem\hybrid_test_images")
OUTPUT_DIR = Path(r"C:\AI Projects\HelmetViolationSystem\outputs\yolo_baseline")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL)

print("=" * 60)
print("YOLO11n BASELINE TEST")
print("=" * 60)

for image_path in sorted(INPUT_DIR.iterdir()):

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        continue

    print(f"\nProcessing: {image_path.name}")

    image = cv2.imread(str(image_path))

    results = model.predict(
        source=image,
        conf=0.20,
        device=0,
        verbose=False
    )

    result = results[0]

    count = 0

    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = model.names[cls]

            print(
                f"  {count + 1}. "
                f"{label} "
                f"confidence={conf:.3f} "
                f"box=({x1},{y1},{x2},{y2})"
            )

            if label == "helmet":
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                image,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            count += 1

    output_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(output_path), image)

    print(f"Total detections: {count}")
    print(f"Saved: {output_path}")

print("\n" + "=" * 60)
print("YOLO BASELINE TEST COMPLETE")
print("=" * 60)
print(f"Results: {OUTPUT_DIR}")