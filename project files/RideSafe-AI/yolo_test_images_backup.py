from ultralytics import YOLO
from pathlib import Path
import cv2

print("=" * 60)
print("YOLO11n - NEW IMAGE TEST")
print("=" * 60)

# Model
model_path = r"C:\AI Projects\HelmetViolationSystem\runs\helmet_yolo11n-3\weights\best.pt"

# Test images
image_dir = Path(r"C:\AI Projects\HelmetViolationSystem\hybrid_test_images")

# Output folder
output_dir = Path(r"C:\AI Projects\HelmetViolationSystem\outputs\yolo_new_tests")
output_dir.mkdir(parents=True, exist_ok=True)

print("\nLoading YOLO11n...")
model = YOLO(model_path)
print("YOLO11n loaded successfully.")
print("Classes:", model.names)

images = sorted(image_dir.glob("*.png"))

print(f"\nImages found: {len(images)}")

for image_path in images:
    print("\n" + "=" * 60)
    print(f"PROCESSING: {image_path.name}")
    print("=" * 60)

    image = cv2.imread(str(image_path))

    if image is None:
        print("ERROR: Could not load image.")
        continue

    print(f"Image size: {image.shape[1]} x {image.shape[0]}")

    print("\nRunning YOLO11n detection...")

    results = model.predict(
        source=image,
        conf=0.15,
        device=0,
        verbose=False
    )

    result = results[0]

    print(f"YOLO detections: {len(result.boxes)}")

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_name = model.names[cls_id]

        print(
            f"Detection {i+1}: "
            f"{class_name} "
            f"confidence={confidence:.3f} "
            f"box=({x1},{y1},{x2},{y2})"
        )

    # Draw detections
    annotated = result.plot()

    output_path = output_dir / f"{image_path.stem}_yolo.png"

    cv2.imwrite(str(output_path), annotated)

    print(f"\nOutput: {output_path}")

print("\n" + "=" * 60)
print("YOLO NEW IMAGE TEST COMPLETE")
print("=" * 60)
print(f"Results saved in:")
print(output_dir)