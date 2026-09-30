import cv2
import os

INPUT_DIR = r"test_frames"
OUTPUT_DIR = r"test_crops"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(INPUT_DIR, filename)

    image = cv2.imread(path)

    if image is None:
        continue

    height, width = image.shape[:2]

    # Crop the central traffic area
    crop = image[
        int(height * 0.10):int(height * 0.90),
        int(width * 0.10):int(width * 0.90)
    ]

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cv2.imwrite(output_path, crop)

    print(f"Created: {output_path}")

print("\nCropping completed!")