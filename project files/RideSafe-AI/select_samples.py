import os
import random
import shutil

# Prediction folder
source_dir = r"C:\AI Projects\HelmetViolationSystem\runs\detect\predict"

# Folder where selected samples will be copied
output_dir = r"C:\AI Projects\HelmetViolationSystem\runs\detect\selected_samples"

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# Get all prediction images
extensions = (".jpg", ".jpeg", ".png", ".webp")

images = [
    f for f in os.listdir(source_dir)
    if f.lower().endswith(extensions)
]

print(f"Found {len(images)} prediction images.")

if len(images) == 0:
    print("No prediction images found!")
    exit()

# Select up to 9 random images
sample_count = min(9, len(images))
selected = random.sample(images, sample_count)

# Copy them
for i, filename in enumerate(selected, 1):
    source = os.path.join(source_dir, filename)
    destination = os.path.join(output_dir, f"sample_{i:02d}_{filename}")

    shutil.copy2(source, destination)

print("\nSelected images:")
for filename in selected:
    print(" -", filename)

print("\nDone!")
print("Samples saved to:")
print(output_dir)