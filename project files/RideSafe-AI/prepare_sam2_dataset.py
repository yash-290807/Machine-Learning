from pathlib import Path
import json
import random

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"C:\AI Projects\HelmetViolationSystem")

SOURCE_DIR = PROJECT_DIR / "sam2_instance_training"

IMAGE_DIR = SOURCE_DIR / "images"
MASK_DIR = SOURCE_DIR / "masks"
METADATA_DIR = SOURCE_DIR / "metadata"

SPLIT_DIR = SOURCE_DIR / "splits"

TRAIN_LIST = SPLIT_DIR / "train.txt"
VAL_LIST = SPLIT_DIR / "val.txt"

# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.90
RANDOM_SEED = 42

# ============================================================
# START
# ============================================================

print("=" * 65)
print("SAM2 DATASET SPLIT PREPARATION")
print("=" * 65)

print()
print("Source:", SOURCE_DIR)
print("Images:", IMAGE_DIR)
print("Masks:", MASK_DIR)
print("Metadata:", METADATA_DIR)
print()

# ============================================================
# CREATE SPLIT DIRECTORY
# ============================================================

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FIND IMAGES
# ============================================================

image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png"))
)

print("Images found:", len(image_files))

# ============================================================
# VERIFY IMAGE / MASK / METADATA PAIRS
# ============================================================

valid_files = []
missing_masks = []
missing_metadata = []

for image_path in image_files:

    mask_path = MASK_DIR / f"{image_path.stem}.png"
    metadata_path = METADATA_DIR / f"{image_path.stem}.json"

    if not mask_path.exists():
        missing_masks.append(image_path.name)
        continue

    if not metadata_path.exists():
        missing_metadata.append(image_path.name)
        continue

    valid_files.append(image_path.name)

print("Valid image/mask/metadata sets:", len(valid_files))
print("Missing masks:", len(missing_masks))
print("Missing metadata:", len(missing_metadata))

if missing_masks:
    print()
    print("First missing masks:")
    print(missing_masks[:10])

if missing_metadata:
    print()
    print("First missing metadata:")
    print(missing_metadata[:10])

if len(valid_files) == 0:
    raise RuntimeError("No valid dataset entries found.")

# ============================================================
# SHUFFLE
# ============================================================

random.seed(RANDOM_SEED)

random.shuffle(valid_files)

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_count = int(len(valid_files) * TRAIN_RATIO)

train_files = valid_files[:train_count]
val_files = valid_files[train_count:]

# ============================================================
# SAVE FILE LISTS
# ============================================================

TRAIN_LIST.write_text(
    "\n".join(train_files),
    encoding="utf-8"
)

VAL_LIST.write_text(
    "\n".join(val_files),
    encoding="utf-8"
)

# ============================================================
# SAVE SPLIT INFORMATION
# ============================================================

split_info = {
    "total_images": len(valid_files),
    "train_images": len(train_files),
    "validation_images": len(val_files),
    "train_ratio": TRAIN_RATIO,
    "random_seed": RANDOM_SEED,
    "train_list": str(TRAIN_LIST),
    "validation_list": str(VAL_LIST)
}

with open(SPLIT_DIR / "split_info.json", "w", encoding="utf-8") as f:
    json.dump(split_info, f, indent=2)

# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 65)
print("DATASET SPLIT COMPLETE")
print("=" * 65)

print("Total:", len(valid_files))
print("Training:", len(train_files))
print("Validation:", len(val_files))

print()
print("Train list:")
print(TRAIN_LIST)

print()
print("Validation list:")
print(VAL_LIST)

print()
print("=" * 65)