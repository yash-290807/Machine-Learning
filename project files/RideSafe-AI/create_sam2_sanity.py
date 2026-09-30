from pathlib import Path
import shutil

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"C:\AI Projects\HelmetViolationSystem")

SOURCE_DIR = PROJECT_DIR / "sam2_instance_training"

IMAGE_DIR = SOURCE_DIR / "images"
MASK_DIR = SOURCE_DIR / "masks"
METADATA_DIR = SOURCE_DIR / "metadata"

SPLIT_DIR = SOURCE_DIR / "splits"

SANITY_DIR = PROJECT_DIR / "sam2_sanity"

SANITY_IMAGES = SANITY_DIR / "images"
SANITY_MASKS = SANITY_DIR / "masks"
SANITY_METADATA = SANITY_DIR / "metadata"

# ============================================================
# SETTINGS
# ============================================================

SANITY_COUNT = 100

# ============================================================
# START
# ============================================================

print("=" * 65)
print("SAM2 SANITY DATASET CREATION")
print("=" * 65)

print()

# ============================================================
# READ TRAINING LIST
# ============================================================

train_list = SPLIT_DIR / "train.txt"

if not train_list.exists():
    raise FileNotFoundError(
        f"Training list not found:\n{train_list}"
    )

train_files = [
    x.strip()
    for x in train_list.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

print("Training images available:", len(train_files))

if len(train_files) < SANITY_COUNT:
    raise RuntimeError(
        "Not enough training images for sanity dataset."
    )

# ============================================================
# SELECT FIRST 100
# ============================================================

selected_files = train_files[:SANITY_COUNT]

print("Sanity images selected:", len(selected_files))
print()

# ============================================================
# CREATE DIRECTORIES
# ============================================================

SANITY_IMAGES.mkdir(
    parents=True,
    exist_ok=True
)

SANITY_MASKS.mkdir(
    parents=True,
    exist_ok=True
)

SANITY_METADATA.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# COPY DATA
# ============================================================

copied_images = 0
copied_masks = 0
copied_metadata = 0

for filename in selected_files:

    image_path = IMAGE_DIR / filename
    mask_path = MASK_DIR / f"{Path(filename).stem}.png"
    metadata_path = METADATA_DIR / f"{Path(filename).stem}.json"

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if not image_path.exists():
        print("WARNING: Missing image:", filename)
        continue

    shutil.copy2(
        image_path,
        SANITY_IMAGES / filename
    )

    copied_images += 1

    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    if not mask_path.exists():
        print("WARNING: Missing mask:", mask_path)
        continue

    shutil.copy2(
        mask_path,
        SANITY_MASKS / mask_path.name
    )

    copied_masks += 1

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    if not metadata_path.exists():
        print(
            "WARNING: Missing metadata:",
            metadata_path
        )
        continue

    shutil.copy2(
        metadata_path,
        SANITY_METADATA / metadata_path.name
    )

    copied_metadata += 1

# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 65)
print("SANITY DATASET CREATION COMPLETE")
print("=" * 65)

print("Images copied:", copied_images)
print("Masks copied:", copied_masks)
print("Metadata copied:", copied_metadata)

print()
print("Sanity dataset:")
print(SANITY_DIR)

print()
print("=" * 65)