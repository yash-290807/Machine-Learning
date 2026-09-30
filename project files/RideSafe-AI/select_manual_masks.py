from pathlib import Path
import random
import shutil
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\AI Projects\HelmetViolationSystem")

DATASET = PROJECT_ROOT / "datasets" / "helmet_combined"

IMAGE_DIR = DATASET / "train" / "images"
LABEL_DIR = DATASET / "train" / "labels"

OUTPUT = PROJECT_ROOT / "manual_selection"

HELMET_OUTPUT = OUTPUT / "helmet_500" / "images"
NO_HELMET_OUTPUT = OUTPUT / "no_helmet_500" / "images"

# ============================================================
# SETTINGS
# ============================================================

NUM_HELMET = 500
NUM_NO_HELMET = 500

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

HELMET_OUTPUT.mkdir(parents=True, exist_ok=True)
NO_HELMET_OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ DATASET
# ============================================================

print("=" * 70)
print("SELECTING MANUAL SAM2 MASKING DATASET")
print("=" * 70)

print("\nDataset:")
print(DATASET)

print("\nScanning annotations...")

helmet_images = set()
no_helmet_images = set()

image_object_counts = {}

all_images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

for image_path in all_images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    helmet_count = 0
    no_helmet_count = 0

    with label_path.open("r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:
                cls = int(parts[0])
            except ValueError:
                continue

            if cls == 0:
                helmet_count += 1

            elif cls == 1:
                no_helmet_count += 1

    if helmet_count > 0:
        helmet_images.add(image_path)

    if no_helmet_count > 0:
        no_helmet_images.add(image_path)

    image_object_counts[image_path] = (
        helmet_count,
        no_helmet_count
    )


print("\nAvailable images:")
print("Helmet images:     ", len(helmet_images))
print("No-helmet images:  ", len(no_helmet_images))


# ============================================================
# SELECT HELMET IMAGES
# ============================================================

print("\nSelecting helmet images...")

helmet_candidates = list(helmet_images)

random.shuffle(helmet_candidates)

selected_helmet = []

for image_path in helmet_candidates:

    # Avoid selecting images that will later be used
    # in the no-helmet group.
    selected_helmet.append(image_path)

    if len(selected_helmet) >= NUM_HELMET:
        break


# ============================================================
# SELECT NO-HELMET IMAGES
# ============================================================

print("Selecting no-helmet images...")

selected_helmet_set = set(selected_helmet)

no_helmet_candidates = [
    p for p in no_helmet_images
    if p not in selected_helmet_set
]

random.shuffle(no_helmet_candidates)

selected_no_helmet = no_helmet_candidates[:NUM_NO_HELMET]


# ============================================================
# CHECK COUNTS
# ============================================================

if len(selected_helmet) < NUM_HELMET:
    raise RuntimeError(
        f"Only found {len(selected_helmet)} helmet images."
    )

if len(selected_no_helmet) < NUM_NO_HELMET:
    raise RuntimeError(
        f"Only found {len(selected_no_helmet)} no-helmet images."
    )


# ============================================================
# COPY IMAGES
# ============================================================

print("\nCopying helmet images...")

for image_path in selected_helmet:

    shutil.copy2(
        image_path,
        HELMET_OUTPUT / image_path.name
    )


print("Copying no-helmet images...")

for image_path in selected_no_helmet:

    shutil.copy2(
        image_path,
        NO_HELMET_OUTPUT / image_path.name
    )


# ============================================================
# SAVE LISTS
# ============================================================

helmet_list = OUTPUT / "helmet_500.txt"
no_helmet_list = OUTPUT / "no_helmet_500.txt"

helmet_list.write_text(
    "\n".join(str(p) for p in selected_helmet),
    encoding="utf-8"
)

no_helmet_list.write_text(
    "\n".join(str(p) for p in selected_no_helmet),
    encoding="utf-8"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SELECTION COMPLETE")
print("=" * 70)

print("\nHelmet images selected:     ", len(selected_helmet))
print("No-helmet images selected:  ", len(selected_no_helmet))

print("\nHelmet folder:")
print(HELMET_OUTPUT)

print("\nNo-helmet folder:")
print(NO_HELMET_OUTPUT)

print("\nOriginal dataset was NOT modified.")

print("\nSelection lists:")
print(helmet_list)
print(no_helmet_list)

print("=" * 70)