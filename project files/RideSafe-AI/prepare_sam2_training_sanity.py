from pathlib import Path
import shutil
import json


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"C:\AI Projects\HelmetViolationSystem")

SOURCE_DIR = PROJECT_DIR / "sam2_instance_training"

IMAGE_DIR = SOURCE_DIR / "images"
MASK_DIR = SOURCE_DIR / "masks"

TRAIN_LIST = SOURCE_DIR / "splits" / "train.txt"

OUTPUT_DIR = PROJECT_DIR / "sam2_training_sanity"

JPEG_DIR = OUTPUT_DIR / "JPEGImages"
ANNOTATION_DIR = OUTPUT_DIR / "Annotations"

FILE_LIST = OUTPUT_DIR / "sanity_train.txt"


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_COUNT = 12
FRAMES_PER_SEQUENCE = 8

TOTAL_FRAMES = SEQUENCE_COUNT * FRAMES_PER_SEQUENCE


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SAM2 TRAINING SANITY DATASET")
print("=" * 70)

print()
print("Source:", SOURCE_DIR)
print("Output:", OUTPUT_DIR)
print()

# ============================================================
# READ TRAIN LIST
# ============================================================

if not TRAIN_LIST.exists():
    raise FileNotFoundError(
        f"Train list not found:\n{TRAIN_LIST}"
    )

train_files = [
    x.strip()
    for x in TRAIN_LIST.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

print("Training images available:", len(train_files))
print("Frames required:", TOTAL_FRAMES)

if len(train_files) < TOTAL_FRAMES:
    raise RuntimeError(
        "Not enough training images."
    )

# ============================================================
# SELECT FIRST 96
# ============================================================

selected_files = train_files[:TOTAL_FRAMES]

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

JPEG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ANNOTATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# CREATE SEQUENCES
# ============================================================

sequence_names = []

for seq_index in range(SEQUENCE_COUNT):

    sequence_name = f"sanity_seq_{seq_index:02d}"

    sequence_names.append(sequence_name)

    image_sequence_dir = JPEG_DIR / sequence_name
    mask_sequence_dir = ANNOTATION_DIR / sequence_name

    image_sequence_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_sequence_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    start = seq_index * FRAMES_PER_SEQUENCE
    end = start + FRAMES_PER_SEQUENCE

    sequence_files = selected_files[start:end]

    print()
    print(
        f"{sequence_name}: "
        f"{len(sequence_files)} frames"
    )

    if len(sequence_files) != FRAMES_PER_SEQUENCE:
        raise RuntimeError(
            f"Incorrect frame count in {sequence_name}"
        )

    # --------------------------------------------------------
    # COPY FRAMES
    # --------------------------------------------------------

    for frame_index, filename in enumerate(sequence_files):

        source_image = IMAGE_DIR / filename
        source_mask = MASK_DIR / f"{Path(filename).stem}.png"

        if not source_image.exists():
            raise FileNotFoundError(
                f"Missing image:\n{source_image}"
            )

        if not source_mask.exists():
            raise FileNotFoundError(
                f"Missing mask:\n{source_mask}"
            )

        frame_name = f"{frame_index:05d}"

        # Image
        output_image = (
            image_sequence_dir /
            f"{frame_name}{source_image.suffix.lower()}"
        )

        shutil.copy2(
            source_image,
            output_image
        )

        # Annotation
        output_mask = (
            mask_sequence_dir /
            f"{frame_name}.png"
        )

        shutil.copy2(
            source_mask,
            output_mask
        )


# ============================================================
# SAVE FILE LIST
# ============================================================

FILE_LIST.write_text(
    "\n".join(sequence_names),
    encoding="utf-8"
)


# ============================================================
# SAVE DATASET INFORMATION
# ============================================================

info = {
    "sequence_count": SEQUENCE_COUNT,
    "frames_per_sequence": FRAMES_PER_SEQUENCE,
    "total_frames": TOTAL_FRAMES,
    "source_train_images": len(train_files),
    "num_frames_for_sanity_training": 2,
    "max_num_objects_for_sanity_training": 3
}

with open(
    OUTPUT_DIR / "dataset_info.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        info,
        f,
        indent=2
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

print()
print("=" * 70)
print("VERIFYING DATASET")
print("=" * 70)

total_images = 0
total_masks = 0

for sequence_name in sequence_names:

    image_dir = JPEG_DIR / sequence_name
    mask_dir = ANNOTATION_DIR / sequence_name

    images = list(image_dir.glob("*"))
    images = [
        f for f in images
        if f.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png"
        ]
    ]

    masks = list(mask_dir.glob("*.png"))

    print(
        f"{sequence_name}: "
        f"images={len(images)}, "
        f"masks={len(masks)}"
    )

    if len(images) != FRAMES_PER_SEQUENCE:
        raise RuntimeError(
            f"Wrong image count in {sequence_name}"
        )

    if len(masks) != FRAMES_PER_SEQUENCE:
        raise RuntimeError(
            f"Wrong mask count in {sequence_name}"
        )

    total_images += len(images)
    total_masks += len(masks)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("SAM2 TRAINING SANITY DATASET READY")
print("=" * 70)

print("Sequences:", SEQUENCE_COUNT)
print("Images:", total_images)
print("Masks:", total_masks)
print("Frames per sequence:", FRAMES_PER_SEQUENCE)

print()
print("File list:")
print(FILE_LIST)

print()
print("Dataset:")
print(OUTPUT_DIR)

print()
print("=" * 70)