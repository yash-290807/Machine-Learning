from pathlib import Path
import shutil
import json
import numpy as np
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"C:\AI Projects\HelmetViolationSystem")

SOURCE_DIR = PROJECT_DIR / "sam2_sanity"

SOURCE_IMAGES = SOURCE_DIR / "images"
SOURCE_MASKS = SOURCE_DIR / "masks"
SOURCE_METADATA = SOURCE_DIR / "metadata"

VOS_DIR = PROJECT_DIR / "sam2_sanity_vos"

JPEG_DIR = VOS_DIR / "JPEGImages"
ANNOTATION_DIR = VOS_DIR / "Annotations"

FILE_LIST = VOS_DIR / "sanity_train.txt"


# ============================================================
# SETTINGS
# ============================================================

FRAME_NAME = "00000.jpg"
MASK_NAME = "00000.png"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SAM2 SANITY DATASET -> VOS FORMAT")
print("=" * 70)

print()
print("Source:", SOURCE_DIR)
print("Output:", VOS_DIR)
print()


# ============================================================
# CREATE DIRECTORIES
# ============================================================

JPEG_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND SOURCE IMAGES
# ============================================================

image_files = sorted(SOURCE_IMAGES.glob("*"))

image_files = [
    f for f in image_files
    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

print("Source images:", len(image_files))


if len(image_files) != 100:
    raise RuntimeError(
        f"Expected 100 sanity images, found {len(image_files)}"
    )


# ============================================================
# PROCESS
# ============================================================

video_names = []

max_objects = 0
total_objects = 0

for index, image_path in enumerate(image_files):

    video_name = image_path.stem

    video_names.append(video_name)

    image_video_dir = JPEG_DIR / video_name
    mask_video_dir = ANNOTATION_DIR / video_name

    image_video_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_video_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SOURCE MASK
    # --------------------------------------------------------

    source_mask = SOURCE_MASKS / f"{video_name}.png"

    if not source_mask.exists():
        raise FileNotFoundError(
            f"Missing mask:\n{source_mask}"
        )

    # --------------------------------------------------------
    # SOURCE METADATA
    # --------------------------------------------------------

    metadata_file = SOURCE_METADATA / f"{video_name}.json"

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Missing metadata:\n{metadata_file}"
        )

    # --------------------------------------------------------
    # READ MASK
    # --------------------------------------------------------

    mask = np.array(
        Image.open(source_mask)
    )

    if mask.ndim != 2:
        raise RuntimeError(
            f"Mask is not single-channel:\n{source_mask}"
        )

    unique_values = np.unique(mask)

    if len(unique_values) == 0:
        raise RuntimeError(
            f"Empty mask:\n{source_mask}"
        )

    if unique_values[0] != 0:
        raise RuntimeError(
            f"Mask does not contain background value 0:\n"
            f"{source_mask}"
        )

    object_ids = unique_values[
        unique_values > 0
    ]

    object_count = len(object_ids)

    max_objects = max(
        max_objects,
        object_count
    )

    total_objects += object_count

    # --------------------------------------------------------
    # COPY IMAGE
    # --------------------------------------------------------

    output_image = image_video_dir / FRAME_NAME

    shutil.copy2(
        image_path,
        output_image
    )

    # --------------------------------------------------------
    # COPY MASK
    # --------------------------------------------------------

    output_mask = mask_video_dir / MASK_NAME

    shutil.copy2(
        source_mask,
        output_mask
    )

    # --------------------------------------------------------
    # VERIFY METADATA
    # --------------------------------------------------------

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    metadata_objects = metadata.get(
        "objects",
        []
    )

    if len(metadata_objects) != object_count:

        raise RuntimeError(
            f"Object count mismatch:\n"
            f"{video_name}\n"
            f"Mask objects: {object_count}\n"
            f"Metadata objects: {len(metadata_objects)}"
        )

    print(
        f"[{index + 1:03d}/100] "
        f"{video_name} -> "
        f"{object_count} objects"
    )


# ============================================================
# SAVE FILE LIST
# ============================================================

FILE_LIST.write_text(
    "\n".join(video_names),
    encoding="utf-8"
)


# ============================================================
# SAVE INFO
# ============================================================

info = {
    "images": len(image_files),
    "videos": len(video_names),
    "total_objects": total_objects,
    "max_objects_per_image": max_objects,
    "frame_name": FRAME_NAME,
    "mask_name": MASK_NAME
}

with open(
    VOS_DIR / "dataset_info.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        info,
        f,
        indent=2
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("VOS SANITY DATASET CREATED")
print("=" * 70)

print("Images:", len(image_files))
print("Sequences:", len(video_names))
print("Total objects:", total_objects)
print("Maximum objects in one image:", max_objects)

print()
print("JPEGImages:")
print(JPEG_DIR)

print()
print("Annotations:")
print(ANNOTATION_DIR)

print()
print("File list:")
print(FILE_LIST)

print()
print("=" * 70)