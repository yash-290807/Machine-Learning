from pathlib import Path
import shutil
import random
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\AI Projects\HelmetViolationSystem")

# Original datasets
DATASET_1 = PROJECT_ROOT / "datasets" / "helmet"
DATASET_2 = PROJECT_ROOT / "datasets" / "helmet_2"
DATASET_3 = Path(r"C:\D3")

# Final dataset
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "helmet_final"

# Reproducible random splitting for Dataset 3
random.seed(42)

# Final classes
# 0 = helmet
# 1 = no_helmet


# ============================================================
# ORIGINAL → FINAL CLASS MAPPINGS
# ============================================================

# Dataset 1
MAPPING_D1 = {
    "0": 0,   # Full-Faced        -> helmet
    "1": 0,   # Half-Faced        -> helmet
    "3": 1,   # Not Wearing      -> no_helmet
    "4": 0,   # motorcycle-helmet -> helmet

    # Class 2 = Invalid          -> EXCLUDE
    # Class 5 = motorcycle-rider -> EXCLUDE
}

# Dataset 2
MAPPING_D2 = {
    "0": 0,   # With Helmet       -> helmet
    "1": 1,   # Without Helmet    -> no_helmet
}

# Dataset 3
MAPPING_D3 = {
    "3": 0,   # Helmet            -> helmet
    "4": 1,   # No Helmet         -> no_helmet

    # Classes 0, 1, 2 -> EXCLUDE
}


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

if OUTPUT_DIR.exists():
    print("=" * 70)
    print("WARNING: Final dataset already exists.")
    print("=" * 70)
    print(f"Location: {OUTPUT_DIR}")
    print()
    answer = input("Delete and recreate it? (yes/no): ").strip().lower()

    if answer != "yes":
        print("\nOperation cancelled.")
        raise SystemExit

    shutil.rmtree(OUTPUT_DIR)

for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FIND IMAGE FILES
# ============================================================

def get_images(images_dir):
    if not images_dir.exists():
        return []

    return sorted([
        p for p in images_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


# ============================================================
# FIND SPLITS
# ============================================================

def find_split_dir(dataset_root, split_name):

    possible_names = {
        "train": ["train"],
        "val": ["valid", "val", "validation"],
        "test": ["test"]
    }

    for name in possible_names[split_name]:

        candidate = dataset_root / name

        if candidate.exists():
            return candidate

    return None


# ============================================================
# CONVERT YOLO LABELS
# ============================================================

def convert_label(label_file, mapping):

    converted = []

    if not label_file.exists():
        return converted

    lines = label_file.read_text(
        encoding="utf-8",
        errors="ignore"
    ).splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Ignore comments
        if line.startswith("#"):
            continue

        parts = line.split()

        # Valid YOLO annotation:
        # class x_center y_center width height
        if len(parts) < 5:
            continue

        original_class = parts[0]

        # Ignore unwanted classes
        if original_class not in mapping:
            continue

        new_class = mapping[original_class]

        # Keep YOLO coordinates
        converted_line = (
            f"{new_class} "
            f"{parts[1]} "
            f"{parts[2]} "
            f"{parts[3]} "
            f"{parts[4]}"
        )

        converted.append(converted_line)

    return converted


# ============================================================
# COLLECT DATASET PAIRS
# ============================================================

def collect_dataset(dataset_name, dataset_root, mapping):

    print()
    print("=" * 70)
    print(f"COLLECTING {dataset_name}")
    print("=" * 70)

    collected = {
        "train": [],
        "val": [],
        "test": []
    }

    statistics = {
        "images_found": 0,
        "usable_images": 0,
        "missing_labels": 0,
        "empty_after_filter": 0,
        "annotations_kept": Counter(),
        "annotations_removed": Counter()
    }

    for split in ["train", "val", "test"]:

        split_dir = find_split_dir(
            dataset_root,
            split
        )

        if split_dir is None:

            print(f"{split.upper():5}: not found")

            continue

        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        images = get_images(images_dir)

        print(
            f"{split.upper():5}: "
            f"{len(images)} images found"
        )

        for image_path in images:

            statistics["images_found"] += 1

            label_path = (
                labels_dir
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():

                statistics["missing_labels"] += 1

                continue

            converted_labels = convert_label(
                label_path,
                mapping
            )

            # Count original labels
            original_lines = label_path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).splitlines()

            for line in original_lines:

                parts = line.strip().split()

                if not parts:
                    continue

                original_class = parts[0]

                if original_class in mapping:
                    statistics["annotations_kept"][
                        mapping[original_class]
                    ] += 1

                elif original_class.isdigit():
                    statistics["annotations_removed"][
                        int(original_class)
                    ] += 1

            # If nothing useful remains, exclude image
            if not converted_labels:

                statistics["empty_after_filter"] += 1

                continue

            statistics["usable_images"] += 1

            collected[split].append({
                "image": image_path,
                "labels": converted_labels
            })

    print(
        f"Usable images: "
        f"{statistics['usable_images']}"
    )

    print(
        f"Missing labels: "
        f"{statistics['missing_labels']}"
    )

    print(
        f"Images with no usable annotations: "
        f"{statistics['empty_after_filter']}"
    )

    return collected, statistics


# ============================================================
# COLLECT DATASET 1
# ============================================================

d1_data, d1_stats = collect_dataset(
    "DATASET 1 - KAGGLE",
    DATASET_1,
    MAPPING_D1
)


# ============================================================
# COLLECT DATASET 2
# ============================================================

d2_data, d2_stats = collect_dataset(
    "DATASET 2 - ROBOFLOW",
    DATASET_2,
    MAPPING_D2
)


# ============================================================
# COLLECT DATASET 3
# ============================================================

d3_data, d3_stats = collect_dataset(
    "DATASET 3 - ROBOFLOW",
    DATASET_3,
    MAPPING_D3
)


# ============================================================
# DATASET 3 ONLY HAS TRAIN
# SPLIT IT INTO 80 / 10 / 10
# ============================================================

d3_train = d3_data["train"]

random.shuffle(d3_train)

d3_total = len(d3_train)

d3_train_end = int(d3_total * 0.80)

d3_val_end = (
    d3_train_end
    + int(d3_total * 0.10)
)

d3_data["train"] = d3_train[:d3_train_end]

d3_data["val"] = d3_train[
    d3_train_end:d3_val_end
]

d3_data["test"] = d3_train[
    d3_val_end:
]

print()
print("=" * 70)
print("DATASET 3 SPLIT")
print("=" * 70)

print(
    f"Train: {len(d3_data['train'])}"
)

print(
    f"Val:   {len(d3_data['val'])}"
)

print(
    f"Test:  {len(d3_data['test'])}"
)


# ============================================================
# COPY DATA INTO FINAL DATASET
# ============================================================

final_statistics = {
    "train": 0,
    "val": 0,
    "test": 0,

    "helmet": 0,
    "no_helmet": 0,

    "Dataset 1": 0,
    "Dataset 2": 0,
    "Dataset 3": 0
}


def copy_dataset_items(
    dataset_items,
    final_split,
    dataset_prefix,
    dataset_name
):

    for index, item in enumerate(dataset_items):

        image_path = item["image"]

        labels = item["labels"]

        # Unique filename
        new_stem = (
            f"{dataset_prefix}_"
            f"{final_split}_"
            f"{index:06d}"
        )

        new_image_path = (
            OUTPUT_DIR
            / final_split
            / "images"
            / f"{new_stem}{image_path.suffix.lower()}"
        )

        new_label_path = (
            OUTPUT_DIR
            / final_split
            / "labels"
            / f"{new_stem}.txt"
        )

        # Copy image
        shutil.copy2(
            image_path,
            new_image_path
        )

        # Write converted label
        new_label_path.write_text(
            "\n".join(labels) + "\n",
            encoding="utf-8"
        )

        final_statistics[final_split] += 1

        final_statistics[dataset_name] += 1

        # Count final classes
        for label in labels:

            class_id = int(
                label.split()[0]
            )

            if class_id == 0:
                final_statistics["helmet"] += 1

            elif class_id == 1:
                final_statistics["no_helmet"] += 1


# ============================================================
# COPY DATASET 1
# ============================================================

for split in ["train", "val", "test"]:

    copy_dataset_items(
        d1_data[split],
        split,
        "d1",
        "Dataset 1"
    )


# ============================================================
# COPY DATASET 2
# ============================================================

for split in ["train", "val", "test"]:

    copy_dataset_items(
        d2_data[split],
        split,
        "d2",
        "Dataset 2"
    )


# ============================================================
# COPY DATASET 3
# ============================================================

for split in ["train", "val", "test"]:

    copy_dataset_items(
        d3_data[split],
        split,
        "d3",
        "Dataset 3"
    )


# ============================================================
# CREATE data.yaml
# ============================================================

yaml_content = """path: C:/AI Projects/HelmetViolationSystem/datasets/helmet_final

train: train/images
val: val/images
test: test/images

nc: 2

names:
  0: helmet
  1: no_helmet
"""

(OUTPUT_DIR / "data.yaml").write_text(
    yaml_content,
    encoding="utf-8"
)


# ============================================================
# CREATE DATASET REPORT
# ============================================================

report = []

report.append("=" * 70)
report.append("HELMET FINAL DATASET REPORT")
report.append("=" * 70)
report.append("")

report.append("FINAL CLASSES")
report.append("0 = helmet")
report.append("1 = no_helmet")
report.append("")

report.append("FINAL IMAGE COUNTS")
report.append(
    f"Train: {final_statistics['train']}"
)
report.append(
    f"Val:   {final_statistics['val']}"
)
report.append(
    f"Test:  {final_statistics['test']}"
)

total_final = (
    final_statistics["train"]
    + final_statistics["val"]
    + final_statistics["test"]
)

report.append(
    f"Total: {total_final}"
)

report.append("")

report.append("FINAL ANNOTATION COUNTS")
report.append(
    f"Helmet:    {final_statistics['helmet']}"
)
report.append(
    f"No Helmet: {final_statistics['no_helmet']}"
)

report.append("")

report.append("SOURCE CONTRIBUTION")

report.append(
    f"Dataset 1: {final_statistics['Dataset 1']}"
)

report.append(
    f"Dataset 2: {final_statistics['Dataset 2']}"
)

report.append(
    f"Dataset 3: {final_statistics['Dataset 3']}"
)

report.append("")

report.append("IMPORTANT")
report.append(
    "Original datasets were not modified."
)

report.append(
    "Dataset 3 was split into train/val/test."
)

report.append(
    "Classes excluded during preprocessing:"
)

report.append(
    "D1 class 2 = Invalid"
)

report.append(
    "D1 class 5 = motorcycle-rider"
)

report.append(
    "D3 classes 0, 1, 2 = excluded"
)

report.append("")

report.append(
    f"Output directory: {OUTPUT_DIR}"
)

(OUTPUT_DIR / "dataset_report.txt").write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("DATASET PREPROCESSING COMPLETE")
print("=" * 70)

print()

print(
    f"Train images : "
    f"{final_statistics['train']}"
)

print(
    f"Val images   : "
    f"{final_statistics['val']}"
)

print(
    f"Test images  : "
    f"{final_statistics['test']}"
)

print(
    f"Total images : "
    f"{total_final}"
)

print()

print(
    f"Helmet annotations    : "
    f"{final_statistics['helmet']}"
)

print(
    f"No-helmet annotations : "
    f"{final_statistics['no_helmet']}"
)

print()

print("SOURCE CONTRIBUTION")

print(
    f"Dataset 1 : "
    f"{final_statistics['Dataset 1']}"
)

print(
    f"Dataset 2 : "
    f"{final_statistics['Dataset 2']}"
)

print(
    f"Dataset 3 : "
    f"{final_statistics['Dataset 3']}"
)

print()

print(
    "Final dataset created at:"
)

print(OUTPUT_DIR)

print()

print(
    "Original datasets were NOT modified."
)

print(
    "Preprocessing finished successfully!"
)
