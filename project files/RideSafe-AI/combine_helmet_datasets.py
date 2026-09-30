from pathlib import Path
import shutil

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\AI Projects\HelmetViolationSystem")

HELMET_FINAL = PROJECT_ROOT / "datasets" / "helmet_final"
D3 = Path(r"C:\D3")

OUTPUT = PROJECT_ROOT / "datasets" / "helmet_combined"

# ============================================================
# SETTINGS
# ============================================================

# helmet_final already uses:
# 0 = helmet
# 1 = no_helmet

# D3 uses:
# 3 = helmet
# 4 = no_helmet
#
# D3 classes 0, 1, 2 are discarded.

D3_CLASS_MAP = {
    3: 0,
    4: 1,
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_dirs(split):
    (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_helmet_final(split):
    src_images = HELMET_FINAL / split / "images"
    src_labels = HELMET_FINAL / split / "labels"

    if not src_images.exists():
        print(f"[WARNING] Missing images folder: {src_images}")
        return 0

    if not src_labels.exists():
        print(f"[WARNING] Missing labels folder: {src_labels}")
        return 0

    count = 0

    for image_path in sorted(src_images.iterdir()):
        if not image_path.is_file():
            continue

        label_path = src_labels / f"{image_path.stem}.txt"

        # We only copy images that have labels.
        if not label_path.exists():
            print(f"[WARNING] Missing label for: {image_path.name}")
            continue

        new_name = f"helmet_final_{image_path.name}"

        shutil.copy2(
            image_path,
            OUTPUT / split / "images" / new_name
        )

        shutil.copy2(
            label_path,
            OUTPUT / split / "labels" / f"helmet_final_{image_path.stem}.txt"
        )

        count += 1

    return count


def process_d3_train():
    src_images = D3 / "train" / "images"
    src_labels = D3 / "train" / "labels"

    if not src_images.exists():
        raise FileNotFoundError(f"Missing D3 images folder: {src_images}")

    if not src_labels.exists():
        raise FileNotFoundError(f"Missing D3 labels folder: {src_labels}")

    count_images = 0
    count_with_useful_labels = 0
    count_discarded = 0
    total_kept_boxes = 0

    output_split = OUTPUT / "train"

    for image_path in sorted(src_images.iterdir()):
        if not image_path.is_file():
            continue

        label_path = src_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"[WARNING] Missing label: {image_path.name}")
            continue

        count_images += 1

        useful_lines = []

        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    print(
                        f"[WARNING] Invalid annotation in "
                        f"{label_path.name}: {line}"
                    )
                    continue

                try:
                    cls = int(parts[0])
                except ValueError:
                    print(
                        f"[WARNING] Invalid class in "
                        f"{label_path.name}: {line}"
                    )
                    continue

                if cls not in D3_CLASS_MAP:
                    # Discard D3 classes 0, 1, 2
                    continue

                new_cls = D3_CLASS_MAP[cls]

                useful_lines.append(
                    f"{new_cls} {parts[1]} {parts[2]} "
                    f"{parts[3]} {parts[4]}"
                )

        # If the image contains no helmet/no_helmet objects,
        # don't include it in the combined training dataset.
        if not useful_lines:
            count_discarded += 1
            continue

        new_name = f"D3_{image_path.name}"
        new_label_name = f"D3_{image_path.stem}.txt"

        shutil.copy2(
            image_path,
            output_split / "images" / new_name
        )

        with (
            output_split / "labels" / new_label_name
        ).open("w", encoding="utf-8") as f:
            f.write("\n".join(useful_lines) + "\n")

        count_with_useful_labels += 1
        total_kept_boxes += len(useful_lines)

    return (
        count_images,
        count_with_useful_labels,
        count_discarded,
        total_kept_boxes,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COMBINING HELMET DATASETS")
    print("=" * 70)

    if OUTPUT.exists():
        raise FileExistsError(
            f"\nOutput folder already exists:\n{OUTPUT}\n\n"
            "Delete it manually if you want to run this script again."
        )

    # Create output directories.
    make_dirs("train")
    make_dirs("val")
    make_dirs("test")

    # --------------------------------------------------------
    # Copy helmet_final
    # --------------------------------------------------------

    print("\n[1/2] Copying helmet_final...")

    train_count = copy_helmet_final("train")
    val_count = copy_helmet_final("val")
    test_count = copy_helmet_final("test")

    print(f"helmet_final train: {train_count}")
    print(f"helmet_final val:   {val_count}")
    print(f"helmet_final test:  {test_count}")

    # --------------------------------------------------------
    # Process D3
    # --------------------------------------------------------

    print("\n[2/2] Processing D3...")
    print("D3 class 3 -> helmet (0)")
    print("D3 class 4 -> no_helmet (1)")
    print("D3 classes 0,1,2 -> discarded")

    (
        d3_total,
        d3_kept,
        d3_discarded,
        d3_boxes,
    ) = process_d3_train()

    print(f"\nD3 total images:              {d3_total}")
    print(f"D3 images with useful labels: {d3_kept}")
    print(f"D3 images discarded:          {d3_discarded}")
    print(f"D3 useful bounding boxes:     {d3_boxes}")

    # --------------------------------------------------------
    # Create data.yaml
    # --------------------------------------------------------

    data_yaml = OUTPUT / "data.yaml"

    yaml_text = """path: C:/AI Projects/HelmetViolationSystem/datasets/helmet_combined
train: train/images
val: val/images
test: test/images

nc: 2

names:
  0: helmet
  1: no_helmet
"""

    data_yaml.write_text(yaml_text, encoding="utf-8")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMBINATION COMPLETE")
    print("=" * 70)

    print(f"\nOutput dataset:")
    print(OUTPUT)

    print("\nStructure:")
    print("helmet_combined/")
    print("├── data.yaml")
    print("├── train/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("├── val/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("└── test/")
    print("    ├── images/")
    print("    └── labels/")

    print("\nOriginal datasets were NOT modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()