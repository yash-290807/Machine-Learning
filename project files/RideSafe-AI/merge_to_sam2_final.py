import json
import hashlib
import shutil
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\AI Projects\HelmetViolationSystem")

DATASETS = {
    "401": ROOT / "datasets" / "helmet_coco",
    "724": ROOT / "datasets" / "helmet_coco_724",
}

OUTPUT = ROOT / "datasets" / "sam2_helmet_final"

# ============================================================
# CLASS MAPPINGS
# ============================================================

# 401:
# 0 = helmet-object       -> ignore
# 1 = helmet              -> keep
# 2 = without-helmet      -> keep

# 724:
# 0 = helmet              -> keep
# 1 = bike                -> ignore
# 2 = with_helmet         -> keep
# 3 = without_helmet      -> keep

CLASS_MAP = {
    "401": {
        1: "helmet",
        2: "without-helmet",
    },
    "724": {
        0: "helmet",
        2: "helmet",
        3: "without-helmet",
    },
}

# Duplicate policy:
# 724 gets priority over 401.
#
# If an image exists in both datasets, keep the 724 copy.
#
# If the same image somehow appears in multiple splits,
# prefer train, then valid, then test to guarantee that
# the final dataset has no cross-split image leakage.

SPLIT_PRIORITY = {
    "train": 0,
    "valid": 1,
    "test": 2,
}

DATASET_PRIORITY = {
    "724": 0,
    "401": 1,
}

# ============================================================
# HELPERS
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_records(dataset_name, root):

    records = []

    for split in ["train", "valid", "test"]:

        split_dir = root / split
        json_path = split_dir / "_annotations.coco.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"Missing COCO annotation file:\n{json_path}"
            )

        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images = {
            image["id"]: image
            for image in coco["images"]
        }

        annotations_by_image = {}

        for ann in coco["annotations"]:

            category_id = ann["category_id"]

            if category_id not in CLASS_MAP[dataset_name]:
                continue

            annotations_by_image.setdefault(
                ann["image_id"], []
            ).append(ann)

        for image_id, image_info in images.items():

            useful_annotations = annotations_by_image.get(
                image_id, []
            )

            if not useful_annotations:
                continue

            image_path = split_dir / image_info["file_name"]

            if not image_path.exists():
                print(
                    f"[WARNING] Missing image: {image_path}"
                )
                continue

            records.append({
                "dataset": dataset_name,
                "split": split,
                "path": image_path,
                "file_name": image_info["file_name"],
                "image_info": image_info,
                "annotations": useful_annotations,
            })

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("401 + 724 -> FINAL SAM2 DATASET")
    print("=" * 70)

    if OUTPUT.exists():
        raise FileExistsError(
            f"\nOutput already exists:\n{OUTPUT}\n\n"
            "Delete it manually only if you intentionally "
            "want to regenerate it."
        )

    # --------------------------------------------------------
    # Load both datasets
    # --------------------------------------------------------

    print("\nLoading 401 dataset...")
    records_401 = load_records(
        "401",
        DATASETS["401"]
    )

    print(
        f"401 useful images: {len(records_401)}"
    )

    print("\nLoading 724 dataset...")
    records_724 = load_records(
        "724",
        DATASETS["724"]
    )

    print(
        f"724 useful images: {len(records_724)}"
    )

    all_records = records_401 + records_724

    # --------------------------------------------------------
    # Hash images
    # --------------------------------------------------------

    print("\nCalculating image hashes...")

    hash_groups = {}

    for record in all_records:

        image_hash = sha256(record["path"])

        record["hash"] = image_hash

        hash_groups.setdefault(
            image_hash,
            []
        ).append(record)

    duplicate_groups = [
        group
        for group in hash_groups.values()
        if len(group) > 1
    ]

    print(
        f"Duplicate groups found: "
        f"{len(duplicate_groups)}"
    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    final_records = []
    removed_records = []

    for image_hash, group in hash_groups.items():

        group_sorted = sorted(
            group,
            key=lambda r: (
                DATASET_PRIORITY[r["dataset"]],
                SPLIT_PRIORITY[r["split"]],
            )
        )

        keeper = group_sorted[0]

        final_records.append(keeper)

        for duplicate in group_sorted[1:]:

            removed_records.append(
                (keeper, duplicate)
            )

    # --------------------------------------------------------
    # Final ordering
    # --------------------------------------------------------

    final_records.sort(
        key=lambda r: (
            SPLIT_PRIORITY[r["split"]],
            DATASET_PRIORITY[r["dataset"]],
            r["file_name"].lower(),
        )
    )

    # --------------------------------------------------------
    # Print final counts
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MERGE SUMMARY")
    print("=" * 70)

    print(
        f"401 useful images       : {len(records_401)}"
    )

    print(
        f"724 useful images       : {len(records_724)}"
    )

    print(
        f"Before deduplication    : {len(all_records)}"
    )

    print(
        f"Duplicate groups        : {len(duplicate_groups)}"
    )

    print(
        f"Duplicate copies removed: {len(removed_records)}"
    )

    print(
        f"Final unique images     : {len(final_records)}"
    )

    print("\nFINAL SPLITS:")

    for split in ["train", "valid", "test"]:

        count = sum(
            r["split"] == split
            for r in final_records
        )

        print(
            f"  {split:5s}: {count}"
        )

    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    for split in ["train", "valid", "test"]:

        (
            OUTPUT
            / "JPEGImages"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            OUTPUT
            / "Annotations"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Global class definition
    # --------------------------------------------------------

    classes = {
        "1": "helmet",
        "2": "without-helmet",
    }

    with open(
        OUTPUT / "classes.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            classes,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    global_sequence_counter = 0

    split_mappings = {
        "train": {},
        "valid": {},
        "test": {},
    }

    total_objects = 0
    total_helmet = 0
    total_without_helmet = 0

    for split in ["train", "valid", "test"]:

        split_records = [
            r
            for r in final_records
            if r["split"] == split
        ]

        print(
            f"\nProcessing {split}: "
            f"{len(split_records)} images"
        )

        sequence_number = 0

        for record in split_records:

            source_image = record["path"]

            image_info = record["image_info"]

            width = image_info["width"]
            height = image_info["height"]

            sequence_name = (
                f"{split}_{sequence_number:06d}"
            )

            image_sequence_dir = (
                OUTPUT
                / "JPEGImages"
                / split
                / sequence_name
            )

            mask_sequence_dir = (
                OUTPUT
                / "Annotations"
                / split
                / sequence_name
            )

            image_sequence_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            mask_sequence_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # Copy image
            # ------------------------------------------------

            output_image = (
                image_sequence_dir / "00000.jpg"
            )

            shutil.copy2(
                source_image,
                output_image
            )

            # ------------------------------------------------
            # Create instance-ID mask
            # ------------------------------------------------

            from PIL import Image, ImageDraw

            mask = Image.new(
                "L",
                (width, height),
                0
            )

            draw = ImageDraw.Draw(mask)

            object_classes = {}

            object_id = 1

            # ------------------------------------------------
            # Process annotations
            # ------------------------------------------------

            for ann in record["annotations"]:

                segmentation = ann.get(
                    "segmentation"
                )

                if not isinstance(
                    segmentation,
                    list
                ):
                    print(
                        f"[WARNING] Non-polygon "
                        f"segmentation: "
                        f"{source_image.name}"
                    )
                    continue

                category_id = ann["category_id"]

                class_name = CLASS_MAP[
                    record["dataset"]
                ][category_id]

                valid_polygon = False

                for polygon in segmentation:

                    if not isinstance(
                        polygon,
                        list
                    ):
                        continue

                    if len(polygon) < 6:
                        continue

                    points = []

                    for i in range(
                        0,
                        len(polygon),
                        2
                    ):

                        x = polygon[i]
                        y = polygon[i + 1]

                        points.append(
                            (x, y)
                        )

                    if len(points) < 3:
                        continue

                    draw.polygon(
                        points,
                        fill=object_id
                    )

                    valid_polygon = True

                if not valid_polygon:

                    print(
                        f"[WARNING] Invalid polygon "
                        f"in {source_image.name}"
                    )

                    continue

                object_classes[
                    str(object_id)
                ] = {
                    "class_id": (
                        1
                        if class_name == "helmet"
                        else 2
                    ),
                    "class_name": class_name,
                    "source_dataset": record[
                        "dataset"
                    ],
                    "source_annotation_id": ann[
                        "id"
                    ],
                }

                total_objects += 1

                if class_name == "helmet":
                    total_helmet += 1
                else:
                    total_without_helmet += 1

                object_id += 1

            if not object_classes:

                print(
                    f"[WARNING] No valid objects: "
                    f"{source_image.name}"
                )

                continue

            # ------------------------------------------------
            # Save mask
            # ------------------------------------------------

            output_mask = (
                mask_sequence_dir / "00000.png"
            )

            mask.save(
                output_mask
            )

            # ------------------------------------------------
            # Store mapping
            # ------------------------------------------------

            split_mappings[split][
                sequence_name
            ] = {
                "source_dataset": record[
                    "dataset"
                ],
                "source_split": record[
                    "split"
                ],
                "source_image": record[
                    "file_name"
                ],
                "source_image_path": str(
                    source_image
                ),
                "width": width,
                "height": height,
                "objects": object_classes,
            }

            sequence_number += 1
            global_sequence_counter += 1

    # --------------------------------------------------------
    # Write split files and class mappings
    # --------------------------------------------------------

    for split in ["train", "valid", "test"]:

        list_path = (
            OUTPUT / f"{split}.txt"
        )

        with open(
            list_path,
            "w",
            encoding="utf-8"
        ) as f:

            for sequence_name in sorted(
                split_mappings[split]
            ):

                f.write(
                    sequence_name + "\n"
                )

        mapping_path = (
            OUTPUT
            / f"{split}_classes.json"
        )

        with open(
            mapping_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                split_mappings[split],
                f,
                indent=2
            )

    # --------------------------------------------------------
    # Duplicate report
    # --------------------------------------------------------

    report_path = (
        OUTPUT / "duplicate_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "401 + 724 DUPLICATE REPORT\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Duplicate groups: "
            f"{len(duplicate_groups)}\n"
        )

        f.write(
            f"Removed copies: "
            f"{len(removed_records)}\n\n"
        )

        for i, (keeper, removed) in enumerate(
            removed_records,
            1
        ):

            f.write(
                f"Duplicate {i}\n"
            )

            f.write(
                f"KEPT   : "
                f"[{keeper['dataset']}/{keeper['split']}] "
                f"{keeper['file_name']}\n"
            )

            f.write(
                f"REMOVED: "
                f"[{removed['dataset']}/{removed['split']}] "
                f"{removed['file_name']}\n\n"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SAM2 DATASET CREATED")
    print("=" * 70)

    print(
        f"\nImages : {len(final_records)}"
    )

    print(
        f"Objects: {total_objects}"
    )

    print(
        f"Helmet : {total_helmet}"
    )

    print(
        f"Without-helmet: {total_without_helmet}"
    )

    print("\nOutput:")
    print(OUTPUT)

    print("\nClasses:")
    print("  1 = helmet")
    print("  2 = without-helmet")

    print("\nDuplicate policy:")
    print("  724 dataset has priority over 401.")
    print("  Cross-split duplicates are not allowed.")

    print("\nOriginal datasets were NOT modified.")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()