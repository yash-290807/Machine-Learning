import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_ROOT = Path(
    r"C:\AI Projects\HelmetViolationSystem\datasets\helmet_coco"
)

OUTPUT_ROOT = Path(
    r"C:\AI Projects\HelmetViolationSystem\datasets\sam2_helmet"
)

# Confirmed from the COCO dataset
# category_id 0 = helmet-object (supercategory, not used)
# category_id 1 = helmet
# category_id 2 = without-helmet
CLASS_NAMES = {
    1: "helmet",
    2: "without-helmet",
}


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name):

    source_dir = SOURCE_ROOT / split_name
    json_path = source_dir / "_annotations.coco.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found:\n{json_path}"
        )

    print()
    print("=" * 70)
    print(f"PROCESSING {split_name.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load COCO JSON
    # --------------------------------------------------------

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {
        image["id"]: image
        for image in coco["images"]
    }

    # --------------------------------------------------------
    # Group annotations by image
    # --------------------------------------------------------

    annotations_by_image = {}

    for annotation in coco["annotations"]:

        image_id = annotation["image_id"]

        annotations_by_image.setdefault(
            image_id,
            []
        ).append(annotation)

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    image_root = (
        OUTPUT_ROOT
        / "JPEGImages"
        / split_name
    )

    mask_root = (
        OUTPUT_ROOT
        / "Annotations"
        / split_name
    )

    image_root.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    processed_images = 0
    processed_objects = 0
    skipped_images = 0
    skipped_annotations = 0

    class_counts = {
        "helmet": 0,
        "without-helmet": 0,
    }

    class_mapping = {}

    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    for image_id in sorted(images):

        image_info = images[image_id]

        file_name = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]

        source_image = source_dir / file_name

        # ----------------------------------------------------
        # Check image exists
        # ----------------------------------------------------

        if not source_image.exists():

            print(
                f"WARNING: Missing image:\n"
                f"  {source_image}"
            )

            skipped_images += 1
            continue

        # ----------------------------------------------------
        # Get annotations for this image
        # ----------------------------------------------------

        annotations = annotations_by_image.get(
            image_id,
            []
        )

        # Only keep:
        # 1 = helmet
        # 2 = without-helmet

        annotations = [
            annotation
            for annotation in annotations
            if annotation.get("category_id") in CLASS_NAMES
        ]

        if not annotations:

            print(
                f"WARNING: No usable annotations: "
                f"{file_name}"
            )

            skipped_images += 1
            continue

        # ----------------------------------------------------
        # One image = one-frame SAM2 sequence
        # ----------------------------------------------------

        sequence_name = (
            f"{split_name}_{image_id:06d}"
        )

        image_sequence_dir = (
            image_root / sequence_name
        )

        mask_sequence_dir = (
            mask_root / sequence_name
        )

        image_sequence_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        mask_sequence_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Copy image as frame 00000
        # ----------------------------------------------------

        output_image = (
            image_sequence_dir / "00000.jpg"
        )

        shutil.copy2(
            source_image,
            output_image
        )

        # ----------------------------------------------------
        # Create instance-ID mask
        #
        # 0 = background
        # 1 = object 1
        # 2 = object 2
        # 3 = object 3
        # ...
        #
        # Maximum objects = 16 in our dataset,
        # therefore 8-bit PNG is sufficient.
        # ----------------------------------------------------

        instance_mask = Image.new(
            "L",
            (width, height),
            0
        )

        draw = ImageDraw.Draw(
            instance_mask
        )

        object_classes = {}

        object_id = 1

        # ----------------------------------------------------
        # Process every object
        # ----------------------------------------------------

        for annotation in annotations:

            segmentation = annotation.get(
                "segmentation"
            )

            # We verified that our dataset uses
            # polygon segmentation.
            if not isinstance(
                segmentation,
                list
            ):

                print(
                    f"WARNING: Invalid segmentation "
                    f"in {file_name}"
                )

                skipped_annotations += 1
                continue

            category_id = annotation["category_id"]

            class_name = CLASS_NAMES[
                category_id
            ]

            # ------------------------------------------------
            # Draw all polygons belonging to this object
            # ------------------------------------------------

            valid_polygon = False

            for polygon in segmentation:

                if not isinstance(
                    polygon,
                    list
                ):
                    continue

                if len(polygon) < 6:
                    continue

                # Convert:
                #
                # [x1,y1,x2,y2,x3,y3,...]
                #
                # into:
                #
                # [(x1,y1),(x2,y2),(x3,y3),...]

                points = [
                    (
                        polygon[i],
                        polygon[i + 1]
                    )
                    for i in range(
                        0,
                        len(polygon),
                        2
                    )
                ]

                if len(points) < 3:
                    continue

                draw.polygon(
                    points,
                    fill=object_id
                )

                valid_polygon = True

            if not valid_polygon:

                print(
                    f"WARNING: No valid polygon "
                    f"for annotation "
                    f"{annotation['id']} "
                    f"in {file_name}"
                )

                skipped_annotations += 1
                continue

            # ------------------------------------------------
            # Preserve class information
            # ------------------------------------------------

            object_classes[
                str(object_id)
            ] = {
                "category_id": category_id,
                "class_name": class_name,
                "source_annotation_id": annotation["id"],
            }

            class_counts[class_name] += 1

            processed_objects += 1
            object_id += 1

        # ----------------------------------------------------
        # Save mask
        # ----------------------------------------------------

        output_mask = (
            mask_sequence_dir / "00000.png"
        )

        instance_mask.save(
            output_mask
        )

        # ----------------------------------------------------
        # Store mapping
        # ----------------------------------------------------

        class_mapping[
            sequence_name
        ] = {
            "image_id": image_id,
            "image": file_name,
            "width": width,
            "height": height,
            "objects": object_classes,
        }

        processed_images += 1

    # --------------------------------------------------------
    # Create sequence list
    # --------------------------------------------------------

    list_file = (
        OUTPUT_ROOT
        / f"{split_name}.txt"
    )

    with open(
        list_file,
        "w",
        encoding="utf-8"
    ) as f:

        for sequence_name in sorted(
            class_mapping
        ):

            f.write(
                sequence_name + "\n"
            )

    # --------------------------------------------------------
    # Save class mapping
    # --------------------------------------------------------

    mapping_file = (
        OUTPUT_ROOT
        / f"{split_name}_classes.json"
    )

    with open(
        mapping_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            class_mapping,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print()
    print(f"Images processed     : {processed_images}")
    print(f"Objects processed    : {processed_objects}")
    print(f"Skipped images       : {skipped_images}")
    print(f"Skipped annotations  : {skipped_annotations}")

    print()
    print("Class counts:")

    print(
        f"  helmet             : "
        f"{class_counts['helmet']}"
    )

    print(
        f"  without-helmet     : "
        f"{class_counts['without-helmet']}"
    )

    print()
    print(f"Sequence list        : {list_file}")
    print(f"Class mapping        : {mapping_file}")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("COCO → SAM2 DATASET CONVERSION")
    print("=" * 70)

    print()
    print(f"Source:")
    print(SOURCE_ROOT)

    print()
    print(f"Output:")
    print(OUTPUT_ROOT)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save global class definition
    # --------------------------------------------------------

    class_file = (
        OUTPUT_ROOT / "classes.json"
    )

    with open(
        class_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            CLASS_NAMES,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Process all splits
    # --------------------------------------------------------

    for split in [
        "train",
        "valid",
        "test"
    ]:

        process_split(split)

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print()
    print("Classes:")
    print("  1 = helmet")
    print("  2 = without-helmet")

    print()
    print("Output:")
    print(OUTPUT_ROOT)

    print()
    print("Expected totals:")
    print("  Train : 280 images / 539 objects")
    print("  Valid : 80 images  / 131 objects")
    print("  Test  : 41 images  / 73 objects")

    print()
    print("IMPORTANT:")
    print("Original COCO dataset was not modified.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()