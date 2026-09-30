import json
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\AI Projects\HelmetViolationSystem"
)

TEST_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "helmet_coco"
    / "test"
)

COCO_JSON = (
    TEST_ROOT
    / "_annotations.coco.json"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "sam2_repo"
    / "sam2_logs"
    / "helmet_small"
    / "checkpoints"
    / "checkpoint_40.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "evaluation"
    / "sam2_helmet_test"
)

VISUAL_ROOT = (
    OUTPUT_ROOT
    / "visualizations"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

CLASS_NAMES = {
    1: "helmet",
    2: "without-helmet",
}


# ============================================================
# HELPERS
# ============================================================

def polygon_to_mask(segmentation, width, height):
    """
    Convert COCO polygon segmentation into a binary mask.
    """
    mask = Image.new(
        "L",
        (width, height),
        0
    )

    draw = ImageDraw.Draw(mask)

    for polygon in segmentation:

        if not isinstance(polygon, list):
            continue

        if len(polygon) < 6:
            continue

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

        if len(points) >= 3:
            draw.polygon(
                points,
                fill=1
            )

    return np.array(
        mask,
        dtype=np.uint8
    )


def calculate_iou(pred_mask, gt_mask):
    """
    Intersection over Union.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    if union == 0:
        return 1.0

    return float(
        intersection / union
    )


def calculate_dice(pred_mask, gt_mask):
    """
    Dice coefficient.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    total = pred.sum() + gt.sum()

    if total == 0:
        return 1.0

    return float(
        2.0 * intersection / total
    )


def save_visualization(
    image,
    gt_mask,
    pred_mask,
    bbox,
    class_name,
    score,
    output_path
):
    """
    Save a simple visual comparison:
    original + GT outline + predicted mask.
    """

    base = image.copy().convert("RGB")

    overlay = Image.new(
        "RGBA",
        base.size,
        (0, 0, 0, 0)
    )

    overlay_pixels = overlay.load()
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    width, height = base.size

    for y in range(height):
        for x in range(width):

            gt = gt_bool[y, x]
            pred = pred_bool[y, x]

            if gt and pred:
                overlay_pixels[x, y] = (
                    255,
                    255,
                    0,
                    100
                )

            elif gt:
                overlay_pixels[x, y] = (
                    0,
                    255,
                    0,
                    100
                )

            elif pred:
                overlay_pixels[x, y] = (
                    255,
                    0,
                    0,
                    100
                )

    result = Image.alpha_composite(
        base.convert("RGBA"),
        overlay
    )

    draw = ImageDraw.Draw(result)

    x, y, w, h = bbox

    draw.rectangle(
        [
            x,
            y,
            x + w,
            y + h
        ],
        outline=(255, 255, 255, 255),
        width=2
    )

    label = (
        f"{class_name} | "
        f"SAM score={score:.3f}"
    )

    draw.rectangle(
        [
            x,
            max(0, y - 20),
            x + 260,
            y
        ],
        fill=(0, 0, 0, 180)
    )

    draw.text(
        (
            x + 3,
            max(0, y - 18)
        ),
        label,
        fill=(255, 255, 255, 255)
    )

    result.convert("RGB").save(
        output_path,
        quality=95
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SAM2 HELMET TEST EVALUATION")
    print("=" * 70)

    print()
    print("Checkpoint:")
    print(CHECKPOINT)

    print()
    print("Test annotations:")
    print(COCO_JSON)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    VISUAL_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load COCO annotations
    # --------------------------------------------------------

    with open(
        COCO_JSON,
        "r",
        encoding="utf-8"
    ) as f:
        coco = json.load(f)

    images = {
        image["id"]: image
        for image in coco["images"]
    }

    annotations = coco["annotations"]

    print()
    print(
        f"Test images in COCO : {len(images)}"
    )

    print(
        f"Test annotations     : {len(annotations)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading SAM2...")

    model = build_sam2(
        MODEL_CONFIG,
        str(CHECKPOINT),
        device="cuda"
    )

    predictor = SAM2ImagePredictor(
        model
    )

    print("SAM2 loaded successfully.")
    print(
        "Device:",
        next(model.parameters()).device
    )

    # --------------------------------------------------------
    # Group annotations by image
    # --------------------------------------------------------

    annotations_by_image = {}

    for annotation in annotations:

        category_id = annotation.get(
            "category_id"
        )

        if category_id not in CLASS_NAMES:
            continue

        image_id = annotation[
            "image_id"
        ]

        annotations_by_image.setdefault(
            image_id,
            []
        ).append(annotation)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    rows = []

    class_metrics = {
        "helmet": {
            "ious": [],
            "dices": [],
            "scores": [],
        },
        "without-helmet": {
            "ious": [],
            "dices": [],
            "scores": [],
        },
    }

    total_objects = 0

    # --------------------------------------------------------
    # Evaluate every test image
    # --------------------------------------------------------

    for image_index, image_id in enumerate(
        sorted(annotations_by_image)
    ):

        image_info = images[image_id]

        file_name = image_info[
            "file_name"
        ]

        image_path = (
            TEST_ROOT
            / file_name
        )

        if not image_path.exists():

            print(
                f"WARNING: Missing image: "
                f"{image_path}"
            )

            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        image_np = np.array(
            image
        )

        width = image_info[
            "width"
        ]

        height = image_info[
            "height"
        ]

        print()
        print(
            f"[{image_index + 1}/"
            f"{len(annotations_by_image)}] "
            f"{file_name}"
        )

        # ----------------------------------------------------
        # Set image once
        # ----------------------------------------------------

        predictor.set_image(
            image_np
        )

        # ----------------------------------------------------
        # Evaluate each object
        # ----------------------------------------------------

        for object_index, annotation in enumerate(
            annotations_by_image[image_id]
        ):

            category_id = annotation[
                "category_id"
            ]

            class_name = CLASS_NAMES[
                category_id
            ]

            bbox = annotation[
                "bbox"
            ]

            gt_mask = polygon_to_mask(
                annotation[
                    "segmentation"
                ],
                width,
                height
            )

            # COCO:
            # [x, y, width, height]
            #
            # SAM2:
            # [x1, y1, x2, y2]

            x, y, w, h = bbox

            sam_box = np.array(
                [
                    x,
                    y,
                    x + w,
                    y + h,
                ],
                dtype=np.float32
            )

            # ------------------------------------------------
            # Predict segmentation
            # ------------------------------------------------

            with torch.inference_mode():

                masks, scores, _ = (
                    predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=sam_box,
                        multimask_output=True,
                    )
                )

            # Best candidate mask
            best_index = int(
                np.argmax(scores)
            )

            pred_mask = (
                masks[best_index]
                .astype(np.uint8)
            )

            score = float(
                scores[best_index]
            )

            iou = calculate_iou(
                pred_mask,
                gt_mask
            )

            dice = calculate_dice(
                pred_mask,
                gt_mask
            )

            # ------------------------------------------------
            # Store metrics
            # ------------------------------------------------

            class_metrics[
                class_name
            ]["ious"].append(iou)

            class_metrics[
                class_name
            ]["dices"].append(dice)

            class_metrics[
                class_name
            ]["scores"].append(score)

            rows.append({
                "image": file_name,
                "image_id": image_id,
                "annotation_id": annotation["id"],
                "class": class_name,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": w,
                "bbox_height": h,
                "sam_score": score,
                "iou": iou,
                "dice": dice,
            })

            total_objects += 1

            print(
                f"  Object {object_index + 1}: "
                f"{class_name:<16} "
                f"IoU={iou:.4f} "
                f"Dice={dice:.4f} "
                f"Score={score:.4f}"
            )

            # ------------------------------------------------
            # Save first few visual examples
            # ------------------------------------------------

            if total_objects <= 15:

                visual_name = (
                    f"{total_objects:03d}_"
                    f"{Path(file_name).stem}_"
                    f"{class_name}.jpg"
                )

                visual_path = (
                    VISUAL_ROOT
                    / visual_name
                )

                save_visualization(
                    image=image,
                    gt_mask=gt_mask,
                    pred_mask=pred_mask,
                    bbox=bbox,
                    class_name=class_name,
                    score=score,
                    output_path=visual_path,
                )

    # --------------------------------------------------------
    # Save per-object results
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_ROOT
        / "object_metrics.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        if rows:

            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys())
            )

            writer.writeheader()
            writer.writerows(rows)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    def mean(values):

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    summary = {
        "total_objects": total_objects,
        "helmet": {
            "objects": len(
                class_metrics["helmet"]["ious"]
            ),
            "mean_iou": mean(
                class_metrics["helmet"]["ious"]
            ),
            "mean_dice": mean(
                class_metrics["helmet"]["dices"]
            ),
            "mean_sam_score": mean(
                class_metrics["helmet"]["scores"]
            ),
        },
        "without-helmet": {
            "objects": len(
                class_metrics["without-helmet"]["ious"]
            ),
            "mean_iou": mean(
                class_metrics["without-helmet"]["ious"]
            ),
            "mean_dice": mean(
                class_metrics["without-helmet"]["dices"]
            ),
            "mean_sam_score": mean(
                class_metrics["without-helmet"]["scores"]
            ),
        },
    }

    summary_path = (
        OUTPUT_ROOT
        / "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Objects evaluated: "
        f"{total_objects}"
    )

    print()
    print("HELMET")
    print(
        f"  Objects      : "
        f"{summary['helmet']['objects']}"
    )
    print(
        f"  Mean IoU     : "
        f"{summary['helmet']['mean_iou']:.4f}"
    )
    print(
        f"  Mean Dice    : "
        f"{summary['helmet']['mean_dice']:.4f}"
    )
    print(
        f"  Mean Score   : "
        f"{summary['helmet']['mean_sam_score']:.4f}"
    )

    print()
    print("WITHOUT-HELMET")
    print(
        f"  Objects      : "
        f"{summary['without-helmet']['objects']}"
    )
    print(
        f"  Mean IoU     : "
        f"{summary['without-helmet']['mean_iou']:.4f}"
    )
    print(
        f"  Mean Dice    : "
        f"{summary['without-helmet']['mean_dice']:.4f}"
    )
    print(
        f"  Mean Score   : "
        f"{summary['without-helmet']['mean_sam_score']:.4f}"
    )

    print()
    print("Results:")
    print(csv_path)

    print()
    print("Summary:")
    print(summary_path)

    print()
    print("Visual examples:")
    print(VISUAL_ROOT)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()