import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\AI Projects\HelmetViolationSystem")

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sam2_helmet_final"
)

VALID_IMAGES = DATASET_ROOT / "JPEGImages" / "test"
VALID_MASKS = DATASET_ROOT / "Annotations" / "test"
TEST_METADATA = DATASET_ROOT / "test_classes.json"

CHECKPOINT = (
    PROJECT_ROOT
    / "sam2_repo"
    / "sam2_logs"
    / "helmet_small_final"
    / "checkpoints"
    / "checkpoint_40.pt"
)

MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "evaluation"
    / "sam2_final_test"
)

VISUAL_ROOT = OUTPUT_ROOT / "visualizations"

CLASS_NAMES = {
    1: "helmet",
    2: "without-helmet",
}


# ============================================================
# METRICS
# ============================================================

def calculate_iou(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def calculate_dice(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()

    if total == 0:
        return 1.0

    return float(2.0 * intersection / total)


# ============================================================
# VISUALIZATION
# ============================================================

def save_visualization(
    image,
    gt_mask,
    pred_mask,
    bbox,
    class_name,
    score,
    output_path,
):
    base = image.copy().convert("RGBA")

    overlay = Image.new(
        "RGBA",
        base.size,
        (0, 0, 0, 0),
    )

    overlay_pixels = overlay.load()

    gt_bool = gt_mask.astype(bool)
    pred_bool = pred_mask.astype(bool)

    width, height = base.size

    for y in range(height):
        for x in range(width):

            gt = gt_bool[y, x]
            pred = pred_bool[y, x]

            if gt and pred:
                overlay_pixels[x, y] = (
                    255, 255, 0, 100
                )

            elif gt:
                overlay_pixels[x, y] = (
                    0, 255, 0, 100
                )

            elif pred:
                overlay_pixels[x, y] = (
                    255, 0, 0, 100
                )

    result = Image.alpha_composite(
        base,
        overlay,
    )

    draw = ImageDraw.Draw(result)

    x, y, w, h = bbox

    draw.rectangle(
        [
            x,
            y,
            x + w,
            y + h,
        ],
        outline=(255, 255, 255, 255),
        width=2,
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
            y,
        ],
        fill=(0, 0, 0, 180),
    )

    draw.text(
        (
            x + 3,
            max(0, y - 18),
        ),
        label,
        fill=(255, 255, 255, 255),
    )

    result.convert("RGB").save(
        output_path,
        quality=95,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SAM2 FINAL VALIDATION EVALUATION")
    print("=" * 70)

    print("\nDataset:")
    print(DATASET_ROOT)

    print("\nCheckpoint:")
    print(CHECKPOINT)

    print("\nTest metadata:")
    print(TEST_METADATA)

    if not TEST_METADATA.exists():
        raise FileNotFoundError(
            f"Missing metadata:\n{TEST_METADATA}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing checkpoint:\n{CHECKPOINT}"
        )

    if not VALID_IMAGES.exists():
        raise FileNotFoundError(
            f"Missing validation images:\n{VALID_IMAGES}"
        )

    if not VALID_MASKS.exists():
        raise FileNotFoundError(
            f"Missing validation masks:\n{VALID_MASKS}"
        )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    with open(
        TEST_METADATA,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    print(
        f"\nTest sequences: {len(metadata)}"
    )

    total_gt_objects = sum(
        len(info["objects"])
        for info in metadata.values()
    )

    print(
        f"Ground-truth test objects : {total_gt_objects}"
    )

    # --------------------------------------------------------
    # Create outputs
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    VISUAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading SAM2...")

    model = build_sam2(
        MODEL_CONFIG,
        str(CHECKPOINT),
        device="cuda",
    )

    predictor = SAM2ImagePredictor(model)

    print("SAM2 loaded successfully.")

    print(
        "Device:",
        next(model.parameters()).device,
    )

    # --------------------------------------------------------
    # Metrics storage
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
    visual_count = 0

    # --------------------------------------------------------
    # Evaluate each sequence
    # --------------------------------------------------------

    for image_index, sequence_name in enumerate(
        sorted(metadata.keys())
    ):

        info = metadata[sequence_name]

        image_path = (
            VALID_IMAGES
            / sequence_name
            / "00000.jpg"
        )

        mask_path = (
            VALID_MASKS
            / sequence_name
            / "00000.png"
        )

        if not image_path.exists():
            print(
                f"WARNING: Missing image: "
                f"{image_path}"
            )
            continue

        if not mask_path.exists():
            print(
                f"WARNING: Missing mask: "
                f"{mask_path}"
            )
            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        image_np = np.array(image)

        gt_instance_mask = np.array(
            Image.open(mask_path)
        )

        print()
        print(
            f"[{image_index + 1}/"
            f"{len(metadata)}] "
            f"{sequence_name} "
            f"{info['source_image']}"
        )

        # ----------------------------------------------------
        # Set image once
        # ----------------------------------------------------

        predictor.set_image(image_np)

        # ----------------------------------------------------
        # Evaluate every GT object
        # ----------------------------------------------------

        for object_id_str, object_info in (
            info["objects"].items()
        ):

            object_id = int(object_id_str)

            class_name = object_info[
                "class_name"
            ]

            # Exact instance mask.
            gt_mask = (
                gt_instance_mask == object_id
            ).astype(np.uint8)

            ys, xs = np.where(gt_mask > 0)

            if len(xs) == 0:
                print(
                    f"WARNING: Empty GT object "
                    f"{object_id} in "
                    f"{sequence_name}"
                )
                continue

            # ------------------------------------------------
            # Derive bounding box from GT mask
            # ------------------------------------------------

            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max()) + 1
            y2 = int(ys.max()) + 1

            sam_box = np.array(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                dtype=np.float32,
            )

            bbox = [
                x1,
                y1,
                x2 - x1,
                y2 - y1,
            ]

            # ------------------------------------------------
            # Predict segmentation
            # ------------------------------------------------

            with torch.inference_mode():

                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=sam_box,
                    multimask_output=True,
                )

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

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            iou = calculate_iou(
                pred_mask,
                gt_mask,
            )

            dice = calculate_dice(
                pred_mask,
                gt_mask,
            )

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
                "sequence": sequence_name,
                "source_dataset": info[
                    "source_dataset"
                ],
                "source_split": info[
                    "source_split"
                ],
                "source_image": info[
                    "source_image"
                ],
                "object_id": object_id,
                "class": class_name,
                "bbox_x": x1,
                "bbox_y": y1,
                "bbox_width": x2 - x1,
                "bbox_height": y2 - y1,
                "sam_score": score,
                "iou": iou,
                "dice": dice,
            })

            total_objects += 1

            print(
                f"  Object {object_id}: "
                f"{class_name:<16} "
                f"IoU={iou:.4f} "
                f"Dice={dice:.4f} "
                f"Score={score:.4f}"
            )

            # ------------------------------------------------
            # Save first 15 visual examples
            # ------------------------------------------------

            if visual_count < 15:

                visual_count += 1

                visual_name = (
                    f"{visual_count:03d}_"
                    f"{sequence_name}_"
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
    # Save per-object CSV
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_ROOT
        / "object_metrics.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        if rows:

            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
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
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nObjects evaluated: "
        f"{total_objects}"
    )

    print("\nHELMET")

    print(
        f"  Objects       : "
        f"{summary['helmet']['objects']}"
    )

    print(
        f"  Mean IoU      : "
        f"{summary['helmet']['mean_iou']:.4f}"
    )

    print(
        f"  Mean Dice     : "
        f"{summary['helmet']['mean_dice']:.4f}"
    )

    print(
        f"  Mean SAM score: "
        f"{summary['helmet']['mean_sam_score']:.4f}"
    )

    print("\nWITHOUT-HELMET")

    print(
        f"  Objects       : "
        f"{summary['without-helmet']['objects']}"
    )

    print(
        f"  Mean IoU      : "
        f"{summary['without-helmet']['mean_iou']:.4f}"
    )

    print(
        f"  Mean Dice     : "
        f"{summary['without-helmet']['mean_dice']:.4f}"
    )

    print(
        f"  Mean SAM score: "
        f"{summary['without-helmet']['mean_sam_score']:.4f}"
    )

    print("\nResults:")
    print(csv_path)

    print("\nSummary:")
    print(summary_path)

    print("\nVisual examples:")
    print(VISUAL_ROOT)


if __name__ == "__main__":
    main()


