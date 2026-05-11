"""
Cloud Coverage Inference & Evaluation Script
=============================================
Run inference with a trained model to:
  1. Predict cloud masks for new images
  2. Compute cloud coverage percentage
  3. Save visualisations (overlay) and a CSV report
  4. Optionally evaluate against ground-truth masks (IoU / Dice)

Usage:
    # Single image
    python predict.py --checkpoint checkpoints/best_model.pth --input photo.jpg

    # Whole directory — inference only (default behaviour)
    python predict.py --checkpoint checkpoints/best_model.pth \\
                      --input ../dataset/test --output_dir ./predictions

    # Test set evaluation with ground-truth masks
    python predict.py --checkpoint checkpoints/best_model.pth \\
                      --input ../dataset/test \\
                      --labels_dir ../dataset/test_labels \\
                      --output_dir ./predictions

    # Single random image only
    python predict.py --checkpoint checkpoints/best_model.pth --input photo.jpg --single
"""

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    """Load a fine-tuned model from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["model_config"]

    ModelClass = getattr(smp, cfg["architecture"])
    model = ModelClass(
        encoder_name    = cfg["encoder_name"],
        encoder_weights = None,   # we load our own fine-tuned weights
        in_channels     = cfg["in_channels"],
        classes         = cfg["classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    image_size = ckpt.get("image_size", 384)
    print(f"Loaded {cfg['architecture']} ({cfg['encoder_name']}) | image_size={image_size}")
    return model, image_size


def get_inference_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_mask(model, image_bgr, transform, device, threshold=0.5):
    """
    Run inference on a single BGR image (as returned by cv2.imread).
    Returns the binary mask and the probability map, both at the original resolution.
    """
    original_h, original_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)
    probs  = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()   # (H, W)

    probs_full = cv2.resize(probs, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    mask = (probs_full > threshold).astype(np.uint8)
    return mask, probs_full


def cloud_coverage(mask, valid_region=None):
    """
    Compute cloud coverage as a percentage.

    valid_region — binary array (same size as mask) that marks the sky area,
                   useful for circular all-sky imagers to exclude the black ring.
    """
    if valid_region is not None:
        total = int(valid_region.sum())
        cloud = int((mask * valid_region).sum())
    else:
        total = mask.size
        cloud = int(mask.sum())
    return 0.0 if total == 0 else 100.0 * cloud / total


def make_side_by_side(image_bgr, mask):
    """Return original image and binary mask side by side."""
    mask_bgr = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    return np.concatenate([image_bgr, mask_bgr], axis=1)


def gather_input_paths(input_path, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """Return a sorted list of image paths from a file or a directory."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in extensions)
    raise FileNotFoundError(f"Input path not found: {input_path}")


# ---------------------------------------------------------------------------
# Ground-truth evaluation helpers
# ---------------------------------------------------------------------------

def load_gt_mask(mask_path):
    """
    Load a ground-truth mask and convert to binary (1=cloud, 0=sky).
    Matches the dataset convention: white(255)=cloud, black(0)=sky.
    """
    gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return (gray >= 128).astype(np.uint8)


def _iou(pred, gt, eps=1e-7):
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    return (inter + eps) / (union + eps)


def _dice(pred, gt, eps=1e-7):
    inter = int((pred & gt).sum())
    return (2 * inter + eps) / (int(pred.sum()) + int(gt.sum()) + eps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cloud coverage inference and evaluation."
    )
    parser.add_argument("--checkpoint",  type=str, default="./checkpoints/best_model.pth",
                        help="Path to trained model .pth (default: ./checkpoints/best_model.pth)")
    parser.add_argument("--input",       type=str, default="../dataset/test",
                        help="Image file or directory of images (default: ../dataset/test)")
    parser.add_argument("--labels_dir",  type=str, default="../dataset/test_labels",
                        help="Directory with ground-truth masks (default: ../dataset/test_labels). "
                             "Set to '' to disable GT evaluation.")
    parser.add_argument("--single",       action="store_true",
                        help="Process only one random image from --input (default: all images)")
    parser.add_argument("--output_dir",  type=str, default="./predictions")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Probability threshold for cloud pixels (default: 0.5)")
    parser.add_argument("--no_overlay",  action="store_true",
                        help="Do not save overlay visualisations")
    parser.add_argument("--no_mask",     action="store_true",
                        help="Do not save raw binary mask PNGs")
    parser.add_argument("--device",      type=str, default=None,
                        help="cuda | mps | cpu (auto-detected if omitted)")
    args = parser.parse_args()

    # Device
    if args.device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
    else:
        device = args.device
    print(f"Using device: {device}")

    save_overlay = not args.no_overlay
    save_mask    = not args.no_mask

    os.makedirs(args.output_dir, exist_ok=True)
    if save_overlay:
        os.makedirs(os.path.join(args.output_dir, "overlays"), exist_ok=True)
    if save_mask:
        os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)

    model, image_size = load_model(args.checkpoint, device)
    transform         = get_inference_transform(image_size)
    image_paths       = gather_input_paths(args.input)

    if args.single:
        image_paths = [random.choice(image_paths)]
        print(f"Random image selected: {image_paths[0].name}")
    else:
        print(f"Processing {len(image_paths)} image(s)...")

    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    mask_extensions = (".png", ".jpg", ".bmp")

    results          = []
    all_ious, all_dices = [], []

    for img_path in image_paths:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [skip] could not read {img_path}")
            continue

        mask, _ = predict_mask(model, image_bgr, transform, device, args.threshold)
        coverage = cloud_coverage(mask)
        stem     = img_path.stem

        if save_mask:
            cv2.imwrite(
                os.path.join(args.output_dir, "masks", f"{stem}_mask.png"),
                mask * 255,
            )
        if save_overlay:
            viz = make_side_by_side(image_bgr, mask)
            cv2.imwrite(
                os.path.join(args.output_dir, "overlays", f"{stem}_viz.jpg"),
                viz,
            )

        row = {"image": img_path.name, "cloud_coverage_percent": round(coverage, 2)}

        # Ground-truth evaluation
        if labels_dir is not None:
            gt_path = next(
                (labels_dir / (stem + ext) for ext in mask_extensions
                 if (labels_dir / (stem + ext)).exists()),
                None,
            )
            if gt_path is not None:
                gt = load_gt_mask(gt_path)
                if gt is not None:
                    # resize GT to match predicted mask dimensions (original image size)
                    gt_resized  = cv2.resize(gt, (mask.shape[1], mask.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                    pred_binary = (mask > 0).astype(np.uint8)
                    batch_iou   = _iou(pred_binary,  gt_resized)
                    batch_dice  = _dice(pred_binary, gt_resized)
                    all_ious.append(batch_iou)
                    all_dices.append(batch_dice)
                    row["iou"]  = round(float(batch_iou),  4)
                    row["dice"] = round(float(batch_dice), 4)
            else:
                print(f"  [warn] no ground-truth mask found for {img_path.name}")

        results.append(row)
        line = f"  {img_path.name:40s}  coverage={coverage:6.2f}%"
        if "iou" in row:
            line += f"  IoU={row['iou']:.4f}  Dice={row['dice']:.4f}"
        print(line)

    # ------------------------------------------------------------------
    # CSV report
    # ------------------------------------------------------------------
    fieldnames = ["image", "cloud_coverage_percent"]
    if all_ious:
        fieldnames += ["iou", "dice"]
    csv_path = os.path.join(args.output_dir, "coverage_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    if results:
        coverages = [r["cloud_coverage_percent"] for r in results]
        print(f"\nProcessed {len(results)} image(s).")
        print(f"  Coverage — mean: {np.mean(coverages):.2f}%  "
              f"median: {np.median(coverages):.2f}%  "
              f"min/max: {min(coverages):.2f}% / {max(coverages):.2f}%")
        if all_ious:
            print(f"  IoU      — mean: {np.mean(all_ious):.4f}  "
                  f"min: {min(all_ious):.4f}  max: {max(all_ious):.4f}")
            print(f"  Dice     — mean: {np.mean(all_dices):.4f}  "
                  f"min: {min(all_dices):.4f}  max: {max(all_dices):.4f}")
    print(f"\nReport saved to {csv_path}")


if __name__ == "__main__":
    main()
