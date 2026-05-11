"""
Cloud Coverage Inference Script
================================
Run inference with a trained model to:
  1. Predict cloud masks for new images
  2. Compute cloud coverage percentage
  3. Save visualizations (overlay) and a CSV report

Usage:
    # Single image
    python predict.py --checkpoint checkpoints/best_model.pth --input photo.jpg

    # Whole directory
    python predict.py --checkpoint checkpoints/best_model.pth \\
                      --input /path/to/images --output_dir ./predictions
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp


def load_model(checkpoint_path, device):
    """Load a model from checkpoint with its config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]

    ModelClass = getattr(smp, cfg["architecture"])
    model = ModelClass(
        encoder_name=cfg["encoder_name"],
        encoder_weights=None,           # we load fine-tuned weights, no need for imagenet
        in_channels=cfg["in_channels"],
        classes=cfg["classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    image_size = ckpt.get("image_size", 384)
    print(f"Loaded {cfg['architecture']} ({cfg['encoder_name']}) | "
          f"image_size={image_size}")
    return model, image_size


def get_inference_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


@torch.no_grad()
def predict_mask(model, image_bgr, transform, device, threshold=0.5):
    """
    Run inference on a single image.
    Returns mask resized back to the original image dimensions.
    """
    original_h, original_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    augmented = transform(image=image_rgb)
    tensor = augmented["image"].unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)

    # Resize back to original
    probs_full = cv2.resize(probs, (original_w, original_h),
                            interpolation=cv2.INTER_LINEAR)
    mask = (probs_full > threshold).astype(np.uint8)
    return mask, probs_full


def cloud_coverage(mask, valid_region=None):
    """
    Compute cloud coverage percentage.

    If valid_region is provided (binary mask of the actual sky area, e.g. circular
    ASI), coverage is computed within that region only. Otherwise it uses the
    full image.
    """
    if valid_region is not None:
        total = valid_region.sum()
        cloud = (mask * valid_region).sum()
    else:
        total = mask.size
        cloud = mask.sum()

    if total == 0:
        return 0.0
    return 100.0 * cloud / total


def make_overlay(image_bgr, mask, alpha=0.45, color=(0, 0, 255)):
    """Overlay the cloud mask in red on the original image."""
    overlay = image_bgr.copy()
    colored_mask = np.zeros_like(image_bgr)
    colored_mask[mask > 0] = color
    return cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)


def gather_input_paths(input_path, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """Return a list of image paths whether input is a file or a directory."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([
            f for f in p.rglob("*") if f.suffix.lower() in extensions
        ])
    raise FileNotFoundError(f"Input path not found: {input_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model .pth")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to an image or a directory of images")
    parser.add_argument("--output_dir", type=str, default="./predictions")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for cloud pixels (0-1)")
    parser.add_argument("--save_overlay", action="store_true", default=True,
                        help="Save red-tinted overlay visualizations")
    parser.add_argument("--save_mask", action="store_true", default=True,
                        help="Save raw binary masks as PNG")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_overlay:
        os.makedirs(os.path.join(args.output_dir, "overlays"), exist_ok=True)
    if args.save_mask:
        os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)

    model, image_size = load_model(args.checkpoint, device)
    transform = get_inference_transform(image_size)

    image_paths = gather_input_paths(args.input)
    print(f"Processing {len(image_paths)} image(s)...")

    results = []
    for img_path in image_paths:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [skip] could not read {img_path}")
            continue

        mask, _ = predict_mask(model, image_bgr, transform, device, args.threshold)
        coverage = cloud_coverage(mask)

        stem = img_path.stem
        if args.save_mask:
            mask_path = os.path.join(args.output_dir, "masks", f"{stem}_mask.png")
            cv2.imwrite(mask_path, mask * 255)
        if args.save_overlay:
            overlay = make_overlay(image_bgr, mask)
            # Annotate coverage on the image
            cv2.putText(
                overlay, f"Cloud cover: {coverage:.1f}%",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                cv2.LINE_AA,
            )
            overlay_path = os.path.join(args.output_dir, "overlays", f"{stem}_overlay.jpg")
            cv2.imwrite(overlay_path, overlay)

        results.append({"image": img_path.name, "cloud_coverage_percent": round(coverage, 2)})
        print(f"  {img_path.name:40s}  coverage = {coverage:6.2f}%")

    # Save CSV report
    csv_path = os.path.join(args.output_dir, "coverage_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "cloud_coverage_percent"])
        writer.writeheader()
        writer.writerows(results)

    if results:
        coverages = [r["cloud_coverage_percent"] for r in results]
        print(f"\nProcessed {len(results)} images.")
        print(f"  Mean coverage:   {np.mean(coverages):.2f}%")
        print(f"  Median coverage: {np.median(coverages):.2f}%")
        print(f"  Min / Max:       {min(coverages):.2f}% / {max(coverages):.2f}%")
    print(f"\nReport saved to {csv_path}")


if __name__ == "__main__":
    main()
