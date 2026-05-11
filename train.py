"""
Cloud Segmentation Training Script
===================================
Fine-tune a U-Net for binary cloud/sky segmentation.

Dataset layout expected (matches the SWIMSEG/sky-image dataset):
    data_dir/
        train/              (--train_subdir)
            0001.png
            0002.png
            ...
        train_labels/       (--masks_subdir)
            0001.png        binary: white(255)=cloud, black(0)=sky
            0002.png
            ...
        val/                (--val_subdir)   — optional
        val_labels/         (--val_labels_subdir) — optional

If val/ and val_labels/ exist the pre-defined split is used; otherwise a
random split from the training set is created (--val_ratio).

Usage:
    python train.py --data_dir ../dataset --epochs 30 --batch_size 8
"""

import argparse
import os
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves to file without needing a display
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_CONFIG = {
    "image_extensions": (".jpg", ".jpeg", ".png", ".bmp"),
    "mask_extensions":  (".png", ".jpg", ".bmp"),
}

MODEL_CONFIG = {
    "architecture":    "Unet",       # Unet | UnetPlusPlus | DeepLabV3Plus | FPN
    "encoder_name":    "resnet34",   # resnet34/50 | efficientnet-b0..b3 | mobilenet_v2
    "encoder_weights": "imagenet",
    "in_channels":     3,
    "classes":         1,            # binary segmentation
}

TRAIN_CONFIG = {
    "image_size":  384,
    # multiprocessing workers: Windows requires 0 (uses spawn, not fork)
    "num_workers": 0 if platform.system() == "Windows" else 4,
    "pin_memory":  True,
}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CloudDataset(Dataset):
    """
    Loads (image, mask) pairs.
    Mask convention (mask images):
        white (255) = cloud → label 1
        black (0)   = sky   → label 0
    """

    def __init__(self, image_paths, mask_paths, transform=None):
        assert len(image_paths) == len(mask_paths), "Mismatched image/mask counts"
        self.image_paths = list(image_paths)
        self.mask_paths  = list(mask_paths)
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            raise RuntimeError(f"Could not read image: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {self.mask_paths[idx]}")
        # bright pixels = cloud (label 1), dark pixels = sky (label 0)
        mask = (mask >= 128).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"].unsqueeze(0)  # (1, H, W)

        return image, mask


def get_transforms(image_size, train=True):
    """Augmentation pipeline with ImageNet normalisation for pretrained backbone."""
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                                 val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def find_pairs(img_dir, msk_dir):
    """Match images with masks by filename stem. Returns sorted list of (img, mask) tuples."""
    img_dir = Path(img_dir)
    msk_dir = Path(msk_dir)

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not msk_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {msk_dir}")

    images = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in DATA_CONFIG["image_extensions"]
    )
    mask_lookup = {
        p.stem: p for p in msk_dir.iterdir()
        if p.suffix.lower() in DATA_CONFIG["mask_extensions"]
    }
    pairs = [(img, mask_lookup[img.stem]) for img in images if img.stem in mask_lookup]
    if not pairs:
        raise RuntimeError(
            f"No matching image/mask pairs found.\n"
            f"  images: {img_dir}\n  masks:  {msk_dir}"
        )
    return pairs

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iou_score(pred, target, threshold=0.5, eps=1e-7):
    pred   = (pred > threshold).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union        = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


def dice_score(pred, target, threshold=0.5, eps=1e-7):
    pred   = (pred > threshold).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    denom        = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (denom + eps)).mean().item()

# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    losses = []
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = loss_fn(model(images), masks)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    losses, ious, dices = [], [], []
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        logits = model(images)
        losses.append(loss_fn(logits, masks).item())
        probs = torch.sigmoid(logits)
        ious.append(iou_score(probs, masks))
        dices.append(dice_score(probs, masks))
    return float(np.mean(losses)), float(np.mean(ious)), float(np.mean(dices))


class BCEDiceLoss(nn.Module):
    """Combo loss: standard for binary segmentation, robust to class imbalance."""
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce       = nn.BCEWithLogitsLoss()
        self.dice      = DiceLoss(mode="binary", from_logits=True)
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        return (self.bce_weight * self.bce(logits, targets)
                + (1 - self.bce_weight) * self.dice(logits, targets))

# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def save_training_curves(history, output_dir):
    """Save a 1×2 figure: loss curves + IoU/Dice curves."""
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    val_iou    = [h["iou"]        for h in history]
    val_dice   = [h["dice"]       for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="train loss")
    ax1.plot(epochs, val_loss,   label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_iou,  label="val IoU")
    ax2.plot(epochs, val_dice, label="val Dice")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a U-Net for cloud segmentation.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root dataset directory (e.g. ../dataset)")
    parser.add_argument("--train_subdir",      type=str, default="train",
                        help="Sub-folder with training images (default: train)")
    parser.add_argument("--masks_subdir",      type=str, default="train_labels",
                        help="Sub-folder with training masks (default: train_labels)")
    parser.add_argument("--val_subdir",        type=str, default="val",
                        help="Sub-folder with validation images (default: val). "
                             "Falls back to random split if not found.")
    parser.add_argument("--val_labels_subdir", type=str, default="val_labels",
                        help="Sub-folder with validation masks (default: val_labels)")
    parser.add_argument("--output_dir",  type=str,   default="./checkpoints")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--image_size",  type=int,   default=TRAIN_CONFIG["image_size"],
                        help="Resize images to this square size (default: 384)")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float, default=1e-5)
    parser.add_argument("--val_ratio",   type=float, default=0.2,
                        help="Validation fraction when no val_subdir is found")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",      type=str,   default=None,
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    data_dir       = Path(args.data_dir)
    train_img_dir  = data_dir / args.train_subdir
    train_msk_dir  = data_dir / args.masks_subdir
    val_img_dir    = data_dir / args.val_subdir
    val_msk_dir    = data_dir / args.val_labels_subdir

    train_pairs = find_pairs(train_img_dir, train_msk_dir)
    print(f"Found {len(train_pairs)} training pairs in '{args.train_subdir}/'.")

    if val_img_dir.exists() and val_msk_dir.exists():
        val_pairs = find_pairs(val_img_dir, val_msk_dir)
        print(f"Found {len(val_pairs)} validation pairs in '{args.val_subdir}/'.")
        train_imgs, train_msks = zip(*train_pairs)
        val_imgs,   val_msks   = zip(*val_pairs)
        train_ds = CloudDataset(train_imgs, train_msks,
                                transform=get_transforms(args.image_size, train=True))
        val_ds   = CloudDataset(val_imgs,   val_msks,
                                transform=get_transforms(args.image_size, train=False))
    else:
        print(f"'{args.val_subdir}/' not found — using {args.val_ratio:.0%} random split.")
        all_imgs, all_msks = zip(*train_pairs)
        n_total  = len(train_pairs)
        n_val    = int(n_total * args.val_ratio)
        n_train  = n_total - n_val
        generator = torch.Generator().manual_seed(args.seed)
        train_idx, val_idx = random_split(range(n_total), [n_train, n_val],
                                          generator=generator)
        train_ds = CloudDataset(
            [all_imgs[i] for i in train_idx.indices],
            [all_msks[i] for i in train_idx.indices],
            transform=get_transforms(args.image_size, train=True),
        )
        val_ds = CloudDataset(
            [all_imgs[i] for i in val_idx.indices],
            [all_msks[i] for i in val_idx.indices],
            transform=get_transforms(args.image_size, train=False),
        )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    num_workers = TRAIN_CONFIG["num_workers"]
    pin_memory  = TRAIN_CONFIG["pin_memory"] and device == "cuda"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    ModelClass = getattr(smp, MODEL_CONFIG["architecture"])
    model = ModelClass(
        encoder_name    = MODEL_CONFIG["encoder_name"],
        encoder_weights = MODEL_CONFIG["encoder_weights"],
        in_channels     = MODEL_CONFIG["in_channels"],
        classes         = MODEL_CONFIG["classes"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {MODEL_CONFIG['architecture']} | "
          f"Encoder: {MODEL_CONFIG['encoder_name']} | "
          f"Params: {n_params/1e6:.2f}M")

    loss_fn   = BCEDiceLoss(bce_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_iou = 0.0
    history  = []
    for epoch in range(1, args.epochs + 1):
        train_loss             = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_iou, val_dice = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"IoU={val_iou:.4f} | Dice={val_dice:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "iou": val_iou, "dice": val_dice})

        if val_iou > best_iou:
            best_iou  = val_iou
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config":     MODEL_CONFIG,
                "image_size":       args.image_size,
                "epoch":            epoch,
                "iou":              val_iou,
                "dice":             val_dice,
            }, ckpt_path)
            print(f"  -> Saved best model (IoU={val_iou:.4f}) to {ckpt_path}")

    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config":     MODEL_CONFIG,
        "image_size":       args.image_size,
        "history":          history,
    }, final_path)
    save_training_curves(history, args.output_dir)
    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
