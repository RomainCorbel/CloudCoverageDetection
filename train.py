"""
Cloud Segmentation Training Script
===================================
Fine-tune a U-Net for binary cloud/sky segmentation on the Sky-Image dataset.

Usage:
    python train.py --data_dir /path/to/dataset --epochs 30 --batch_size 8

The dataset directory should be structured as:
    data_dir/
        images/
            img001.jpg
            img002.jpg
            ...
        masks/
            img001.png  (binary: 0=sky, 255=cloud, or 0/1)
            img002.png
            ...

If your dataset has a different structure, adjust DATA_CONFIG below or pass
custom --images_subdir and --masks_subdir arguments.
"""

import argparse
import os
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
from PIL import Image

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DATA_CONFIG = {
    "image_extensions": (".jpg", ".jpeg", ".png", ".bmp"),
    "mask_extensions": (".png", ".jpg", ".bmp"),
}

MODEL_CONFIG = {
    "architecture": "Unet",          # Unet, UnetPlusPlus, DeepLabV3Plus, FPN, ...
    "encoder_name": "resnet34",      # resnet34/50, efficientnet-b0..b3, mobilenet_v2
    "encoder_weights": "imagenet",
    "in_channels": 3,
    "classes": 1,                    # binary segmentation
}

TRAIN_CONFIG = {
    "image_size": 384,               # resize input to this size
    "num_workers": 4,
    "pin_memory": True,
}

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class CloudDataset(Dataset):
    """Loads (image, mask) pairs. Masks are binarized: 0=sky, 1=cloud."""

    def __init__(self, image_paths, mask_paths, transform=None):
        assert len(image_paths) == len(mask_paths), "Mismatched image/mask counts"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image (BGR -> RGB)
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        # Binarize: anything > 127 is cloud
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)  # (1, H, W)

        return image, mask


def get_transforms(image_size, train=True):
    """Build augmentation pipeline. ImageNet normalization for pretrained backbone."""
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


def find_image_mask_pairs(data_dir, images_subdir="images", masks_subdir="masks"):
    """Match images with their corresponding masks by filename stem."""
    data_dir = Path(data_dir)
    img_dir = data_dir / images_subdir
    msk_dir = data_dir / masks_subdir

    if not img_dir.exists() or not msk_dir.exists():
        raise FileNotFoundError(
            f"Expected {img_dir} and {msk_dir}. Adjust --images_subdir/--masks_subdir."
        )

    images = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in DATA_CONFIG["image_extensions"]
    ])

    # Build a lookup of mask stems -> mask path
    mask_lookup = {
        p.stem: p for p in msk_dir.iterdir()
        if p.suffix.lower() in DATA_CONFIG["mask_extensions"]
    }

    pairs = []
    for img_path in images:
        if img_path.stem in mask_lookup:
            pairs.append((img_path, mask_lookup[img_path.stem]))

    if not pairs:
        raise RuntimeError(
            f"No matching pairs found. Check that mask filenames match image stems."
        )

    print(f"Found {len(pairs)} image/mask pairs.")
    return pairs

# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def iou_score(pred, target, threshold=0.5, eps=1e-7):
    """Intersection over Union (Jaccard index)."""
    pred = (pred > threshold).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


def dice_score(pred, target, threshold=0.5, eps=1e-7):
    """Dice coefficient (F1 for segmentation)."""
    pred = (pred > threshold).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (denom + eps)).mean().item()

# ----------------------------------------------------------------------------
# Training / validation loops
# ----------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    losses = []
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return np.mean(losses)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    losses, ious, dices = [], [], []
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, masks)
        probs = torch.sigmoid(logits)

        losses.append(loss.item())
        ious.append(iou_score(probs, masks))
        dices.append(dice_score(probs, masks))

    return np.mean(losses), np.mean(ious), np.mean(dices)


class BCEDiceLoss(nn.Module):
    """Combo loss: standard for binary segmentation, robust to class imbalance."""
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(mode="binary", from_logits=True)
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) \
             + (1 - self.bce_weight) * self.dice(logits, targets)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing 'images/' and 'masks/'")
    parser.add_argument("--images_subdir", type=str, default="images")
    parser.add_argument("--masks_subdir", type=str, default="masks")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda, mps, or cpu. Auto-detected if not provided.")
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

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Build dataset
    pairs = find_image_mask_pairs(
        args.data_dir, args.images_subdir, args.masks_subdir
    )
    image_paths, mask_paths = zip(*pairs)

    full_dataset_train_tf = CloudDataset(
        image_paths, mask_paths,
        transform=get_transforms(TRAIN_CONFIG["image_size"], train=True)
    )
    full_dataset_val_tf = CloudDataset(
        image_paths, mask_paths,
        transform=get_transforms(TRAIN_CONFIG["image_size"], train=False)
    )

    # Split indices
    n_total = len(pairs)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_idx, val_idx = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )
    train_ds = torch.utils.data.Subset(full_dataset_train_tf, train_idx.indices)
    val_ds = torch.utils.data.Subset(full_dataset_val_tf, val_idx.indices)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=TRAIN_CONFIG["pin_memory"] and device == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=TRAIN_CONFIG["pin_memory"] and device == "cuda",
    )

    # Model
    ModelClass = getattr(smp, MODEL_CONFIG["architecture"])
    model = ModelClass(
        encoder_name=MODEL_CONFIG["encoder_name"],
        encoder_weights=MODEL_CONFIG["encoder_weights"],
        in_channels=MODEL_CONFIG["in_channels"],
        classes=MODEL_CONFIG["classes"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {MODEL_CONFIG['architecture']} | "
          f"Encoder: {MODEL_CONFIG['encoder_name']} | "
          f"Params: {n_params/1e6:.2f}M")

    # Loss, optimizer, scheduler
    loss_fn = BCEDiceLoss(bce_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_iou = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_iou, val_dice = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"IoU={val_iou:.4f} | Dice={val_dice:.4f}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "iou": val_iou,
            "dice": val_dice,
        })

        # Save best
        if val_iou > best_iou:
            best_iou = val_iou
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": MODEL_CONFIG,
                "image_size": TRAIN_CONFIG["image_size"],
                "epoch": epoch,
                "iou": val_iou,
                "dice": val_dice,
            }, ckpt_path)
            print(f"  -> Saved best model (IoU={val_iou:.4f}) to {ckpt_path}")

    # Save final
    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": MODEL_CONFIG,
        "image_size": TRAIN_CONFIG["image_size"],
        "history": history,
    }, final_path)
    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
