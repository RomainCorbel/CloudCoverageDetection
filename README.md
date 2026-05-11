# Cloud Segmentation — Sky-Image Dataset

Fine-tunes a U-Net (pretrained on ImageNet) for binary cloud/sky segmentation,
then computes cloud coverage percentage on new images.

## Setup

### 1 — Install PyTorch with CUDA support

Your RTX 40/50-series GPU requires PyTorch built against **CUDA 12.8**:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> If you have an older NVIDIA GPU (RTX 20/30-series), use `cu121` instead of `cu128`.

### 2 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Verify GPU is detected

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 5060 Laptop GPU
```

## Dataset structure

The scripts are designed for the SWIMSEG sky-image dataset layout (located at
`../dataset` relative to this project):

```
dataset/
    train/              ← 861 training images   (.png)
    train_labels/       ← 861 training masks    (.png)
    val/                ← 101 validation images (.png)
    val_labels/         ← 101 validation masks  (.png)
    test/               ←  51 test images       (.png)
    test_labels/        ←  51 test masks        (.png)
    class_dict.csv      ← cloud=(0,0,0) black | nocloud=(255,255,255) white
```

**Mask convention** (`class_dict.csv`):
- `black (0, 0, 0)` = **cloud** → label 1
- `white (255, 255, 255)` = **sky** → label 0

Image and mask filenames share the same stem (`0001.png` ↔ `0001.png`).

### Using a different dataset

Pass custom sub-directory names to override the defaults:
```bash
python train.py --data_dir /path/to/data \
                --train_subdir images \
                --masks_subdir masks \
                --val_subdir val_images \
                --val_labels_subdir val_masks
```

If `val_subdir` does not exist, a random split is made from the training set
(controlled by `--val_ratio`, default 0.2).

If your masks use a different colour convention (e.g. white=cloud), adjust the
single binarisation line in `train.py` inside `CloudDataset.__getitem__`:
```python
# current (black=cloud):
mask = (mask < 128).astype(np.float32)

# for white=cloud datasets:
mask = (mask > 127).astype(np.float32)
```

## Training

```bash
python train.py --data_dir ../dataset --epochs 30 --batch_size 8
```

Key flags:
| Flag | Default | Description |
|---|---|---|
| `--data_dir` | *(required)* | Root dataset directory |
| `--train_subdir` | `train` | Sub-folder with training images |
| `--masks_subdir` | `train_labels` | Sub-folder with training masks |
| `--val_subdir` | `val` | Sub-folder with validation images |
| `--val_labels_subdir` | `val_labels` | Sub-folder with validation masks |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `8` | Batch size |
| `--image_size` | `384` | Resize images to this square size |
| `--lr` | `1e-4` | Learning rate |
| `--val_ratio` | `0.2` | Val fraction when no val_subdir found |
| `--output_dir` | `./checkpoints` | Where to save model checkpoints |
| `--device` | auto | `cuda` \| `mps` \| `cpu` |

Checkpoints saved to:
- `checkpoints/best_model.pth` — best validation IoU
- `checkpoints/final_model.pth` — last epoch

### Switching model architecture

Edit `MODEL_CONFIG` at the top of `train.py`:

```python
MODEL_CONFIG = {
    "architecture":    "DeepLabV3Plus",   # Unet | UnetPlusPlus | FPN | Linknet
    "encoder_name":    "efficientnet-b0", # resnet50 | mobilenet_v2 | ...
    "encoder_weights": "imagenet",
    ...
}
```

## Inference — compute cloud coverage

### Single image

```bash
python predict.py --checkpoint checkpoints/best_model.pth --input photo.jpg
```

### Whole directory

```bash
python predict.py --checkpoint checkpoints/best_model.pth \
                  --input ../dataset/test \
                  --output_dir ./predictions
```

### Test set evaluation (with ground-truth masks)

```bash
python predict.py --checkpoint checkpoints/best_model.pth \
                  --input ../dataset/test \
                  --labels_dir ../dataset/test_labels \
                  --output_dir ./predictions
```

When `--labels_dir` is provided, per-image **IoU** and **Dice** scores are
computed and included in the CSV report, and mean/min/max metrics are printed at
the end.

### Inference flags

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to `.pth` checkpoint |
| `--input` | *(required)* | Image file or directory |
| `--labels_dir` | `None` | Ground-truth mask directory for evaluation |
| `--output_dir` | `./predictions` | Where to save results |
| `--threshold` | `0.5` | Probability threshold for cloud pixels |
| `--no_overlay` | off | Disable saving overlay visualisations |
| `--no_mask` | off | Disable saving raw binary mask PNGs |
| `--device` | auto | `cuda` \| `mps` \| `cpu` |

### Outputs in `./predictions/`

```
predictions/
├── masks/
│   ├── 0913_mask.png      ← binary mask (0 = sky, 255 = cloud)
│   └── ...
├── overlays/
│   ├── 0913_overlay.jpg   ← original image + red cloud overlay
│   └── ...
└── coverage_report.csv    ← per-image coverage % (+ IoU/Dice if --labels_dir given)
```

## Notes

- **Windows**: `num_workers` is automatically set to 0 to avoid multiprocessing
  issues. Training is otherwise identical to Linux/macOS.
- **CPU-only**: training works but is slow. Reduce `--batch_size 4` and
  `--image_size 256` if needed. Inference is fast on CPU.
- **All-sky cameras**: if your images have a circular field of view, the black
  border may be predicted as sky. Pass a `valid_region` binary mask to
  `cloud_coverage()` in `predict.py` to restrict the coverage calculation to the
  circular sky region.
- **Small datasets** (< 500 images): increase augmentation strength or train
  fewer epochs to avoid overfitting. Monitor val IoU — stop if it plateaus.
