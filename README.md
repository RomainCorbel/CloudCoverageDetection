# Cloud Segmentation — Sky-Image Dataset

Fine-tunes a U-Net (pretrained ImageNet) for binary cloud/sky segmentation,
then computes the cloud coverage percentage on new images.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset structure

The scripts expect:

```
dataset/
    images/
        img001.jpg
        img002.jpg
        ...
    masks/
        img001.png      # 0 = sky, 255 (or 1) = cloud
        img002.png
        ...
```

Image and mask filenames must share the same stem (`img001.jpg` ↔ `img001.png`).
If your dataset uses different folder names, pass `--images_subdir` and
`--masks_subdir`.

If your masks have a different convention (different file structure, multi-class,
etc.), the only thing to adjust is the `CloudDataset.__getitem__` method in
`train.py`, line where masks are binarized:
```python
mask = (mask > 127).astype(np.float32)
```

## Training

```bash
python train.py --data_dir /path/to/dataset --epochs 30 --batch_size 8
```

Useful flags:
- `--lr 1e-4` (default, good starting point)
- `--val_ratio 0.2` (80/20 train/val split)
- `--device cuda|mps|cpu` (auto-detected by default)

Checkpoints go to `./checkpoints/best_model.pth` (best val IoU) and
`./checkpoints/final_model.pth` (last epoch).

### Switching model architecture

Edit `MODEL_CONFIG` at the top of `train.py`. Anything from
`segmentation_models_pytorch` works:

```python
MODEL_CONFIG = {
    "architecture": "DeepLabV3Plus",         # or UnetPlusPlus, FPN, Linknet, ...
    "encoder_name": "efficientnet-b0",       # or resnet50, mobilenet_v2, ...
    "encoder_weights": "imagenet",
    ...
}
```

## Inference — compute cloud coverage

Single image:
```bash
python predict.py --checkpoint checkpoints/best_model.pth --input photo.jpg
```

Whole directory (recursive):
```bash
python predict.py --checkpoint checkpoints/best_model.pth \
                  --input /path/to/images \
                  --output_dir ./predictions
```

Outputs in `./predictions/`:
- `masks/` — binary masks (PNG, 0/255)
- `overlays/` — original image with cloud mask in red + coverage annotation
- `coverage_report.csv` — per-image coverage percentages

## Notes

- **All-sky cameras with circular field of view**: the model will also segment
  the black border as "sky". To get accurate coverage, mask out the black ring
  first (or pass a `valid_region` mask in `cloud_coverage()` — see the function
  signature in `predict.py`).
- **CPU-only**: training works but is slow. Reduce `--batch_size 4` and
  `image_size` (in `TRAIN_CONFIG`) to 256 if needed. Inference is fine on CPU.
- **Small dataset**: with <500 images, increase augmentation strength or train
  fewer epochs to avoid overfitting. Monitor val IoU — if it plateaus, stop.
