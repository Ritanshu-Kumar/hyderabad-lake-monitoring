"""
train_unet.py

Trains a U-Net (pretrained ResNet34 encoder) to segment water vs. land
from Sentinel-2 imagery, using NDWI-derived masks as preliminary
(pseudo) ground truth labels.

NOTE: These NDWI-derived masks are a temporary stand-in for manually
verified ground truth. They are used here to build and validate a
working end-to-end pipeline before the manual QGIS labeling phase
(which will specifically correct algae-related misclassification in
Hussain Sagar).

Expected folder structure (edit DATA_DIR below to match your setup):

    0 Raw_Images/
        HussainSagar_2025_March.tif
        OsmanSagar_2025_March.tif
        HimayatSagar_2025_March.tif
    2 Labeled_patches/
        HussainSagar_2025_Mask.tif
        OsmanSagar_2025_Mask.tif
        HimayatSagar_2025_Mask.tif

Install dependencies first:
    pip install -r requirements.txt
"""

import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# CONFIG — edit these paths to match your folder locations
# ------------------------------------------------------------------
RAW_IMAGES_DIR = r"C:\Ritanshu\10 Capstone\Data\0 Raw_Images"
MASKS_DIR = r"C:\Ritanshu\10 Capstone\Data\4 mndwi_Labeled_patches"

# Automatically discover all (image, mask) pairs instead of hardcoding
# a fixed list of lakes/years. For every raw image "LakeName_Year_March.tif"
# in RAW_IMAGES_DIR, we look for a matching mask "LakeName_Year_Mask.tif"
# in MASKS_DIR. Any raw image without a matching mask is skipped (with
# a printed warning) rather than crashing the whole run.
def discover_pairs(raw_dir, mask_dir):
    pairs = []
    for filename in sorted(os.listdir(raw_dir)):
        if not filename.lower().endswith(".tif"):
            continue

        base_name = os.path.splitext(filename)[0]  # e.g. "HussainSagar_2020_March"
        if base_name.endswith("_March"):
            base_name = base_name[: -len("_March")]

        mask_filename = f"{base_name}_Mask.tif"
        mask_path = os.path.join(mask_dir, mask_filename)

        if os.path.exists(mask_path):
            pairs.append((filename, mask_filename))
        else:
            print(f"Skipping {filename} — no matching mask found ({mask_filename})")

    return pairs


PAIRS = discover_pairs(RAW_IMAGES_DIR, MASKS_DIR)
print(f"Discovered {len(PAIRS)} image/mask pairs for training.\n")

PATCH_SIZE = 256       # crop size fed into the model
EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------
# STEP 1 — Dataset: cuts each big image/mask pair into small patches
# ------------------------------------------------------------------
class LakePatchDataset(Dataset):
    def __init__(self, pairs, raw_dir, mask_dir, patch_size=256):
        self.patches = []
        self.masks = []

        for img_name, mask_name in pairs:
            img_path = os.path.join(raw_dir, img_name)
            mask_path = os.path.join(mask_dir, mask_name)

            print(f"Processing: {img_name} / {mask_name}")

            if not os.path.exists(img_path):
                print(f"  !! IMAGE FILE NOT FOUND: {img_path}")
                continue
            if not os.path.exists(mask_path):
                print(f"  !! MASK FILE NOT FOUND: {mask_path}")
                continue

            with rasterio.open(img_path) as src:
                # Read 5 bands: Blue, Green, Red, NIR, SWIR
                img = src.read().astype(np.float32)  # shape: (5, H, W)

            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.float32)  # shape: (H, W)

            # Normalize reflectance values roughly to 0-1
            img = np.clip(img / 3000.0, 0, 1)

            h, w = mask.shape
            step = patch_size
            print(f"  Image size: {w}x{h}")

            before_count = len(self.patches)

            # Slide a window across the image, cutting non-overlapping patches
            for y in range(0, h - step + 1, step):
                for x in range(0, w - step + 1, step):
                    img_patch = img[:, y:y + step, x:x + step]
                    mask_patch = mask[y:y + step, x:x + step]

                    # Skip empty/degenerate patches (e.g., all zeros)
                    if img_patch.shape[1:] != (step, step):
                        continue

                    self.patches.append(img_patch)
                    self.masks.append(mask_patch)

            added = len(self.patches) - before_count
            print(f"  -> Added {added} patches from this image")

        print(f"Loaded {len(self.patches)} patches of size {patch_size}x{patch_size}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img = torch.tensor(self.patches[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32).unsqueeze(0)
        return img, mask


# ------------------------------------------------------------------
# STEP 2 — Build the U-Net model (pretrained ResNet34 encoder)
# ------------------------------------------------------------------
def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=5,      # Blue, Green, Red, NIR, SWIR
        classes=1,          # binary: water vs. land
        activation=None,    # raw logits; sigmoid applied in loss/inference
    )
    return model.to(DEVICE)


# ------------------------------------------------------------------
# STEP 3 — Training loop
# ------------------------------------------------------------------
def train():
    dataset = LakePatchDataset(PAIRS, RAW_IMAGES_DIR, MASKS_DIR, PATCH_SIZE)

    if len(dataset) == 0:
        raise RuntimeError(
            "No patches were loaded. Check that your image dimensions are "
            "at least as large as PATCH_SIZE, and that file paths are correct."
        )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = build_model()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "unet_lake_model_complete.pth")
    print("Model saved to unet_lake_model_complete.pth")

    return model, dataset


# ------------------------------------------------------------------
# STEP 4 — Visualize one prediction to sanity-check the pipeline
# ------------------------------------------------------------------
def visualize_prediction(model, dataset, index=0):
    model.eval()
    img, true_mask = dataset[index]
    img_batch = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_batch))
        pred_mask = (pred > 0.5).float().cpu().squeeze().numpy()

    # For display: use Red, Green, Blue bands (indices 2, 1, 0)
    rgb = img[[2, 1, 0], :, :].permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Input (RGB)")
    axes[1].imshow(true_mask.squeeze(), cmap="gray")
    axes[1].set_title("NDWI Pseudo-Label (ground truth)")
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("U-Net Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("complete_prediction_check.png", dpi=150)
    print("Saved visualization to complete_prediction_check.png")
    plt.show()


if __name__ == "__main__":
    trained_model, dataset = train()
    visualize_prediction(trained_model, dataset, index=0)
