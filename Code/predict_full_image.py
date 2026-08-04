"""
predict_full_image.py

Runs a trained U-Net on a full Sentinel-2 image and compares the
prediction with the corresponding MNDWI mask.
"""

import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
IMAGE_PATH = r"C:\Ritanshu\10 Capstone\Data\Testing\OsmanSagar_832x810.tif"
COMP_PATH = r"C:\Ritanshu\10 Capstone\Data\Testing\OsmanSagar_832x810_mndwi_mask.tif"

MODEL_PATH = "unet_lake_model_complete.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------------
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=5,
        classes=1,
        activation=None,
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


# ------------------------------------------------------------------
# Pad image so dimensions are divisible by 32
# ------------------------------------------------------------------
def pad_to_multiple(img, multiple=32):
    _, h, w = img.shape

    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    padded = np.pad(
        img,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode="reflect"
    )

    return padded, h, w


# ------------------------------------------------------------------
# Main prediction function
# ------------------------------------------------------------------
def predict_full_image():

    model = load_model()

    # --------------------------------------------------------------
    # Load Sentinel-2 image
    # --------------------------------------------------------------
    with rasterio.open(IMAGE_PATH) as src:
        img = src.read().astype(np.float32)

    # Normalize
    img = np.clip(img / 3000.0, 0, 1)

    # --------------------------------------------------------------
    # Load Ground Truth MNDWI Mask
    # --------------------------------------------------------------
    with rasterio.open(COMP_PATH) as src:
        gt_mask = src.read(1)

    gt_mask = (gt_mask > 0).astype(np.uint8)

    # --------------------------------------------------------------
    # Predict
    # --------------------------------------------------------------
    padded_img, orig_h, orig_w = pad_to_multiple(img)

    img_tensor = (
        torch.tensor(padded_img, dtype=torch.float32)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
        pred_mask = (pred > 0.5).float().cpu().numpy().squeeze()

    pred_mask = pred_mask[:orig_h, :orig_w].astype(np.uint8)

    # --------------------------------------------------------------
    # Surface Area
    # --------------------------------------------------------------
    water_pixels = np.sum(pred_mask)

    surface_area_km2 = (water_pixels * 100) / 1_000_000

    print(f"\nPredicted Water Surface Area : {surface_area_km2:.3f} km²")

    # --------------------------------------------------------------
    # Evaluation Metrics
    # --------------------------------------------------------------
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()

    iou = intersection / (union + 1e-8)

    dice = (
        2 * intersection
        / (gt_mask.sum() + pred_mask.sum() + 1e-8)
    )

    pixel_acc = (gt_mask == pred_mask).mean()

    print(f"IoU            : {iou:.4f}")
    print(f"Dice Score     : {dice:.4f}")
    print(f"Pixel Accuracy : {pixel_acc:.4f}")

    # --------------------------------------------------------------
    # RGB Visualization
    # --------------------------------------------------------------
    rgb = img[[2, 1, 0], :, :].transpose(1, 2, 0)
    rgb = np.clip(rgb, 0, 1)

    # --------------------------------------------------------------
    # Plot Comparison
    # --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(rgb)
    axes[0].set_title("Raw Sentinel-2 RGB", fontsize=14)

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth MNDWI Mask", fontsize=14)

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title(
        f"U-Net Prediction\nArea = {surface_area_km2:.3f} km²",
        fontsize=14,
    )

    for ax in axes:
        ax.axis("off")

    plt.suptitle(
        f"IoU = {iou:.4f}    |    Dice = {dice:.4f}    |    Pixel Accuracy = {pixel_acc:.4f}",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    plt.savefig(
        r"C:\Ritanshu\10 Capstone\Data\Testing\comparison_result.png",
        dpi=300,
        bbox_inches="tight",
    )

    print("\nSaved comparison_result.png")

    plt.show()


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
if __name__ == "__main__":
    predict_full_image()