"""
unet_manual_mndwi_v3.py

CAPSTONE — Manual-GT U-Net + MNDWI Water Segmentation

TRAINING:
    Uses manually verified 2018 ground truth for:
        - Hussain Sagar
        - Osman Sagar
        - Himayat Sagar

INFERENCE:
    Applies the trained U-Net to every available year (2018–2025).

    U-Net:
        Learns lake shape / boundary / water appearance.

    MNDWI:
        Provides additional water-vs-land evidence INSIDE the
        U-Net predicted lake region.

    Final:
        Adaptive U-Net + MNDWI fusion followed by connected-component
        cleanup.

INPUT STRUCTURE:

Data/
├── 0 Raw_Images/
│   ├── HussainSagar_2018_March.tif
│   ├── ...
│   ├── OsmanSagar_2018_March.tif
│   ├── ...
│   └── HimayatSagar_2025_March.tif
│
├── 3 mndwi_masks/
│   ├── HussainSagar_2018_MNDWI.tif
│   ├── HussainSagar_2019_MNDWI.tif
│   ├── ...
│   └── HimayatSagar_2025_MNDWI.tif
│
└── 5 manual_ground_truth/
    ├── HussainSagar_GT.tif
    ├── OsmanSagar_GT.tif
    └── HimayatSagar_GT.tif

OUTPUT:

Results/
└── U-Net_Manual_MNDWI_v3/
    ├── model/
    ├── probability_maps/
    ├── raw_fusion_masks/
    ├── final_water_masks/
    ├── visualizations/
    └── tables/
        ├── water_area_summary.csv
        └── training_history.csv
"""

import os
import csv
import random
import numpy as np
import rasterio
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from rasterio.enums import Resampling
from scipy import ndimage


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = r"C:\Users\ritan\Downloads\Capstone_project"

RAW_IMAGES_DIR = os.path.join(
    BASE_DIR, "Data", "0 Raw_Images"
)

MNDWI_DIR = os.path.join(
    BASE_DIR, "Data", "3 mndwi_masks"
)

MANUAL_GT_DIR = os.path.join(
    BASE_DIR, "Data", "5 manual_ground_truth"
)

RESULTS_DIR = os.path.join(
    BASE_DIR, "Results", "U-Net_Manual_MNDWI_v3"
)

MODEL_DIR = os.path.join(RESULTS_DIR, "model")
PROB_DIR = os.path.join(RESULTS_DIR, "probability_maps")
RAW_FUSION_DIR = os.path.join(RESULTS_DIR, "raw_fusion_masks")
FINAL_MASK_DIR = os.path.join(RESULTS_DIR, "final_water_masks")
VIS_DIR = os.path.join(RESULTS_DIR, "visualizations")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")


# Create output folders
for folder in [
    RESULTS_DIR,
    MODEL_DIR,
    PROB_DIR,
    RAW_FUSION_DIR,
    FINAL_MASK_DIR,
    VIS_DIR,
    TABLE_DIR,
]:
    os.makedirs(folder, exist_ok=True)


# ------------------------------------------------
# Model / training parameters
# ------------------------------------------------

PATCH_SIZE = 256

# Overlap between patches
STRIDE = 128

EPOCHS = 40
BATCH_SIZE = 4

LEARNING_RATE = 1e-4

VAL_SPLIT = 0.15

EARLY_STOPPING_PATIENCE = 8

RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------
# Fusion parameters
# ------------------------------------------------

# U-Net probability below this is considered outside
# the learned lake boundary.
BOUNDARY_THRESHOLD = 0.15

# Final fused probability threshold.
FUSION_THRESHOLD = 0.50

# U-Net contribution
UNET_WEIGHT = 0.70

# MNDWI contribution
MNDWI_WEIGHT = 0.30


# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ================================================================
# LAKE INFORMATION
# ================================================================

LAKES = {
    "HussainSagar": {
        "manual_image": "HussainSagar_2018_March.tif",
        "manual_gt": "HussainSagar_GT.tif",
    },

    "OsmanSagar": {
        "manual_image": "OsmanSagar_2018_March.tif",
        "manual_gt": "OsmanSagar_GT.tif",
    },

    "HimayatSagar": {
        "manual_image": "HimayatSagar_2018_March.tif",
        "manual_gt": "HimayatSagar_GT.tif",
    },
}


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def print_separator():
    print("\n" + "=" * 70)


def normalize_image(img):
    """
    Per-image percentile normalization.

    This reduces differences between years caused by
    different reflectance distributions.

    Expected input:
        (5, H, W)

    Output:
        (5, H, W), values approximately 0–1
    """

    img = img.astype(np.float32)

    output = np.zeros_like(img, dtype=np.float32)

    for band in range(img.shape[0]):

        data = img[band]

        valid = np.isfinite(data)

        if not np.any(valid):
            continue

        values = data[valid]

        low = np.percentile(values, 2)
        high = np.percentile(values, 98)

        if high <= low:
            high = low + 1.0

        output[band] = np.clip(
            (data - low) / (high - low),
            0,
            1,
        )

    output[~np.isfinite(output)] = 0

    return output


def get_patch_starts(length, patch_size, stride):
    """
    Generates patch starts and guarantees that the final
    edge of the image is covered.
    """

    if length <= patch_size:
        return [0]

    starts = list(
        range(
            0,
            length - patch_size + 1,
            stride,
        )
    )

    final_start = length - patch_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def resize_mask(mask, height, width):
    """
    Resize a mask to image dimensions using nearest-neighbour
    interpolation.
    """

    if mask.shape == (height, width):
        return mask

    from PIL import Image

    mask_img = Image.fromarray(
        mask.astype(np.float32)
    )

    mask_img = mask_img.resize(
        (width, height),
        Image.Resampling.NEAREST,
    )

    return np.array(mask_img)


# ================================================================
# DATA AUGMENTATION
# ================================================================

def augment_patch(image, mask):
    """
    Lightweight spatial + radiometric augmentation.

    Both image and mask receive identical spatial transformations.
    """

    # Horizontal flip
    if random.random() < 0.5:
        image = np.flip(image, axis=2).copy()
        mask = np.flip(mask, axis=1).copy()

    # Vertical flip
    if random.random() < 0.5:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=0).copy()

    # 90-degree rotation
    if random.random() < 0.5:

        k = random.randint(1, 3)

        image = np.rot90(
            image,
            k=k,
            axes=(1, 2),
        ).copy()

        mask = np.rot90(
            mask,
            k=k,
            axes=(0, 1),
        ).copy()

    # Brightness
    if random.random() < 0.5:

        factor = random.uniform(
            0.90,
            1.10,
        )

        image = np.clip(
            image * factor,
            0,
            1,
        )

    # Contrast
    if random.random() < 0.5:

        factor = random.uniform(
            0.90,
            1.10,
        )

        mean = image.mean(
            axis=(1, 2),
            keepdims=True,
        )

        image = np.clip(
            (image - mean) * factor + mean,
            0,
            1,
        )

    return image.astype(np.float32), mask.astype(np.float32)


# ================================================================
# TRAINING DATASET
# ================================================================

class ManualGroundTruthDataset(Dataset):

    def __init__(
        self,
        training_pairs,
        augment=False,
    ):

        self.patches = []
        self.masks = []

        self.augment = augment

        for image_name, gt_name in training_pairs:

            print_separator()

            print("Loading manual training pair")

            image_path = os.path.join(
                RAW_IMAGES_DIR,
                image_name,
            )

            gt_path = os.path.join(
                MANUAL_GT_DIR,
                gt_name,
            )

            print("Image:")
            print(image_path)

            print("GT:")
            print(gt_path)

            if not os.path.exists(image_path):
                print("WARNING: Image not found.")
                continue

            if not os.path.exists(gt_path):
                print("WARNING: GT not found.")
                continue

            # ------------------------------------------------
            # Read image
            # ------------------------------------------------

            with rasterio.open(image_path) as src:

                img = src.read().astype(
                    np.float32
                )

                height = src.height
                width = src.width

            # ------------------------------------------------
            # Read manual GT
            # ------------------------------------------------

            with rasterio.open(gt_path) as src:

                gt = src.read(1).astype(
                    np.float32
                )

            gt = resize_mask(
                gt,
                height,
                width,
            )

            # Convert GT to binary
            gt = (gt > 0.5).astype(
                np.float32
            )

            print(
                f"Image dimensions: "
                f"{width} x {height}"
            )

            print(
                "GT unique values:",
                np.unique(gt),
            )

            # ------------------------------------------------
            # Normalize image
            # ------------------------------------------------

            img = normalize_image(img)

            # ------------------------------------------------
            # Extract overlapping patches
            # ------------------------------------------------

            y_starts = get_patch_starts(
                height,
                PATCH_SIZE,
                STRIDE,
            )

            x_starts = get_patch_starts(
                width,
                PATCH_SIZE,
                STRIDE,
            )

            before = len(self.patches)

            for y in y_starts:

                for x in x_starts:

                    img_patch = img[
                        :,
                        y:y + PATCH_SIZE,
                        x:x + PATCH_SIZE,
                    ]

                    gt_patch = gt[
                        y:y + PATCH_SIZE,
                        x:x + PATCH_SIZE,
                    ]

                    if img_patch.shape != (
                        img.shape[0],
                        PATCH_SIZE,
                        PATCH_SIZE,
                    ):
                        continue

                    if gt_patch.shape != (
                        PATCH_SIZE,
                        PATCH_SIZE,
                    ):
                        continue

                    self.patches.append(
                        img_patch
                    )

                    self.masks.append(
                        gt_patch
                    )

            added = len(self.patches) - before

            print(
                f"Patches added: {added}"
            )

        print_separator()

        print(
            f"TOTAL BASE TRAINING PATCHES: "
            f"{len(self.patches)}"
        )

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):

        image = self.patches[index].copy()

        mask = self.masks[index].copy()

        if self.augment:

            image, mask = augment_patch(
                image,
                mask,
            )

        image = torch.from_numpy(
            image
        ).float()

        mask = torch.from_numpy(
            mask
        ).float().unsqueeze(0)

        return image, mask


# ================================================================
# LOSS FUNCTIONS
# ================================================================

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        smooth = 1.0

        intersection = (
            probs * targets
        ).sum(dim=(1, 2, 3))

        denominator = (
            probs.sum(dim=(1, 2, 3))
            +
            targets.sum(dim=(1, 2, 3))
        )

        dice = (
            2.0 * intersection
            + smooth
        ) / (
            denominator
            + smooth
        )

        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(
            logits,
            targets,
        )

        dice_loss = self.dice(
            logits,
            targets,
        )

        return (
            0.5 * bce_loss
            +
            0.5 * dice_loss
        )


# ================================================================
# MODEL
# ================================================================

def build_model():

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",

        # Sentinel-2:
        # Blue, Green, Red, NIR, SWIR
        in_channels=5,

        classes=1,

        activation=None,
    )

    return model.to(DEVICE)


# ================================================================
# VALIDATION
# ================================================================

def evaluate(
    model,
    loader,
    loss_fn,
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True,
            )

            logits = model(images)

            loss = loss_fn(
                logits,
                masks,
            )

            total_loss += loss.item()

    if len(loader) == 0:
        return 0.0

    return total_loss / len(loader)


# ================================================================
# TRAINING
# ================================================================

def train_model():

    print_separator()

    print("CAPSTONE U-NET V3")

    print_separator()

    print(
        "Training using manually verified 2018 GT:"
    )

    for lake, info in LAKES.items():

        print(
            f"  {lake}: "
            f"{info['manual_image']} "
            f"<-> "
            f"{info['manual_gt']}"
        )

    # ------------------------------------------------
    # Create base dataset
    # ------------------------------------------------

    training_pairs = []

    for info in LAKES.values():

        training_pairs.append(
            (
                info["manual_image"],
                info["manual_gt"],
            )
        )

    full_dataset = ManualGroundTruthDataset(
        training_pairs,
        augment=False,
    )

    if len(full_dataset) == 0:

        raise RuntimeError(
            "No training patches found."
        )

    # ------------------------------------------------
    # Split into train / validation
    # ------------------------------------------------

    indices = list(
        range(len(full_dataset))
    )

    random.shuffle(indices)

    val_count = max(
        1,
        int(
            len(indices)
            * VAL_SPLIT
        ),
    )

    val_indices = indices[
        :val_count
    ]

    train_indices = indices[
        val_count:
    ]

    # ------------------------------------------------
    # Create separate augmented dataset
    # ------------------------------------------------

    augmented_dataset = ManualGroundTruthDataset(
        training_pairs,
        augment=True,
    )

    train_subset = torch.utils.data.Subset(
        augmented_dataset,
        train_indices,
    )

    val_subset = torch.utils.data.Subset(
        full_dataset,
        val_indices,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print_separator()

    print(
        f"Total patches: {len(full_dataset)}"
    )

    print(
        f"Training patches: {len(train_subset)}"
    )

    print(
        f"Validation patches: {len(val_subset)}"
    )

    print(
        f"Device: {DEVICE}"
    )

    # ------------------------------------------------
    # Model
    # ------------------------------------------------

    model = build_model()

    loss_fn = CombinedLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_val_loss = float("inf")

    patience_counter = 0

    history = []

    best_model_path = os.path.join(
        MODEL_DIR,
        "unet_manual_boundary_v3_best.pth",
    )

    # ------------------------------------------------
    # Training
    # ------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        train_loss = 0.0

        for images, masks in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = loss_fn(
                logits,
                masks,
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(
            len(train_loader),
            1,
        )

        val_loss = evaluate(
            model,
            val_loader,
            loss_fn,
        )

        scheduler.step(
            val_loss
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Val Loss: {val_loss:.4f} "
            f"- LR: {current_lr:.2e}"
        )

        history.append(
            [
                epoch,
                train_loss,
                val_loss,
                current_lr,
            ]
        )

        # ------------------------------------------------
        # Best model
        # ------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            print(
                "  -> Best model saved."
            )

        else:

            patience_counter += 1

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # ------------------------------------------------
    # Save history
    # ------------------------------------------------

    history_path = os.path.join(
        TABLE_DIR,
        "training_history.csv",
    )

    with open(
        history_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "validation_loss",
                "learning_rate",
            ]
        )

        writer.writerows(history)

    # ------------------------------------------------
    # Load best model
    # ------------------------------------------------

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=DEVICE,
        )
    )

    print_separator()

    print(
        "BEST MODEL LOADED"
    )

    print(
        best_model_path
    )

    return model


# ================================================================
# FULL IMAGE U-NET INFERENCE
# ================================================================

def predict_full_image(
    model,
    image,
):

    model.eval()

    bands, height, width = image.shape

    y_starts = get_patch_starts(
        height,
        PATCH_SIZE,
        STRIDE,
    )

    x_starts = get_patch_starts(
        width,
        PATCH_SIZE,
        STRIDE,
    )

    probability_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    probability_count = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    with torch.no_grad():

        for y in y_starts:

            for x in x_starts:

                patch = image[
                    :,
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE,
                ]

                if patch.shape != (
                    bands,
                    PATCH_SIZE,
                    PATCH_SIZE,
                ):
                    continue

                tensor = torch.from_numpy(
                    patch
                ).float().unsqueeze(0)

                tensor = tensor.to(
                    DEVICE
                )

                logits = model(
                    tensor
                )

                probability = torch.sigmoid(
                    logits
                )

                probability = (
                    probability
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                probability_sum[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE,
                ] += probability

                probability_count[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE,
                ] += 1.0

    probability_count[
        probability_count == 0
    ] = 1.0

    probability = (
        probability_sum
        /
        probability_count
    )

    return probability


# ================================================================
# READ MNDWI MASK
# ================================================================

def load_mndwi_mask(
    lake,
    year,
    height,
    width,
):

    filename = (
        f"{lake}_{year}_MNDWI.tif"
    )

    path = os.path.join(
        MNDWI_DIR,
        filename,
    )

    if not os.path.exists(path):

        print(
            "WARNING: MNDWI not found:"
        )

        print(path)

        return None

    with rasterio.open(path) as src:

        mask = src.read(
            1,
            out_shape=(
                height,
                width,
            ),
            resampling=Resampling.nearest,
        ).astype(np.float32)

    # Existing MNDWI masks are binary.
    # This also handles masks stored as 0/255.
    if np.nanmax(mask) > 1:

        mask = mask / np.nanmax(mask)

    mask = (
        mask > 0.5
    ).astype(np.float32)

    return mask


# ================================================================
# CLEAN FINAL WATER MASK
# ================================================================

def clean_water_mask(
    mask,
):

    mask = mask.astype(bool)

    # ------------------------------------------------
    # Remove tiny isolated pixels
    # ------------------------------------------------

    structure = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    mask = ndimage.binary_opening(
        mask,
        structure=structure,
        iterations=1,
    )

    # ------------------------------------------------
    # Close tiny gaps
    # ------------------------------------------------

    mask = ndimage.binary_closing(
        mask,
        structure=structure,
        iterations=2,
    )

    # ------------------------------------------------
    # Connected components
    # ------------------------------------------------

    labeled, number = ndimage.label(
        mask,
        structure=structure,
    )

    if number == 0:

        return np.zeros_like(
            mask,
            dtype=np.uint8,
        )

    component_sizes = np.bincount(
        labeled.ravel()
    )

    component_sizes[0] = 0

    largest_label = (
        np.argmax(component_sizes)
    )

    largest_component = (
        labeled == largest_label
    )

    return largest_component.astype(
        np.uint8
    )


# ================================================================
# FUSION
# ================================================================

def fuse_unet_mndwi(
    probability,
    mndwi,
):

    # ------------------------------------------------
    # Step 1:
    # U-Net defines the broad possible lake region.
    #
    # This prevents completely unrelated water bodies
    # from being selected.
    # ------------------------------------------------

    boundary_region = (
        probability
        >= BOUNDARY_THRESHOLD
    )

    # ------------------------------------------------
    # Step 2:
    # Adaptive fusion.
    #
    # U-Net is the primary signal.
    # MNDWI provides additional evidence.
    #
    # If MNDWI says water, a lower U-Net probability
    # can still pass, but ONLY inside the U-Net region.
    # ------------------------------------------------

    fused_score = (
        UNET_WEIGHT * probability
        +
        MNDWI_WEIGHT * mndwi
    )

    raw_mask = (
        boundary_region
        &
        (
            fused_score
            >= FUSION_THRESHOLD
        )
    )

    # ------------------------------------------------
    # Step 3:
    # Always retain very strong U-Net predictions.
    #
    # This prevents MNDWI errors from deleting
    # genuine water detected confidently by U-Net.
    # ------------------------------------------------

    strong_unet = (
        probability >= 0.70
    )

    raw_mask = (
        raw_mask
        |
        strong_unet
    )

    raw_mask = raw_mask.astype(
        np.uint8
    )

    # ------------------------------------------------
    # Step 4:
    # Remove disconnected water bodies.
    # ------------------------------------------------

    final_mask = clean_water_mask(
        raw_mask
    )

    return (
        fused_score,
        raw_mask,
        final_mask,
    )


# ================================================================
# RGB VISUALIZATION
# ================================================================

def make_rgb(image):

    # Bands:
    # 0 = Blue
    # 1 = Green
    # 2 = Red
    # 3 = NIR
    # 4 = SWIR

    rgb = image[
        [2, 1, 0]
    ]

    rgb = np.transpose(
        rgb,
        (1, 2, 0),
    )

    return np.clip(
        rgb,
        0,
        1,
    )


# ================================================================
# SAVE RASTER
# ================================================================

def save_raster(
    path,
    array,
    reference_path,
    dtype,
):

    with rasterio.open(
        reference_path
    ) as src:

        profile = src.profile.copy()

        profile.update(
            count=1,
            dtype=dtype,
            compress="lzw",
        )

        with rasterio.open(
            path,
            "w",
            **profile,
        ) as dst:

            dst.write(
                array.astype(dtype),
                1,
            )


# ================================================================
# VISUALIZATION
# ================================================================

def save_visualization(
    image,
    probability,
    mndwi,
    raw_mask,
    final_mask,
    lake,
    year,
):

    rgb = make_rgb(
        image
    )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5),
    )

    axes[0].imshow(
        rgb
    )

    axes[0].set_title(
        "Sentinel-2 RGB"
    )

    axes[1].imshow(
        probability,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "U-Net Probability"
    )

    axes[2].imshow(
        mndwi,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        "MNDWI Mask"
    )

    axes[3].imshow(
        raw_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[3].set_title(
        "Raw U-Net + MNDWI"
    )

    axes[4].imshow(
        final_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[4].set_title(
        "Final Cleaned Lake Mask"
    )

    for ax in axes:

        ax.axis(
            "off"
        )

    plt.tight_layout()

    filename = (
        f"{lake}_{year}_"
        f"U-Net_MNDWI_final_check.png"
    )

    path = os.path.join(
        VIS_DIR,
        filename,
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved visualization:"
    )

    print(path)


# ================================================================
# PROCESS ONE IMAGE
# ================================================================

def process_image(
    model,
    lake,
    year,
):

    image_name = (
        f"{lake}_{year}_March.tif"
    )

    image_path = os.path.join(
        RAW_IMAGES_DIR,
        image_name,
    )

    if not os.path.exists(
        image_path
    ):

        print(
            f"SKIPPING: {image_name}"
        )

        return None

    print_separator()

    print(
        f"PROCESSING: {image_name}"
    )

    # ------------------------------------------------
    # Read image
    # ------------------------------------------------

    with rasterio.open(
        image_path
    ) as src:

        image = src.read().astype(
            np.float32
        )

        height = src.height
        width = src.width

    # ------------------------------------------------
    # Normalize exactly as training
    # ------------------------------------------------

    image = normalize_image(
        image
    )

    print(
        f"Image size: {width} x {height}"
    )

    # ------------------------------------------------
    # U-Net
    # ------------------------------------------------

    print(
        "Running U-Net..."
    )

    probability = predict_full_image(
        model,
        image,
    )

    # ------------------------------------------------
    # MNDWI
    # ------------------------------------------------

    print(
        "Loading existing MNDWI mask..."
    )

    mndwi = load_mndwi_mask(
        lake,
        year,
        height,
        width,
    )

    if mndwi is None:

        print(
            "No MNDWI available. "
            "Using U-Net only."
        )

        mndwi = np.zeros_like(
            probability,
            dtype=np.float32,
        )

    # ------------------------------------------------
    # Fusion
    # ------------------------------------------------

    print(
        "Fusing U-Net + MNDWI..."
    )

    fused_score, raw_mask, final_mask = (
        fuse_unet_mndwi(
            probability,
            mndwi,
        )
    )

    # ------------------------------------------------
    # Statistics
    # ------------------------------------------------

    total_pixels = (
        final_mask.size
    )

    water_pixels = int(
        final_mask.sum()
    )

    water_percentage = (
        water_pixels
        /
        total_pixels
        *
        100
    )

    print(
        f"Final water pixels: "
        f"{water_pixels}"
    )

    print(
        f"Total pixels: "
        f"{total_pixels}"
    )

    print(
        f"Final water percentage: "
        f"{water_percentage:.2f}%"
    )

    # ------------------------------------------------
    # Save probability
    # ------------------------------------------------

    probability_path = os.path.join(
        PROB_DIR,
        f"{lake}_{year}_U-Net_probability.tif",
    )

    save_raster(
        probability_path,
        probability,
        image_path,
        "float32",
    )

    # ------------------------------------------------
    # Save raw fusion
    # ------------------------------------------------

    raw_path = os.path.join(
        RAW_FUSION_DIR,
        f"{lake}_{year}_raw_U-Net_MNDWI_mask.tif",
    )

    save_raster(
        raw_path,
        raw_mask,
        image_path,
        "uint8",
    )

    # ------------------------------------------------
    # Save final mask
    # ------------------------------------------------

    final_path = os.path.join(
        FINAL_MASK_DIR,
        f"{lake}_{year}_FINAL_water_mask.tif",
    )

    save_raster(
        final_path,
        final_mask,
        image_path,
        "uint8",
    )

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------

    save_visualization(
        image,
        probability,
        mndwi,
        raw_mask,
        final_mask,
        lake,
        year,
    )

    print(
        "Saved probability:"
    )

    print(probability_path)

    print(
        "Saved raw fusion:"
    )

    print(raw_path)

    print(
        "Saved FINAL mask:"
    )

    print(final_path)

    return {
        "lake": lake,
        "year": year,
        "total_pixels": total_pixels,
        "water_pixels": water_pixels,
        "water_percentage": water_percentage,
        "probability_path": probability_path,
        "raw_mask_path": raw_path,
        "final_mask_path": final_path,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print("\n")
    print("#" * 70)

    print(
        "CAPSTONE U-NET V3"
    )

    print(
        "MANUAL GT + MNDWI WATER SEGMENTATION"
    )

    print("#" * 70)

    print()

    print(
        "Training data:"
    )

    print(
        "2018 manually verified GT "
        "for all three lakes."
    )

    print()

    print(
        "Inference:"
    )

    print(
        "2018–2025 Sentinel-2 imagery."
    )

    print()

    print(
        "MNDWI source:"
    )

    print(MNDWI_DIR)

    print()

    print(
        "Results:"
    )

    print(RESULTS_DIR)

    # ------------------------------------------------
    # Train
    # ------------------------------------------------

    model = train_model()

    # ------------------------------------------------
    # Process every lake/year
    # ------------------------------------------------

    summary = []

    for lake in LAKES:

        print("\n")

        print(
            "#" * 70
        )

        print(
            f"LAKE: {lake}"
        )

        print(
            "#" * 70
        )

        for year in range(
            2018,
            2026,
        ):

            result = process_image(
                model,
                lake,
                year,
            )

            if result is not None:

                summary.append(
                    result
                )

    # ------------------------------------------------
    # Save summary CSV
    # ------------------------------------------------

    summary_path = os.path.join(
        TABLE_DIR,
        "water_area_summary.csv",
    )

    with open(
        summary_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "Lake",
                "Year",
                "Total Pixels",
                "Water Pixels",
                "Water Percentage",
                "Probability Map",
                "Raw Fusion Mask",
                "Final Water Mask",
            ]
        )

        for row in summary:

            writer.writerow(
                [
                    row["lake"],
                    row["year"],
                    row["total_pixels"],
                    row["water_pixels"],
                    f'{row["water_percentage"]:.4f}',
                    row["probability_path"],
                    row["raw_mask_path"],
                    row["final_mask_path"],
                ]
            )

    # ------------------------------------------------
    # Print final summary
    # ------------------------------------------------

    print("\n")

    print(
        "#" * 70
    )

    print(
        "FINAL WATER-AREA SUMMARY"
    )

    print(
        "#" * 70
    )

    print()

    print(
        f"{'Lake':<18}"
        f"{'Year':<8}"
        f"{'Water %':>12}"
    )

    print(
        "-" * 40
    )

    for row in summary:

        print(
            f"{row['lake']:<18}"
            f"{row['year']:<8}"
            f"{row['water_percentage']:>11.2f}%"
        )

    print()

    print(
        "Summary saved:"
    )

    print(summary_path)

    print()

    print(
        "MODEL:"
    )

    print(
        os.path.join(
            MODEL_DIR,
            "unet_manual_boundary_v3_best.pth",
        )
    )

    print()

    print(
        "DONE."
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()