"""
unet_manual_mndwi_fusion_v2.py

CAPSTONE:
U-Net + MNDWI Water Segmentation

Purpose
-------
1. Train U-Net using the manually verified 2018 lake masks.
2. Apply the trained U-Net to every available year (2018-2025).
3. Use MNDWI masks only as supporting evidence.
4. DO NOT force the final result to be the intersection of U-Net and MNDWI.
5. Preserve genuine lake-area predictions even when MNDWI misses water.
6. Remove small disconnected water-like objects.
7. Save all outputs into a separate results folder.

Training:
    Hussain Sagar 2018 -> manual GT
    Osman Sagar   2018 -> manual GT
    Himayat Sagar 2018 -> manual GT

Prediction:
    All available years for all three lakes.

Fusion philosophy:
    U-Net = PRIMARY
    MNDWI = SUPPORTING EVIDENCE

Final score:
    0.75 * U-Net probability
    + 0.25 * MNDWI evidence

This means:
    - Strong U-Net water prediction can remain water even if MNDWI misses it.
    - MNDWI can strengthen uncertain U-Net predictions.
    - Small isolated objects are removed afterward.
"""

import os
import glob
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from scipy import ndimage


# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"C:\Users\ritan\Downloads\Capstone_project"

RAW_IMAGES_DIR = os.path.join(
    BASE_DIR, "Data", "0 Raw_Images"
)

MANUAL_GT_DIR = os.path.join(
    BASE_DIR, "Data", "5 manual_ground_truth"
)

MNDWI_DIR = os.path.join(
    BASE_DIR, "Data", "3 mndwi_masks"
)

RESULTS_DIR = os.path.join(
    BASE_DIR, "Results", "U-Net_Manual_MNDWI_v2"
)

MODEL_DIR = os.path.join(
    RESULTS_DIR, "model"
)

PROBABILITY_DIR = os.path.join(
    RESULTS_DIR, "probability_maps"
)

RAW_MASK_DIR = os.path.join(
    RESULTS_DIR, "raw_fusion_masks"
)

FINAL_MASK_DIR = os.path.join(
    RESULTS_DIR, "final_water_masks"
)

VIS_DIR = os.path.join(
    RESULTS_DIR, "visualizations"
)


# Create output folders
for folder in [
    RESULTS_DIR,
    MODEL_DIR,
    PROBABILITY_DIR,
    RAW_MASK_DIR,
    FINAL_MASK_DIR,
    VIS_DIR,
]:
    os.makedirs(folder, exist_ok=True)


# ================================================================
# CONFIGURATION
# ================================================================

PATCH_SIZE = 256

EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# U-Net is deliberately given more weight.
UNET_WEIGHT = 0.75
MNDWI_WEIGHT = 0.25

# Main segmentation threshold.
# Lowering this slightly allows U-Net to retain lake boundary pixels.
FINAL_THRESHOLD = 0.50

# Strong U-Net prediction:
# if U-Net is above this value, water can survive even when
# MNDWI does not detect it.
STRONG_UNET_THRESHOLD = 0.65

# Remove very small disconnected components.
MIN_COMPONENT_PIXELS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# TRAINING PAIRS
# ================================================================

TRAINING_PAIRS = [
    (
        "HussainSagar_2018_March.tif",
        "HussainSagar_GT.tif",
    ),
    (
        "OsmanSagar_2018_March.tif",
        "OsmanSagar_GT.tif",
    ),
    (
        "HimayatSagar_2018_March.tif",
        "HimayatSagar_GT.tif",
    ),
]


# ================================================================
# UTILITY
# ================================================================

def print_header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ================================================================
# DATASET
# ================================================================

class ManualGroundTruthDataset(Dataset):

    def __init__(
        self,
        pairs,
        raw_dir,
        gt_dir,
        patch_size=256,
    ):

        self.images = []
        self.masks = []

        print_header("LOADING MANUAL GROUND-TRUTH TRAINING DATA")

        for image_name, gt_name in pairs:

            image_path = os.path.join(
                raw_dir,
                image_name
            )

            gt_path = os.path.join(
                gt_dir,
                gt_name
            )

            print("\n" + "-" * 60)
            print("Image:", image_path)
            print("GT   :", gt_path)

            if not os.path.exists(image_path):
                print("WARNING: Image not found.")
                continue

            if not os.path.exists(gt_path):
                print("WARNING: Ground truth not found.")
                continue

            with rasterio.open(image_path) as src:

                image = src.read().astype(
                    np.float32
                )

            with rasterio.open(gt_path) as src:

                mask = src.read(1).astype(
                    np.float32
                )

            print(
                f"Image dimensions: "
                f"{image.shape[2]} x {image.shape[1]}"
            )

            print(
                "GT unique values:",
                np.unique(mask)
            )

            # ----------------------------------------------------
            # Make sure GT is binary
            # ----------------------------------------------------

            mask = (mask > 0.5).astype(
                np.float32
            )

            # ----------------------------------------------------
            # Normalize Sentinel-2 reflectance
            # ----------------------------------------------------

            image = image / 3000.0
            image = np.clip(
                image,
                0,
                1
            )

            h, w = mask.shape

            added = 0

            # ----------------------------------------------------
            # Non-overlapping patches
            # ----------------------------------------------------

            for y in range(
                0,
                h - patch_size + 1,
                patch_size
            ):

                for x in range(
                    0,
                    w - patch_size + 1,
                    patch_size
                ):

                    image_patch = image[
                        :,
                        y:y + patch_size,
                        x:x + patch_size
                    ]

                    mask_patch = mask[
                        y:y + patch_size,
                        x:x + patch_size
                    ]

                    if image_patch.shape[1:] != (
                        patch_size,
                        patch_size
                    ):
                        continue

                    self.images.append(
                        image_patch
                    )

                    self.masks.append(
                        mask_patch
                    )

                    added += 1

            print(
                f"Patches added: {added}"
            )

        print_header(
            f"TOTAL TRAINING PATCHES: {len(self.images)}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index].copy()
        mask = self.masks[index].copy()

        # --------------------------------------------------------
        # Data augmentation
        #
        # This is especially useful because we only have three
        # manually labelled 2018 images.
        # --------------------------------------------------------

        if np.random.random() < 0.5:

            image = np.flip(
                image,
                axis=2
            ).copy()

            mask = np.flip(
                mask,
                axis=1
            ).copy()

        if np.random.random() < 0.5:

            image = np.flip(
                image,
                axis=1
            ).copy()

            mask = np.flip(
                mask,
                axis=0
            ).copy()

        # 90-degree rotation
        if np.random.random() < 0.5:

            k = np.random.randint(
                1,
                4
            )

            image = np.rot90(
                image,
                k,
                axes=(1, 2)
            ).copy()

            mask = np.rot90(
                mask,
                k
            ).copy()

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask


# ================================================================
# LOSS
# ================================================================

class DiceBCELoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits,
        targets
    ):

        bce_loss = self.bce(
            logits,
            targets
        )

        probabilities = torch.sigmoid(
            logits
        )

        smooth = 1.0

        intersection = (
            probabilities * targets
        ).sum()

        dice = (
            2 * intersection + smooth
        ) / (
            probabilities.sum()
            + targets.sum()
            + smooth
        )

        dice_loss = 1 - dice

        return bce_loss + dice_loss


# ================================================================
# BUILD U-NET
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
# TRAIN
# ================================================================

def train_model():

    dataset = ManualGroundTruthDataset(
        TRAINING_PAIRS,
        RAW_IMAGES_DIR,
        MANUAL_GT_DIR,
        PATCH_SIZE
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "No manual-ground-truth patches were loaded."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = build_model()

    loss_function = DiceBCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print_header(
        "STAGE 1 — TRAINING U-NET USING MANUAL 2018 GT"
    )

    print(
        "Device:",
        DEVICE
    )

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        total_loss = 0.0

        for images, masks in loader:

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            optimizer.zero_grad()

            predictions = model(
                images
            )

            loss = loss_function(
                predictions,
                masks
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss /
            len(loader)
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{EPOCHS} "
            f"- Loss: "
            f"{average_loss:.4f}"
        )

    model_path = os.path.join(
        MODEL_DIR,
        "unet_manual_boundary_model_v2.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print_header(
        "MODEL SAVED"
    )

    print(model_path)

    return model


# ================================================================
# FIND MNDWI MASK
# ================================================================

def find_mndwi_mask(
    lake_name,
    year
):

    filename = (
        f"{lake_name}_{year}_MNDWI.tif"
    )

    path = os.path.join(
        MNDWI_DIR,
        filename
    )

    if os.path.exists(path):
        return path

    return None


# ================================================================
# READ SENTINEL IMAGE
# ================================================================

def read_image(
    image_path
):

    with rasterio.open(
        image_path
    ) as src:

        image = src.read().astype(
            np.float32
        )

        profile = src.profile.copy()

    return image, profile


# ================================================================
# RUN U-NET ON FULL IMAGE
# ================================================================

def run_unet_full_image(
    model,
    image
):

    model.eval()

    # Normalize
    image_normalized = (
        image / 3000.0
    )

    image_normalized = np.clip(
        image_normalized,
        0,
        1
    )

    _, h, w = image.shape

    probability = np.zeros(
        (h, w),
        dtype=np.float32
    )

    count = np.zeros(
        (h, w),
        dtype=np.float32
    )

    step = PATCH_SIZE

    with torch.no_grad():

        for y in range(
            0,
            h,
            step
        ):

            for x in range(
                0,
                w,
                step
            ):

                y2 = min(
                    y + PATCH_SIZE,
                    h
                )

                x2 = min(
                    x + PATCH_SIZE,
                    w
                )

                patch = image_normalized[
                    :,
                    y:y2,
                    x:x2
                ]

                ph, pw = patch.shape[1:]

                # ------------------------------------------------
                # Ignore tiny edge patches.
                # ------------------------------------------------

                if ph < 32 or pw < 32:
                    continue

                padded = np.zeros(
                    (
                        5,
                        PATCH_SIZE,
                        PATCH_SIZE
                    ),
                    dtype=np.float32
                )

                padded[
                    :,
                    :ph,
                    :pw
                ] = patch

                tensor = torch.tensor(
                    padded,
                    dtype=torch.float32
                ).unsqueeze(0).to(
                    DEVICE
                )

                prediction = torch.sigmoid(
                    model(tensor)
                )

                prediction = (
                    prediction
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                prediction = prediction[
                    :ph,
                    :pw
                ]

                probability[
                    y:y2,
                    x:x2
                ] += prediction

                count[
                    y:y2,
                    x:x2
                ] += 1

    count[count == 0] = 1

    probability /= count

    return probability


# ================================================================
# READ MNDWI MASK
# ================================================================

def read_mndwi_mask(
    mndwi_path,
    target_shape
):

    with rasterio.open(
        mndwi_path
    ) as src:

        mndwi = src.read(1).astype(
            np.float32
        )

    # ------------------------------------------------------------
    # Normally the dimensions should already match.
    # ------------------------------------------------------------

    if mndwi.shape != target_shape:

        print(
            "WARNING: MNDWI dimensions do not match "
            "the Sentinel-2 image."
        )

        print(
            "MNDWI:",
            mndwi.shape,
            "Image:",
            target_shape
        )

        # Nearest-neighbour resizing
        zoom_y = (
            target_shape[0] /
            mndwi.shape[0]
        )

        zoom_x = (
            target_shape[1] /
            mndwi.shape[1]
        )

        mndwi = ndimage.zoom(
            mndwi,
            (
                zoom_y,
                zoom_x
            ),
            order=0
        )

        mndwi = mndwi[
            :target_shape[0],
            :target_shape[1]
        ]

    # Binary MNDWI mask
    mndwi_binary = (
        mndwi > 0.5
    ).astype(
        np.float32
    )

    return mndwi_binary


# ================================================================
# FUSION
# ================================================================

def fuse_unet_and_mndwi(
    unet_probability,
    mndwi_mask
):

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # MNDWI is SUPPORTING evidence.
    #
    # We do NOT use:
    #
    #     U-Net AND MNDWI
    #
    # because that would throw away real water whenever MNDWI
    # makes a false-negative.
    #
    # Instead we use a weighted score.
    # ------------------------------------------------------------

    fused_score = (
        UNET_WEIGHT *
        unet_probability
        +
        MNDWI_WEIGHT *
        mndwi_mask
    )

    # ------------------------------------------------------------
    # Strong U-Net prediction override
    #
    # If U-Net is very confident that a pixel is water,
    # MNDWI is NOT allowed to remove it.
    # ------------------------------------------------------------

    strong_unet_water = (
        unet_probability >=
        STRONG_UNET_THRESHOLD
    )

    final_mask = (
        fused_score >=
        FINAL_THRESHOLD
    )

    final_mask[
        strong_unet_water
    ] = 1

    return fused_score, final_mask.astype(
        np.uint8
    )


# ================================================================
# REMOVE SMALL COMPONENTS
# ================================================================

def clean_mask(
    mask
):

    # Connected component labelling
    labeled, number = ndimage.label(
        mask
    )

    if number == 0:
        return mask.astype(
            np.uint8
        )

    component_sizes = np.bincount(
        labeled.ravel()
    )

    # Keep only components above threshold
    keep = (
        component_sizes >=
        MIN_COMPONENT_PIXELS
    )

    keep[0] = False

    cleaned = keep[
        labeled
    ]

    # ------------------------------------------------------------
    # Keep the largest component if possible.
    #
    # This prevents isolated water-looking objects outside the
    # lake from becoming part of the final lake mask.
    # ------------------------------------------------------------

    labeled2, number2 = ndimage.label(
        cleaned
    )

    if number2 > 0:

        sizes = np.bincount(
            labeled2.ravel()
        )

        sizes[0] = 0

        largest_label = np.argmax(
            sizes
        )

        largest = (
            labeled2 ==
            largest_label
        )

        # --------------------------------------------------------
        # We keep the largest component as the main lake.
        # Small components that passed the first threshold are
        # removed.
        # --------------------------------------------------------

        cleaned = largest

    # ------------------------------------------------------------
    # Fill tiny holes inside the lake.
    # ------------------------------------------------------------

    cleaned = ndimage.binary_closing(
        cleaned,
        structure=np.ones(
            (3, 3)
        ),
        iterations=1
    )

    return cleaned.astype(
        np.uint8
    )


# ================================================================
# SAVE RASTER
# ================================================================

def save_raster(
    path,
    array,
    profile,
    dtype
):

    output_profile = profile.copy()

    output_profile.update(
        count=1,
        dtype=dtype,
        compress="lzw"
    )

    with rasterio.open(
        path,
        "w",
        **output_profile
    ) as dst:

        dst.write(
            array.astype(dtype),
            1
        )


# ================================================================
# VISUALIZATION
# ================================================================

def create_visualization(
    image,
    unet_probability,
    mndwi_mask,
    fused_mask,
    final_mask,
    output_path,
    lake_name,
    year
):

    # RGB
    rgb = image[
        [2, 1, 0],
        :,
        :
    ].transpose(
        1,
        2,
        0
    )

    rgb = rgb / 3000.0
    rgb = np.clip(
        rgb,
        0,
        1
    )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 5)
    )

    axes[0].imshow(
        rgb
    )

    axes[0].set_title(
        "Sentinel-2 RGB"
    )

    axes[1].imshow(
        unet_probability,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        "U-Net Probability"
    )

    axes[2].imshow(
        mndwi_mask,
        cmap="gray"
    )

    axes[2].set_title(
        "MNDWI Mask"
    )

    axes[3].imshow(
        fused_mask,
        cmap="gray"
    )

    axes[3].set_title(
        "Raw U-Net + MNDWI"
    )

    axes[4].imshow(
        final_mask,
        cmap="gray"
    )

    axes[4].set_title(
        "Final Cleaned Lake Mask"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{lake_name} — {year}",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# PROCESS ONE IMAGE
# ================================================================

def process_image(
    model,
    image_path
):

    filename = os.path.basename(
        image_path
    )

    # Example:
    # HussainSagar_2019_March.tif

    name_without_ext = os.path.splitext(
        filename
    )[0]

    parts = name_without_ext.split(
        "_"
    )

    lake_name = parts[0]
    year = parts[1]

    print_header(
        f"PROCESSING {lake_name} — {year}"
    )

    # ------------------------------------------------------------
    # Read image
    # ------------------------------------------------------------

    image, profile = read_image(
        image_path
    )

    print(
        "Image size:",
        image.shape[2],
        "x",
        image.shape[1]
    )

    # ------------------------------------------------------------
    # Find MNDWI
    # ------------------------------------------------------------

    mndwi_path = find_mndwi_mask(
        lake_name,
        year
    )

    if mndwi_path is None:

        print(
            "MNDWI mask not found."
        )

        print(
            "Skipping this image."
        )

        return

    print(
        "MNDWI:",
        mndwi_path
    )

    # ------------------------------------------------------------
    # U-NET
    # ------------------------------------------------------------

    print(
        "Running U-Net..."
    )

    unet_probability = run_unet_full_image(
        model,
        image
    )

    # ------------------------------------------------------------
    # MNDWI
    # ------------------------------------------------------------

    print(
        "Reading MNDWI..."
    )

    mndwi_mask = read_mndwi_mask(
        mndwi_path,
        (
            image.shape[1],
            image.shape[2]
        )
    )

    # ------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------

    print(
        "Fusing U-Net + MNDWI..."
    )

    fused_score, raw_mask = (
        fuse_unet_and_mndwi(
            unet_probability,
            mndwi_mask
        )
    )

    # ------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------

    print(
        "Removing disconnected water bodies..."
    )

    final_mask = clean_mask(
        raw_mask
    )

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    total_pixels = final_mask.size

    water_pixels = int(
        final_mask.sum()
    )

    water_percentage = (
        water_pixels /
        total_pixels *
        100
    )

    print()
    print(
        "Final water pixels:",
        water_pixels
    )

    print(
        "Total pixels:",
        total_pixels
    )

    print(
        f"Final water percentage: "
        f"{water_percentage:.2f}%"
    )

    # ------------------------------------------------------------
    # Output names
    # ------------------------------------------------------------

    probability_path = os.path.join(
        PROBABILITY_DIR,
        f"{lake_name}_{year}_U-Net_probability.tif"
    )

    raw_mask_path = os.path.join(
        RAW_MASK_DIR,
        f"{lake_name}_{year}_U-Net_MNDWI_raw_mask.tif"
    )

    final_mask_path = os.path.join(
        FINAL_MASK_DIR,
        f"{lake_name}_{year}_FINAL_water_mask.tif"
    )

    visualization_path = os.path.join(
        VIS_DIR,
        f"{lake_name}_{year}_segmentation_check.png"
    )

    # ------------------------------------------------------------
    # Save probability
    # ------------------------------------------------------------

    save_raster(
        probability_path,
        unet_probability,
        profile,
        "float32"
    )

    # ------------------------------------------------------------
    # Save raw fusion
    # ------------------------------------------------------------

    save_raster(
        raw_mask_path,
        raw_mask,
        profile,
        "uint8"
    )

    # ------------------------------------------------------------
    # Save final mask
    # ------------------------------------------------------------

    save_raster(
        final_mask_path,
        final_mask,
        profile,
        "uint8"
    )

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------

    create_visualization(
        image,
        unet_probability,
        mndwi_mask,
        raw_mask,
        final_mask,
        visualization_path,
        lake_name,
        year
    )

    print()
    print(
        "Saved probability:"
    )

    print(
        probability_path
    )

    print(
        "Saved raw fusion mask:"
    )

    print(
        raw_mask_path
    )

    print(
        "Saved FINAL lake mask:"
    )

    print(
        final_mask_path
    )

    print(
        "Saved visualization:"
    )

    print(
        visualization_path
    )


# ================================================================
# FIND ALL SENTINEL-2 IMAGES
# ================================================================

def find_all_images():

    images = []

    for filename in sorted(
        os.listdir(
            RAW_IMAGES_DIR
        )
    ):

        if not filename.lower().endswith(
            ".tif"
        ):
            continue

        # Ignore auxiliary files
        if ".tif." in filename.lower():
            continue

        # Only our March Sentinel images
        if "_March" not in filename:
            continue

        images.append(
            os.path.join(
                RAW_IMAGES_DIR,
                filename
            )
        )

    return images


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print_header(
        "CAPSTONE U-NET + MNDWI WATER SEGMENTATION v2"
    )

    print(
        "Training source:"
    )

    print(
        "2018 manually verified ground truth"
    )

    print()

    print(
        "U-Net role:"
    )

    print(
        "PRIMARY lake boundary and water segmentation"
    )

    print()

    print(
        "MNDWI role:"
    )

    print(
        "SUPPORTING water evidence"
    )

    print()

    print(
        "Fusion:"
    )

    print(
        f"{UNET_WEIGHT:.2f} U-Net + "
        f"{MNDWI_WEIGHT:.2f} MNDWI"
    )

    print()

    print(
        "Results directory:"
    )

    print(
        RESULTS_DIR
    )

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    model = train_model()

    # ------------------------------------------------------------
    # PROCESS ALL YEARS
    # ------------------------------------------------------------

    images = find_all_images()

    print_header(
        f"FOUND {len(images)} SENTINEL-2 IMAGES"
    )

    for image_path in images:

        try:

            process_image(
                model,
                image_path
            )

        except Exception as error:

            print()
            print(
                "ERROR PROCESSING:",
                image_path
            )

            print(
                error
            )

            print(
                "Continuing with next image..."
            )

    # ------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------

    print_header(
        "PROCESSING COMPLETE"
    )

    print(
        "All results are stored in:"
    )

    print(
        RESULTS_DIR
    )