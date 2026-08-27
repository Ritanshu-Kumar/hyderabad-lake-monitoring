"""
unet_manual_mndwi_fusion.py

CAPSTONE U-NET + MNDWI WATER SEGMENTATION

Pipeline
--------
1. Train U-Net using manually verified 2018 ground truth for:
       - Hussain Sagar
       - Osman Sagar
       - Himayat Sagar

2. The U-Net learns the spatial characteristics and approximate
   boundary of the lake.

3. For every later year:
       Raw Sentinel-2 image
              |
              v
          U-Net
              |
              v
       Lake probability
              |
              v
       U-Net lake region
              |
              +-------- MNDWI mask
              |
              v
       Final water mask
              |
              v
       Remove disconnected water bodies

The manually created masks are NOT used for every year.
They are used only to train the initial U-Net boundary model.

The MNDWI masks are already generated and are read from:

    Data/3 mndwi_masks/

Output:

    Results/U-Net_Manual_MNDWI/

        Model/
            unet_manual_boundary_model_final.pth

        HussainSagar/
            2019/
            2020/
            ...
            2025/

        OsmanSagar/
            2019/
            ...
            2025/

        HimayatSagar/
            2019/
            ...
            2025/
"""

import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from scipy import ndimage


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = r"C:\Users\ritan\Downloads\Capstone_project"

RAW_IMAGES_DIR = os.path.join(
    BASE_DIR,
    "Data",
    "0 Raw_Images"
)

MNDWI_MASKS_DIR = os.path.join(
    BASE_DIR,
    "Data",
    "3 mndwi_masks"
)

MANUAL_GT_DIR = os.path.join(
    BASE_DIR,
    "Data",
    "5 manual_ground_truth"
)

# Completely separate output directory
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Results",
    "U-Net_Manual_MNDWI"
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "Model"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ================================================================
# MODEL / TRAINING PARAMETERS
# ================================================================

PATCH_SIZE = 256

# Overlapping patches are important because the original images
# are smaller than / not exact multiples of 256.
STRIDE = 128

EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# U-Net probability threshold
UNET_THRESHOLD = 0.50

# MNDWI mask is already binary.
# Pixels inside the learned U-Net region are allowed to remain
# water only when the MNDWI mask identifies them as water.

# Remove very small disconnected components.
MIN_COMPONENT_SIZE = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# TRAINING PAIRS
# ================================================================

MANUAL_TRAINING_PAIRS = [
    (
        "HussainSagar_2018_March.tif",
        "HussainSagar_GT.tif"
    ),
    (
        "OsmanSagar_2018_March.tif",
        "OsmanSagar_GT.tif"
    ),
    (
        "HimayatSagar_2018_March.tif",
        "HimayatSagar_GT.tif"
    ),
]


# ================================================================
# YEAR / LAKE CONFIGURATION
# ================================================================

LAKES = [
    "HussainSagar",
    "OsmanSagar",
    "HimayatSagar"
]

YEARS = list(range(2019, 2026))


# ================================================================
# UTILITY
# ================================================================

def print_separator():
    print("\n" + "=" * 70)


# ================================================================
# PATCH POSITION GENERATION
# ================================================================

def get_patch_positions(length, patch_size, stride):
    """
    Generate patch starting positions.

    Ensures that the final part of the image is also included,
    padding is used when necessary.
    """

    if length <= patch_size:
        return [0]

    positions = list(
        range(
            0,
            length - patch_size + 1,
            stride
        )
    )

    final_position = length - patch_size

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions


# ================================================================
# IMAGE NORMALIZATION
# ================================================================

def normalize_image(img):
    """
    Normalize Sentinel-2 reflectance values.

    Expected input:
        (bands, height, width)

    Expected bands:
        Blue, Green, Red, NIR, SWIR
    """

    img = img.astype(np.float32)

    # Sentinel-2 reflectance scaling
    img = np.clip(img / 3000.0, 0.0, 1.0)

    return img


# ================================================================
# TRAINING DATASET
# ================================================================

class ManualGroundTruthDataset(Dataset):

    def __init__(
        self,
        training_pairs,
        raw_dir,
        gt_dir,
        patch_size=256,
        stride=128
    ):

        self.images = []
        self.masks = []

        for image_name, gt_name in training_pairs:

            print_separator()
            print("Loading training pair")

            image_path = os.path.join(
                raw_dir,
                image_name
            )

            gt_path = os.path.join(
                gt_dir,
                gt_name
            )

            print(f"Image: {image_path}")
            print(f"GT   : {gt_path}")

            if not os.path.exists(image_path):
                raise FileNotFoundError(
                    f"Image not found:\n{image_path}"
                )

            if not os.path.exists(gt_path):
                raise FileNotFoundError(
                    f"Manual GT not found:\n{gt_path}"
                )

            # ----------------------------------------------------
            # Read image
            # ----------------------------------------------------

            with rasterio.open(image_path) as src:

                img = src.read().astype(
                    np.float32
                )

            # ----------------------------------------------------
            # Read manual ground truth
            # ----------------------------------------------------

            with rasterio.open(gt_path) as src:

                gt = src.read(1).astype(
                    np.float32
                )

            print(
                f"Image dimensions: "
                f"{img.shape[2]} x {img.shape[1]}"
            )

            print(
                "GT unique values:",
                np.unique(gt)
            )

            # ----------------------------------------------------
            # Check dimensions
            # ----------------------------------------------------

            if img.shape[1:] != gt.shape:

                raise ValueError(
                    f"Image/GT dimension mismatch:\n"
                    f"Image: {img.shape[1:]}\n"
                    f"GT:    {gt.shape}\n"
                    f"Pair: {image_name}"
                )

            # ----------------------------------------------------
            # Normalize image
            # ----------------------------------------------------

            img = normalize_image(img)

            height, width = gt.shape

            y_positions = get_patch_positions(
                height,
                patch_size,
                stride
            )

            x_positions = get_patch_positions(
                width,
                patch_size,
                stride
            )

            before = len(self.images)

            # ----------------------------------------------------
            # Extract patches
            # ----------------------------------------------------

            for y in y_positions:

                for x in x_positions:

                    img_patch = np.zeros(
                        (
                            img.shape[0],
                            patch_size,
                            patch_size
                        ),
                        dtype=np.float32
                    )

                    gt_patch = np.zeros(
                        (
                            patch_size,
                            patch_size
                        ),
                        dtype=np.float32
                    )

                    y_end = min(
                        y + patch_size,
                        height
                    )

                    x_end = min(
                        x + patch_size,
                        width
                    )

                    actual_h = y_end - y
                    actual_w = x_end - x

                    img_patch[
                        :,
                        :actual_h,
                        :actual_w
                    ] = img[
                        :,
                        y:y_end,
                        x:x_end
                    ]

                    gt_patch[
                        :actual_h,
                        :actual_w
                    ] = gt[
                        y:y_end,
                        x:x_end
                    ]

                    # Make absolutely sure the GT is binary.
                    gt_patch = (
                        gt_patch > 0.5
                    ).astype(
                        np.float32
                    )

                    self.images.append(
                        img_patch
                    )

                    self.masks.append(
                        gt_patch
                    )

            added = len(self.images) - before

            print(
                f"Patches added: {added}"
            )

        print_separator()

        print(
            f"TOTAL TRAINING PATCHES: "
            f"{len(self.images)}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = torch.tensor(
            self.images[index],
            dtype=torch.float32
        )

        mask = torch.tensor(
            self.masks[index],
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask


# ================================================================
# BUILD U-NET
# ================================================================

def build_model():

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=5,
        classes=1,
        activation=None
    )

    return model.to(DEVICE)


# ================================================================
# TRAIN U-NET
# ================================================================

def train_manual_boundary_model():

    print_separator()
    print("STAGE 1 — MANUAL-GT U-NET TRAINING")
    print_separator()

    dataset = ManualGroundTruthDataset(
        MANUAL_TRAINING_PAIRS,
        RAW_IMAGES_DIR,
        MANUAL_GT_DIR,
        PATCH_SIZE,
        STRIDE
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "No training patches were created."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = build_model()

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(f"Device: {DEVICE}")

    model.train()

    for epoch in range(1, EPOCHS + 1):

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

            loss = loss_fn(
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
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"- Loss: {average_loss:.4f}"
        )

    # ------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "unet_manual_boundary_model_final.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print_separator()
    print("MODEL SAVED")
    print(model_path)
    print_separator()

    return model


# ================================================================
# RUN U-NET ON FULL IMAGE
# ================================================================

def predict_full_image(
    model,
    image,
    patch_size=256,
    stride=128
):

    height = image.shape[1]
    width = image.shape[2]

    probability_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    probability_count = np.zeros(
        (height, width),
        dtype=np.float32
    )

    y_positions = get_patch_positions(
        height,
        patch_size,
        stride
    )

    x_positions = get_patch_positions(
        width,
        patch_size,
        stride
    )

    model.eval()

    with torch.no_grad():

        for y in y_positions:

            for x in x_positions:

                patch = np.zeros(
                    (
                        image.shape[0],
                        patch_size,
                        patch_size
                    ),
                    dtype=np.float32
                )

                y_end = min(
                    y + patch_size,
                    height
                )

                x_end = min(
                    x + patch_size,
                    width
                )

                actual_h = y_end - y
                actual_w = x_end - x

                patch[
                    :,
                    :actual_h,
                    :actual_w
                ] = image[
                    :,
                    y:y_end,
                    x:x_end
                ]

                tensor = torch.tensor(
                    patch,
                    dtype=torch.float32
                ).unsqueeze(0).to(
                    DEVICE
                )

                output = model(
                    tensor
                )

                probability = torch.sigmoid(
                    output
                )

                probability = (
                    probability
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                probability_sum[
                    y:y_end,
                    x:x_end
                ] += probability[
                    :actual_h,
                    :actual_w
                ]

                probability_count[
                    y:y_end,
                    x:x_end
                ] += 1.0

    probability_count[
        probability_count == 0
    ] = 1.0

    probability = (
        probability_sum /
        probability_count
    )

    return probability


# ================================================================
# LOAD MNDWI MASK
# ================================================================

def load_mndwi_mask(
    lake,
    year,
    expected_shape
):

    filename = (
        f"{lake}_{year}_MNDWI.tif"
    )

    path = os.path.join(
        MNDWI_MASKS_DIR,
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nMNDWI mask not found:\n{path}"
        )

    print(
        f"MNDWI mask: {path}"
    )

    with rasterio.open(path) as src:

        mndwi = src.read(1)

    if mndwi.shape != expected_shape:

        raise ValueError(
            f"\nMNDWI dimension mismatch:\n"
            f"Expected: {expected_shape}\n"
            f"Found:    {mndwi.shape}\n"
            f"File:     {path}"
        )

    # ------------------------------------------------------------
    # The files in 3 mndwi_masks are already masks.
    #
    # Convert them to binary:
    #
    # 0 = land
    # 1 = water
    # ------------------------------------------------------------

    mndwi_mask = (
        mndwi > 0.5
    ).astype(
        np.uint8
    )

    return mndwi_mask


# ================================================================
# REMOVE DISCONNECTED WATER BODIES
# ================================================================

def clean_connected_components(
    mask,
    min_component_size=100
):

    mask = mask.astype(
        np.uint8
    )

    labeled, number = ndimage.label(
        mask
    )

    if number == 0:
        return mask

    component_sizes = np.bincount(
        labeled.ravel()
    )

    cleaned = np.zeros_like(
        mask,
        dtype=np.uint8
    )

    for label_id in range(
        1,
        len(component_sizes)
    ):

        size = component_sizes[
            label_id
        ]

        if size >= min_component_size:

            cleaned[
                labeled == label_id
            ] = 1

    return cleaned


# ================================================================
# CREATE OUTPUT PROFILE
# ================================================================

def create_output_profile(
    image_path,
    dtype,
    count=1
):

    with rasterio.open(
        image_path
    ) as src:

        profile = src.profile.copy()

    profile.update(
        dtype=dtype,
        count=count,
        compress="lzw"
    )

    return profile


# ================================================================
# SAVE RASTER
# ================================================================

def save_raster(
    path,
    data,
    profile
):

    with rasterio.open(
        path,
        "w",
        **profile
    ) as dst:

        dst.write(
            data,
            1
        )


# ================================================================
# VISUALIZATION
# ================================================================

def save_visualization(
    image,
    probability,
    mndwi,
    raw_fusion,
    final_mask,
    output_path
):

    # RGB = Red, Green, Blue
    # Input bands = Blue, Green, Red, NIR, SWIR

    rgb = image[
        [2, 1, 0],
        :,
        :
    ]

    rgb = np.transpose(
        rgb,
        (1, 2, 0)
    )

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
        probability,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        "U-Net Probability"
    )

    axes[2].imshow(
        mndwi,
        cmap="gray"
    )

    axes[2].set_title(
        "MNDWI Mask"
    )

    axes[3].imshow(
        raw_fusion,
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

        ax.axis(
            "off"
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# PROCESS ONE YEAR
# ================================================================

def process_year(
    model,
    lake,
    year
):

    print_separator()

    image_name = (
        f"{lake}_{year}_March.tif"
    )

    image_path = os.path.join(
        RAW_IMAGES_DIR,
        image_name
    )

    if not os.path.exists(
        image_path
    ):

        print(
            f"Skipping {image_name} "
            f"— image not found."
        )

        return

    print(
        f"PROCESSING: {image_name}"
    )

    # ------------------------------------------------------------
    # Create separate output directory
    # ------------------------------------------------------------

    year_output_dir = os.path.join(
        OUTPUT_DIR,
        lake,
        str(year)
    )

    os.makedirs(
        year_output_dir,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Read Sentinel-2 image
    # ------------------------------------------------------------

    with rasterio.open(
        image_path
    ) as src:

        raw_image = src.read().astype(
            np.float32
        )

        source_profile = src.profile.copy()

    image = normalize_image(
        raw_image
    )

    height = image.shape[1]
    width = image.shape[2]

    print(
        f"Image size: {width} x {height}"
    )

    # ------------------------------------------------------------
    # Verify five input bands
    # ------------------------------------------------------------

    if image.shape[0] != 5:

        raise ValueError(
            f"Expected 5 bands, "
            f"but found {image.shape[0]} "
            f"in {image_name}"
        )

    # ------------------------------------------------------------
    # STAGE 1: U-NET
    # ------------------------------------------------------------

    print(
        "Running U-Net..."
    )

    probability = predict_full_image(
        model,
        image,
        PATCH_SIZE,
        STRIDE
    )

    # ------------------------------------------------------------
    # U-NET REGION
    # ------------------------------------------------------------

    unet_region = (
        probability >=
        UNET_THRESHOLD
    ).astype(
        np.uint8
    )

    # ------------------------------------------------------------
    # Load existing MNDWI mask
    # ------------------------------------------------------------

    print(
        "Loading MNDWI mask..."
    )

    mndwi_mask = load_mndwi_mask(
        lake,
        year,
        (height, width)
    )

    # ------------------------------------------------------------
    # FUSION
    #
    # U-Net determines WHERE the lake is allowed to exist.
    #
    # MNDWI determines WHICH pixels inside that region are water.
    #
    # Therefore:
    #
    # final/raw fusion =
    #       U-Net region AND MNDWI water
    #
    # This prevents isolated water-like pixels outside the
    # learned lake boundary from becoming water.
    # ------------------------------------------------------------

    print(
        "Fusing U-Net + MNDWI..."
    )

    raw_fusion = (
        unet_region &
        mndwi_mask
    ).astype(
        np.uint8
    )

    # ------------------------------------------------------------
    # CLEAN DISCONNECTED COMPONENTS
    # ------------------------------------------------------------

    print(
        "Removing disconnected water bodies..."
    )

    final_mask = clean_connected_components(
        raw_fusion,
        MIN_COMPONENT_SIZE
    )

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    water_pixels = int(
        np.sum(final_mask)
    )

    total_pixels = (
        final_mask.shape[0] *
        final_mask.shape[1]
    )

    water_percentage = (
        water_pixels /
        total_pixels *
        100.0
    )

    print(
        f"\nFinal water pixels: "
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

    # ============================================================
    # OUTPUT PATHS
    # ============================================================

    probability_path = os.path.join(
        year_output_dir,
        f"{lake}_{year}_U-Net_Probability.tif"
    )

    unet_boundary_path = os.path.join(
        year_output_dir,
        f"{lake}_{year}_U-Net_Boundary_Mask.tif"
    )

    raw_fusion_path = os.path.join(
        year_output_dir,
        f"{lake}_{year}_Raw_Fusion_Mask.tif"
    )

    final_mask_path = os.path.join(
        year_output_dir,
        f"{lake}_{year}_Final_Water_Mask.tif"
    )

    visualization_path = os.path.join(
        year_output_dir,
        f"{lake}_{year}_Segmentation_Check.png"
    )

    # ============================================================
    # OUTPUT PROFILE
    # ============================================================

    float_profile = source_profile.copy()

    float_profile.update(
        dtype="float32",
        count=1,
        compress="lzw"
    )

    uint8_profile = source_profile.copy()

    uint8_profile.update(
        dtype="uint8",
        count=1,
        compress="lzw"
    )

    # ============================================================
    # SAVE U-NET PROBABILITY
    # ============================================================

    save_raster(
        probability_path,
        probability.astype(
            np.float32
        ),
        float_profile
    )

    print(
        f"\nSaved U-Net probability:"
        f"\n{probability_path}"
    )

    # ============================================================
    # SAVE U-NET BOUNDARY MASK
    # ============================================================

    save_raster(
        unet_boundary_path,
        unet_region.astype(
            np.uint8
        ),
        uint8_profile
    )

    print(
        f"Saved U-Net boundary mask:"
        f"\n{unet_boundary_path}"
    )

    # ============================================================
    # SAVE RAW FUSION
    # ============================================================

    save_raster(
        raw_fusion_path,
        raw_fusion.astype(
            np.uint8
        ),
        uint8_profile
    )

    print(
        f"Saved raw fusion mask:"
        f"\n{raw_fusion_path}"
    )

    # ============================================================
    # SAVE FINAL MASK
    # ============================================================

    save_raster(
        final_mask_path,
        final_mask.astype(
            np.uint8
        ),
        uint8_profile
    )

    print(
        f"Saved FINAL water mask:"
        f"\n{final_mask_path}"
    )

    # ============================================================
    # SAVE VISUALIZATION
    # ============================================================

    save_visualization(
        image,
        probability,
        mndwi_mask,
        raw_fusion,
        final_mask,
        visualization_path
    )

    print(
        f"Saved visualization:"
        f"\n{visualization_path}"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("\n" + "=" * 70)
    print(
        "CAPSTONE U-NET + MNDWI WATER SEGMENTATION"
    )
    print("=" * 70)

    print(
        "\nTraining on manually verified 2018 GT:"
    )

    for image_name, gt_name in (
        MANUAL_TRAINING_PAIRS
    ):

        print(
            f"  {image_name} <-> {gt_name}"
        )

    print(
        "\nMNDWI masks are read from:"
    )

    print(
        f"  {MNDWI_MASKS_DIR}"
    )

    print(
        "\nResults will be saved to:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    # ============================================================
    # STAGE 1
    # ============================================================

    model = train_manual_boundary_model()

    # ============================================================
    # STAGE 2
    # ============================================================

    print_separator()
    print(
        "STAGE 2 — YEAR-WISE WATER SEGMENTATION"
    )
    print_separator()

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

        for year in YEARS:

            try:

                process_year(
                    model,
                    lake,
                    year
                )

            except Exception as error:

                print(
                    "\nERROR while processing:"
                )

                print(
                    f"{lake} {year}"
                )

                print(
                    error
                )

                print(
                    "\nContinuing with next year..."
                )

    # ============================================================
    # COMPLETE
    # ============================================================

    print("\n" + "=" * 70)
    print(
        "PROCESSING COMPLETE"
    )
    print("=" * 70)

    print(
        "\nModel:"
    )

    print(
        os.path.join(
            MODEL_DIR,
            "unet_manual_boundary_model_final.pth"
        )
    )

    print(
        "\nResults:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nEach lake/year contains:"
    )

    print(
        "  - U-Net probability"
    )

    print(
        "  - U-Net boundary mask"
    )

    print(
        "  - Raw U-Net + MNDWI fusion"
    )

    print(
        "  - Final cleaned water mask"
    )

    print(
        "  - Segmentation visualization"
    )

    print("=" * 70)


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()