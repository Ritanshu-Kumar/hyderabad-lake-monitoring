"""
test_v3_unseen_years.py

INFERENCE-ONLY EVALUATION OF THE ALREADY TRAINED V3 MODEL.

The V3 model was trained using:
    - Hussain Sagar 2018 manual GT
    - Osman Sagar 2018 manual GT
    - Himayat Sagar 2018 manual GT

This script DOES NOT TRAIN the model.

It loads:
    unet_manual_boundary_v3_best.pth

and applies it to previously unseen years:

    2019
    2020
    2021

for:

    Hussain Sagar
    Osman Sagar
    Himayat Sagar

Purpose:
    Demonstrate temporal generalization:
        2018 manual training
                  ↓
             Frozen V3
                  ↓
        unseen 2019/2020/2021
                  ↓
             predictions

IMPORTANT:
    There is no numerical IoU/Dice calculation for 2019–2021
    because we do not have manually verified GT for those years.

This script produces:
    - U-Net probability maps
    - U-Net masks
    - RGB + probability + MNDWI + final visualizations
    - water percentage CSV

The existing V3 model and previous results are NOT modified.
"""

import os
import csv

import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from scipy import ndimage


# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"C:\Users\ritan\Downloads\Capstone_project"

RAW_IMAGES_DIR = os.path.join(
    BASE_DIR,
    "Data",
    "0 Raw_Images",
)

MNDWI_DIR = os.path.join(
    BASE_DIR,
    "Data",
    "3 mndwi_masks",
)

# Existing trained V3 model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "Results",
    "U-Net_Manual_MNDWI_v3",
    "model",
    "unet_manual_boundary_v3_best.pth",
)

# Completely separate output folder
RESULTS_DIR = os.path.join(
    BASE_DIR,
    "Results",
    "V3_Unseen_Year_Test",
)

PROBABILITY_DIR = os.path.join(
    RESULTS_DIR,
    "probability_maps",
)

UNET_MASK_DIR = os.path.join(
    RESULTS_DIR,
    "unet_masks",
)

FINAL_MASK_DIR = os.path.join(
    RESULTS_DIR,
    "final_masks",
)

VIS_DIR = os.path.join(
    RESULTS_DIR,
    "visualizations",
)

TABLE_DIR = os.path.join(
    RESULTS_DIR,
    "tables",
)


for folder in [
    RESULTS_DIR,
    PROBABILITY_DIR,
    UNET_MASK_DIR,
    FINAL_MASK_DIR,
    VIS_DIR,
    TABLE_DIR,
]:

    os.makedirs(
        folder,
        exist_ok=True,
    )


# ================================================================
# CONFIGURATION
# ================================================================

PATCH_SIZE = 256

STRIDE = 128

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# These are deliberately unseen during training.
TEST_YEARS = [
    2019,
    2020,
    2021,
]

LAKES = [
    "HussainSagar",
    "OsmanSagar",
    "HimayatSagar",
]


# ================================================================
# V3 FUSION PARAMETERS
# ================================================================

BOUNDARY_THRESHOLD = 0.15

FUSION_THRESHOLD = 0.50

UNET_WEIGHT = 0.70

MNDWI_WEIGHT = 0.30


# ================================================================
# MODEL
# ================================================================

def build_model():

    print(
        "Creating V3 architecture..."
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=5,
        classes=1,
        activation=None,
    )

    model = model.to(
        DEVICE
    )

    return model


def load_v3_model():

    print(
        "\nLoading existing V3 model:"
    )

    print(
        MODEL_PATH
    )

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "\nV3 model not found:\n"
            + MODEL_PATH
            + "\n\n"
            "Make sure unet_manual_mndwi_v3.py "
            "has already been run."
        )

    model = build_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    print(
        "V3 model loaded successfully."
    )

    print(
        "NO TRAINING WILL BE PERFORMED."
    )

    return model


# ================================================================
# NORMALIZATION
# ================================================================

def normalize_image(image):

    """
    Same normalization used by V3.
    """

    image = image.astype(
        np.float32
    )

    output = np.zeros_like(
        image,
        dtype=np.float32,
    )

    for band in range(
        image.shape[0]
    ):

        data = image[band]

        valid = np.isfinite(
            data
        )

        if not np.any(valid):
            continue

        values = data[valid]

        low = np.percentile(
            values,
            2,
        )

        high = np.percentile(
            values,
            98,
        )

        if high <= low:

            high = low + 1.0

        output[band] = np.clip(
            (data - low)
            /
            (high - low),
            0,
            1,
        )

    output[
        ~np.isfinite(output)
    ] = 0

    return output


# ================================================================
# PATCH POSITIONS
# ================================================================

def get_patch_starts(
    length,
    patch_size,
    stride,
):

    if length <= patch_size:

        return [0]

    starts = list(
        range(
            0,
            length - patch_size + 1,
            stride,
        )
    )

    final_start = (
        length - patch_size
    )

    if starts[-1] != final_start:

        starts.append(
            final_start
        )

    return starts


# ================================================================
# FULL IMAGE U-NET PREDICTION
# ================================================================

def predict_full_image(
    model,
    image,
):

    model.eval()

    bands, height, width = (
        image.shape
    )

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

    count = np.zeros(
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

                count[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE,
                ] += 1.0

    count[
        count == 0
    ] = 1.0

    return (
        probability_sum
        /
        count
    )


# ================================================================
# LOAD MNDWI
# ================================================================

def load_mndwi(
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

    if not os.path.exists(
        path
    ):

        print(
            "WARNING: MNDWI file not found:"
        )

        print(path)

        return np.zeros(
            (height, width),
            dtype=np.float32,
        )

    with rasterio.open(
        path
    ) as src:

        mndwi = src.read(
            1
        ).astype(
            np.float32
        )

    if mndwi.shape != (
        height,
        width,
    ):

        raise RuntimeError(
            f"MNDWI dimensions do not match "
            f"image for {lake} {year}.\n"
            f"Image: {width} x {height}\n"
            f"MNDWI: {mndwi.shape[1]} x "
            f"{mndwi.shape[0]}"
        )

    # Convert 0/255 masks to 0/1
    if np.nanmax(mndwi) > 1:

        mndwi = (
            mndwi
            /
            np.nanmax(mndwi)
        )

    return (
        mndwi > 0.5
    ).astype(
        np.float32
    )


# ================================================================
# FUSION
# ================================================================

def fuse_predictions(
    probability,
    mndwi,
):

    # ------------------------------------------------
    # U-Net defines broad possible lake region.
    # ------------------------------------------------

    boundary_region = (
        probability
        >= BOUNDARY_THRESHOLD
    )

    # ------------------------------------------------
    # U-Net + MNDWI
    # ------------------------------------------------

    fused_score = (
        UNET_WEIGHT
        * probability
        +
        MNDWI_WEIGHT
        * mndwi
    )

    raw_mask = (
        boundary_region
        &
        (
            fused_score
            >= FUSION_THRESHOLD
        )
    )

    # Preserve strong U-Net detections.
    strong_unet = (
        probability
        >= 0.70
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
    # Connected component cleanup
    # ------------------------------------------------

    structure = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    # Remove tiny isolated pixels.
    cleaned = ndimage.binary_opening(
        raw_mask,
        structure=structure,
        iterations=1,
    )

    # Close small gaps.
    cleaned = ndimage.binary_closing(
        cleaned,
        structure=structure,
        iterations=2,
    )

    labeled, number = ndimage.label(
        cleaned,
        structure=structure,
    )

    if number == 0:

        final_mask = np.zeros_like(
            raw_mask,
            dtype=np.uint8,
        )

    else:

        component_sizes = np.bincount(
            labeled.ravel()
        )

        component_sizes[0] = 0

        largest_label = np.argmax(
            component_sizes
        )

        final_mask = (
            labeled
            ==
            largest_label
        ).astype(
            np.uint8
        )

    return (
        fused_score,
        raw_mask,
        final_mask,
    )


# ================================================================
# RGB
# ================================================================

def make_rgb(
    image,
):

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
    lake,
    year,
    image,
    probability,
    mndwi,
    raw_mask,
    final_mask,
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
        f"{lake} {year}\nSentinel-2 RGB"
    )

    axes[1].imshow(
        probability,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "V3 U-Net Probability"
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
        "U-Net + MNDWI"
    )

    axes[4].imshow(
        final_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[4].set_title(
        "Final Water Mask"
    )

    for ax in axes:

        ax.axis(
            "off"
        )

    plt.tight_layout()

    path = os.path.join(
        VIS_DIR,
        f"{lake}_{year}_V3_unseen_year.png",
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
# PROCESS ONE YEAR
# ================================================================

def process_year(
    model,
    lake,
    year,
):

    print("\n")
    print("-" * 70)

    print(
        f"UNSEEN YEAR: {lake} {year}"
    )

    print("-" * 70)

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
            "Image not found:"
        )

        print(image_path)

        return None

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

    print(
        f"Image size: "
        f"{width} x {height}"
    )

    # ------------------------------------------------
    # Normalize
    # ------------------------------------------------

    image = normalize_image(
        image
    )

    # ------------------------------------------------
    # V3 prediction
    # ------------------------------------------------

    print(
        "Running frozen V3 U-Net..."
    )

    probability = predict_full_image(
        model,
        image,
    )

    # ------------------------------------------------
    # MNDWI
    # ------------------------------------------------

    print(
        "Loading year's MNDWI mask..."
    )

    mndwi = load_mndwi(
        lake,
        year,
        height,
        width,
    )

    # ------------------------------------------------
    # Fusion
    # ------------------------------------------------

    print(
        "Running V3 fusion..."
    )

    (
        fused_score,
        raw_mask,
        final_mask,
    ) = fuse_predictions(
        probability,
        mndwi,
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

    print()

    print(
        f"Final water pixels: "
        f"{water_pixels}"
    )

    print(
        f"Total pixels: "
        f"{total_pixels}"
    )

    print(
        f"Water percentage: "
        f"{water_percentage:.2f}%"
    )

    # ------------------------------------------------
    # Save probability
    # ------------------------------------------------

    probability_path = os.path.join(
        PROBABILITY_DIR,
        f"{lake}_{year}_V3_probability.tif",
    )

    save_raster(
        probability_path,
        probability,
        image_path,
        "float32",
    )

    # ------------------------------------------------
    # Save U-Net binary mask
    # ------------------------------------------------

    unet_mask = (
        probability
        >= 0.50
    ).astype(
        np.uint8
    )

    unet_mask_path = os.path.join(
        UNET_MASK_DIR,
        f"{lake}_{year}_V3_U-Net_mask.tif",
    )

    save_raster(
        unet_mask_path,
        unet_mask,
        image_path,
        "uint8",
    )

    # ------------------------------------------------
    # Save final mask
    # ------------------------------------------------

    final_path = os.path.join(
        FINAL_MASK_DIR,
        f"{lake}_{year}_V3_FINAL_water_mask.tif",
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
        lake,
        year,
        image,
        probability,
        mndwi,
        raw_mask,
        final_mask,
    )

    return {
        "lake": lake,
        "year": year,
        "total_pixels": total_pixels,
        "water_pixels": water_pixels,
        "water_percentage": water_percentage,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print("\n")

    print(
        "#" * 70
    )

    print(
        "V3 UNSEEN-YEAR GENERALIZATION TEST"
    )

    print(
        "#" * 70
    )

    print()

    print(
        "TRAINING:"
    )

    print(
        "2018 manually verified GT"
    )

    print()

    print(
        "MODEL:"
    )

    print(
        "Frozen V3 U-Net"
    )

    print()

    print(
        "UNSEEN YEARS:"
    )

    print(
        "2019, 2020, 2021"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "No training is performed in this script."
    )

    print(
        "No model weights are changed."
    )

    print()

    print(
        "Device:",
        DEVICE,
    )

    # ------------------------------------------------
    # Load frozen V3
    # ------------------------------------------------

    model = load_v3_model()

    # ------------------------------------------------
    # Process all unseen years
    # ------------------------------------------------

    summary = []

    for lake in LAKES:

        for year in TEST_YEARS:

            result = process_year(
                model,
                lake,
                year,
            )

            if result is not None:

                summary.append(
                    result
                )

    # ------------------------------------------------
    # Save CSV
    # ------------------------------------------------

    csv_path = os.path.join(
        TABLE_DIR,
        "v3_unseen_year_results.csv",
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "Lake",
                "Year",
                "Total Pixels",
                "Water Pixels",
                "Water Percentage",
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
                ]
            )

    # ------------------------------------------------
    # Final summary
    # ------------------------------------------------

    print("\n")

    print(
        "#" * 70
    )

    print(
        "UNSEEN-YEAR TEST COMPLETE"
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
        "Results saved to:"
    )

    print(
        RESULTS_DIR
    )

    print()

    print(
        "CSV:"
    )

    print(
        csv_path
    )

    print()

    print(
        "This was inference only."
    )

    print(
        "The V3 model was NOT retrained."
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()