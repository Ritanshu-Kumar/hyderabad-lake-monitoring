"""
compare_baseline_vs_v3.py

CAPSTONE MODEL COMPARISON

BASELINE:
    train_unet_compete.py

    U-Net trained using MNDWI-derived masks as pseudo-ground truth.

FINAL:
    unet_manual_mndwi_v3.py

    U-Net trained using manually verified 2018 ground truth,
    followed by MNDWI-supported segmentation.

PURPOSE:
    Compare the two approaches without modifying their existing
    source files or result folders.

EVALUATION:
    2018 manually verified ground truth is used as the reference
    evaluation mask.

METRICS:
    - IoU / Jaccard
    - Dice / F1
    - Precision
    - Recall
    - Pixel Accuracy

QUALITATIVE COMPARISON:
    RGB
    Manual GT
    MNDWI
    Baseline U-Net
    V3 U-Net

OUTPUT:
    Results/
    └── Model_Comparison/
        ├── metrics.csv
        ├── summary.txt
        ├── HussainSagar_2018_comparison.png
        ├── OsmanSagar_2018_comparison.png
        └── HimayatSagar_2018_comparison.png

IMPORTANT:
    This script is independent of the existing V1/V2/V3 result files.
    It creates its own comparison models/checkpoints.
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

MANUAL_GT_DIR = os.path.join(
    BASE_DIR,
    "Data",
    "5 manual_ground_truth",
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "Results",
    "Model_Comparison",
)

MODEL_DIR = os.path.join(
    RESULTS_DIR,
    "models",
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
    MODEL_DIR,
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

# Same overlapping-patch strategy used by V3.
STRIDE = 128

EPOCHS = 30

BATCH_SIZE = 4

LEARNING_RATE = 1e-4

RANDOM_SEED = 42

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# LAKES
# ================================================================

LAKES = {

    "HussainSagar": {
        "image": "HussainSagar_2018_March.tif",
        "manual_gt": "HussainSagar_GT.tif",
        "mndwi": "HussainSagar_2018_MNDWI.tif",
    },

    "OsmanSagar": {
        "image": "OsmanSagar_2018_March.tif",
        "manual_gt": "OsmanSagar_GT.tif",
        "mndwi": "OsmanSagar_2018_MNDWI.tif",
    },

    "HimayatSagar": {
        "image": "HimayatSagar_2018_March.tif",
        "manual_gt": "HimayatSagar_GT.tif",
        "mndwi": "HimayatSagar_2018_MNDWI.tif",
    },
}


# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ================================================================
# UTILITY
# ================================================================

def print_header(text):

    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def normalize_image(image):

    """
    Same per-band percentile normalization used by V3.
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
# DATASET
# ================================================================

class TrainingDataset(Dataset):

    def __init__(
        self,
        pairs,
        label_type,
    ):

        self.images = []
        self.labels = []

        print_header(
            f"BUILDING {label_type.upper()} TRAINING DATA"
        )

        for image_path, label_path in pairs:

            print(
                "\nImage:",
                image_path,
            )

            print(
                "Label:",
                label_path,
            )

            with rasterio.open(
                image_path
            ) as src:

                image = src.read().astype(
                    np.float32
                )

                height = src.height
                width = src.width

            with rasterio.open(
                label_path
            ) as src:

                label = src.read(
                    1
                ).astype(
                    np.float32
                )

            # Ensure same dimensions.
            if label.shape != (
                height,
                width,
            ):

                raise RuntimeError(
                    "Image and label dimensions "
                    "do not match:\n"
                    f"Image: {image.shape}\n"
                    f"Label: {label.shape}"
                )

            image = normalize_image(
                image
            )

            # Binary labels.
            label = (
                label > 0.5
            ).astype(
                np.float32
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

            added = 0

            for y in y_starts:

                for x in x_starts:

                    image_patch = image[
                        :,
                        y:y + PATCH_SIZE,
                        x:x + PATCH_SIZE,
                    ]

                    label_patch = label[
                        y:y + PATCH_SIZE,
                        x:x + PATCH_SIZE,
                    ]

                    if image_patch.shape != (
                        image.shape[0],
                        PATCH_SIZE,
                        PATCH_SIZE,
                    ):
                        continue

                    self.images.append(
                        image_patch
                    )

                    self.labels.append(
                        label_patch
                    )

                    added += 1

            print(
                "Patches added:",
                added,
            )

        print(
            "\nTotal patches:",
            len(self.images),
        )

    def __len__(self):

        return len(
            self.images
        )

    def __getitem__(
        self,
        index,
    ):

        image = torch.from_numpy(
            self.images[index].copy()
        ).float()

        label = torch.from_numpy(
            self.labels[index].copy()
        ).float().unsqueeze(0)

        return image, label


# ================================================================
# MODEL
# ================================================================

def build_model():

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=5,
        classes=1,
        activation=None,
    )

    return model.to(
        DEVICE
    )


# ================================================================
# LOSS
# ================================================================

class DiceLoss(nn.Module):

    def forward(
        self,
        logits,
        targets,
    ):

        probabilities = torch.sigmoid(
            logits
        )

        smooth = 1.0

        intersection = (
            probabilities
            * targets
        ).sum(
            dim=(1, 2, 3)
        )

        denominator = (
            probabilities.sum(
                dim=(1, 2, 3)
            )
            +
            targets.sum(
                dim=(1, 2, 3)
            )
        )

        dice = (
            2 * intersection
            + smooth
        ) / (
            denominator
            + smooth
        )

        return 1 - dice.mean()


class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = (
            nn.BCEWithLogitsLoss()
        )

        self.dice = DiceLoss()

    def forward(
        self,
        logits,
        targets,
    ):

        return (
            0.5
            * self.bce(
                logits,
                targets,
            )
            +
            0.5
            * self.dice(
                logits,
                targets,
            )
        )


# ================================================================
# TRAIN MODEL
# ================================================================

def train_model(
    name,
    pairs,
):

    dataset = TrainingDataset(
        pairs,
        name,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = build_model()

    loss_fn = CombinedLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_loss = float(
        "inf"
    )

    best_path = os.path.join(
        MODEL_DIR,
        f"{name}_model.pth",
    )

    print_header(
        f"TRAINING {name.upper()} MODEL"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad()

            predictions = model(
                images
            )

            loss = loss_fn(
                predictions,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss
            /
            max(
                len(loader),
                1,
            )
        )

        print(
            f"{name} | "
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"| Loss: "
            f"{average_loss:.4f}"
        )

        if average_loss < best_loss:

            best_loss = average_loss

            torch.save(
                model.state_dict(),
                best_path,
            )

    model.load_state_dict(
        torch.load(
            best_path,
            map_location=DEVICE,
        )
    )

    print(
        "\nBest model:",
        best_path,
    )

    return model


# ================================================================
# FULL IMAGE PREDICTION
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

                prediction = torch.sigmoid(
                    model(tensor)
                )

                prediction = (
                    prediction
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                probability_sum[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE,
                ] += prediction

                count[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE,
                ] += 1

    count[
        count == 0
    ] = 1

    return (
        probability_sum
        /
        count
    )


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(
    prediction,
    ground_truth,
):

    prediction = (
        prediction.astype(bool)
    )

    ground_truth = (
        ground_truth.astype(bool)
    )

    tp = np.logical_and(
        prediction,
        ground_truth,
    ).sum()

    fp = np.logical_and(
        prediction,
        ~ground_truth,
    ).sum()

    fn = np.logical_and(
        ~prediction,
        ground_truth,
    ).sum()

    tn = np.logical_and(
        ~prediction,
        ~ground_truth,
    ).sum()

    epsilon = 1e-8

    iou = (
        tp
        /
        (
            tp
            + fp
            + fn
            + epsilon
        )
    )

    dice = (
        2 * tp
        /
        (
            2 * tp
            + fp
            + fn
            + epsilon
        )
    )

    precision = (
        tp
        /
        (
            tp
            + fp
            + epsilon
        )
    )

    recall = (
        tp
        /
        (
            tp
            + fn
            + epsilon
        )
    )

    accuracy = (
        tp + tn
    ) / (
        tp
        + tn
        + fp
        + fn
        + epsilon
    )

    return {
        "IoU": float(iou),
        "Dice_F1": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "Pixel_Accuracy": float(accuracy),
    }


# ================================================================
# LOAD MASK
# ================================================================

def load_binary_mask(
    path,
):

    with rasterio.open(
        path
    ) as src:

        mask = src.read(
            1
        ).astype(
            np.float32
        )

    return (
        mask > 0.5
    ).astype(
        np.uint8
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
# SAVE COMPARISON IMAGE
# ================================================================

def save_comparison_image(
    lake,
    image,
    manual_gt,
    mndwi,
    baseline_probability,
    baseline_mask,
    v3_probability,
    v3_mask,
):

    rgb = make_rgb(
        image
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 9),
    )

    # ------------------------------------------------
    # Row 1
    # ------------------------------------------------

    axes[0, 0].imshow(
        rgb
    )

    axes[0, 0].set_title(
        "Sentinel-2 RGB"
    )

    axes[0, 1].imshow(
        manual_gt,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0, 1].set_title(
        "Manual Ground Truth"
    )

    axes[0, 2].imshow(
        mndwi,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0, 2].set_title(
        "MNDWI Pseudo-Label"
    )

    axes[0, 3].imshow(
        baseline_probability,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0, 3].set_title(
        "MNDWI-U-Net Probability"
    )

    # ------------------------------------------------
    # Row 2
    # ------------------------------------------------

    axes[1, 0].imshow(
        baseline_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1, 0].set_title(
        "Baseline Prediction"
    )

    axes[1, 1].imshow(
        v3_probability,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1, 1].set_title(
        "Manual-GT U-Net Probability"
    )

    axes[1, 2].imshow(
        v3_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1, 2].set_title(
        "V3 Prediction"
    )

    # Difference map
    difference = (
        v3_mask.astype(np.int8)
        -
        baseline_mask.astype(np.int8)
    )

    axes[1, 3].imshow(
        difference,
        cmap="gray",
        vmin=-1,
        vmax=1,
    )

    axes[1, 3].set_title(
        "V3 − Baseline"
    )

    for row in axes:

        for ax in row:

            ax.axis(
                "off"
            )

    fig.suptitle(
        f"{lake} 2018 — Baseline vs Manual-GT V3",
        fontsize=15,
    )

    plt.tight_layout()

    output_path = os.path.join(
        VIS_DIR,
        f"{lake}_2018_comparison.png",
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        output_path,
    )


# ================================================================
# MAIN COMPARISON
# ================================================================

def main():

    print_header(
        "CAPSTONE MODEL COMPARISON"
    )

    print(
        "BASELINE:"
    )

    print(
        "train_unet_compete.py"
    )

    print()

    print(
        "FINAL:"
    )

    print(
        "unet_manual_mndwi_v3.py"
    )

    print()

    print(
        "Evaluation reference:"
    )

    print(
        "2018 manually verified ground truth"
    )

    print()

    print(
        "Device:",
        DEVICE,
    )

    # ============================================================
    # BUILD BASELINE TRAINING LABELS
    #
    # Baseline learns directly from MNDWI pseudo-labels.
    # ============================================================

    baseline_pairs = []

    manual_pairs = []

    for lake, info in LAKES.items():

        image_path = os.path.join(
            RAW_IMAGES_DIR,
            info["image"],
        )

        mndwi_path = os.path.join(
            MNDWI_DIR,
            info["mndwi"],
        )

        gt_path = os.path.join(
            MANUAL_GT_DIR,
            info["manual_gt"],
        )

        if not os.path.exists(
            image_path
        ):

            raise FileNotFoundError(
                image_path
            )

        if not os.path.exists(
            mndwi_path
        ):

            raise FileNotFoundError(
                mndwi_path
            )

        if not os.path.exists(
            gt_path
        ):

            raise FileNotFoundError(
                gt_path
            )

        baseline_pairs.append(
            (
                image_path,
                mndwi_path,
            )
        )

        manual_pairs.append(
            (
                image_path,
                gt_path,
            )
        )

    # ============================================================
    # TRAIN BASELINE
    # ============================================================

    baseline_model = train_model(
        "MNDWI_Baseline",
        baseline_pairs,
    )

    # ============================================================
    # TRAIN MANUAL-GT MODEL
    # ============================================================

    manual_model = train_model(
        "Manual_GT_V3",
        manual_pairs,
    )

    # ============================================================
    # EVALUATE
    # ============================================================

    print_header(
        "EVALUATING BOTH MODELS AGAINST MANUAL GT"
    )

    all_results = []

    for lake, info in LAKES.items():

        print_header(
            f"EVALUATION: {lake} 2018"
        )

        image_path = os.path.join(
            RAW_IMAGES_DIR,
            info["image"],
        )

        gt_path = os.path.join(
            MANUAL_GT_DIR,
            info["manual_gt"],
        )

        mndwi_path = os.path.join(
            MNDWI_DIR,
            info["mndwi"],
        )

        # --------------------------------------------------------
        # Read image
        # --------------------------------------------------------

        with rasterio.open(
            image_path
        ) as src:

            image = src.read().astype(
                np.float32
            )

        image = normalize_image(
            image
        )

        # --------------------------------------------------------
        # Read manual GT
        # --------------------------------------------------------

        manual_gt = load_binary_mask(
            gt_path
        )

        # --------------------------------------------------------
        # Read MNDWI
        # --------------------------------------------------------

        mndwi = load_binary_mask(
            mndwi_path
        )

        # --------------------------------------------------------
        # Baseline prediction
        # --------------------------------------------------------

        print(
            "Running MNDWI baseline..."
        )

        baseline_probability = (
            predict_full_image(
                baseline_model,
                image,
            )
        )

        baseline_mask = (
            baseline_probability
            >= 0.50
        ).astype(
            np.uint8
        )

        # --------------------------------------------------------
        # V3 prediction
        # --------------------------------------------------------

        print(
            "Running Manual-GT V3..."
        )

        v3_probability = (
            predict_full_image(
                manual_model,
                image,
            )
        )

        v3_mask = (
            v3_probability
            >= 0.50
        ).astype(
            np.uint8
        )

        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------

        baseline_metrics = (
            calculate_metrics(
                baseline_mask,
                manual_gt,
            )
        )

        v3_metrics = (
            calculate_metrics(
                v3_mask,
                manual_gt,
            )
        )

        print()

        print(
            "MNDWI BASELINE"
        )

        for key, value in (
            baseline_metrics.items()
        ):

            print(
                f"{key:<18}: "
                f"{value:.4f}"
            )

        print()

        print(
            "MANUAL-GT V3"
        )

        for key, value in (
            v3_metrics.items()
        ):

            print(
                f"{key:<18}: "
                f"{value:.4f}"
            )

        # --------------------------------------------------------
        # Store results
        # --------------------------------------------------------

        for metric in baseline_metrics:

            all_results.append(
                [
                    lake,
                    "MNDWI_Baseline",
                    metric,
                    baseline_metrics[
                        metric
                    ],
                ]
            )

        for metric in v3_metrics:

            all_results.append(
                [
                    lake,
                    "Manual_GT_V3",
                    metric,
                    v3_metrics[
                        metric
                    ],
                ]
            )

        # --------------------------------------------------------
        # Visual comparison
        # --------------------------------------------------------

        save_comparison_image(
            lake,
            image,
            manual_gt,
            mndwi,
            baseline_probability,
            baseline_mask,
            v3_probability,
            v3_mask,
        )

    # ============================================================
    # SAVE METRICS CSV
    # ============================================================

    metrics_path = os.path.join(
        TABLE_DIR,
        "metrics.csv",
    )

    with open(
        metrics_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "Lake",
                "Model",
                "Metric",
                "Value",
            ]
        )

        writer.writerows(
            all_results
        )

    # ============================================================
    # CREATE SUMMARY
    # ============================================================

    summary_path = os.path.join(
        RESULTS_DIR,
        "summary.txt",
    )

    # Calculate averages
    models = [
        "MNDWI_Baseline",
        "Manual_GT_V3",
    ]

    metric_names = [
        "IoU",
        "Dice_F1",
        "Precision",
        "Recall",
        "Pixel_Accuracy",
    ]

    with open(
        summary_path,
        "w",
    ) as f:

        f.write(
            "CAPSTONE MODEL COMPARISON\n"
        )

        f.write(
            "=========================\n\n"
        )

        f.write(
            "Baseline:\n"
        )

        f.write(
            "MNDWI pseudo-ground-truth U-Net\n\n"
        )

        f.write(
            "Final model:\n"
        )

        f.write(
            "Manual-GT U-Net V3\n\n"
        )

        f.write(
            "Evaluation reference:\n"
        )

        f.write(
            "2018 manually verified ground truth\n\n"
        )

        f.write(
            "AVERAGE METRICS\n"
        )

        f.write(
            "===============\n\n"
        )

        for model_name in models:

            f.write(
                f"{model_name}\n"
            )

            f.write(
                "-" * len(model_name)
                + "\n"
            )

            for metric in metric_names:

                values = [
                    row[3]
                    for row in all_results
                    if (
                        row[1] == model_name
                        and row[2] == metric
                    )
                ]

                if values:

                    average = np.mean(
                        values
                    )

                    f.write(
                        f"{metric}: "
                        f"{average:.4f}\n"
                    )

            f.write("\n")

    # ============================================================
    # PRINT FINAL SUMMARY
    # ============================================================

    print_header(
        "COMPARISON COMPLETE"
    )

    print(
        "Metrics:"
    )

    print(
        metrics_path
    )

    print()

    print(
        "Summary:"
    )

    print(
        summary_path
    )

    print()

    print(
        "Visualizations:"
    )

    print(
        VIS_DIR
    )

    print()

    print(
        "The existing V3 model and its results "
        "were NOT modified."
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()