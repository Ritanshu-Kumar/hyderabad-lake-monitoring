"""
batch_mndwi_all_lakes.py

Generates MNDWI (and binary mask) for EVERY raw image in your
0 Raw_Images folder -- all three lakes, all years -- using SWIR
instead of NIR.

MNDWI = (Green - SWIR) / (Green + SWIR)   -> band 2, band 5
Mask  = 1 where MNDWI > 0 (water), else 0

This overwrites any existing NDWI-based masks with the same
filename, since we're standardizing on MNDWI across all lakes
for consistency (see project notes / report methodology).

Install dependencies first (same as train_unet.py):
    pip install -r requirements.txt
"""

import os
import glob
import numpy as np
import rasterio

# ------------------------------------------------------------------
# CONFIG — edit these paths to match your folder locations
# ------------------------------------------------------------------
RAW_IMAGES_DIR = r"C:\Ritanshu\10 Capstone\Data\0 Raw_Images"
MNDWI_OUTPUT_DIR = r"C:\Ritanshu\10 Capstone\Data\3 mndwi_masks"
MASK_OUTPUT_DIR = r"C:\Ritanshu\10 Capstone\Data\4 mndwi_Labeled_patches"

GREEN_BAND_INDEX = 2   # B3
SWIR_BAND_INDEX = 5    # B11


def compute_mndwi_and_mask(image_path, mndwi_out_path, mask_out_path):
    with rasterio.open(image_path) as src:
        green = src.read(GREEN_BAND_INDEX).astype(np.float32)
        swir = src.read(SWIR_BAND_INDEX).astype(np.float32)
        profile = src.profile  # preserves georeferencing info for output files

    # Avoid divide-by-zero issues
    denom = (green + swir)
    denom[denom == 0] = 1e-6

    mndwi = (green - swir) / denom
    mask = (mndwi > 0).astype(np.uint8)

    # Output profile: single band, float32 for MNDWI
    mndwi_profile = profile.copy()
    mndwi_profile.update(count=1, dtype=rasterio.float32)

    with rasterio.open(mndwi_out_path, "w", **mndwi_profile) as dst:
        dst.write(mndwi.astype(np.float32), 1)

    # Output profile: single band, uint8 for binary mask
    mask_profile = profile.copy()
    mask_profile.update(count=1, dtype=rasterio.uint8)

    with rasterio.open(mask_out_path, "w", **mask_profile) as dst:
        dst.write(mask, 1)

    water_pct = 100 * mask.sum() / mask.size
    print(f"  MNDWI + mask saved. Water coverage: {water_pct:.1f}%")


def main():
    os.makedirs(MNDWI_OUTPUT_DIR, exist_ok=True)
    os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)

    image_paths = glob.glob(os.path.join(RAW_IMAGES_DIR, "*.tif"))

    if not image_paths:
        print(f"No .tif files found in {RAW_IMAGES_DIR}")
        return

    print(f"Found {len(image_paths)} raw images. Processing with MNDWI...\n")

    for image_path in sorted(image_paths):
        filename = os.path.basename(image_path)

        if not filename.lower().endswith(".tif"):
            continue

        base_name = os.path.splitext(filename)[0]  # e.g. "HussainSagar_2020_March"

        if base_name.endswith("_March"):
            base_name = base_name[: -len("_March")]

        mndwi_out_path = os.path.join(MNDWI_OUTPUT_DIR, f"{base_name}_MNDWI.tif")
        mask_out_path = os.path.join(MASK_OUTPUT_DIR, f"{base_name}_Mask.tif")

        print(f"Processing: {filename}")

        try:
            compute_mndwi_and_mask(image_path, mndwi_out_path, mask_out_path)
        except Exception as e:
            print(f"  !! FAILED: {e}")

    print("\nDone.")
    print("MNDWI files saved to:", MNDWI_OUTPUT_DIR)
    print("Mask files saved to:", MASK_OUTPUT_DIR)
    print("\nNote: existing NDWI-based masks with the same filename have")
    print("been overwritten with MNDWI-based versions for consistency.")


if __name__ == "__main__":
    main()
