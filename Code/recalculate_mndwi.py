"""
recalculate_mndwi.py

Recalculates MNDWI masks directly from the original Sentinel-2
5-band GeoTIFF images.

Band order:
    1 = Blue
    2 = Green
    3 = Red
    4 = NIR
    5 = SWIR

MNDWI:
    (Green - SWIR) / (Green + SWIR)

Water:
    MNDWI > 0

IMPORTANT:
    Existing MNDWI masks are NOT overwritten.
    New masks are saved to:

    Data/3 mndwi_masks_recalculated/
"""

import os
import numpy as np
import rasterio


# ================================================================
# PATHS
# ================================================================

RAW_IMAGES_DIR = r"C:\Users\ritan\Downloads\Capstone_project\Data\0 Raw_Images"

OUTPUT_DIR = r"C:\Users\ritan\Downloads\Capstone_project\Data\3 mndwi_masks_recalculated"

# Standard MNDWI threshold
MNDWI_THRESHOLD = 0.0


# ================================================================
# CREATE OUTPUT DIRECTORY
# ================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# CALCULATE MNDWI
# ================================================================

def calculate_mndwi(image_path, output_path):

    print("\n" + "=" * 70)
    print("PROCESSING")
    print("=" * 70)

    print("Input :", image_path)
    print("Output:", output_path)

    with rasterio.open(image_path) as src:

        print(f"Dimensions: {src.width} x {src.height}")
        print(f"Bands: {src.count}")

        if src.count < 5:
            raise RuntimeError(
                f"{os.path.basename(image_path)} has only "
                f"{src.count} bands. Expected at least 5."
            )

        # --------------------------------------------------------
        # Read bands
        # --------------------------------------------------------

        green = src.read(2).astype(np.float32)
        swir = src.read(5).astype(np.float32)

        # --------------------------------------------------------
        # Valid pixels
        # --------------------------------------------------------

        valid = np.isfinite(green) & np.isfinite(swir)

        if src.nodata is not None:
            valid &= green != src.nodata
            valid &= swir != src.nodata

        # --------------------------------------------------------
        # MNDWI
        # --------------------------------------------------------

        denominator = green + swir

        valid &= denominator != 0

        mndwi = np.zeros_like(green, dtype=np.float32)

        mndwi[valid] = (
            (green[valid] - swir[valid])
            / denominator[valid]
        )

        # Remove numerical problems
        mndwi = np.nan_to_num(
            mndwi,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # --------------------------------------------------------
        # Binary water mask
        # --------------------------------------------------------

        water_mask = (
            (mndwi > MNDWI_THRESHOLD) &
            valid
        ).astype(np.uint8)

        # --------------------------------------------------------
        # Statistics
        # --------------------------------------------------------

        valid_pixels = np.count_nonzero(valid)
        water_pixels = np.count_nonzero(water_mask)

        if valid_pixels > 0:
            water_percentage = (
                water_pixels / valid_pixels
            ) * 100
        else:
            water_percentage = 0

        print(f"Valid pixels : {valid_pixels}")
        print(f"Water pixels : {water_pixels}")
        print(f"Water %      : {water_percentage:.2f}%")

        valid_values = mndwi[valid]

        if len(valid_values) > 0:
            print(f"MNDWI min    : {valid_values.min():.4f}")
            print(f"MNDWI max    : {valid_values.max():.4f}")
            print(f"MNDWI mean   : {valid_values.mean():.4f}")


        # ========================================================
        # SAVE MNDWI
        # ========================================================

        mndwi_profile = src.profile.copy()

        mndwi_profile.update(
            dtype=rasterio.float32,
            count=1,
            compress="lzw",
            nodata=-9999
        )

        mndwi_output = mndwi.copy()
        mndwi_output[~valid] = -9999

        with rasterio.open(
            output_path,
            "w",
            **mndwi_profile
        ) as dst:

            dst.write(
                mndwi_output.astype(np.float32),
                1
            )


        # ========================================================
        # SAVE BINARY MASK
        # ========================================================

        mask_output_path = output_path.replace(
            "_MNDWI.tif",
            "_MNDWI_Mask.tif"
        )

        mask_profile = src.profile.copy()

        mask_profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress="lzw",
            nodata=0
        )

        with rasterio.open(
            mask_output_path,
            "w",
            **mask_profile
        ) as dst:

            dst.write(
                water_mask,
                1
            )

        print("Saved MNDWI     :", output_path)
        print("Saved water mask:", mask_output_path)


# ================================================================
# PROCESS ALL SENTINEL-2 IMAGES
# ================================================================

files = sorted(
    f for f in os.listdir(RAW_IMAGES_DIR)
    if f.lower().endswith(".tif")
    and "_March" in f
)

print("\n")
print("=" * 70)
print("MNDWI RECALCULATION")
print("=" * 70)

print(f"Input directory : {RAW_IMAGES_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Threshold       : {MNDWI_THRESHOLD}")

print(f"\nFound {len(files)} Sentinel-2 images.")

for filename in files:

    input_path = os.path.join(
        RAW_IMAGES_DIR,
        filename
    )

    base = filename.replace(
        "_March.tif",
        ""
    )

    output_filename = (
        f"{base}_MNDWI.tif"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    try:

        calculate_mndwi(
            input_path,
            output_path
        )

    except Exception as e:

        print("\n!!! ERROR !!!")
        print(filename)
        print(str(e))


# ================================================================
# COMPLETE
# ================================================================

print("\n")
print("=" * 70)
print("MNDWI RECALCULATION COMPLETE")
print("=" * 70)

print(f"New masks saved in:")
print(OUTPUT_DIR)

print("\nExisting folder was NOT modified:")
print(
    r"C:\Users\ritan\Downloads\Capstone_project\Data\3 mndwi_masks"
)

print("\nNext step:")
print("Compare the recalculated masks with the old MNDWI masks.")