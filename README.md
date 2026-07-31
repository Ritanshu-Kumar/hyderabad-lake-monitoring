# High-Precision Multi-Temporal Quantification of Urban Lake Surface Area Change in Hyderabad Using Deep Learning-Based Segmentation of Sentinel-2 Imagery

## One-Line Summary

Using deep learning to measure exactly how much Hyderabad's lakes/reservoirs — Hussain Sagar, Osman Sagar, and Himayat Sagar — have shrunk over recent years, using Sentinel-2 satellite imagery and a U-Net segmentation model. No prior study has applied this specific, rigorous, multi-temporal analysis to these three lakes.

## Problem Statement

Hyderabad's urban lakes have reportedly lost significant surface area due to encroachment and sewage-driven pollution, but this shrinkage has never been systematically quantified using a deep-learning-based remote sensing pipeline. This project builds that pipeline: a U-Net convolutional neural network trained to detect water body boundaries from multi-temporal Sentinel-2 imagery, producing a year-by-year, data-driven measurement of surface area change.

## Tech Stack

- **Data & Access:** Google Earth Engine (GEE), Sentinel-2 SR (`COPERNICUS/S2_SR_HARMONIZED`), QGIS
- **Core ML/DL:** Python 3.10+, PyTorch, `segmentation-models-pytorch` (U-Net w/ pretrained ResNet34 encoder), Albumentations, rasterio, geopandas
- **Compute & Tracking:** Google Colab, Weights & Biases
- **Visualization/Reporting:** Matplotlib/Seaborn, Folium/Kepler.gl

## Core Deliverables

1. A trained U-Net model (transfer learning: pretrained ResNet34 encoder, fine-tuned decoder) that segments water vs. land from multi-band Sentinel-2 imagery, explicitly handling algae/turbidity ambiguity via MNDWI + multi-band input.
2. Per-year water/land segmentation masks for Hussain Sagar, Osman Sagar, and Himayat Sagar (2018–2025).
3. A quantified surface area trend analysis and report across all three lakes.

## Optional Extensions (attempted only after the core is complete)

1. FTL (Full Tank Level)-based encroachment analysis, if official boundary data can be sourced.
2. Grad-CAM interpretability visualizations.
3. Transformer-CNN hybrid comparison against the U-Net baseline.
4. Shoreline change classification (stable / seasonal fluctuation / permanent encroachment / new construction).

## Current Progress

- ✅ Google Earth Engine project set up (`hyderabad-lakes-fyp`)
- ✅ Confirmed accurate coordinates for all three lakes
- ✅ Verified Sentinel-2 Surface Reflectance data availability: usable from **2018–2025** (2016–2017 have no L2A coverage for this region)
- ✅ Cloud-cover checked across all lakes/years (near-0% cloud cover for March, most years)
- ✅ Exported 24 satellite images (3 lakes × 8 years) to Google Drive as GeoTIFFs
- ✅ Initial NDWI computation and visual comparison in QGIS (confirms algae-driven misclassification risk in Hussain Sagar vs. clean results in Himayat Sagar)
- ⬜ Manual ground-truth labeling (QGIS)
- ⬜ U-Net training
- ⬜ Surface area extraction + trend analysis

## Repository Structure

```
gee_scripts/          → Google Earth Engine JavaScript scripts (data scouting + export)
docs/                 → Reference planning document, reports
notebooks/            → (upcoming) Python/Colab notebooks for U-Net training
sample_data/          → A few small example images (full dataset stored separately — see note below)
```

## Note on Data Storage

Raw Sentinel-2 GeoTIFF exports (24 files, 3 lakes × 8 years) and derived NDWI/MNDWI masks are **not stored in this repository** due to file size. They are stored in Google Drive and can be regenerated at any time by re-running the scripts in `gee_scripts/` inside the [Earth Engine Code Editor](https://code.earthengine.google.com).

## How to Reproduce

1. Create a Google Earth Engine account and register a Cloud project (Community/academic tier).
2. Open the [Earth Engine Code Editor](https://code.earthengine.google.com).
3. Paste and run `gee_scripts/01_cloud_cover_check.js` to verify data availability for your target lakes/years.
4. Paste and run `gee_scripts/02_batch_export.js` to queue image exports to Google Drive.
5. Confirm and run each task in the **Tasks** tab.
6. Download the resulting GeoTIFFs from Google Drive for use in QGIS / the training pipeline.

## Author

[Your Name] — [Reg. No.]
Final Year Project, under the guidance of [Guide's Name]
