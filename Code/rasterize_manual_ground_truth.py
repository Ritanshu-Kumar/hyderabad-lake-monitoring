"""Rasterize a manually digitized water polygon onto a reference satellite grid.

The reference raster defines every spatial property of the output: dimensions,
geotransform, extent, and CRS.  The result is a Byte training mask where
background is 0 and water is 1.  Zero is deliberately *not* stored as NoData.

Run from the project root:
    "C:\\Program Files\\QGIS 3.44.12\\apps\\Python312\\python.exe" \
        Code\\rasterize_manual_ground_truth.py

After saving at least one feature in the `water_body` layer, the default output
is Data/5 manual_ground_truth/HussainSagar_GT.tif.  Existing outputs are
protected unless --overwrite is explicitly supplied.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from osgeo import gdal, ogr, osr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE = PROJECT_ROOT / "Data" / "0 Raw_Images" / "HussainSagar_2018_March.tif"
DEFAULT_VECTOR = PROJECT_ROOT / "Data" / "5 manual_ground_truth" / "HussainSagar_2018_ManualGroundTruth.gpkg"
DEFAULT_OUTPUT = PROJECT_ROOT / "Data" / "5 manual_ground_truth" / "HussainSagar_GT.tif"
DEFAULT_LAYER = "water_body"


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--vector", type=Path, default=DEFAULT_VECTOR)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_target_layer(
    source_layer: ogr.Layer, target_srs: osr.SpatialReference
) -> tuple[ogr.DataSource, ogr.Layer]:
    """Return an in-memory copy of valid source features in the target CRS."""
    source_srs_raw = source_layer.GetSpatialRef()
    if source_srs_raw is None:
        fail("The water_body layer has no CRS. Assign its correct CRS in QGIS first.")
    # GeoPackages and GeoTIFFs both commonly advertise EPSG:4326 with
    # latitude-first axis metadata while storing GIS coordinates as X=longitude,
    # Y=latitude.  Use the conventional GIS order consistently when comparing
    # and, if needed, transforming geometries.
    source_srs = source_srs_raw.Clone()
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_srs = target_srs.Clone()
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transform = None
    if not source_srs.IsSame(target_srs):
        transform = osr.CoordinateTransformation(source_srs, target_srs)

    memory_driver = ogr.GetDriverByName("Memory")
    memory_ds = memory_driver.CreateDataSource("water_features")
    target_layer = memory_ds.CreateLayer("water_body", target_srs, ogr.wkbUnknown)

    source_layer.ResetReading()
    copied = 0
    for source_feature in source_layer:
        geometry = source_feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            fail(f"Feature {source_feature.GetFID()} has no usable geometry.")
        if not geometry.IsValid():
            fail(f"Feature {source_feature.GetFID()} is invalid; repair it in QGIS before rasterizing.")

        target_geometry = geometry.Clone()
        if transform is not None and target_geometry.Transform(transform) != ogr.OGRERR_NONE:
            fail(f"Could not reproject feature {source_feature.GetFID()} to the reference CRS.")

        output_feature = ogr.Feature(target_layer.GetLayerDefn())
        output_feature.SetGeometry(target_geometry)
        if target_layer.CreateFeature(output_feature) != ogr.OGRERR_NONE:
            fail(f"Could not copy feature {source_feature.GetFID()} for rasterization.")
        copied += 1

    if copied == 0:
        fail(
            "Layer 'water_body' contains zero saved features. No mask was created. "
            "In QGIS, save the manually digitized polygon to the GeoPackage, then rerun this script."
        )

    # The caller keeps memory_ds alive until GDAL has rasterized target_layer.
    return memory_ds, target_layer


def assert_features_intersect_grid(layer: ogr.Layer, geotransform: tuple[float, ...], width: int, height: int) -> None:
    corners = [
        (geotransform[0], geotransform[3]),
        (geotransform[0] + width * geotransform[1], geotransform[3] + width * geotransform[4]),
        (geotransform[0] + height * geotransform[2], geotransform[3] + height * geotransform[5]),
        (
            geotransform[0] + width * geotransform[1] + height * geotransform[2],
            geotransform[3] + width * geotransform[4] + height * geotransform[5],
        ),
    ]
    grid_min_x, grid_max_x = min(x for x, _ in corners), max(x for x, _ in corners)
    grid_min_y, grid_max_y = min(y for _, y in corners), max(y for _, y in corners)
    layer.ResetReading()
    intersects = False
    for feature in layer:
        min_x, max_x, min_y, max_y = feature.GetGeometryRef().GetEnvelope()
        if max_x >= grid_min_x and min_x <= grid_max_x and max_y >= grid_min_y and min_y <= grid_max_y:
            intersects = True
            break
    if not intersects:
        fail("The saved water_body feature(s) do not intersect the reference raster extent.")


def verify_output(output_path: Path, reference: gdal.Dataset) -> dict[str, int]:
    output = gdal.Open(str(output_path), gdal.GA_ReadOnly)
    if output is None:
        fail("Output could not be reopened for verification.")

    if (output.RasterXSize, output.RasterYSize) != (reference.RasterXSize, reference.RasterYSize):
        fail("Verification failed: output dimensions differ from the reference raster.")
    if output.GetGeoTransform() != reference.GetGeoTransform():
        fail("Verification failed: output geotransform differs from the reference raster.")
    if output.GetProjectionRef() != reference.GetProjectionRef():
        fail("Verification failed: output CRS differs from the reference raster.")

    band = output.GetRasterBand(1)
    if band.DataType != gdal.GDT_Byte:
        fail("Verification failed: output is not Byte/uint8.")
    if band.GetNoDataValue() is not None:
        fail("Verification failed: output has an unintended NoData value.")

    data = band.ReadRaster(0, 0, output.RasterXSize, output.RasterYSize, buf_type=gdal.GDT_Byte)
    counts = Counter(data)
    unexpected = set(counts) - {0, 1}
    if unexpected:
        fail(f"Verification failed: unexpected pixel values: {sorted(unexpected)}")
    if counts[0] == 0 or counts[1] == 0:
        fail("Verification failed: both background (0) and water (1) must be present.")
    return {"background": counts[0], "water": counts[1]}


def main() -> None:
    args = parse_args()
    gdal.UseExceptions()
    ogr.UseExceptions()

    if not args.reference.is_file():
        fail(f"Reference raster not found: {args.reference}")
    if not args.vector.is_file():
        fail(f"Vector GeoPackage not found: {args.vector}")
    if args.output.exists() and not args.overwrite:
        fail(f"Output already exists: {args.output}. Use --overwrite only after reviewing it.")

    reference = gdal.Open(str(args.reference), gdal.GA_ReadOnly)
    if reference is None:
        fail(f"Could not open reference raster: {args.reference}")
    if reference.GetProjectionRef() == "":
        fail("Reference raster has no CRS.")

    vector_ds = ogr.Open(str(args.vector), 0)
    if vector_ds is None:
        fail(f"Could not open GeoPackage: {args.vector}")
    source_layer = vector_ds.GetLayerByName(args.layer)
    if source_layer is None:
        fail(f"Layer '{args.layer}' was not found in {args.vector}.")
    if source_layer.GetFeatureCount() == 0:
        fail(
            f"Layer '{args.layer}' contains zero saved features. No mask was created. "
            "In QGIS, save the manually digitized polygon to the GeoPackage, then rerun this script."
        )

    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(reference.GetProjectionRef())
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_memory_ds, target_layer = make_target_layer(source_layer, target_srs)
    geotransform = reference.GetGeoTransform()
    assert_features_intersect_grid(target_layer, geotransform, reference.RasterXSize, reference.RasterYSize)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.tif")
    if temporary.exists():
        temporary.unlink()

    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(
        str(temporary),
        reference.RasterXSize,
        reference.RasterYSize,
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    output.SetGeoTransform(geotransform)
    output.SetProjection(reference.GetProjectionRef())
    output_band = output.GetRasterBand(1)
    output_band.Fill(0)
    gdal.RasterizeLayer(output, [1], target_layer, burn_values=[1])
    output_band.FlushCache()
    output.FlushCache()
    output = None
    target_layer = None
    target_memory_ds = None

    counts = verify_output(temporary, reference)
    if args.output.exists() and args.overwrite:
        args.output.unlink()
    os.replace(temporary, args.output)

    print(f"Created: {args.output}")
    print(f"Reference grid: {reference.RasterXSize} x {reference.RasterYSize}")
    print("NoData: none")
    print(f"Pixel counts: background=0: {counts['background']}, water=1: {counts['water']}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
