// ============================================================
// 03_single_lake_visualize.js
// Quick visual check of a single lake/year in true color.
// Useful for eyeballing cloud cover and algae patches before
// committing to a batch export.
//
// Edit the point, date range, and title below as needed.
// ============================================================

// Example: Hussain Sagar, March 2023
var lakePoint = ee.Geometry.Point([78.4738, 17.4239]);

var image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(lakePoint)
  .filterDate('2023-03-01', '2023-03-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .sort('CLOUDY_PIXEL_PERCENTAGE')
  .first();

// True color visualization
var visParams = {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: 3000
};

Map.centerObject(lakePoint, 13);
Map.addLayer(image, visParams, 'Hussain Sagar - March 2023');

// Print image metadata (date, cloud %, etc.)
print('Image info:', image);
