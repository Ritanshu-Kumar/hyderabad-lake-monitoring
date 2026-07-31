// ============================================================
// 02_batch_export.js
// Queues Sentinel-2 image exports (5 bands: Blue, Green, Red,
// NIR, SWIR) to Google Drive for all three lakes, 2018-2025.
//
// After running this script, go to the "Tasks" tab in the
// Earth Engine Code Editor and click "Run" on each queued
// task individually to actually start the export.
// ============================================================

// Lake bounding boxes (buffer sized per lake's actual extent)
var lakes = {
  'HussainSagar': ee.Geometry.Point([78.4738, 17.4239]).buffer(2000).bounds(),
  'OsmanSagar': ee.Geometry.Point([78.2989, 17.3763]).buffer(4000).bounds(),
  'HimayatSagar': ee.Geometry.Point([78.3571, 17.3115]).buffer(4000).bounds()
};

var years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];

Object.keys(lakes).forEach(function(lakeName) {
  var area = lakes[lakeName];

  years.forEach(function(year) {
    var startDate = year + '-03-01';
    var endDate = year + '-03-31';

    var image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(area)
      .filterDate(startDate, endDate)
      .sort('CLOUDY_PIXEL_PERCENTAGE')
      .first()
      .select(['B2', 'B3', 'B4', 'B8', 'B11']); // Blue, Green, Red, NIR, SWIR

    var taskName = lakeName + '_' + year + '_March';

    Export.image.toDrive({
      image: image.clip(area),
      description: taskName,
      folder: 'FYP_Lake_Imagery',
      region: area,
      scale: 10,
      crs: 'EPSG:4326',
      maxPixels: 1e9
    });
  });
});
