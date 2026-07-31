// ============================================================
// 01_cloud_cover_check.js
// Checks Sentinel-2 SR data availability and cloud cover
// for all three lakes across the target year range.
// Run this FIRST, before exporting any imagery.
// ============================================================

// Lake coordinates (confirmed centers)
var lakes = {
  'Hussain Sagar': ee.Geometry.Point([78.4738, 17.4239]),
  'Osman Sagar': ee.Geometry.Point([78.2989, 17.3763]),
  'Himayat Sagar': ee.Geometry.Point([78.3571, 17.3115])
};

// Usable year range: 2018-2025
// (2016-2017 excluded: no Sentinel-2 L2A/Surface Reflectance
// coverage available for this region in those years)
var years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];

Object.keys(lakes).forEach(function(lakeName) {
  var point = lakes[lakeName];
  years.forEach(function(year) {
    var startDate = year + '-03-01';
    var endDate = year + '-03-31';

    var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(point)
      .filterDate(startDate, endDate)
      .sort('CLOUDY_PIXEL_PERCENTAGE');

    var count = collection.size();
    var best = collection.first();
    var cloudPct = best.get('CLOUDY_PIXEL_PERCENTAGE');
    var dateAcquired = best.get('DATATAKE_IDENTIFIER');

    print(
      lakeName + ' | ' + year + ' | Images:', count,
      '| Cloud %:', cloudPct,
      '| Datatake:', dateAcquired
    );
  });
});
