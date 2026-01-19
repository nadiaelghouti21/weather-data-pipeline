-- ============================================================================
-- Final Transformed Table Query
-- ============================================================================
-- Purpose: Joins monthly weather data with geonames mapping
--          Produces the final output schema with all required columns
-- 
-- Input Tables: 
--   - weather_monthly_yoy (monthly data with YoY calculations)
--   - station_geoname_mapping (station to geoname matches)
-- Output: Final table matching the required schema
-- ============================================================================

SELECT 
    w."Station Name" AS station_name,
    CAST(w."Climate ID" AS TEXT) AS climate_id,
    w."Latitude (y)" AS latitude,
    w."Longitude (x)" AS longitude,
    w.date_month,
    g.feature_id,
    g.map,
    w.temperature_celsius_avg,
    w.temperature_celsius_min,
    w.temperature_celsius_max,
    w.temperature_celsius_yoy_avg
FROM weather_monthly_yoy w
LEFT JOIN station_geoname_mapping g
    ON w."Station Name" = g."Station Name"
ORDER BY 
    station_name,
    date_month;

