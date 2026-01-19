-- ============================================================================
-- Monthly Temperature Aggregation Query
-- ============================================================================
-- Purpose: Groups hourly weather readings by station, year, and month
--          Calculates average, minimum, and maximum temperatures
-- 
-- Input Table: weather_raw (hourly data)
-- Output: Monthly aggregated temperature statistics
-- ============================================================================

SELECT 
    "Station Name",
    "Climate ID",
    "Latitude (y)",
    "Longitude (x)",
    Year,
    Month,
    -- Create date_month in YYYY-MM format
    Year || '-' || CASE 
        WHEN Month < 10 THEN '0' || Month 
        ELSE CAST(Month AS TEXT) 
    END AS date_month,
    -- Temperature aggregations
    ROUND(AVG("Temp (°C)"), 2) AS temperature_celsius_avg,
    ROUND(MIN("Temp (°C)"), 2) AS temperature_celsius_min,
    ROUND(MAX("Temp (°C)"), 2) AS temperature_celsius_max,
    COUNT("Temp (°C)") AS observation_count
FROM weather_raw
WHERE "Temp (°C)" IS NOT NULL
GROUP BY 
    "Station Name",
    "Climate ID", 
    "Latitude (y)",
    "Longitude (x)",
    Year,
    Month
ORDER BY 
    "Station Name",
    Year,
    Month;

