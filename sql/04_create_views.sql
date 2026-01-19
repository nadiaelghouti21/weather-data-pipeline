-- ============================================================================
-- Create Database Views
-- ============================================================================
-- Purpose: Creates useful views for querying the transformed data
-- ============================================================================

-- View: Final weather data with all transformations
DROP VIEW IF EXISTS v_weather_final;
CREATE VIEW v_weather_final AS
SELECT 
    station_name,
    climate_id,
    latitude,
    longitude,
    date_month,
    feature_id,
    map,
    temperature_celsius_avg,
    temperature_celsius_min,
    temperature_celsius_max,
    temperature_celsius_yoy_avg
FROM weather_final
ORDER BY station_name, date_month;

-- View: Year-over-year comparison
DROP VIEW IF EXISTS v_yoy_comparison;
CREATE VIEW v_yoy_comparison AS
SELECT 
    station_name,
    date_month,
    temperature_celsius_avg,
    temperature_celsius_yoy_avg,
    CASE 
        WHEN temperature_celsius_yoy_avg > 0 THEN 'Warmer'
        WHEN temperature_celsius_yoy_avg < 0 THEN 'Colder'
        ELSE 'Same'
    END AS yoy_trend
FROM weather_final
WHERE temperature_celsius_yoy_avg IS NOT NULL
ORDER BY station_name, date_month;

