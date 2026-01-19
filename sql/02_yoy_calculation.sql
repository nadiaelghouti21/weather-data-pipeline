-- ============================================================================
-- Year-over-Year Temperature Difference Query
-- ============================================================================
-- Purpose: Compares each month's average temperature to the same month 
--          in the previous year using a self-join
-- 
-- Input Table: weather_monthly_agg (monthly aggregations)
-- Output: Monthly data with YoY temperature difference
-- ============================================================================

SELECT 
    curr."Station Name",
    curr."Climate ID",
    curr."Latitude (y)",
    curr."Longitude (x)",
    curr.Year,
    curr.Month,
    curr.date_month,
    curr.temperature_celsius_avg,
    curr.temperature_celsius_min,
    curr.temperature_celsius_max,
    curr.observation_count,
    -- YoY difference: current year avg - previous year avg (same month)
    -- Example: Jan 2024 avg - Jan 2023 avg = YoY difference
    ROUND(
        curr.temperature_celsius_avg - prev.temperature_celsius_avg, 
        2
    ) AS temperature_celsius_yoy_avg
FROM weather_monthly_agg curr
LEFT JOIN weather_monthly_agg prev
    ON curr."Station Name" = prev."Station Name"
    AND curr.Month = prev.Month
    AND curr.Year = prev.Year + 1
ORDER BY 
    curr."Station Name",
    curr.Year,
    curr.Month;

