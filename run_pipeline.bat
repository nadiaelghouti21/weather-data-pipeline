@echo off
REM ============================================================================
REM Weather Data Pipeline - Windows Batch Script
REM ============================================================================
REM 
REM Usage:
REM   run_pipeline.bat                           # Run with defaults
REM   run_pipeline.bat --stations 26953          # Specific station
REM   run_pipeline.bat --years 2024              # Specific year
REM   run_pipeline.bat --stations 31688 --years 2023 2024
REM
REM Parameters are passed directly to the Python scripts.
REM ============================================================================

echo ======================================================================
echo WEATHER DATA PIPELINE
echo ======================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

echo Step 1: Data Extraction
echo ----------------------------------------------------------------------
echo Running: python extract_weather_data.py %*
echo.

python extract_weather_data.py %*
if errorlevel 1 (
    echo.
    echo ERROR: Extraction failed!
    exit /b 1
)

echo.
echo Step 2: Data Transformation
echo ----------------------------------------------------------------------
echo Running: python transform_weather_data.py
echo.

python transform_weather_data.py
if errorlevel 1 (
    echo.
    echo ERROR: Transformation failed!
    exit /b 1
)

echo.
echo ======================================================================
echo PIPELINE COMPLETED SUCCESSFULLY
echo ======================================================================
echo.
echo Output files:
echo   - raw_data\weather_raw_*.csv
echo   - raw_data\weather_raw_*.parquet
echo   - transformed_data\weather_transformed_*.csv
echo   - transformed_data\weather_transformed_*.parquet
echo   - weather_data.db
echo.

