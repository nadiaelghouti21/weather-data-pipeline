#!/bin/bash
# ============================================================================
# Weather Data Pipeline - Unix Shell Script
# ============================================================================
#
# Usage:
#   ./run_pipeline.sh                           # Run with defaults
#   ./run_pipeline.sh --stations 26953          # Specific station
#   ./run_pipeline.sh --years 2024              # Specific year
#   ./run_pipeline.sh --stations 31688 --years 2023 2024
#
# Parameters are passed directly to the Python scripts.
# ============================================================================

set -e  # Exit on error

echo "======================================================================"
echo "WEATHER DATA PIPELINE"
echo "======================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Step 1: Data Extraction"
echo "----------------------------------------------------------------------"
echo "Running: $PYTHON_CMD extract_weather_data.py $@"
echo ""

$PYTHON_CMD extract_weather_data.py "$@"

echo ""
echo "Step 2: Data Transformation"
echo "----------------------------------------------------------------------"
echo "Running: $PYTHON_CMD transform_weather_data.py"
echo ""

$PYTHON_CMD transform_weather_data.py

echo ""
echo "======================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "======================================================================"
echo ""
echo "Output files:"
echo "  - raw_data/weather_raw_*.csv"
echo "  - raw_data/weather_raw_*.parquet"
echo "  - transformed_data/weather_transformed_*.csv"
echo "  - transformed_data/weather_transformed_*.parquet"
echo "  - weather_data.db"
echo ""

