# Weather Data Pipeline

An ETL pipeline that extracts weather data from Environment Canada, transforms it with geolocation enrichment, and outputs aggregated monthly summaries.

## Features

- **Extract**: Fetches hourly weather data from Environment Canada's historical data API
- **Transform**: 
  - Joins weather stations with geonames using proximity matching (Haversine formula)
  - Aggregates hourly data to monthly summaries
  - Calculates year-over-year temperature differences
- **Load**: Outputs to SQLite, CSV, and Parquet formats

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
# Windows
run_pipeline.bat

# Linux/Mac
./run_pipeline.sh
```

Or run each step individually:

```bash
python extract_weather_data.py
python transform_weather_data.py
```

## Project Structure

```
├── extract_weather_data.py   # Data extraction from Environment Canada
├── transform_weather_data.py # Data transformation and aggregation
├── sql/                      # SQL transformation queries
├── tests/                    # Unit tests
├── geonames.csv              # Geonames dimension table
└── requirements.txt          # Python dependencies
```

## Output

- `weather_data.db` - SQLite database with transformed data
- `transformed_data/` - CSV and Parquet exports

## Tests

```bash
pytest tests/
```
