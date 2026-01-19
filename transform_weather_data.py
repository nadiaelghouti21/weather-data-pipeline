"""
Weather Data Transformation Script
===================================
This script transforms raw weather data extracted from Environment Canada
and joins it with the geonames dimension table.

Transformation steps:
1. Load raw weather data and geonames dimension table
2. Perform data quality checks
3. Join weather stations with geonames using proximity matching (Haversine)
4. Aggregate hourly data to monthly summaries
5. Calculate year-over-year temperature differences
6. Output final transformed table

Usage:
    python transform_weather_data.py

"""

import os
import sys
import sqlite3
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from glob import glob

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TransformConfig:
    """Configuration for data transformation."""
    
    # Input paths
    raw_data_dir: str = "raw_data"
    geonames_path: str = "geonames.csv"
    sql_dir: str = "sql"  # Directory for external SQL files
    
    # Output paths
    output_dir: str = "transformed_data"
    database_path: str = "weather_data.db"
    log_dir: str = "logs"
    
    # Matching settings
    max_distance_km: float = 50.0  # Maximum distance for proximity matching
    
    # SQL file names
    sql_files: Dict[str, str] = field(default_factory=lambda: {
        "monthly_aggregation": "01_monthly_aggregation.sql",
        "yoy_calculation": "02_yoy_calculation.sql",
        "final_table": "03_final_table.sql",
        "create_views": "04_create_views.sql",
    })


# =============================================================================
# DATA QUALITY CHECKER
# =============================================================================

class DataQualityChecker:
    """Performs data quality checks on weather data."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.issues: List[Dict[str, Any]] = []
    
    def check_nulls(self, df: pd.DataFrame, column: str, threshold: float = 0.1) -> bool:
        """Check if null percentage exceeds threshold."""
        null_pct = df[column].isna().sum() / len(df) # calculate the percentage of null values in the column
        if null_pct > threshold: 
            self.issues.append({ # add dectionary to the issues list
                "check": "null_check",
                "column": column,
                "null_percentage": round(null_pct * 100, 2),
                "threshold": threshold * 100,
                "passed": False
            })
            self.logger.warning(f"Column '{column}' has {null_pct:.1%} null values (threshold: {threshold:.1%})")
            return False
        self.logger.info(f"Column '{column}' null check passed ({null_pct:.1%} nulls)")
        return True
    
    def check_range(self, df: pd.DataFrame, column: str, min_val: float, max_val: float) -> bool:
        """Check if values are within expected range."""
        out_of_range = ((df[column] < min_val) | (df[column] > max_val)).sum()
        if out_of_range > 0:
            self.issues.append({
                "check": "range_check",
                "column": column,
                "out_of_range_count": int(out_of_range),
                "expected_range": [min_val, max_val],
                "passed": False
            })
            self.logger.warning(f"Column '{column}' has {out_of_range} values outside range [{min_val}, {max_val}]")
            return False
        self.logger.info(f"Column '{column}' range check passed")
        return True
    
    
    
    def run_all_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all quality checks on the DataFrame."""
        self.logger.info("=" * 50)
        self.logger.info("RUNNING DATA QUALITY CHECKS")
        self.logger.info("=" * 50)
        
        results = {
            "total_records": len(df),
            "checks_passed": 0,
            "checks_failed": 0,
            "issues": []
        }
        
        # Check for null temperatures
        if "Temp (°C)" in df.columns:
            if self.check_nulls(df, "Temp (°C)", threshold=0.3): # threshold is 30 % null values
                results["checks_passed"] += 1
            else:
                results["checks_failed"] += 1
        
        # Check temperature range (-60°C to 50°C is reasonable for Canada)
        if "Temp (°C)" in df.columns:
            temp_data = df["Temp (°C)"].dropna()
            if len(temp_data) > 0:
                if self.check_range(df.dropna(subset=["Temp (°C)"]), "Temp (°C)", -60, 50):
                    results["checks_passed"] += 1
                else:
                    results["checks_failed"] += 1
        
        # Check latitude range (41°N to 84°N is valid for Canada)
        if "Latitude (y)" in df.columns:
            if self.check_nulls(df, "Latitude (y)", threshold=0.01):  # Coordinates should rarely be null so threshold is 1%
                results["checks_passed"] += 1
            else:
                results["checks_failed"] += 1
            lat_data = df["Latitude (y)"].dropna()
            if len(lat_data) > 0:
                if self.check_range(df.dropna(subset=["Latitude (y)"]), "Latitude (y)", 41, 84):
                    results["checks_passed"] += 1
                else:
                    results["checks_failed"] += 1
        
        # Check longitude range (-141°W to -52°W is valid for Canada)
        if "Longitude (x)" in df.columns:
            if self.check_nulls(df, "Longitude (x)", threshold=0.01):  # Coordinates should rarely be null
                results["checks_passed"] += 1
            else:
                results["checks_failed"] += 1
            lon_data = df["Longitude (x)"].dropna()
            if len(lon_data) > 0:
                if self.check_range(df.dropna(subset=["Longitude (x)"]), "Longitude (x)", -141, -52):
                    results["checks_passed"] += 1
                else:
                    results["checks_failed"] += 1
        
        # Check for required columns
        required = ["Station Name", "Climate ID", "Year", "Month", "Temp (°C)"]
        for col in required:
            if col not in df.columns:
                self.logger.error(f"Missing required column: {col}")
                results["checks_failed"] += 1
            else:
                results["checks_passed"] += 1
        
        results["issues"] = self.issues
        
        self.logger.info(f"Quality checks complete: {results['checks_passed']} passed, {results['checks_failed']} failed")
        return results


# =============================================================================
# GEONAMES LOADER
# =============================================================================

class GeonamesLoader:
    """Handles loading and processing geonames dimension table."""
    
    def __init__(self, filepath: str, logger: logging.Logger):
        self.filepath = filepath
        self.logger = logger
        self.df: Optional[pd.DataFrame] = None
    
    def load(self) -> pd.DataFrame:
        """Load geonames data from CSV."""
        self.logger.info(f"Loading geonames from: {self.filepath}")
        
        self.df = pd.read_csv(self.filepath)
        self.logger.info(f"Loaded {len(self.df)} geonames records")
        self.logger.info(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def get_relevant_columns(self) -> pd.DataFrame:
        """Get only the columns needed for matching."""
        if self.df is None:
            self.load()
        
        # Select relevant columns
        cols = ["id", "name", "feature.id", "map", "latitude", "longitude"]
        available_cols = [c for c in cols if c in self.df.columns]
        
        return self.df[available_cols].copy()


# =============================================================================
# HAVERSINE DISTANCE CALCULATOR
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Coordinates of first point (degrees)
        lat2, lon2: Coordinates of second point (degrees)
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def find_nearest_geoname(
    station_lat: float,
    station_lon: float,
    geonames_df: pd.DataFrame,
    max_distance_km: float = 50.0
) -> Optional[Dict[str, Any]]:
    """
    Find the nearest geoname entry to a weather station.
    
    Args:
        station_lat: Station latitude
        station_lon: Station longitude
        geonames_df: DataFrame with geonames data
        max_distance_km: Maximum distance to consider a match
    
    Returns:
        Dictionary with matched geoname info, or None if no match found
    """
    min_distance = float('inf')
    best_match = None
    
    for _, row in geonames_df.iterrows():
        geo_lat = row.get("latitude")
        geo_lon = row.get("longitude")
        
        if pd.isna(geo_lat) or pd.isna(geo_lon):
            continue
        
        distance = haversine_distance(station_lat, station_lon, geo_lat, geo_lon)
        
        if distance < min_distance and distance <= max_distance_km:
            min_distance = distance
            best_match = {
                "geoname_id": row.get("id"),
                "geoname_name": row.get("name"),
                "feature_id": row.get("feature.id"),
                "map": row.get("map"),
                "distance_km": round(distance, 2)
            }
    
    return best_match


# =============================================================================
# DATABASE HANDLER
# =============================================================================

class DatabaseHandler:
    """Handles SQLite database operations."""
    
    def __init__(self, db_path: str, logger: logging.Logger):
        self.db_path = db_path
        self.logger = logger
        self.conn: Optional[sqlite3.Connection] = None
    
    def connect(self) -> sqlite3.Connection:
        """Create database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.logger.info(f"Connected to database: {self.db_path}")
        return self.conn
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        if self.conn is None:
            self.connect()
        
        self.logger.debug(f"Executing query: {query[:100]}...")
        
        if params:
            return pd.read_sql_query(query, self.conn, params=params)
        return pd.read_sql_query(query, self.conn)
    
    def load_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> None:
        """Load DataFrame into database table."""
        if self.conn is None:
            self.connect()
        
        df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
        self.logger.info(f"Loaded {len(df)} records into table '{table_name}'")
    
    def execute_sql_file(self, filepath: str) -> None:
        """Execute SQL statements from a file (for DDL/scripts)."""
        if self.conn is None:
            self.connect()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            sql = f.read()
        
        self.conn.executescript(sql)
        self.conn.commit()
        self.logger.info(f"Executed SQL file: {filepath}")
    
    def load_sql_file(self, filepath: str) -> str:
        """Load SQL query from external file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            sql = f.read()
        self.logger.info(f"Loaded SQL from: {filepath}")
        return sql
    
    def execute_query_from_file(self, filepath: str) -> pd.DataFrame:
        """Load SQL from file and execute as query, returning DataFrame."""
        sql = self.load_sql_file(filepath)
        self.logger.debug(f"SQL Query:\n{sql[:200]}...")
        return self.execute_query(sql)


# =============================================================================
# MAIN TRANSFORMER CLASS
# =============================================================================

class WeatherDataTransformer:
    """
    Main class for transforming weather data.
    
    Performs:
    - Data loading and validation
    - Quality checks
    - Proximity-based join with geonames
    - Monthly aggregation
    - Year-over-year calculations
    """
    
    def __init__(self, config: TransformConfig = None):
        self.config = config or TransformConfig()
        self.logger = self._setup_logging()
        self.db = DatabaseHandler(self.config.database_path, self.logger)
        self.quality_checker = DataQualityChecker(self.logger)
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.config.log_dir, f"transformation_{timestamp}.log")
        
        log_format = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
        
        logger = logging.getLogger("weather_transformation")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
        
        logger.info(f"Logging initialized. Log file: {log_file}")
        return logger
    
    def _find_latest_raw_data(self) -> str:
        """Find the most recent raw data CSV file."""
        pattern = os.path.join(self.config.raw_data_dir, "weather_raw_*.csv")
        files = glob(pattern) # find all files matching the pattern
        
        if not files:
            raise FileNotFoundError(f"No raw data files found matching: {pattern}")
        
        # Sort by modification time, get most recent
        latest = max(files, key=os.path.getmtime)
        self.logger.info(f"Found latest raw data: {latest}")
        return latest
    
    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw weather data."""
        filepath = self._find_latest_raw_data()
        self.logger.info(f"Loading raw data from: {filepath}")
        
        df = pd.read_csv(filepath)
        self.logger.info(f"Loaded {len(df)} raw records")
        return df
    
    def _match_stations_to_geonames(
        self,
        weather_df: pd.DataFrame,
        geonames_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Match weather stations to nearest geonames entries."""
        self.logger.info("Matching weather stations to geonames...")
        
        # Get unique stations
        stations = weather_df[["Station Name", "Latitude (y)", "Longitude (x)"]].drop_duplicates()
        
        station_matches = []
        for _, station in stations.iterrows(): # iterate over each unique station
            station_name = station["Station Name"]
            lat = station["Latitude (y)"]
            lon = station["Longitude (x)"]
            
            match = find_nearest_geoname(lat, lon, geonames_df, self.config.max_distance_km) # Call the Haversine-based function to find nearest geoname within 50km
            
            if match:
                self.logger.info(
                    f"Station '{station_name}' matched to '{match['geoname_name']}' "
                    f"(distance: {match['distance_km']} km)"
                )
            else:
                self.logger.warning(f"No geoname match found for station '{station_name}'")
                match = {
                    "geoname_id": None,
                    "geoname_name": None,
                    "feature_id": None,
                    "map": None,
                    "distance_km": None
                }
            
            station_matches.append({
                "Station Name": station_name,
                **match
            })
        
        return pd.DataFrame(station_matches)
    
    def _get_sql_path(self, sql_key: str) -> str:
        """Get full path to SQL file."""
        filename = self.config.sql_files.get(sql_key)
        if not filename:
            raise ValueError(f"Unknown SQL file key: {sql_key}")
        return os.path.join(self.config.sql_dir, filename)
    
    def _aggregate_monthly_sql(self) -> pd.DataFrame:
        """
        Aggregate hourly data to monthly summaries using SQL.
        
        Loads SQL from external file: sql/01_monthly_aggregation.sql
        """
        self.logger.info("Aggregating to monthly summaries using SQL...")
        
        # Load and execute SQL from external file
        sql_path = self._get_sql_path("monthly_aggregation")
        self.logger.info(f"Loading SQL from: {sql_path}")
        
        monthly_df = self.db.execute_query_from_file(sql_path)
        
        self.logger.info(f"SQL query returned {len(monthly_df)} monthly aggregations")
        
        # Save the aggregated data to a table
        self.db.load_dataframe(monthly_df, "weather_monthly_agg")
        
        return monthly_df
    
    def _calculate_yoy_difference_sql(self) -> pd.DataFrame:
        """
        Calculate year-over-year temperature difference using SQL.
        
        Loads SQL from external file: sql/02_yoy_calculation.sql
        """
        self.logger.info("Calculating year-over-year temperature differences using SQL...")
        
        # Load and execute SQL from external file
        sql_path = self._get_sql_path("yoy_calculation")
        self.logger.info(f"Loading SQL from: {sql_path}")
        
        yoy_df = self.db.execute_query_from_file(sql_path)
        
        yoy_count = yoy_df["temperature_celsius_yoy_avg"].notna().sum()
        self.logger.info(f"SQL query calculated YoY differences for {yoy_count} records")
        
        # Save to table
        self.db.load_dataframe(yoy_df, "weather_monthly_yoy")
        
        return yoy_df
    
    def _create_final_table_sql(self, station_matches: pd.DataFrame) -> pd.DataFrame:
        """
        Create final transformed table using SQL join with geonames.
        
        Loads SQL from external file: sql/03_final_table.sql
        """
        self.logger.info("Creating final transformed table using SQL...")
        
        # First ensure station mapping is in database
        self.db.load_dataframe(station_matches, "station_geoname_mapping")
        
        # Load and execute SQL from external file
        sql_path = self._get_sql_path("final_table")
        self.logger.info(f"Loading SQL from: {sql_path}")
        
        final_df = self.db.execute_query_from_file(sql_path)
        
        self.logger.info(f"SQL query returned final table with {len(final_df)} records and {len(final_df.columns)} columns")
        
        # Save final table to database
        self.db.load_dataframe(final_df, "weather_final")
        
        # Create database views
        views_path = self._get_sql_path("create_views")
        self.logger.info(f"Creating database views from: {views_path}")
        self.db.execute_sql_file(views_path)
        
        return final_df
    
    def _save_outputs(self, df: pd.DataFrame) -> List[str]:
        """Save transformed data to multiple formats."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        output_files = []
        
        # Save as CSV
        csv_path = os.path.join(self.config.output_dir, f"weather_transformed_{self.timestamp}.csv")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved CSV: {csv_path}")
        output_files.append(csv_path)
        
        # Save as Parquet
        parquet_path = os.path.join(self.config.output_dir, f"weather_transformed_{self.timestamp}.parquet")
        df.to_parquet(parquet_path, index=False)
        self.logger.info(f"Saved Parquet: {parquet_path}")
        output_files.append(parquet_path)
        
        # Save to SQLite database
        self.db.connect()
        self.db.load_dataframe(df, "weather_monthly")
        
        # Also create a view with the final schema
        create_view_sql = """
        CREATE VIEW IF NOT EXISTS v_weather_final AS
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
        FROM weather_monthly
        ORDER BY station_name, date_month;
        """
        self.db.conn.execute(create_view_sql)
        self.db.conn.commit()
        self.logger.info(f"Created database view: v_weather_final")
        
        self.db.close()
        output_files.append(self.config.database_path)
        
        return output_files
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the full transformation process.
        
        Returns:
            Dictionary with transformation results
        """
        self.logger.info("=" * 70)
        self.logger.info("WEATHER DATA TRANSFORMATION STARTED")
        self.logger.info("=" * 70)
        
        results = {
            "timestamp": self.timestamp,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "success": False,
            "output_files": [],
            "quality_report": None,
            "error": None
        }
        
        # Step 1: Load raw data
        raw_df = self._load_raw_data()
        results["raw_records"] = len(raw_df)
        
        # Step 2: Run quality checks
        quality_report = self.quality_checker.run_all_checks(raw_df)
        results["quality_report"] = quality_report
        
        # Step 3: Load geonames
        geonames_loader = GeonamesLoader(self.config.geonames_path, self.logger)
        geonames_df = geonames_loader.get_relevant_columns()
        
        # Step 4: Match stations to geonames
        station_matches = self._match_stations_to_geonames(raw_df, geonames_df)
        
        # Step 5: Store raw data in database for SQL operations
        self.db.connect()
        self.db.load_dataframe(raw_df, "weather_raw")
        self.db.load_dataframe(geonames_df, "geonames")
        
        # Step 6: Aggregate to monthly using SQL
        self.logger.info("\n" + "=" * 50)
        self.logger.info("EXECUTING SQL TRANSFORMATIONS")
        self.logger.info("=" * 50)
        
        monthly_df = self._aggregate_monthly_sql()
        
        # Step 7: Calculate YoY differences using SQL
        monthly_df = self._calculate_yoy_difference_sql()
        
        # Step 8: Create final table using SQL join
        final_df = self._create_final_table_sql(station_matches)
        results["final_records"] = len(final_df)
        
        # Step 9: Save outputs
        output_files = self._save_outputs(final_df)
        results["output_files"] = output_files
        
        # Log completion
        results["success"] = True
        results["end_time"] = datetime.now(timezone.utc).isoformat()
        
        self.logger.info("=" * 70)
        self.logger.info("TRANSFORMATION COMPLETE")
        self.logger.info(f"Raw records: {results['raw_records']}")
        self.logger.info(f"Final records: {results['final_records']}")
        self.logger.info(f"Output files: {len(output_files)}")
        self.logger.info("=" * 70)
        
        return results


# =============================================================================
# CLI FUNCTIONS
# =============================================================================

def print_header() -> None:
    """Print script header."""
    print("\n" + "=" * 70)
    print("WEATHER DATA TRANSFORMATION SCRIPT")
    print("Using SQL queries executed programmatically from Python")
    print("=" * 70)
    print("\nTransformation steps:")
    print("  1. Load raw weather data into SQLite database")
    print("  2. Run data quality checks")
    print("  3. Match stations to geonames (proximity)")
    print("  4. [SQL] Aggregate to monthly summaries")
    print("  5. [SQL] Calculate year-over-year differences")
    print("  6. [SQL] Join with geonames for final table")
    print("  7. Output final transformed table")
    print("\nStarting transformation...\n")


def print_results(results: Dict[str, Any]) -> None:
    """Print transformation results."""
    print("\n" + "-" * 50)
    print("Transformation Results:")
    print("-" * 50)
    print(f"  Raw records processed: {results.get('raw_records', 'N/A')}")
    print(f"  Final records created: {results.get('final_records', 'N/A')}")
    
    print("\n" + "-" * 50)
    print("Output files:")
    print("-" * 50)
    for f in results.get("output_files", []):
        print(f"  [OK] {f}")
    
    if results.get("quality_report"):
        qr = results["quality_report"]
        print("\n" + "-" * 50)
        print("Quality Check Summary:")
        print("-" * 50)
        print(f"  Checks passed: {qr.get('checks_passed', 0)}")
        print(f"  Checks failed: {qr.get('checks_failed', 0)}")


def main() -> int:
    """Main entry point."""
    print_header()
    
    config = TransformConfig()
    transformer = WeatherDataTransformer(config)
    results = transformer.run()
    
    if results["success"]:
        print_results(results)
        print("\nTransformation completed successfully!")
        return 0
    
    print(f"\nTransformation FAILED: {results.get('error', 'Unknown error')}")
    return 1


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Usage:
        python transform_weather_data.py
    
    The script will:
    1. Load the most recent raw weather data from raw_data/
    2. Join with geonames.csv using proximity matching
    3. Aggregate to monthly temperature statistics
    4. Calculate year-over-year temperature differences
    5. Output final table to transformed_data/ and SQLite database
    """
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n\nTransformation interrupted by user.")
        exit_code = 130
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        logging.exception("Unhandled exception during transformation")
        exit_code = 1
    
    sys.exit(exit_code)
