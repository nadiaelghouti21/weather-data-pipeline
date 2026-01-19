"""
Weather Data Extraction Script
==============================
This script extracts weather data from Environment and Climate Change Canada (ECCC)
by dynamically constructing API URLs to fetch hourly climate observations.

Data Source: Environment and Climate Change Canada
API Endpoint: https://climate.weather.gc.ca/climate_data/bulk_data_e.html

Usage:
    python extract_weather_data.py

"""

import os
import sys
import json
import logging
import time
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from io import StringIO

import requests
import pandas as pd


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class ExtractionConfig:
    """Configuration for weather data extraction."""
    
    # API settings
    base_url: str = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    request_timeout: int = 60
    retry_count: int = 3
    retry_delay: int = 2
    rate_limit_delay: float = 0.5
    
    # Data parameters
    station_ids: List[int] = field(default_factory=lambda: [26953, 31688])
    years: List[int] = field(default_factory=lambda: [2022, 2023, 2024])
    months: List[int] = field(default_factory=lambda: list(range(1, 13)))
    
    # Output settings
    output_dir: str = "raw_data"
    log_dir: str = "logs"
    
    def total_requests(self) -> int:
        """Calculate total number of API requests."""
        return len(self.station_ids) * len(self.years) * len(self.months)


# =============================================================================
# EXTRACTION SUMMARY DATACLASS
# =============================================================================

@dataclass
class ExtractionSummary:
    """Summary of extraction results."""
    
    timestamp: str = ""
    extraction_start: str = ""
    extraction_end: str = ""
    api_endpoint: str = ""
    station_ids: List[int] = field(default_factory=list)
    years: List[int] = field(default_factory=list)
    months: List[int] = field(default_factory=list)
    output_dir: str = ""
    requests_made: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    records_extracted: int = 0
    output_files: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    stations_info: List[Dict[str, Any]] = field(default_factory=list)
    data_range: Dict[str, str] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "timestamp": self.timestamp,
            "extraction_start": self.extraction_start,
            "extraction_end": self.extraction_end,
            "api_endpoint": self.api_endpoint,
            "station_ids": self.station_ids,
            "years": self.years,
            "months": self.months,
            "output_dir": self.output_dir,
            "requests_made": self.requests_made,
            "requests_successful": self.requests_successful,
            "requests_failed": self.requests_failed,
            "records_extracted": self.records_extracted,
            "output_files": self.output_files,
            "columns": self.columns,
            "stations_info": self.stations_info,
            "data_range": self.data_range,
            "success": self.success,
            "error": self.error,
        }


# =============================================================================
# FILE EXPORTER CLASS
# =============================================================================

class FileExporter:
    """Handles exporting data to various file formats."""
    
    def __init__(self, output_dir: str, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger
        self._ensure_output_dir()
    
    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.logger.info(f"Output directory ready: {self.output_dir}")
        except OSError as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise
    
    def save_csv(self, df: pd.DataFrame, filename: str) -> Optional[str]:
        """Save DataFrame as CSV file."""
        filepath = os.path.join(self.output_dir, filename)
        try:
            df.to_csv(filepath, index=False, encoding="utf-8")
            self.logger.info(f"Saved CSV: {filepath} ({len(df)} records)")
            return filepath
        except IOError as e:
            self.logger.error(f"Failed to save CSV {filepath}: {e}")
            return None
    
    def save_parquet(self, df: pd.DataFrame, filename: str) -> Optional[str]:
        """Save DataFrame as Parquet file."""
        filepath = os.path.join(self.output_dir, filename)
        try:
            df.to_parquet(filepath, index=False, engine="pyarrow")
            self.logger.info(f"Saved Parquet: {filepath} ({len(df)} records)")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save Parquet {filepath}: {e}")
            return None
    
    def save_json(self, data: Dict[str, Any], filename: str) -> Optional[str]:
        """Save data as JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Saved JSON: {filepath}")
            return filepath
        except IOError as e:
            self.logger.error(f"Failed to save JSON {filepath}: {e}")
            return None


# =============================================================================
# API CLIENT CLASS
# =============================================================================

class WeatherAPIClient:
    """Client for fetching weather data from Environment Canada API."""
    
    def __init__(self, config: ExtractionConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def construct_url(self, station_id: int, year: int, month: int, day: int = 1) -> str:
        """Dynamically construct the API URL for fetching weather data."""
        return (
            f"{self.config.base_url}?"
            f"format=csv&"
            f"stationID={station_id}&"
            f"Year={year}&"
            f"Month={month}&"
            f"Day={day}&"
            f"time=LST&"
            f"timeframe=1&"
            f"submit=Download+Data"
        )
    
    def _make_request(self, url: str) -> requests.Response:
        """Make HTTP GET request to the API."""
        response = requests.get(url, timeout=self.config.request_timeout)
        response.raise_for_status()
        return response
    
    def _parse_csv_response(self, response_text: str) -> Optional[pd.DataFrame]:
        """Parse CSV data from API response text."""
        try:
            df = pd.read_csv(StringIO(response_text), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(StringIO(response_text), encoding="latin-1")
        
        return df if not df.empty else None
    
    def fetch_data(self, station_id: int, year: int, month: int) -> Optional[pd.DataFrame]:
        """
        Fetch weather data for a specific station/year/month.
        
        Implements retry logic for resilient data fetching.
        """
        url = self.construct_url(station_id, year, month)
        self.logger.debug(f"Fetching data from: {url}")
        
        for attempt in range(1, self.config.retry_count + 1):
            # Step 1: Make HTTP request
            try:
                response = self._make_request(url)
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout on attempt {attempt}/{self.config.retry_count} for station {station_id}, {year}-{month:02d}")
                if attempt < self.config.retry_count:
                    time.sleep(self.config.retry_delay)
                continue
            except requests.exceptions.ConnectionError as e:
                self.logger.warning(f"Connection error on attempt {attempt}/{self.config.retry_count}: {e}")
                if attempt < self.config.retry_count:
                    time.sleep(self.config.retry_delay)
                continue
            except requests.exceptions.HTTPError as e:
                self.logger.error(f"HTTP error for station {station_id}, {year}-{month:02d}: {e}")
                return None  # Don't retry on HTTP errors
            
            # Step 2: Validate response content
            if not response.text or len(response.text) < 100:
                self.logger.warning(f"Empty response for station {station_id}, {year}-{month:02d}")
                return None
            
            # Step 3: Parse CSV data
            try:
                df = self._parse_csv_response(response.text)
            except pd.errors.EmptyDataError:
                self.logger.warning(f"Empty CSV for station {station_id}, {year}-{month:02d}")
                return None
            
            # Step 4: Validate and return
            if df is None:
                self.logger.warning(f"No data for station {station_id}, {year}-{month:02d}")
                return None
            
            df["Station ID"] = station_id
            self.logger.info(f"Fetched {len(df)} records for station {station_id}, {year}-{month:02d}")
            return df
        
        self.logger.error(f"Failed after {self.config.retry_count} attempts")
        return None


# =============================================================================
# MAIN EXTRACTOR CLASS
# =============================================================================

class WeatherDataExtractor:
    """
    Main class for extracting weather data from Environment Canada.
    
    Usage:
        config = ExtractionConfig(station_ids=[26953, 31688], years=[2022, 2023, 2024])
        extractor = WeatherDataExtractor(config)
        summary = extractor.run()
    """
    
    def __init__(self, config: ExtractionConfig = None): # if no config is provided, use the default config
        self.config = config or ExtractionConfig()
        self.logger = self._setup_logging()
        self.api_client = WeatherAPIClient(self.config, self.logger)
        self.file_exporter = FileExporter(self.config.output_dir, self.logger)
        self.timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging to both file and console."""
        os.makedirs(self.config.log_dir, exist_ok=True) # create the log directory if it doesn't exist
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.config.log_dir, f"extraction_{timestamp}.log") # create the log file
        
        log_format = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        
        logger = logging.getLogger("weather_extraction") # create the logger
        logger.setLevel(logging.DEBUG) # set the log level to DEBUG (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(file_handler) # add the file handler to the logger
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO) # only log important stuff 
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(console_handler)
        
        logger.info(f"Logging initialized. Log file: {log_file}")
        return logger
    
    def _log_start(self) -> None:  
        """Log extraction start information.The header"""
        self.logger.info("=" * 70)
        self.logger.info("WEATHER DATA EXTRACTION STARTED")
        self.logger.info(f"Data Source: Environment and Climate Change Canada")
        self.logger.info(f"API Endpoint: {self.config.base_url}")
        self.logger.info(f"Station IDs: {self.config.station_ids}")
        self.logger.info(f"Years: {self.config.years}")
        self.logger.info(f"Months: {self.config.months}")
        self.logger.info("=" * 70)
    
    def _log_complete(self, summary: ExtractionSummary) -> None:
        """Log extraction completion summary."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("EXTRACTION COMPLETE")
        self.logger.info(f"Total API requests: {summary.requests_made}")
        self.logger.info(f"Successful requests: {summary.requests_successful}")
        self.logger.info(f"Failed requests: {summary.requests_failed}")
        self.logger.info(f"Total records extracted: {summary.records_extracted}")
        self.logger.info(f"Output files generated: {len(summary.output_files)}")
        self.logger.info("=" * 70)
    
    def _fetch_all_data(self, summary: ExtractionSummary) -> List[pd.DataFrame]:
        """Fetch weather data for all station/year/month combinations."""
        all_data: List[pd.DataFrame] = []
        total = self.config.total_requests()
        current = 0
        
        for station_id in self.config.station_ids:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Processing Station ID: {station_id}")
            self.logger.info(f"{'='*50}")
            
            for year in self.config.years:
                for month in self.config.months:
                    current += 1
                    summary.requests_made += 1
                    
                    self.logger.info(f"[{current}/{total}] Fetching station {station_id}, {year}-{month:02d}...")
                    
                    df = self.api_client.fetch_data(station_id, year, month) # actual API call
                    
                    if df is not None and not df.empty:
                        all_data.append(df)
                        summary.requests_successful += 1
                        summary.records_extracted += len(df)
                    else:
                        summary.requests_failed += 1
                        self.logger.warning(f"No data for station {station_id}, {year}-{month:02d}")
                    
                    time.sleep(self.config.rate_limit_delay) # wait for the rate limit to avoid being blocked
        
        return all_data
    
    def _extract_station_info(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract unique station metadata from DataFrame."""
        station_info = []
        
        if "Station Name" not in df.columns or "Station ID" not in df.columns:
            return station_info
        
        for sid in df["Station ID"].unique():
            row = df[df["Station ID"] == sid].iloc[0]
            station_info.append({
                "Station ID": int(sid),
                "Station Name": row.get("Station Name", "Unknown"),
                "Climate ID": row.get("Climate ID", "Unknown"),
                "Latitude (y)": row.get("Latitude (y)", None),
                "Longitude (x)": row.get("Longitude (x)", None),
            })
        
        return station_info
    
    def _prepare_metadata(self, df: pd.DataFrame, station_info: List[Dict]) -> Dict[str, Any]:
        """Prepare JSON metadata for the extraction."""
        date_col = "Date/Time (LST)"
        
        return {
            "extraction_metadata": {
                "timestamp": self.timestamp,
                "api_endpoint": self.config.base_url,
                "data_source": "Environment and Climate Change Canada",
                "extraction_time_utc": datetime.utcnow().isoformat(),
                "station_ids": self.config.station_ids,
                "years": self.config.years,
                "total_records": len(df),
            },
            "stations": station_info,
            "data_range": {
                "start_date": str(df[date_col].min()) if date_col in df.columns else None,
                "end_date": str(df[date_col].max()) if date_col in df.columns else None,
            },
            "columns": list(df.columns),
            "sample_records": json.loads(df.head(5).to_json(orient="records", date_format="iso")),
        }
    
    def _save_outputs(self, df: pd.DataFrame, metadata: Dict, summary: ExtractionSummary) -> None:
        """Save all output files."""
        self.logger.info("\nSaving data in multiple formats...")
        
        # Save CSV
        csv_file = self.file_exporter.save_csv(df, f"weather_raw_{self.timestamp}.csv")
        if csv_file:
            summary.output_files.append(csv_file)
        
        # Save Parquet
        parquet_file = self.file_exporter.save_parquet(df, f"weather_raw_{self.timestamp}.parquet")
        if parquet_file:
            summary.output_files.append(parquet_file)
        
        # Save JSON metadata
        json_file = self.file_exporter.save_json(metadata, f"weather_raw_{self.timestamp}.json")
        if json_file:
            summary.output_files.append(json_file)
        
        # Save extraction summary
        summary_file = self.file_exporter.save_json(summary.to_dict(), f"extraction_summary_{self.timestamp}.json")
        if summary_file:
            summary.output_files.append(summary_file)
    
    def run(self) -> ExtractionSummary:
        """
        Execute the full extraction process.
        
        Returns:
            ExtractionSummary with results and metadata
        """
        # Initialize summary
        summary = ExtractionSummary(
            timestamp=self.timestamp,
            extraction_start=datetime.utcnow().isoformat(),
            api_endpoint=self.config.base_url,
            station_ids=self.config.station_ids,
            years=self.config.years,
            months=self.config.months,
            output_dir=self.config.output_dir,
        )
        
        # Log start
        self._log_start()
        
        # Fetch all data
        all_data = self._fetch_all_data(summary)
        
        # Handle no data case
        if not all_data:
            self.logger.error("No data was successfully extracted!")
            summary.error = "No data extracted from any station/year/month combination"
            return summary
        
        # Combine data
        self.logger.info(f"\nCombining {len(all_data)} data chunks...")
        combined_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Total combined records: {len(combined_df)}")
        self.logger.info(f"Columns (original API names): {list(combined_df.columns)}")
        
        # Prepare metadata
        station_info = self._extract_station_info(combined_df)
        metadata = self._prepare_metadata(combined_df, station_info)
        
        # Update summary
        summary.records_extracted = len(combined_df)
        summary.stations_info = station_info
        summary.data_range = metadata["data_range"]
        summary.columns = list(combined_df.columns)
        summary.success = True
        summary.extraction_end = datetime.utcnow().isoformat()
        
        # Save outputs
        self._save_outputs(combined_df, metadata, summary)
        
        # Log completion
        self._log_complete(summary)
        
        return summary


# =============================================================================
# CLI FUNCTIONS
# =============================================================================

def print_header(config: ExtractionConfig) -> None:
    """Print the script header with configuration info."""
    print("\n" + "=" * 70)
    print("WEATHER DATA EXTRACTION SCRIPT")
    print("Data Source: Environment and Climate Change Canada")
    print("=" * 70)
    print(f"\nStation IDs to fetch: {config.station_ids}")
    print(f"Years to fetch: {config.years}")
    print(f"Total API requests: {config.total_requests()}")
    print("\nStarting extraction...\n")


def print_results(summary: ExtractionSummary) -> None:
    """Print the extraction results summary."""
    print("\n" + "-" * 50)
    print("Generated files:")
    print("-" * 50)
    for filepath in summary.output_files:
        print(f"  [OK] {filepath}")
    
    print("\n" + "-" * 50)
    print("Columns (Original API Names):")
    print("-" * 50)
    for col in summary.columns:
        print(f"  - {col}")
    
    print("\n" + "-" * 50)
    print("Extraction Summary:")
    print("-" * 50)
    print(f"  Station IDs: {summary.station_ids}")
    print(f"  Years: {summary.years}")
    print(f"  API requests made: {summary.requests_made}")
    print(f"  Successful requests: {summary.requests_successful}")
    print(f"  Failed requests: {summary.requests_failed}")
    print(f"  Total records extracted: {summary.records_extracted}")
    
    if summary.data_range:
        print(f"  Date range: {summary.data_range.get('start_date', 'N/A')} to {summary.data_range.get('end_date', 'N/A')}")
    
    if summary.stations_info:
        print("\n  Stations:")
        for station in summary.stations_info:
            print(f"    - {station.get('Station Name', 'Unknown')} (ID: {station.get('Station ID')})")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Extract weather data from Environment Canada API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        
    )
    
    parser.add_argument(
        "--stations", "-s",
        type=int,
        nargs="+",
        default=None,
        help="Station IDs to fetch (default: 26953, 31688)"
    )
    
    parser.add_argument(
        "--years", "-y",
        type=int,
        nargs="+",
        default=None,
        help="Years to fetch (default: 2022, 2023, 2024)"
    )
    
    parser.add_argument(
        "--months", "-m",
        type=int,
        nargs="+",
        default=None,
        help="Months to fetch, 1-12 (default: all months)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="raw_data",
        help="Output directory for raw data files (default: raw_data)"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create config from arguments
    config = ExtractionConfig(
        station_ids=args.stations if args.stations else [26953, 31688],
        years=args.years if args.years else [2022, 2023, 2024],
        months=args.months if args.months else list(range(1, 13)),
        output_dir=args.output_dir
    )
    
    print_header(config)
    
    extractor = WeatherDataExtractor(config)
    summary = extractor.run()
    
    if summary.success:
        print_results(summary)
        print("\nExtraction completed successfully!")
        return 0
    
    print(f"\nExtraction FAILED: {summary.error or 'Unknown error'}")
    return 1


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Usage:
        python extract_weather_data.py [OPTIONS]
    
    Options:
        --stations, -s    Station IDs to fetch
        --years, -y       Years to fetch
        --months, -m      Months to fetch (1-12)
        --output-dir, -o  Output directory
    
    Examples:
        python extract_weather_data.py --stations 26953 --years 2024
        python extract_weather_data.py -s 31688 -y 2023 2024 -m 6 7 8
    """
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n\nExtraction interrupted by user.")
        exit_code = 130
    
    sys.exit(exit_code)
