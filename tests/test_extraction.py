"""
Unit Tests for Weather Data Extraction
=======================================
Tests for the extract_weather_data.py module.

Run tests:
    python -m pytest tests/test_extraction.py -v
    
Or run all tests:
    python -m pytest tests/ -v
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import requests
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract_weather_data import (
    ExtractionConfig,
    ExtractionSummary,
    FileExporter,
    WeatherAPIClient,
    WeatherDataExtractor,
)


class TestExtractionConfig(unittest.TestCase):
    """Tests for config extraction processes"""
    
    def test_default_values(self):
        """Test if the default configuration values are set correctly."""
        config = ExtractionConfig() # create a config object without arguments 
        
        self.assertEqual(config.station_ids, [26953, 31688]) # test if the station ids are set to the default values
        self.assertEqual(config.years, [2022, 2023, 2024]) # test if the years are set to the default values
        self.assertEqual(len(config.months), 12) # test if the months are set to the default values
        self.assertEqual(config.output_dir, "raw_data") # test if the output directory is set to the default values
        self.assertEqual(config.retry_count, 3) # test if the retry count is set to the default values
    
    def test_custom_values(self):
        """Test that custom configuration values are applied."""
        config = ExtractionConfig(  # config object with custom values
            station_ids=[12345],
            years=[2024],
            months=[1, 2, 3],
            output_dir="custom_output"
        )
        
        self.assertEqual(config.station_ids, [12345]) # test if the station ids are set to the custom values        
        self.assertEqual(config.years, [2024]) # test if the years are set to the custom values
        self.assertEqual(config.months, [1, 2, 3]) # test if the months are set to the custom values
        self.assertEqual(config.output_dir, "custom_output") # test if the output directory is set to the custom values
    
    def test_total_requests_calculation(self):
        """Test total_requests() calculates, it tests if the number of API requests is calculated correctly."""
        config = ExtractionConfig(
            station_ids=[1, 2],
            years=[2023, 2024],
            months=[1, 2, 3]
        )
        
        # 2 stations × 2 years × 3 months = 12 requests - test if the number of API requests is calculated correctly
        self.assertEqual(config.total_requests(), 12)


class TestExtractionSummary(unittest.TestCase):
    """Tests for ExtractionSummary dataclass."""
    
    def test_default_values(self):
        """Test that summary starts with correct defaults."""
        summary = ExtractionSummary() # create a summary object without arguments
        
        self.assertEqual(summary.requests_made, 0)
        self.assertEqual(summary.requests_successful, 0)
        self.assertEqual(summary.requests_failed, 0)
        self.assertFalse(summary.success)
        self.assertEqual(summary.output_files, [])
    
    def test_to_dict(self):
        """Test to_dict() returns proper dictionary."""
        summary = ExtractionSummary(     # create a summary with specific values 
            timestamp="20240101_120000",
            requests_made=10,
            success=True
        )
        
        result = summary.to_dict() # convert the summary to a dictionary to test if the conversion is correct 
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["timestamp"], "20240101_120000")
        self.assertEqual(result["requests_made"], 10)
        self.assertTrue(result["success"])


class TestWeatherAPIClient(unittest.TestCase):
    """Tests for WeatherAPIClient class."""
    
    def setUp(self):
        """Set up test exnvironoments."""
        self.config = ExtractionConfig()
        self.logger = Mock()
        self.client = WeatherAPIClient(self.config, self.logger)
    
    def test_construct_url(self):
        """Test URL construction with parameters."""
        url = self.client.construct_url(  
            station_id=31688,
            year=2024,
            month=6
        )
        
        self.assertIn("stationID=31688", url) # test if the station id is included in the URL
        self.assertIn("Year=2024", url) # test if the year is included in the URL
        self.assertIn("Month=6", url) # test if the month is included in the URL
        self.assertIn("format=csv", url) # test if the format is included in the URL
        self.assertIn("timeframe=1", url) # test if the timeframe is included in the URL
    
    @patch('extract_weather_data.requests.get')
    def test_fetch_data_success(self, mock_get):
        """Test successful data fetch."""
        # Mock the API response 
        csv_content = (
            '"Longitude (x)","Latitude (y)","Station Name","Climate ID","Date/Time (LST)","Year","Month","Day","Time (LST)","Temp (°C)"\n'
            '-79.40,43.67,"TORONTO CITY",6158355,"2024-06-01 00:00","2024","6","1","00:00",15.5\n'
            '-79.40,43.67,"TORONTO CITY",6158355,"2024-06-01 01:00","2024","6","1","01:00",16.0\n'
        )
        mock_response = Mock() #empty object to mock the API response
        mock_response.text = csv_content # set the text of the response to the mock CSV content
        mock_response.raise_for_status = Mock() # set the raise_for_status method of the response to the mock method
        mock_get.return_value = mock_response
        
        result = self.client.fetch_data(31688, 2024, 6) # fetch the data from the API id/year/month
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn("Station ID", result.columns) # test if the Station ID column is included in the result
    
    @patch('extract_weather_data.requests.get')
    def test_fetch_data_timeout(self, mock_get): # Replace requests.get in the extract_weather_data file with a fake method.
        """Test handling of timeout errors."""
        mock_get.side_effect = requests.exceptions.Timeout() # When requests.get() is called, throw a Timeout error
        
        result = self.client.fetch_data(31688, 2024, 6)
        
        self.assertIsNone(result) # Check result is None 
        
        self.assertTrue(self.logger.warning.called) # a warning was logged about the timeout.
    
    @patch('extract_weather_data.requests.get')
    def test_fetch_data_empty_response(self, mock_get):
        """Test handling of empty response."""
        mock_response = Mock() #empty object to mock the API response
        mock_response.text = "" # set the text of the response to an empty string
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = self.client.fetch_data(31688, 2024, 6)
        
        self.assertIsNone(result)


class TestFileExporter(unittest.TestCase):
    """Tests for FileExporter class - saves data to files."""
    
    def setUp(self):
        """Runs BEFORE each test - create fake logger and temp folder name."""
        self.logger = Mock()  
        self.test_dir = "test_output_temp"  # Temporary folder for test files
    
    def tearDown(self):
        """Runs AFTER each test - delete temp folder and files."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)  # Delete folder and all contents
    
    def test_save_csv(self):
        """Test: Can we save a DataFrame as CSV file?"""
        exporter = FileExporter(self.test_dir, self.logger)  # Create exporter
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})  # Fake data
        
        result = exporter.save_csv(df, "test.csv")  # Save to CSV
        
        self.assertIsNotNone(result)  # Check: returned a path (not None)
        self.assertTrue(os.path.exists(result))  # Check: file exists on disk
        
        # Extra check: read the file back and verify content
        loaded = pd.read_csv(result)
        self.assertEqual(len(loaded), 2)  # Check: has 2 rows
    
    def test_save_json(self):
        """Test: Can we save a dictionary as JSON file?"""
        exporter = FileExporter(self.test_dir, self.logger)  # Create exporter
        data = {"key": "value", "number": 42}  # Fake data (dictionary)
        
        result = exporter.save_json(data, "test.json")  # Save to JSON
        
        self.assertIsNotNone(result)  # Check: returned a path (not None)
        self.assertTrue(os.path.exists(result))  # Check: file exists on disk


class TestArgumentParsing(unittest.TestCase):
    """Tests for command line argument parsing (--stations, --years, etc.)."""
    
    def test_parse_stations(self):
        """Test: Does --stations 26953 31688 work correctly?"""
        from extract_weather_data import parse_arguments  # Import inside the method so it sees our fake arguments
        
        # Fake command line: python script.py --stations 26953 31688
        with patch('sys.argv', ['script', '--stations', '26953', '31688']):
            args = parse_arguments()
        
        self.assertEqual(args.stations, [26953, 31688])  # Check: parsed as list of ints
    
    def test_parse_years(self):
        """Test: Does --years 2023 2024 work correctly?"""
        from extract_weather_data import parse_arguments
        
        # Fake command line: python script.py --years 2023 2024
        with patch('sys.argv', ['script', '--years', '2023', '2024']):
            args = parse_arguments()
        
        self.assertEqual(args.years, [2023, 2024])  # Check: parsed as list of ints
    
    def test_parse_output_dir(self):
        """Test: Does --output-dir my_output work correctly?"""
        from extract_weather_data import parse_arguments
        
        # Fake command line: python script.py --output-dir my_output
        with patch('sys.argv', ['script', '--output-dir', 'my_output']):
            args = parse_arguments()
        
        self.assertEqual(args.output_dir, 'my_output')  # Check: parsed correctly
    
    def test_default_values(self):
        """Test: What happens when NO arguments are provided?"""
        from extract_weather_data import parse_arguments
        
        # Fake command line: python script.py (no arguments)
        with patch('sys.argv', ['script']):
            args = parse_arguments()
        
        
        self.assertIsNone(args.stations)  
        self.assertIsNone(args.years) 
        self.assertEqual(args.output_dir, 'raw_data')  # Default output folder


if __name__ == "__main__":
    unittest.main()  # Run all tests when file is executed directly

