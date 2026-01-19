"""
Unit Tests for Weather Data Transformation
==========================================
Tests for the transform_weather_data.py module.

Run tests:
    python -m pytest tests/test_transformation.py -v
    
Or run all tests:
    python -m pytest tests/ -v
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch
import math

import pandas as pd

# Add parent directory to path so we can import from transform_weather_data.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the classes and functions we want to test
from transform_weather_data import (
    TransformConfig,
    DataQualityChecker,
    GeonamesLoader,
    DatabaseHandler,
    haversine_distance,
    find_nearest_geoname,
)



class TestTransformConfig(unittest.TestCase):
    """Tests for TransformConfig dataclass - stores transformation settings."""
    
    def test_default_values(self):
        """Test: Are default values correct when no arguments given?"""
        config = TransformConfig()  # Create with defaults
        
        # Check each default value
        self.assertEqual(config.raw_data_dir, "raw_data")  # Where raw data is stored
        self.assertEqual(config.geonames_path, "geonames.csv")  # Geonames file location
        self.assertEqual(config.output_dir, "transformed_data")  # Where to save output
        self.assertEqual(config.max_distance_km, 50.0)  # Max distance for location matching
        self.assertEqual(config.sql_dir, "sql")  # Where SQL files are stored
    
    def test_sql_files_mapping(self):
        """Test: Does config have correct SQL file name mappings?"""
        config = TransformConfig()
        
        # Check that expected SQL keys exist
        self.assertIn("monthly_aggregation", config.sql_files)
        self.assertIn("yoy_calculation", config.sql_files)
        self.assertIn("final_table", config.sql_files)
        # Check one specific mapping
        self.assertEqual(config.sql_files["monthly_aggregation"], "01_monthly_aggregation.sql")



class TestHaversineDistance(unittest.TestCase):
    """Tests for haversine_distance function - calculates distance between two coordinates."""
    
    def test_same_point(self):
        """Test: Distance from a point to itself should be zero."""
        # Same coordinates for both points
        distance = haversine_distance(43.67, -79.40, 43.67, -79.40)
        self.assertEqual(distance, 0.0)  # Should be exactly 0
    
    def test_known_distance(self):
        """Test: Distance between Toronto and Montreal should be ~500 km."""
        # Toronto coordinates: 43.67, -79.40
        # Montreal coordinates: 45.50, -73.57
        distance = haversine_distance(43.67, -79.40, 45.50, -73.57)
        
        # We know this distance is approximately 500-550 km
        self.assertGreater(distance, 450)  # At least 450 km
        self.assertLess(distance, 600)  # No more than 600 km
    




class TestFindNearestGeoname(unittest.TestCase):
    """Tests for find_nearest_geoname function - matches weather station to nearest city."""
    
    def setUp(self):
        """Create fake geonames data for testing (3 cities)."""
        self.geonames_df = pd.DataFrame({
            "id": ["A", "B", "C"],
            "name": ["Toronto", "Montreal", "Ottawa"],
            "feature.id": ["feat1", "feat2", "feat3"],
            "map": ["map1", "map2", "map3"],
            "latitude": [43.65, 45.50, 45.42],
            "longitude": [-79.38, -73.57, -75.69]
        })
    
    def test_find_nearest_match(self):
        """Test: Can we find Toronto when given coordinates near Toronto?"""
        # Station coordinates near Toronto (43.67, -79.40)
        result = find_nearest_geoname(43.67, -79.40, self.geonames_df)
        
        self.assertIsNotNone(result)  # Should find a match
        self.assertEqual(result["geoname_name"], "Toronto")  # Should be Toronto
        self.assertLess(result["distance_km"], 5)  # Should be very close (<5 km)
    
    def test_no_match_outside_threshold(self):
        """Test: Returns None when no city is close enough."""
        # Coordinates in the middle of the ocean (0, 0) - far from any city
        result = find_nearest_geoname(0.0, 0.0, self.geonames_df, max_distance_km=50)
        
        self.assertIsNone(result)  # No match within 50 km
    
    def test_custom_max_distance(self):
        """Test: Can we set a custom max distance threshold?"""
        # Coordinates exactly at Montreal (45.50, -73.57)
        # With max_distance=1 km, should only match Montreal (exact location)
        result = find_nearest_geoname(45.50, -73.57, self.geonames_df, max_distance_km=1)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["geoname_name"], "Montreal")  # Exact match
    
    def test_result_contains_required_fields(self):
        """Test: Does the result have all the fields we need?"""
        result = find_nearest_geoname(43.67, -79.40, self.geonames_df)
        
        # Check all required fields exist
        self.assertIn("geoname_id", result)
        self.assertIn("geoname_name", result)
        self.assertIn("feature_id", result)  # Needed for final table
        self.assertIn("map", result)  # Needed for final table
        self.assertIn("distance_km", result)  # How far was the match?



class TestDataQualityChecker(unittest.TestCase):
    """Tests for DataQualityChecker class - validates data quality."""
    
    def setUp(self):
        """Create a quality checker with fake logger before each test."""
        self.logger = Mock()  # Fake logger (silent)
        self.checker = DataQualityChecker(self.logger)
    
    def test_check_nulls_pass(self):
        """Test: Null check PASSES when null % is below threshold."""
        df = pd.DataFrame({
            "col": [1, 2, 3, None, 5, 6, 7, 8, 9, 10]  # 1 null out of 10 = 10% nulls
        })
        
        # Threshold is 20%, actual is 10% → should PASS
        result = self.checker.check_nulls(df, "col", threshold=0.2)
        
        self.assertTrue(result)  # 10% < 20% → Pass
    
    def test_check_nulls_fail(self):
        """Test: Null check FAILS when null % is above threshold."""
        df = pd.DataFrame({
            "col": [1, None, None, None, 5, None, None, None, None, 10]  # 7 nulls = 70%
        })
        
        # Threshold is 30%, actual is 70% → should FAIL
        result = self.checker.check_nulls(df, "col", threshold=0.3)
        
        self.assertFalse(result)  # 70% > 30% → Fail
        self.assertEqual(len(self.checker.issues), 1)  # Should record 1 issue
    
    def test_check_range_pass(self):
        """Test: Range check PASSES when all values are within min/max."""
        df = pd.DataFrame({
            "temp": [10, 15, 20, 25, 30]  # All between 0 and 40
        })
        
        result = self.checker.check_range(df, "temp", min_val=0, max_val=40)
        
        self.assertTrue(result)  # All values in range → Pass
    
    def test_check_range_fail(self):
        """Test: Range check FAILS when values are outside min/max."""
        df = pd.DataFrame({
            "temp": [10, 15, 100, 25, -100]  # 100 and -100 are outside -50 to 50
        })
        
        result = self.checker.check_range(df, "temp", min_val=-50, max_val=50)
        
        self.assertFalse(result)  # Some values out of range → Fail


class TestDatabaseHandler(unittest.TestCase):
    """Tests for DatabaseHandler class - manages SQLite database."""
    
    def setUp(self):
        """Create a temporary test database before each test."""
        self.logger = Mock()  # Fake logger (silent)
        self.test_db = "test_temp.db"  # Temporary database file
        self.handler = DatabaseHandler(self.test_db, self.logger)
    
    def tearDown(self):
        """Delete the test database after each test (cleanup)."""
        self.handler.close()  # Close database connection
        if os.path.exists(self.test_db):
            os.remove(self.test_db)  # Delete the .db file
    
    def test_connect(self):
        """Test: Can we connect to the database?"""
        conn = self.handler.connect()
        
        self.assertIsNotNone(conn)  # Connection object exists
        self.assertIsNotNone(self.handler.conn)  # Stored in handler
    
    def test_load_dataframe(self):
        """Test: Can we load a DataFrame into a database table?"""
        # Create test data
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"]
        })
        
        # Load into database as table "test_table"
        self.handler.load_dataframe(df, "test_table")
        
        # Query the table to verify data was loaded
        result = self.handler.execute_query("SELECT * FROM test_table")
        
        self.assertEqual(len(result), 3)  # Should have 3 rows
        self.assertIn("id", result.columns)  # Should have "id" column
    
    def test_execute_query(self):
        """Test: Can we execute SQL and get results?"""
        # Load some numbers into database
        df = pd.DataFrame({"value": [10, 20, 30]})
        self.handler.load_dataframe(df, "numbers")
        
        # Execute a SUM query
        result = self.handler.execute_query("SELECT SUM(value) as total FROM numbers")
        
        self.assertEqual(result["total"].iloc[0], 60)  # 10 + 20 + 30 = 60
    
    def test_load_sql_file(self):
        """Test: Can we load SQL from an external .sql file?"""
        # Create a temporary SQL file
        sql_content = "SELECT 1 + 1 AS result;"
        sql_file = "test_temp.sql"
        
        with open(sql_file, "w") as f:
            f.write(sql_content)
        
        try:
            # Load the SQL from file
            loaded = self.handler.load_sql_file(sql_file)
            self.assertEqual(loaded.strip(), sql_content)  # Content should match
        finally:
            os.remove(sql_file)  # Clean up the temp file


class TestSQLFiles(unittest.TestCase):
    """Tests for SQL file existence - ensures all required .sql files are present."""
    
    def test_sql_files_exist(self):
        """Test: Do all required SQL files exist in the sql/ folder?"""
        sql_dir = "sql"
        
        # List of SQL files that MUST exist for the pipeline to work
        required_files = [
            "01_monthly_aggregation.sql",  # Aggregates hourly → monthly
            "02_yoy_calculation.sql",  # Calculates year-over-year change
            "03_final_table.sql",  # Creates the final output table
            "04_create_views.sql"  # Creates database views
        ]
        
        # Check each file exists
        for filename in required_files:
            filepath = os.path.join(sql_dir, filename)
            self.assertTrue(
                os.path.exists(filepath),
                f"Missing SQL file: {filepath}"  # Error message if missing
            )
    
    def test_sql_files_not_empty(self):
        """Test: Are the SQL files actually populated (not empty)?"""
        sql_dir = "sql"
        
        # Check each .sql file in the folder
        for filename in os.listdir(sql_dir):
            if filename.endswith(".sql"):
                filepath = os.path.join(sql_dir, filename)
                with open(filepath, "r") as f:
                    content = f.read()
                
                # Each file should have at least 50 characters of SQL
                self.assertGreater(
                    len(content), 50,
                    f"SQL file too short (possibly empty): {filepath}"
                )



if __name__ == "__main__":
    unittest.main()

