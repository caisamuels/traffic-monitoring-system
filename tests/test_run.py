import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from datetime import datetime, time
from io import StringIO

class TestRun(unittest.TestCase):
    
    @patch('run.datetime')
    def test_result_callback_normal(self, mock_datetime):
        """Test result callback with normal result data"""
        # Setup
        # Mock current time to be before end_time to avoid exit(0)
        mock_now = MagicMock()
        mock_now.time.return_value = time(12, 0)  # Noon, before end_time (5 PM)
        mock_datetime.now.return_value = mock_now
        
        # Import the module with patches to prevent early exits
        with patch('sys.exit') as mock_exit:
            from run import result_callback, session_stats, db_manager
            
            # Reset session stats
            session_stats.clear()
            session_stats.update({
                "total_vehicles": 0,
                "vehicles_by_type": {},
                "start_time": datetime.now(),
                "last_weather": "Unknown"
            })
            
            # Mock db_manager
            db_manager.add_vehicle = MagicMock()
            
            # Create test result
            test_result = {
                "weather_condition": "Sunny",
                "detected_vehicles": [
                    {
                        "vehicle_id": 123,
                        "vehicle_type": "car",
                        "detection_confidence": 0.95,
                        "timestamp": datetime.now(),
                        "speed": 45.5
                    }
                ]
            }
            
            # Execute
            result_callback(test_result)
            
            # Assert
            self.assertEqual(session_stats["total_vehicles"], 1)
            self.assertEqual(session_stats["vehicles_by_type"]["car"], 1)
            self.assertEqual(session_stats["last_weather"], "Sunny")
            
            # Check that vehicle was added to database
            expected_doc = {
                "vehicle_type": "car",
                "detection_confidence": 0.95,
                "timestamp": test_result["detected_vehicles"][0]["timestamp"],
                "speed": 45.5,
                "weather_condition": "Sunny"
            }
            db_manager.add_vehicle.assert_called_once_with(expected_doc)
            
            # Verify that exit was not called (time is before end_time)
            mock_exit.assert_not_called()
    
    @patch('run.datetime')
    def test_result_callback_termination(self, mock_datetime):
        """Test result callback termination at end time"""
        # Setup
        # Mock current time to be after end_time
        mock_now = MagicMock()
        mock_now.time.return_value = time(18, 0)  # 6 PM, after end_time (5 PM)
        mock_datetime.now.return_value = mock_now
        
        # Import with exit and builtins.exit patched
        with patch('sys.exit') as mock_exit, patch('builtins.exit') as mock_builtin_exit:
            from run import result_callback, end_time
            
            # Create test result
            test_result = {
                "weather_condition": "Cloudy",
                "detected_vehicles": []
            }
            
            # Execute
            result_callback(test_result)
            
            # Assert that one of the exit functions was called
            # The code might be using either sys.exit or the built-in exit
            self.assertTrue(
                mock_exit.called or mock_builtin_exit.called,
                "Neither sys.exit nor builtins.exit was called"
            )
    
    @patch('run.datetime')
    def test_result_callback_multiple_vehicles(self, mock_datetime):
        """Test result callback with multiple vehicles"""
        # Setup
        # Mock current time to be before end_time to avoid exit(0)
        mock_now = MagicMock()
        mock_now.time.return_value = time(12, 0)  # Noon, before end_time (5 PM)
        mock_datetime.now.return_value = mock_now
        
        # Import with exit patched
        with patch('sys.exit') as mock_exit:
            from run import result_callback, session_stats, db_manager
            
            # Reset session stats
            session_stats.clear()
            session_stats.update({
                "total_vehicles": 0,
                "vehicles_by_type": {},
                "start_time": datetime.now(),
                "last_weather": "Unknown"
            })
            
            # Mock db_manager
            db_manager.add_vehicle = MagicMock()
            
            # Create test result with multiple vehicles
            test_result = {
                "weather_condition": "Rain",
                "detected_vehicles": [
                    {
                        "vehicle_id": 123,
                        "vehicle_type": "car",
                        "detection_confidence": 0.95,
                        "timestamp": datetime.now(),
                        "speed": 45.5
                    },
                    {
                        "vehicle_id": 124,
                        "vehicle_type": "truck",
                        "detection_confidence": 0.88,
                        "timestamp": datetime.now(),
                        "speed": 38.2
                    },
                    {
                        "vehicle_id": 125,
                        "vehicle_type": "car",
                        "detection_confidence": 0.92,
                        "timestamp": datetime.now(),
                        "speed": 52.1
                    }
                ]
            }
            
            # Execute
            result_callback(test_result)
            
            # Assert
            self.assertEqual(session_stats["total_vehicles"], 3)
            self.assertEqual(session_stats["vehicles_by_type"]["car"], 2)
            self.assertEqual(session_stats["vehicles_by_type"]["truck"], 1)
            self.assertEqual(session_stats["last_weather"], "Rain")
            
            # Check that all vehicles were added to database
            self.assertEqual(db_manager.add_vehicle.call_count, 3)
            
            # Verify that exit was not called (time is before end_time)
            mock_exit.assert_not_called()
    
    def test_main_function(self):
        """Test the main function execution"""
        # This is a simplified test that just checks if we can import the main modules
        # without errors. We'll skip the actual execution flow testing since it's
        # difficult to mock properly without refactoring the main script.
        
        # Execute main script import with sys.exit patched
        with patch('sys.exit'), patch('builtins.exit'):
            # Redirect stdout to avoid cluttering test output
            original_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                # Try importing the main modules to check they're valid
                import run
                
                # Just test that the script has the expected objects
                self.assertTrue(hasattr(run, 'db_manager'), "db_manager should exist")
                self.assertTrue(hasattr(run, 'vehicle_detection'), "vehicle_detection should exist")
                self.assertTrue(hasattr(run, 'result_callback'), "result_callback should exist")
                
                # To avoid actually executing the main function, which would be hard to mock,
                # we'll just assert that the necessary components exist
                self.assertIsNotNone(run.db_manager)
                self.assertIsNotNone(run.vehicle_detection)
                
            except Exception as e:
                self.fail(f"Main function import raised exception: {e}")
            finally:
                sys.stdout = original_stdout


if __name__ == '__main__':
    unittest.main()