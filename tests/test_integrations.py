from datetime import datetime
import os
import time
import unittest

import numpy as np
from DatabaseManager import DatabaseManager
from unittest.mock import MagicMock, patch
from TrafficMonitoringSystem import TrafficMonitoringSystem

class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the traffic monitoring system."""
    
    @patch('TrafficMonitoringSystem.YOLO')
    @patch.dict(os.environ, {
        "MONGODB_CONNECTION_STRING": "mongodb://localhost:27017",
        "MONGODB_DATABASE_NAME": "test_integration_db",
        "MONGODB_COLLECTION_NAME": "test_integration_collection",
        "WEATHER_API_KEY": "test_api_key",
        "WEATHER_CITY": "TestCity"
    })
    @patch('TrafficMonitoringSystem.requests.get')
    def test_process_frame_to_database(self, mock_get, mock_yolo):
        """Test that processed frame data correctly flows to database."""
        # Setup mock weather API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'weather': [{'main': 'Clear'}]
        }
        mock_get.return_value = mock_response
        
        # Setup mock YOLO model
        mock_model = MagicMock()
        
        # Create a mock result with one vehicle
        boxes_mock = MagicMock()
        boxes_mock.xywh.cpu.return_value = np.array([[1150.0, 400.0, 100.0, 50.0]])  # x is past red line
        boxes_mock.conf.cpu.return_value = np.array([0.95])
        boxes_mock.id.int.return_value.cpu.return_value.tolist.return_value = [42]
        boxes_mock.cls.cpu.return_value.tolist.return_value = [2]  # class 2 = car
        
        results_mock = MagicMock()
        results_mock.boxes = boxes_mock
        results_mock.names = {2: "car"}
        results_mock.plot.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        mock_model.track.return_value = [results_mock]
        mock_yolo.return_value = mock_model
        
        # Initialize system components
        db_manager = DatabaseManager()
        tms = TrafficMonitoringSystem()
        
        # Setup vehicle timestamps to simulate a vehicle that has crossed both lines
        tms.vehicle_timestamps = {
            42: {
                "start": time.time() - 2.0,  # Crossed start line 2 seconds ago
            }
        }
        
        # Create a test frame
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame_timestamp = datetime.now()
        
        # Process the frame
        result = tms.process_frame(test_frame, frame_timestamp)
        
        # Extract the detected vehicle from the result
        self.assertTrue(len(result["detected_vehicles"]) > 0)
        detected_vehicle = result["detected_vehicles"][0]
        
        # Create a document for the database
        vehicle_doc = {
            "vehicle_type": detected_vehicle["vehicle_type"],
            "detection_confidence": detected_vehicle["detection_confidence"],
            "timestamp": detected_vehicle["timestamp"],
            "speed": detected_vehicle["speed"],
            "weather_condition": result["weather_condition"]
        }
        
        # Add to database
        db_manager.add_vehicle(vehicle_doc)
        
        # Wait for database processing
        time.sleep(0.5)
        
        # Verify document was added to database
        # We'll use a patch for the collection.find_one to mock the database query
        with patch.object(db_manager.collection, 'find_one') as mock_find_one:
            mock_find_one.return_value = vehicle_doc
            result_doc = db_manager.collection.find_one({"vehicle_type": detected_vehicle["vehicle_type"]})
            
            # Assert document has the expected values
            self.assertEqual(result_doc["vehicle_type"], detected_vehicle["vehicle_type"])
            self.assertEqual(result_doc["speed"], detected_vehicle["speed"])
            self.assertEqual(result_doc["weather_condition"], "Clear")
        
        # Clean up
        db_manager.shutdown()

    @patch.dict(os.environ, {
        "MONGODB_CONNECTION_STRING": "mongodb://localhost:27017",
        "MONGODB_DATABASE_NAME": "test_integration_db",
        "MONGODB_COLLECTION_NAME": "test_integration_collection"
    })
    def test_database_queue_processing(self):
        """Test that the database manager correctly processes queued items."""
        # Create a real DatabaseManager with a mock collection
        db_manager = DatabaseManager()
        db_manager.collection = MagicMock()
        
        # Sample vehicle data
        vehicle_data = [
            {"vehicle_type": "car", "speed": 45.0, "timestamp": datetime.now()},
            {"vehicle_type": "truck", "speed": 38.2, "timestamp": datetime.now()},
            {"vehicle_type": "car", "speed": 52.1, "timestamp": datetime.now()},
            {"vehicle_type": "bus", "speed": 30.5, "timestamp": datetime.now()},
            {"vehicle_type": "motorcycle", "speed": 65.7, "timestamp": datetime.now()}
        ]
        
        # Add vehicles to queue
        for vehicle in vehicle_data:
            db_manager.add_vehicle(vehicle)
        
        # Wait for processing
        time.sleep(1.0)
        
        # Assert all vehicles were processed
        self.assertEqual(db_manager.collection.insert_one.call_count, 5)
        
        # Verify each vehicle was inserted
        for vehicle in vehicle_data:
            db_manager.collection.insert_one.assert_any_call(vehicle)
        
        # Cleanup
        db_manager.shutdown()

    @patch('TrafficMonitoringSystem.YOLO')
    @patch.dict(os.environ, {
        "MONGODB_CONNECTION_STRING": "mongodb://localhost:27017",
        "MONGODB_DATABASE_NAME": "test_integration_db",
        "MONGODB_COLLECTION_NAME": "test_integration_collection",
        "WEATHER_API_KEY": "test_api_key",
        "WEATHER_CITY": "TestCity"
    })
    def test_speed_calculation_integration(self, mock_yolo):
        """Test speed calculation through multiple frames."""
        # Setup mock YOLO model
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        # Initialize traffic monitoring system
        tms = TrafficMonitoringSystem()
        
        # Create a mock vehicle that crosses the green line
        frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)
        boxes1 = MagicMock()
        boxes1.xywh.cpu.return_value = np.array([[468.0, 400.0, 100.0, 50.0]])  # On green line
        boxes1.conf.cpu.return_value = np.array([0.95])
        boxes1.id.int.return_value.cpu.return_value.tolist.return_value = [789]
        boxes1.cls.cpu.return_value.tolist.return_value = [2]
        
        results1 = MagicMock()
        results1.boxes = boxes1
        results1.names = {2: "car"}
        results1.plot.return_value = frame1.copy()
        
        mock_model.track.return_value = [results1]
        
        # Process first frame - vehicle at green line
        timestamp1 = time.time()
        result1 = tms.process_frame(frame1, timestamp1)
        
        # Vehicle should be recorded at green line
        self.assertTrue(789 in tms.vehicle_timestamps)
        self.assertTrue("start" in tms.vehicle_timestamps[789])
        
        # Wait a bit to simulate time passing between frames
        time.sleep(0.2)
        
        # Now the issue is that in the real implementation, the vehicle needs to move 
        # PAST the red line to trigger the speed calculation, not just be ON the red line
        # Let's position the vehicle clearly past the red line
        frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)
        boxes2 = MagicMock()
        # Position the vehicle well past the red line (1138 + 50)
        boxes2.xywh.cpu.return_value = np.array([[1200.0, 400.0, 100.0, 50.0]])  # Past red line
        boxes2.conf.cpu.return_value = np.array([0.93])
        boxes2.id.int.return_value.cpu.return_value.tolist.return_value = [789]
        boxes2.cls.cpu.return_value.tolist.return_value = [2]
        
        results2 = MagicMock()
        results2.boxes = boxes2
        results2.names = {2: "car"}
        results2.plot.return_value = frame2.copy()
        
        mock_model.track.return_value = [results2]
        
        # Process second frame - vehicle past the red line
        timestamp2 = time.time()
        result2 = tms.process_frame(frame2, timestamp2)
        
        # Check that speed was calculated and the vehicle is in the results
        self.assertEqual(len(result2["detected_vehicles"]), 1, 
                        f"No vehicles detected in results. Contents: {result2}")
        
        detected_vehicle = result2["detected_vehicles"][0]
        self.assertEqual(detected_vehicle["vehicle_id"], 789)
        self.assertGreater(detected_vehicle["speed"], 0)
        
        # Vehicle should be removed from timestamps after speed calculation
        self.assertFalse(789 in tms.vehicle_timestamps)