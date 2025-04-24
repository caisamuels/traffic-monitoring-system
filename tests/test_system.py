import base64
from datetime import datetime
import os
import time
import unittest

import cv2
import numpy as np
from TrafficMonitoringSystem import TrafficMonitoringSystem
from unittest.mock import MagicMock, patch

class TestTrafficMonitoringSystem(unittest.TestCase):
    
    @patch('TrafficMonitoringSystem.YOLO')
    def test_initialization(self, mock_yolo):
        """Test that TrafficMonitoringSystem initializes correctly"""
        # Setup
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        # Execute
        tms = TrafficMonitoringSystem(model_path="test_model.pt")
        
        # Assert
        mock_yolo.assert_called_once_with("test_model.pt")
        self.assertEqual(tms.distance, 17)
        self.assertEqual(tms.green_line_y, 468)
        self.assertEqual(tms.red_line_y, 1138)

    @patch('TrafficMonitoringSystem.YOLO')
    @patch('TrafficMonitoringSystem.requests.get')
    @patch.dict(os.environ, {
        "WEATHER_API_KEY": "test_api_key",
        "WEATHER_CITY": "TestCity"
    })
    def test_get_weather_condition_caching(self, mock_get, mock_yolo):
        """Test weather condition caching"""
        # Setup
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'weather': [{'main': 'Cloudy'}]
        }
        mock_get.return_value = mock_response
        
        tms = TrafficMonitoringSystem()
        
        # Execute - first call
        result1 = tms._get_weather_condition()
        
        # Should use cached version
        result2 = tms._get_weather_condition()
        
        # Assert
        self.assertEqual(result1, 'Cloudy')
        self.assertEqual(result2, 'Cloudy')
        # API should only be called once
        mock_get.assert_called_once()

    @patch('TrafficMonitoringSystem.YOLO')
    @patch('TrafficMonitoringSystem.requests.get')
    @patch.dict(os.environ, {
        "WEATHER_API_KEY": "test_api_key",
        "WEATHER_CITY": "TestCity"
    })
    def test_get_weather_condition_caching(self, mock_get, mock_yolo):
        """Test weather condition caching"""
        # Setup
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'weather': [{'main': 'Cloudy'}]
        }
        mock_get.return_value = mock_response
        
        tms = TrafficMonitoringSystem()
        
        # Execute - first call
        result1 = tms._get_weather_condition()
        
        # Should use cached version
        result2 = tms._get_weather_condition()
        
        # Assert
        self.assertEqual(result1, 'Cloudy')
        self.assertEqual(result2, 'Cloudy')
        # API should only be called once
        mock_get.assert_called_once()
    
    @patch('TrafficMonitoringSystem.YOLO')
    @patch('TrafficMonitoringSystem.requests.get')
    @patch.dict(os.environ, {
        "WEATHER_API_KEY": "test_api_key",
        "WEATHER_CITY": "TestCity"
    })
    def test_get_weather_condition_api_error(self, mock_get, mock_yolo):
        """Test handling of API errors"""
        # Setup
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            'message': 'Invalid API key'
        }
        mock_get.return_value = mock_response
        
        tms = TrafficMonitoringSystem()
        
        # Execute
        result = tms._get_weather_condition()
        
        # Assert
        self.assertEqual(result, 'API Error: Invalid API key')
    
    @patch('TrafficMonitoringSystem.YOLO')
    def test_convert_image_to_base64(self, mock_yolo):
        """Test base64 encoding of image"""
        # Setup
        tms = TrafficMonitoringSystem()
        
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[30:70, 30:70] = [255, 255, 255]  # White square
        
        # Execute
        base64_string = tms._convert_image_to_base64(test_image)
        
        # Assert
        self.assertIsInstance(base64_string, str)
        # Verify we can decode it
        decoded_data = base64.b64decode(base64_string)
        self.assertIsInstance(decoded_data, bytes)
    
    @patch('TrafficMonitoringSystem.YOLO')
    def test_convert_base64_to_image(self, mock_yolo):
        """Test base64 decoding to image"""
        # Setup
        tms = TrafficMonitoringSystem()
        
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[30:70, 30:70] = [255, 255, 255]  # White square
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', test_image)
        base64_string = base64.b64encode(buffer).decode()
        
        # Execute
        decoded_image = tms._convert_base64_to_image(base64_string)
        
        # Assert
        self.assertIsInstance(decoded_image, np.ndarray)
        self.assertEqual(decoded_image.shape[0], 100)  # Height
        self.assertEqual(decoded_image.shape[1], 100)  # Width
    
    @patch('TrafficMonitoringSystem.YOLO')
    def test_convert_base64_to_image_invalid_input(self, mock_yolo):
        """Test handling of invalid base64 input"""
        # Setup
        tms = TrafficMonitoringSystem()
        
        # Execute with invalid base64
        result = tms._convert_base64_to_image("invalid_base64_data")
        
        # Assert
        self.assertIsNone(result)
    
    @patch('TrafficMonitoringSystem.YOLO')
    def test_calculate_speed_normal(self, mock_yolo):
        """Test speed calculation with normal inputs"""
        # Setup
        tms = TrafficMonitoringSystem()
        start_time = 100  # seconds
        end_time = 101.5  # seconds (1.5 seconds later)
        
        # Execute
        speed = tms._calculate_speed(start_time, end_time)
        
        # Assert - expected speed: 17m / 1.5s = 11.33 m/s = 25.36 mph
        self.assertAlmostEqual(speed, 25.4, places=1)
    
    @patch('TrafficMonitoringSystem.YOLO')
    def test_calculate_speed_edge_case(self, mock_yolo):
        """Test speed calculation with zero or negative time difference"""
        # Setup
        tms = TrafficMonitoringSystem()
        
        # Execute with zero time difference
        speed1 = tms._calculate_speed(100, 100)
        
        # Execute with negative time difference (shouldn't happen in real system)
        speed2 = tms._calculate_speed(100, 99)
        
        # Assert
        self.assertEqual(speed1, 0)
        self.assertEqual(speed2, 0)
    
    @patch('TrafficMonitoringSystem.YOLO')
    @patch('TrafficMonitoringSystem.TrafficMonitoringSystem._get_weather_condition')
    def test_process_frame_no_vehicles(self, mock_get_weather, mock_yolo):
        """Test frame processing with no vehicles"""
        # Setup
        mock_get_weather.return_value = "Clear"
        
        # Mock the YOLO model
        mock_model = MagicMock()
        mock_results = [MagicMock()]
        mock_results[0].boxes = MagicMock()
        mock_results[0].boxes.id = None  # No tracked objects
        mock_model.track.return_value = mock_results
        
        mock_yolo.return_value = mock_model
        
        tms = TrafficMonitoringSystem()
        
        # Create a test frame
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Execute
        result = tms.process_frame(test_frame, datetime.now())
        
        # Assert
        self.assertEqual(result["number_of_vehicles_detected"], 0)
        self.assertEqual(len(result["detected_vehicles"]), 0)
        self.assertEqual(result["weather_condition"], "Clear")
        self.assertIsNotNone(result["original_frame_base64"])
    
    @patch('TrafficMonitoringSystem.YOLO')
    @patch('TrafficMonitoringSystem.TrafficMonitoringSystem._get_weather_condition')
    def test_process_frame_with_vehicles(self, mock_get_weather, mock_yolo):
        """Test frame processing with vehicles"""
        # Setup
        mock_get_weather.return_value = "Rain"
        
        # Create a complex mock for YOLO results
        mock_model = MagicMock()
        
        # Create boxes mock
        boxes_mock = MagicMock()
        boxes_mock.xywh.cpu.return_value = torch.tensor([[500.0, 400.0, 100.0, 50.0]])  # x, y, width, height
        boxes_mock.conf.cpu.return_value = torch.tensor([0.95])  # confidence
        boxes_mock.id.int.return_value.cpu.return_value.tolist.return_value = [123]  # track ID
        boxes_mock.cls.cpu.return_value.tolist.return_value = [2]  # class (car)
        
        # Create results mock
        results_mock = MagicMock()
        results_mock.boxes = boxes_mock
        results_mock.names = {2: "car"}
        results_mock.plot.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)  # Annotated frame
        
        # Set up track method to return our mocked results
        mock_model.track.return_value = [results_mock]
        mock_yolo.return_value = mock_model
        
        # Create TMS instance
        tms = TrafficMonitoringSystem()
        
        # Create test frame
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Set up vehicle timestamps to simulate crossing first and second lines
        tms.vehicle_timestamps = {
            123: {
                "start": time.time() - 2.0,  # Crossed first line 2 seconds ago
                "end": None
            }
        }
        
        # Execute
        result = tms.process_frame(test_frame, datetime.now())
        
        # Assert
        mock_model.track.assert_called_once()
        self.assertGreaterEqual(len(result["detected_vehicles"]), 0)
        self.assertIsNotNone(result["annotated_frame_base64"])
        self.assertEqual(result["weather_condition"], "Rain")
    
    # @patch('TrafficMonitoringSystem.YOLO')
    # @patch('TrafficMonitoringSystem.cv2.VideoCapture')
    # def test_process_video(self, mock_video_capture, mock_yolo):
    #     """Test video processing"""
    #     # Setup
    #     mock_model = MagicMock()
    #     mock_yolo.return_value = mock_model
        
    #     # Mock video capture
    #     mock_cap = MagicMock()
    #     mock_cap.isOpened.side_effect = [True, True, False]  # Return True twice, then False to end loop
    #     mock_cap.read.side_effect = [
    #         (True, np.zeros((720, 1280, 3), dtype=np.uint8)),  # First frame
    #         (True, np.zeros((720, 1280, 3), dtype=np.uint8)),  # Second frame
    #     ]
    #     mock_video_capture.return_value = mock_cap
        
    #     # Mock waitKey to immediately exit loop
    #     with patch('TrafficMonitoringSystem.cv2.waitKey', return_value=ord('q')):
    #         tms = TrafficMonitoringSystem()
            
    #         # Create a mock callback
    #         mock_callback = MagicMock()
            
    #         # Execute
    #         tms.process_video("test_video.mp4", mock_callback)
            
    #         # Assert
    #         mock_video_capture.assert_called_once_with("test_video.mp4")
    #         self.assertEqual(mock_cap.read.call_count, 1)  # Should have read one frame
    #         mock_callback.assert_called_once()  # Callback should have been called once
    #         mock_cap.release.assert_called_once()  # Should release the video capture


# Additional imports for testing with PyTorch tensors
import torch

if __name__ == '__main__':
    unittest.main()