import os
from queue import Queue
import time
import unittest
from DatabaseManager import DatabaseManager
from unittest.mock import MagicMock, patch

class TestDatabaseManager(unittest.TestCase):

    @patch.dict(os.environ, {
        "MONGODB_CONNECTION_STRING": "mongodb://localhost:27017",
        "MONGODB_DATABASE_NAME": "test_db",
        "MONGODB_COLLECTION_NAME": "test_collection"
    })
    @patch('DatabaseManager.MongoClient')
    def test_initialization(self, mock_mongo_client):
        """Test that DatabaseManager initializes correctly with environment variables"""
        # Setup mock
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        mock_mongo_client.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        
        # Execute
        db_manager = DatabaseManager()
        
        # Assert
        mock_mongo_client.assert_called_once_with("mongodb://localhost:27017")
        mock_client.__getitem__.assert_called_once_with("test_db")
        mock_db.__getitem__.assert_called_once_with("test_collection")
        
        # Cleanup
        db_manager.shutdown()

    @patch.dict(os.environ, {}, clear=True) 
    @patch('DatabaseManager.MongoClient')
    def test_initialization_missing_env_vars(self, mock_mongo_client):
        """Test handling of missing environment variables"""
        # This should raise an error or use defaults
        with self.assertRaises(ValueError):
            db_manager = DatabaseManager()

    @patch.dict(os.environ, {
    "MONGODB_CONNECTION_STRING": "mongodb://localhost:27017",
    "MONGODB_DATABASE_NAME": "test_db",
    "MONGODB_COLLECTION_NAME": "test_collection"
    })
    @patch('DatabaseManager.MongoClient')
    def test_add_vehicle(self, mock_mongo_client):
        """Test that add_vehicle adds to the queue"""
        # Setup
        db_manager = DatabaseManager()
        db_manager.db_queue = MagicMock(spec=Queue)
        
        vehicle_data = {"type": "car", "speed": 45.5}
        
        # Execute
        db_manager.add_vehicle(vehicle_data)
        
        # Assert
        db_manager.db_queue.put.assert_called_once_with(vehicle_data)
        
        # Cleanup
        db_manager.shutdown()

    @patch.dict(os.environ, {
        "MONGODB_CONNECTION_STRING": "mongodb://localhost:27017",
        "MONGODB_DATABASE_NAME": "test_db",
        "MONGODB_COLLECTION_NAME": "test_collection"
    })
    @patch('DatabaseManager.MongoClient')
    def test_db_worker(self, mock_mongo_client):
        """Test that the worker processes items from the queue"""
        # Setup mocks
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        mock_mongo_client.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        
        # Setup test class with controlled queue
        db_manager = DatabaseManager()
        test_queue = Queue()
        db_manager.db_queue = test_queue
        
        # Add test data to queue
        vehicle_data = {"type": "car", "speed": 45.5}
        test_queue.put(vehicle_data)
        
        # Let the worker run for a short time
        time.sleep(2)
        
        # Signal the worker to terminate
        db_manager.terminate_worker = True
        
        # Wait for the worker to finish
        db_manager.db_thread.join(timeout=1.0)
        
        # Assert
        mock_collection.insert_one.assert_called_once_with(vehicle_data)

if __name__ == '__main__':
    unittest.main()