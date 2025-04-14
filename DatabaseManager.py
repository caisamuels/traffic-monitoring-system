from pymongo import MongoClient
import threading
from queue import Queue, Empty
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseManager:
    """
    Manages database operations for the traffic monitoring system using a separate worker thread.
    """
    
    def __init__(self):
        """
        Initialize the DatabaseManager using only environment variables.
        """
        # Get configuration from environment variables
        self.connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("MONGODB_CONNECTION_STRING environment variable is required")
        self.database_name = os.getenv("MONGODB_DATABASE_NAME")
        if not self.database_name:
            raise ValueError("MONGODB_DATABASE_NAME environment variable is required")
        self.collection_name = os.getenv("MONGODB_COLLECTION_NAME")
        if not self.collection_name:
            raise ValueError("MONGODB_COLLECTION_NAME environment variable is required")
        
        # MongoDB connection
        self.client = MongoClient(self.connection_string)
        self.db = self.client[self.database_name]
        self.collection = self.db[self.collection_name]
        
        # Queue for database operations
        self.db_queue = Queue()
        
        # Flag to signal the worker thread to terminate
        self.terminate_worker = False
        
        # Start the worker thread
        self.db_thread = threading.Thread(target=self._db_worker, daemon=True)
        self.db_thread.start()
    
    def _db_worker(self):
        """Worker thread function that processes database operations from the queue."""
        while not self.terminate_worker:
            try:
                # Get the next vehicle document from the queue with a timeout
                # This allows the thread to check the terminate flag periodically
                vehicle_doc = self.db_queue.get(timeout=1.0)
                
                # Insert the vehicle into MongoDB
                self.collection.insert_one(vehicle_doc)
                
                # Mark the task as done
                self.db_queue.task_done()
            except Empty:
                # Timeout occurred, just continue to check the terminate flag
                continue
    
    def add_vehicle(self, vehicle_data):
        """
        Add a vehicle document to the queue for database insertion.
        
        Args:
            vehicle_data (dict): Vehicle data to be stored
        """
        # Put the vehicle document in the queue for the worker thread to process
        self.db_queue.put(vehicle_data)
    
    def shutdown(self, timeout=5.0):
        """
        Shutdown the database worker thread gracefully.
        
        Args:
            timeout (float): Maximum time to wait for thread termination
        """
        # Signal the worker thread to terminate
        self.terminate_worker = True
        
        # Wait for the worker thread to finish
        self.db_thread.join(timeout=timeout)
        
        # Wait for any remaining database operations to complete
        self.db_queue.join()

        # Close the MongoDB client connection
        self.client.close()