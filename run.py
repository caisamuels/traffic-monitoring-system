from datetime import datetime, time
from pymongo import MongoClient
import threading
from queue import Queue, Empty
from TrafficMonitoringSystem import TrafficMonitoringSystem

video_path = "rtsp://127.0.0.1:8554/"

end_time = time(19, 00) # The time for the script to terminate

connection_string="mongodb://localhost:27017/"
database_name="trafficMonitoring"

client = MongoClient(connection_string)
db = client[database_name]
collection = db["vehicles"]

vehicle_detection = TrafficMonitoringSystem()

# For tracking stats
session_stats = {
    "total_vehicles": 0,
    "vehicles_by_type": {},
    "start_time": datetime.now(),
    "last_weather": "Unknown"
}

# Create a queue for database operations
db_queue = Queue()

# Flag to signal the worker thread to terminate
terminate_worker = False

def db_worker():
    """Worker thread function that processes database operations from the queue."""
    while not terminate_worker:
        try:
            # Get the next vehicle document from the queue with a timeout
            # This allows the thread to check the terminate flag periodically
            vehicle_doc = db_queue.get(timeout=1.0)
            
            # Insert the vehicle into MongoDB
            collection.insert_one(vehicle_doc)
            
            # Mark the task as done
            db_queue.task_done()
        except Empty:
            # Timeout occurred, just continue to check the terminate flag
            continue

# Start the worker thread
db_thread = threading.Thread(target=db_worker, daemon=True)
db_thread.start()

def result_callback(result):
    current_time = datetime.now().time()
    # Terminate script at specified time
    if current_time >= end_time:
        print("\n" + "=" * 50)
        print("  SCHEDULED TERMINATION TIME REACHED")
        print("=" * 50)
        exit(0)  # Terminate the script
    # Update weather status for summary display
    session_stats["last_weather"] = result["weather_condition"]
    
    # Process each detected vehicle
    for vehicle in result['detected_vehicles']:
        # Create a document to store in MongoDB
        vehicle_doc = {
            "vehicle_type": vehicle["vehicle_type"],
            "detection_confidence": vehicle["detection_confidence"],
            "timestamp": vehicle["timestamp"],
            "speed": vehicle["speed"],
            "weather_condition": result["weather_condition"]
        }
        
        # Update session statistics
        session_stats["total_vehicles"] += 1
        vehicle_type = vehicle["vehicle_type"]
        if vehicle_type in session_stats["vehicles_by_type"]:
            session_stats["vehicles_by_type"][vehicle_type] += 1
        else:
            session_stats["vehicles_by_type"][vehicle_type] = 1
        
        # Put the vehicle document in the queue for the worker thread to process
        db_queue.put(vehicle_doc)
        
        # Clear the console (works on most terminals)
        print("\033c", end="")
        
        # Display a clean summary
        runtime = datetime.now() - session_stats["start_time"]
        runtime_str = str(runtime).split('.')[0]  # Remove microseconds
        
        print("=" * 50)
        print(f"  TRAFFIC MONITORING SYSTEM - RUNTIME: {runtime_str}")
        print("=" * 50)
        print(f"  Weather: {session_stats['last_weather']}")
        print(f"  Total vehicles detected: {session_stats['total_vehicles']}")
        print("-" * 50)
        print("  VEHICLE BREAKDOWN:")
        for vtype, count in sorted(session_stats["vehicles_by_type"].items()):
            percentage = (count / session_stats["total_vehicles"]) * 100
            print(f"  • {vtype}: {count} ({percentage:.1f}%)")
        print("-" * 50)
        print(f"  Last vehicle: #{vehicle['vehicle_id']} ({vehicle['vehicle_type']})")
        print(f"  Speed: {vehicle['speed']} MPH")
        print(f"  Confidence: {vehicle['detection_confidence']:.2f}")
        print("=" * 50)

try:
    # Start processing the video
    vehicle_detection.process_video(video_path, result_callback)
finally:
    # Signal the worker thread to terminate and wait for it to finish
    terminate_worker = True
    db_thread.join(timeout=5.0)
    
    # Wait for any remaining database operations to complete
    db_queue.join()