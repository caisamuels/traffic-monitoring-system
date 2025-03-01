from datetime import datetime
from pymongo import MongoClient
from TrafficMonitoringSystem import TrafficMonitoringSystem

video_path = "rtsp://150.204.195.58:8554/cam"

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

def result_callback(result):
    # Update weather status for summary display
    session_stats["last_weather"] = result["weather_condition"]
    
    # Save each vehicle to MongoDB
    for vehicle in result['detected_vehicles']:
        # Create a document to store in MongoDB
        vehicle_doc = {
            "vehicle_id": vehicle["vehicle_id"],
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
        
        # Insert the vehicle into MongoDB
        collection.insert_one(vehicle_doc)
        
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

# Start processing the video
vehicle_detection.process_video(video_path, result_callback)