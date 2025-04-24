from datetime import datetime, time
from TrafficMonitoringSystem import TrafficMonitoringSystem
from DatabaseManager import DatabaseManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

video_path = "rtsp://150.204.195.58:8554/cam"

end_time = time(17, 00)  # The time for the script to terminate

# Initialize database manager (uses environment variables)
db_manager = DatabaseManager()

vehicle_detection = TrafficMonitoringSystem()

# For tracking stats
session_stats = {
    "total_vehicles": 0,
    "vehicles_by_type": {},
    "start_time": datetime.now(),
    "last_weather": "Unknown"
}

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
        
        # Add vehicle to database queue
        db_manager.add_vehicle(vehicle_doc)
        
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


if __name__ == "__main__":
    try:
        # Start processing the video
        vehicle_detection.process_video(video_path, result_callback)
    finally:
        # Shutdown the database manager
        db_manager.shutdown()