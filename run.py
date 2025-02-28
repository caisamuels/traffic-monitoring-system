from pymongo import MongoClient
from TrafficMonitoringSystem import TrafficMonitoringSystem

video_path = "rtsp://150.204.195.58:8554/cam"

connection_string="mongodb://localhost:27017/"
database_name="trafficMonitoring"

client = MongoClient(connection_string)
db = client[database_name]
collection = db["vehicles"]

vehicle_detection = TrafficMonitoringSystem()
def result_callback(result):
    # Print the detection results
    print({
        "number_of_vehicles_detected": result["number_of_vehicles_detected"],
        "detected_vehicles": [
            {
                "vehicle_id": vehicle["vehicle_id"],
                "vehicle_type": vehicle["vehicle_type"],
                "detection_confidence": vehicle["detection_confidence"],
                "timestamp": vehicle["timestamp"],
                "speed": vehicle["speed"]
            }
            for vehicle in result['detected_vehicles']
        ]
    })
    
    # Save each vehicle to MongoDB
    for vehicle in result['detected_vehicles']:
        # Create a document to store in MongoDB
        vehicle_doc = {
            "vehicle_id": vehicle["vehicle_id"],
            "vehicle_type": vehicle["vehicle_type"],
            "detection_confidence": vehicle["detection_confidence"],
            "timestamp": vehicle["timestamp"],
            "speed": vehicle["speed"]
        }
        
        # Insert the vehicle into MongoDB
        collection.insert_one(vehicle_doc)
        print("Written vehicle ", vehicle["vehicle_id"], "to database!")

# Start processing the video
vehicle_detection.process_video(video_path, result_callback)