import base64
from collections import defaultdict
from datetime import datetime
import time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

class TrafficMonitoringSystem:

    def __init__(self, model_path="yolo11x.pt"):
        """
        Initialize the TrafficMonitoringSystem class.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        self.model = YOLO(model_path)
        self.model.to('cuda')
        print(self.model.device)
        self.detected_vehicles = set()  # Set of detected vehicles
        self.track_history = {}  # History of vehicle tracking
        self.vehicle_timestamps = {}  # Keep track of timestamps for each tracked vehicle
        self.distance = 17 # Distance between lines
        self.green_line_y = 480 # First line where speed tracking starts
        self.red_line_y = 1145 # Second line where speed tracking ends

    def _convert_image_to_base64(self, image):
        """
        Converts an image to a base64-encoded string.
 
        Args:
            image (numpy.ndarray): The image to be encoded.

        Returns:
            str: The base64-encoded representation of the image.
        """
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode()
    
    def _convert_base64_to_image(self, image_base64):
        """
        Decodes a base64-encoded string into an image.

        Args:
            image_base64 (str): The base64-encoded image data.

        Returns:
            numpy.ndarray or None: The decoded image as a NumPy array, or None if decoding fails.
        """
        try:
            image_data = base64.b64decode(image_base64)
            image_np = np.frombuffer(image_data, dtype=np.uint8)
            return cv2.imdecode(image_np, flags=cv2.IMREAD_COLOR)
        except Exception:
            return None
        
    def _calculate_speed(self, start_time, end_time):
        """
        Calculate the speed of a vehicle in MPH.
        
        Args:
            start_time (datetime): Time when vehicle crossed the first line.
            end_time (datetime): Time when vehicle crossed the second line.
            
        Returns:
            float: Speed in MPH.
        """
        # Calculate time difference in seconds
        time_diff = (end_time - start_time)
        
        if time_diff <= 0:
            return 0
            
        # Calculate speed in meters per second
        speed_mps = self.distance / time_diff
        
        # Convert to MPH (1 m/s = 2.23694 mph)
        speed_mph = speed_mps * 2.23694
    
        return round(speed_mph, 1)
    
    def process_frame(self, frame, frame_timestamp):
        """
        Processes a single video frame to detect and track vehicles.

        Args:
            frame (numpy.ndarray): The input frame to be processed.
            frame_timestamp (float): The timestamp of the frame.

        Returns:
            dict: A dictionary containing details of detected vehicles, the annotated frame in base64, and the original frame in base64.
        """
        response = {
            "number_of_vehicles_detected": 0,  # Count of detected vehicles in this frame
            "detected_vehicles": [],  # List of detected vehicle details
            "annotated_frame_base64": None,  # Base64-encoded annotated frame
            "original_frame_base64": None  # Base64-encoded original frame
        }

        # Draw start speed estimation line on screen
        cv2.line(frame,(self.green_line_y,380), (620,710),(0, 255, 0),6)
        # Draw end speed estiamtion line on screen
        cv2.line(frame,(self.red_line_y,370), (1435,700),(0, 0, 255),6)

        current_time = time.time()

        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[2, 3, 5, 7], verbose=False) # Perform vehicle tracking in the frame
        if results is not None and results[0] is not None and results[0].boxes is not None and results[0].boxes.id is not None:
            # Obtain bounding boxes (xywh format) of detected objects
            boxes = results[0].boxes.xywh.cpu()
            # Extract confidence scores for each detected object
            conf_list = results[0].boxes.conf.cpu()
            # Get unique IDs assigned to each tracked object
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # Obtain the class labels (e.g., 'car', 'truck') for detected objects
            clss = results[0].boxes.cls.cpu().tolist()
            # Retrieve the names of the detected objects based on class labels
            names = results[0].names
            # Get the annotated frame using results[0].plot() and encode it as base64
            annotated_frame = results[0].plot()

            for box, track_id, cls, conf in zip(boxes, track_ids, clss, conf_list):
                x, y, w, h = box
                label = str(names[cls])

                if track_id not in self.vehicle_timestamps:
                    self.vehicle_timestamps[track_id] = {}

                if (self.green_line_y - 50 <= float(x) <= self.green_line_y + 50) and "start" not in self.vehicle_timestamps[track_id]:
                    self.vehicle_timestamps[track_id]["start"] = current_time

                if float(x) > self.red_line_y and "end" not in self.vehicle_timestamps[track_id] and "start" in self.vehicle_timestamps[track_id]:
                    self.vehicle_timestamps[track_id]["end"] = current_time
                    start_time = self.vehicle_timestamps[track_id]["start"]
                    end_time = self.vehicle_timestamps[track_id]["end"]
                    speed = self._calculate_speed(start_time, end_time)

                
                    response["detected_vehicles"].append({
                        "vehicle_id": track_id,
                        "vehicle_type": label,
                        "detection_confidence": conf.item(),
                        "timestamp": frame_timestamp,
                        "speed": speed
                    })

                    # Remove the vehicle from the dictionary after speed calculation.
                    del self.vehicle_timestamps[track_id]

            annotated_frame_base64 = self._convert_image_to_base64(annotated_frame)
            response["annotated_frame_base64"] = annotated_frame_base64

        # Encode the original frame as base64
        original_frame_base64 = self._convert_image_to_base64(frame)
        response["original_frame_base64"] = original_frame_base64

        return response
    
    def process_video(self, video_path, result_callback):
        """
        Analyzes a video by processing each frame and invoking a callback with the results.

        Args:
            video_path (str): The file path of the video to be processed.
            result_callback (function): A callback function to handle the results for each processed frame.
        """
        # Process a video frame by frame, calling a callback with the results.
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
                # print(f"Frame rate: {frame_rate} FPS")
                timestamp = datetime.now()
                response = self.process_frame(frame, timestamp)
                if 'annotated_frame_base64' in response:
                    annotated_frame = self._convert_base64_to_image(response['annotated_frame_base64'])
                    if annotated_frame is not None:
                        # Display the annotated frame in a window
                        cv2.imshow("Traffic Monitoring System", annotated_frame)
                    else:
                        # Also display original frame if there are no annotations
                        cv2.imshow("Traffic Monitoring System", frame)
                # Call the callback with the response
                result_callback(response)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()