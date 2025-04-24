# Traffic Monitoring System

This project implements a real-time traffic monitoring system using computer vision. It analyzes a video stream (e.g., from an RTSP camera) to detect vehicles, track them, estimate their speed, and log the data along with weather conditions into a MongoDB database.

The data collected by this system and stored in MongoDB can be visualized using the [Traffic Monitoring Dashboard](https://github.com/caisamuels/traffic-monitoring-dashboard).

## Features

* **Vehicle Detection and Tracking**: Utilizes the YOLO (You Only Look Once) object detection model (specifically configured for vehicles like cars, trucks, buses, etc.) to identify and track vehicles in the video feed.
* **Speed Estimation**: Calculates the approximate speed of detected vehicles based on the time they take to cross predefined lines in the camera's view.
* **Weather Integration**: Fetches current weather conditions using the OpenWeatherMap API and logs it alongside vehicle data.
* **Asynchronous Database Logging**: Stores detected vehicle information (type, speed, timestamp, detection confidence) and weather conditions into a MongoDB database using a non-blocking, threaded approach.
* **Real-time Statistics**: Displays running statistics in the console, including total vehicles detected, a breakdown by vehicle type, current weather, and details of the last detected vehicle.
* **Configuration via Environment Variables**: Easily configure database connection details and API keys using a `.env` file.
* **Scheduled Termination**: The main script can be set to stop processing at a specific time.
* **Testing**: Includes unit and integration tests using Python's `unittest` framework to ensure component reliability.

## Core Components

1.  **`TrafficMonitoringSystem.py`**: Handles video processing, object detection/tracking (using Ultralytics YOLO), speed calculation, and weather data fetching.
2.  **`DatabaseManager.py`**: Manages asynchronous communication with the MongoDB database for data storage.
3.  **`run.py`**: The main executable script that initializes the system, processes the video feed, manages statistics, and orchestrates data logging.
4.  **`tests/`**: Contains `unittest` unit and integration tests for the system components.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd traffic-monitoring-system-main
    ```
2.  **Install dependencies:** (Assuming you have Python and pip installed)
    ```bash
    pip install -r requirements.txt
    ```
3.  **YOLO Model:** Download the YOLO model weights file (e.g., `yolo11x.pt` as mentioned in `TrafficMonitoringSystem.py`) and place it in the project directory or provide the correct path.
4.  **Environment Variables:** Create a `.env` file in the root directory with the following information:
    ```dotenv
    # MongoDB Configuration
    MONGODB_CONNECTION_STRING=mongodb://your_mongo_host:port
    MONGODB_DATABASE_NAME=traffic_db
    MONGODB_COLLECTION_NAME=vehicle_data

    # Weather API Configuration
    WEATHER_API_KEY=your_openweathermap_api_key
    WEATHER_CITY=YourCity # e.g., Liverpool
    ```
    * Replace placeholders with your actual MongoDB connection string, desired database/collection names, OpenWeatherMap API key, and the target city for weather data.

## Usage

1.  **Configure Video Source:** Update the `video_path` variable in `run.py` to your video source (e.g., RTSP URL, local video file path).
    ```python
    # In run.py
    video_path = "rtsp://your_camera_ip:port/stream" # Or "path/to/your/video.mp4"
    ```
2.  **Configure End Time (Optional):** Adjust the `end_time` in `run.py` if you want the script to terminate automatically at a specific time.
    ```python
    # In run.py
    end_time = time(17, 00)  # Stop at 5:00 PM
    ```
3.  **Run the system:**
    ```bash
    python run.py
    ```
    * The system will start processing the video feed.
    * An OpenCV window will display the video with annotations (bounding boxes, tracking IDs, speed estimation lines).
    * Real-time statistics will be printed to the console.
    * Detected vehicle data will be logged to your MongoDB database.
    * Press 'q' in the OpenCV window to stop the process manually.

## Testing

The project uses Python's built-in `unittest` framework for tests. To run the tests, navigate to the project's root directory and use the following command:

```bash
python -m unittest discover tests