"""
Configuration Module for Real-Time Traffic Analytics
=====================================================
Contains all configurable parameters for the traffic analytics system.
Modify these values based on your specific video and road geometry.
"""

import numpy as np

# =============================================================================
# VIDEO CONFIGURATION
# =============================================================================
# Sample highway dashcam video URL (public domain)
# You can replace this with your own video URL or local path
SAMPLE_VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
VIDEO_PATH = "highway.mp4"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Using YOLOv11 (latest as of 2025) - Medium version for balance of speed/accuracy
# Options: yolo11n.pt (nano), yolo11s.pt (small), yolo11m.pt (medium), 
#          yolo11l.pt (large), yolo11x.pt (extra-large)
MODEL_NAME = "yolo11m.pt"

# Vehicle classes to detect (COCO dataset class IDs)
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.3

# =============================================================================
# PERSPECTIVE TRANSFORM CONFIGURATION (THE MATHEMATICAL CORE)
# =============================================================================
"""
HOMOGRAPHY EXPLAINED:
====================
Homography is a 3x3 transformation matrix that maps points from one plane to another.
In our case, we map points from the IMAGE PLANE (2D pixels) to the GROUND PLANE 
(real-world coordinates in meters).

Why do we need this?
- In a camera image, objects appear smaller as they get farther away (perspective)
- A car moving 10 pixels near the camera travels less real distance than 
  a car moving 10 pixels far from the camera
- Homography corrects for this perspective distortion

The 4 source points define a quadrilateral on the road (in pixels)
The 4 destination points define what that quadrilateral represents in real-world meters
"""

# SOURCE_POLYGON: 4 points defining the detection zone on the road (in pixels)
# These points should form a trapezoid that represents a lane segment
# Format: [top-left, top-right, bottom-right, bottom-left]
# NOTE: You MUST adjust these values based on your specific video!
SOURCE_POLYGON = np.array([
    [400, 300],   # Top-left corner of the zone
    [880, 300],   # Top-right corner of the zone  
    [1100, 600],  # Bottom-right corner of the zone
    [180, 600],   # Bottom-left corner of the zone
], dtype=np.float32)

# TARGET_RECTANGLE: Real-world dimensions that SOURCE_POLYGON maps to (in meters)
# This represents a rectangular section of road in bird's-eye view
# Width: typical lane width ~3.7 meters, Length: 20 meters for speed calculation
TARGET_WIDTH_METERS = 3.7   # Lane width in meters
TARGET_LENGTH_METERS = 20.0  # Length of the zone for speed calculation

TARGET_RECTANGLE = np.array([
    [0, 0],                              # Top-left
    [TARGET_WIDTH_METERS, 0],            # Top-right
    [TARGET_WIDTH_METERS, TARGET_LENGTH_METERS],  # Bottom-right
    [0, TARGET_LENGTH_METERS],           # Bottom-left
], dtype=np.float32)

# =============================================================================
# SPEED CALCULATION CONFIGURATION
# =============================================================================
# Speed threshold for "SPEEDING" violation (km/h)
SPEED_LIMIT_KMH = 100.0

# Minimum number of frames a vehicle must be tracked before calculating speed
# This helps filter out noisy detections
MIN_TRACKING_FRAMES = 5

# Speed smoothing - number of recent speeds to average
SPEED_SMOOTHING_WINDOW = 3

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================
# Colors (BGR format for OpenCV)
COLOR_NORMAL = (0, 255, 0)      # Green for normal vehicles
COLOR_SPEEDING = (0, 0, 255)    # Red for speeding vehicles
COLOR_ZONE = (255, 255, 0)      # Cyan for detection zone
COLOR_TEXT = (255, 255, 255)    # White for text

# Text settings
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Zone opacity (0.0 to 1.0)
ZONE_OPACITY = 0.3

# Dashboard position
DASHBOARD_POSITION = (10, 30)

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
OUTPUT_VIDEO_PATH = "output_traffic_analytics.mp4"
OUTPUT_FPS = None  # None means use source video FPS
