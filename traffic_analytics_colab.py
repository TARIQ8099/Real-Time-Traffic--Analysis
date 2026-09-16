# ============================================================================
# REAL-TIME TRAFFIC ANALYTICS - GOOGLE COLAB NOTEBOOK
# ============================================================================
# Copy each section below into separate Colab cells
# ============================================================================

# =============================================================================
# CELL 1: INSTALLATION & GPU SETUP
# =============================================================================
# !pip install ultralytics supervision --quiet

# Check GPU availability (run this in Colab)
# import torch
# print(f"CUDA Available: {torch.cuda.is_available()}")
# print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
# NOTE: In Colab, go to Runtime > Change runtime type > GPU (T4) for free GPU

# =============================================================================
# CELL 2: IMPORTS
# =============================================================================
import cv2
import numpy as np
from collections import defaultdict
from google.colab import files
from IPython.display import HTML, display
from base64 import b64encode

# =============================================================================
# CELL 3: DOWNLOAD SAMPLE VIDEO OR UPLOAD YOUR OWN
# =============================================================================
# RECOMMENDED VIDEO TYPES FOR SPEED ESTIMATION:
# - Elevated/overpass view looking down at highway
# - Fixed traffic camera (not moving dashcam)
# - Straight road section (not curved)
# - Clear lane markings visible

# Option A: Roboflow vehicles sample (RECOMMENDED - works great!)
# !wget -q "https://media.roboflow.com/supervision/video-examples/vehicles.mp4" -O highway.mp4

# Option B: Pexels highway footage (download manually from pexels.com)
# Search: "highway traffic aerial" at https://www.pexels.com/search/videos/

# Option C: Upload your own video
# from google.colab import files
# uploaded = files.upload()
# VIDEO_PATH = list(uploaded.keys())[0]

# WHERE TO FIND GOOD VIDEOS:
# 1. Pexels: https://www.pexels.com/search/videos/highway%20traffic/
# 2. Pixabay: https://pixabay.com/videos/search/highway/
# 3. Search YouTube: "traffic camera footage raw" or "highway CCTV"

VIDEO_PATH = "highway.mp4"

# =============================================================================
# CELL 4: CONFIGURATION
# =============================================================================
# Model settings
MODEL_NAME = "yolo11m.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CONFIDENCE = 0.3

# Speed settings
SPEED_LIMIT_KMH = 100.0
MIN_TRACK_FRAMES = 5

# Zone points (ADJUST FOR YOUR VIDEO!)
# Format: [top-left, top-right, bottom-right, bottom-left]
SOURCE_POLYGON = np.array([
    [400, 300], [880, 300], [1100, 600], [180, 600]
], dtype=np.float32)

# Real-world size: 3.7m wide x 20m long
TARGET_RECT = np.array([
    [0, 0], [3.7, 0], [3.7, 20], [0, 20]
], dtype=np.float32)

# Colors (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
CYAN = (255, 255, 0)

# =============================================================================
# CELL 5: HOMOGRAPHY MATH (THE CORE!)
# =============================================================================
"""
HOMOGRAPHY EXPLAINED:
====================
Maps 2D image pixels to real-world meters using a 3x3 matrix.
This corrects perspective distortion - objects far away look smaller
but we need their REAL distance traveled to calculate speed.
"""

# Compute homography matrix
HOMOGRAPHY = cv2.getPerspectiveTransform(SOURCE_POLYGON, TARGET_RECT)

def transform_point(point):
    """Transform image point (pixels) to real-world (meters)."""
    pt = np.array([[point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, HOMOGRAPHY)
    return (transformed[0][0][0], transformed[0][0][1])

def get_real_distance(p1, p2):
    """Calculate real-world distance between two image points."""
    r1 = transform_point(p1)
    r2 = transform_point(p2)
    return np.sqrt((r2[0]-r1[0])**2 + (r2[1]-r1[1])**2)

# =============================================================================
# CELL 6: SPEED CALCULATOR CLASS
# =============================================================================
class SpeedCalculator:
    def __init__(self, fps):
        self.fps = fps
        self.positions = defaultdict(list)
        self.speeds = defaultdict(list)
        self.violations = set()
        self.all_ids = set()
    
    def update(self, track_id, center, frame_num):
        self.positions[track_id].append((frame_num, center))
        self.all_ids.add(track_id)
        
        if len(self.positions[track_id]) < MIN_TRACK_FRAMES:
            return None
        
        curr = self.positions[track_id][-1]
        prev = self.positions[track_id][-MIN_TRACK_FRAMES]
        
        dist = get_real_distance(prev[1], curr[1])
        time = (curr[0] - prev[0]) / self.fps
        
        if time <= 0:
            return None
        
        speed = (dist / time) * 3.6  # m/s to km/h
        
        if speed > 300 or speed < 0:
            return self.get_speed(track_id)
        
        self.speeds[track_id].append(speed)
        self.speeds[track_id] = self.speeds[track_id][-3:]  # Keep last 3
        
        avg_speed = np.mean(self.speeds[track_id])
        
        if avg_speed > SPEED_LIMIT_KMH:
            self.violations.add(track_id)
        
        return avg_speed
    
    def get_speed(self, track_id):
        if self.speeds[track_id]:
            return np.mean(self.speeds[track_id])
        return None
    
    def get_stats(self):
        return len(self.all_ids), len(self.violations)

# =============================================================================
# CELL 7: VISUALIZATION FUNCTIONS
# =============================================================================
def draw_zone(frame):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [SOURCE_POLYGON.astype(np.int32)], CYAN)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    cv2.polylines(frame, [SOURCE_POLYGON.astype(np.int32)], True, CYAN, 2)
    return frame

def draw_vehicle(frame, bbox, track_id, speed, is_speeding):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = RED if is_speeding else GREEN
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_speeding else 2)
    
    label = f"ID:{track_id}"
    if speed:
        label += f" {speed:.0f}km/h"
        if is_speeding:
            label += " SPEEDING!"
    
    cv2.rectangle(frame, (x1, y1-25), (x1+len(label)*10, y1), color, -1)
    cv2.putText(frame, label, (x1+2, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return frame

def draw_dashboard(frame, total, violations, fps):
    cv2.rectangle(frame, (10, 10), (280, 110), (0,0,0), -1)
    cv2.rectangle(frame, (10, 10), (280, 110), CYAN, 2)
    cv2.putText(frame, "TRAFFIC ANALYTICS", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
    cv2.putText(frame, f"Vehicles: {total}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"Violations: {violations}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED if violations else (255,255,255), 1)
    cv2.putText(frame, f"Limit: {SPEED_LIMIT_KMH}km/h | FPS: {fps:.1f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    return frame

# =============================================================================
# CELL 8: MAIN PROCESSING LOOP (GPU ACCELERATED)
# =============================================================================
from ultralytics import YOLO
import supervision as sv
import torch

# Check and set device (GPU if available)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model and move to GPU
model = YOLO(MODEL_NAME)
model.to(DEVICE)  # Move model to GPU

tracker = sv.ByteTrack()

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize
speed_calc = SpeedCalculator(fps)
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_num = 0
import time
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect (GPU inference with half precision for speed)
    results = model(frame, verbose=False, device=DEVICE, half=(DEVICE=='cuda'))[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Filter vehicles
    mask = np.isin(detections.class_id, VEHICLE_CLASSES) if len(detections) > 0 else np.array([])
    if len(mask) > 0:
        detections = detections[mask]
        detections = detections[detections.confidence >= CONFIDENCE]
    
    # Track
    if len(detections) > 0:
        detections = tracker.update_with_detections(detections)
    
    # Draw zone
    frame = draw_zone(frame)
    
    # Process each vehicle
    if len(detections) > 0 and detections.tracker_id is not None:
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            track_id = detections.tracker_id[i]
            center = ((bbox[0]+bbox[2])/2, bbox[3])  # bottom center
            
            speed = speed_calc.update(track_id, center, frame_num)
            is_speeding = speed and speed > SPEED_LIMIT_KMH
            frame = draw_vehicle(frame, bbox, track_id, speed, is_speeding)
    
    # Dashboard
    total, violations = speed_calc.get_stats()
    current_fps = frame_num / (time.time() - start_time + 0.001)
    frame = draw_dashboard(frame, total, violations, current_fps)
    
    out.write(frame)
    frame_num += 1
    
    if frame_num % 30 == 0:
        print(f"Processed {frame_num} frames...")

cap.release()
out.release()
print(f"Done! Processed {frame_num} frames. Output saved to output.mp4")

# =============================================================================
# CELL 9: DISPLAY VIDEO IN COLAB
# =============================================================================
def show_video(path):
    mp4 = open(path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display(HTML(f'<video width=800 controls><source src="{data_url}" type="video/mp4"></video>'))

show_video('output.mp4')
