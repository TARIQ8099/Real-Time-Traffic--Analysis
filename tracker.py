"""
Vehicle Tracker Module
======================
Handles vehicle detection and tracking using YOLO and ByteTrack.

This module integrates:
- YOLO for object detection (detecting vehicles in each frame)
- ByteTrack for multi-object tracking (maintaining vehicle IDs across frames)
- Supervision library for easy annotation and filtering
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# These will be imported when running in Colab
# from ultralytics import YOLO
# import supervision as sv

from config import (
    MODEL_NAME,
    VEHICLE_CLASSES,
    CONFIDENCE_THRESHOLD,
    SOURCE_POLYGON
)


@dataclass
class TrackedVehicle:
    """Data class to store information about a tracked vehicle."""
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]
    confidence: float
    speed_kmh: Optional[float] = None
    is_speeding: bool = False


class VehicleTracker:
    """
    Detects and tracks vehicles using YOLO + ByteTrack.
    
    The tracking pipeline:
    1. YOLO detects all objects in the frame
    2. Filter to keep only vehicle classes (car, truck, motorcycle, bus)
    3. ByteTrack associates detections across frames to maintain IDs
    4. Return tracked vehicles with their positions
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the vehicle tracker.
        
        Args:
            model_name: YOLO model to use (e.g., 'yolo11m.pt')
        """
        from ultralytics import YOLO
        import supervision as sv
        
        # Load YOLO model
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        
        # Initialize ByteTrack tracker
        # ByteTrack is a simple yet effective multi-object tracker
        self.tracker = sv.ByteTrack(
            track_activation_threshold=CONFIDENCE_THRESHOLD,
            lost_track_buffer=30,  # frames to keep lost tracks
            minimum_matching_threshold=0.8,
            frame_rate=30  # approximate, will be updated
        )
        
        # Class name mapping (COCO dataset)
        self.class_names = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Detection zone polygon for filtering
        self.zone_polygon = SOURCE_POLYGON
        
        # Store supervision module reference
        self.sv = sv
        
    def update_fps(self, fps: float):
        """Update the tracker's frame rate setting."""
        self.tracker = self.sv.ByteTrack(
            track_activation_threshold=CONFIDENCE_THRESHOLD,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=int(fps)
        )
        
    def detect_and_track(self, frame: np.ndarray) -> Tuple[List[TrackedVehicle], any]:
        """
        Detect and track vehicles in a frame.
        
        Pipeline:
        1. Run YOLO inference
        2. Convert to supervision Detections format
        3. Filter to vehicle classes only
        4. Apply ByteTrack for consistent IDs
        5. Package results as TrackedVehicle objects
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Tuple of (list of TrackedVehicle, raw supervision Detections)
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)[0]
        
        # Convert to supervision Detections format
        detections = self.sv.Detections.from_ultralytics(results)
        
        # Filter to keep only vehicle classes
        vehicle_mask = np.isin(detections.class_id, VEHICLE_CLASSES)
        detections = detections[vehicle_mask]
        
        # Apply confidence threshold
        if len(detections) > 0:
            confidence_mask = detections.confidence >= CONFIDENCE_THRESHOLD
            detections = detections[confidence_mask]
        
        # Apply ByteTrack for tracking
        if len(detections) > 0:
            detections = self.tracker.update_with_detections(detections)
        
        # Convert to TrackedVehicle objects
        tracked_vehicles = []
        
        if len(detections) > 0 and detections.tracker_id is not None:
            for i in range(len(detections)):
                bbox = detections.xyxy[i]
                center = (
                    (bbox[0] + bbox[2]) / 2,  # center x
                    (bbox[1] + bbox[3]) / 2   # center y
                )
                
                # Use bottom center for more accurate ground position
                ground_point = (
                    (bbox[0] + bbox[2]) / 2,  # center x
                    bbox[3]                     # bottom y
                )
                
                class_id = detections.class_id[i]
                
                vehicle = TrackedVehicle(
                    track_id=detections.tracker_id[i],
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, 'vehicle'),
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    center=ground_point,  # Use ground point for speed calculation
                    confidence=detections.confidence[i]
                )
                
                tracked_vehicles.append(vehicle)
        
        return tracked_vehicles, detections
    
    def is_in_zone(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside the detection zone.
        
        Args:
            point: (x, y) coordinates
            
        Returns:
            True if point is in zone
        """
        import cv2
        result = cv2.pointPolygonTest(
            self.zone_polygon.astype(np.int32),
            point,
            measureDist=False
        )
        return result >= 0
