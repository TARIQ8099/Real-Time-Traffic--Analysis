"""
Perspective Transform Module for Speed Estimation
==================================================
This module implements the mathematical core of the speed estimation system.

HOMOGRAPHY THEORY:
==================
A homography (also called a perspective transform or projective transformation) 
is a 3x3 matrix H that maps points from one plane to another.

Given a point (x, y) in the source plane, the transformed point (x', y') is:
    [x']   [h11 h12 h13]   [x]
    [y'] = [h21 h22 h23] * [y]
    [w']   [h31 h32 h33]   [1]

    x' = (h11*x + h12*y + h13) / (h31*x + h32*y + h33)
    y' = (h21*x + h22*y + h23) / (h31*x + h32*y + h33)

WHY WE NEED THIS FOR SPEED ESTIMATION:
=====================================
1. In a dashcam image, the road appears to converge at a vanishing point
2. Objects farther away appear smaller (perspective distortion)
3. 10 pixels of movement near the camera ≠ 10 pixels far from camera in real distance
4. By mapping image coordinates to real-world coordinates (meters), we can:
   - Calculate actual distance traveled in meters
   - Divide by time to get speed in m/s
   - Convert to km/h for display

CALIBRATION PROCESS:
===================
1. Identify 4 points on the road that form a known shape (e.g., lane markings)
2. Measure or estimate the real-world dimensions of this shape
3. Use cv2.getPerspectiveTransform() to compute the homography matrix
4. Apply cv2.perspectiveTransform() to convert any image point to real-world coordinates
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from collections import defaultdict

# Import configuration
from config import (
    SOURCE_POLYGON, 
    TARGET_RECTANGLE,
    TARGET_LENGTH_METERS,
    MIN_TRACKING_FRAMES,
    SPEED_SMOOTHING_WINDOW,
    SPEED_LIMIT_KMH
)


class PerspectiveTransformer:
    """
    Handles perspective transformation from image coordinates to real-world coordinates.
    
    This class computes the homography matrix once during initialization and provides
    methods to transform points and calculate distances in real-world units (meters).
    """
    
    def __init__(self, source_points: np.ndarray = None, target_points: np.ndarray = None):
        """
        Initialize the perspective transformer.
        
        Args:
            source_points: 4 points defining the zone in image coordinates (pixels)
            target_points: 4 points defining the zone in real-world coordinates (meters)
        """
        self.source_points = source_points if source_points is not None else SOURCE_POLYGON
        self.target_points = target_points if target_points is not None else TARGET_RECTANGLE
        
        # Compute the homography matrix
        # cv2.getPerspectiveTransform requires exactly 4 points
        self.homography_matrix = cv2.getPerspectiveTransform(
            self.source_points.astype(np.float32),
            self.target_points.astype(np.float32)
        )
        
        # Compute inverse homography for transforming back if needed
        self.inverse_homography = cv2.getPerspectiveTransform(
            self.target_points.astype(np.float32),
            self.source_points.astype(np.float32)
        )
        
    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a single point from image coordinates to real-world coordinates.
        
        Mathematical Process:
        --------------------
        1. Convert point to homogeneous coordinates: [x, y, 1]
        2. Multiply by homography matrix: [x', y', w'] = H * [x, y, 1]
        3. Normalize by w': (x'/w', y'/w')
        
        Args:
            point: (x, y) coordinates in image space (pixels)
            
        Returns:
            (x, y) coordinates in real-world space (meters)
        """
        # Reshape point for cv2.perspectiveTransform: shape must be (1, 1, 2)
        point_array = np.array([[point]], dtype=np.float32)
        
        # Apply the transformation
        transformed = cv2.perspectiveTransform(point_array, self.homography_matrix)
        
        # Extract the transformed coordinates
        return (transformed[0][0][0], transformed[0][0][1])
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform multiple points from image to real-world coordinates.
        
        Args:
            points: Array of shape (N, 2) with N points
            
        Returns:
            Transformed points array of shape (N, 2)
        """
        if len(points) == 0:
            return np.array([])
            
        # Reshape for cv2.perspectiveTransform: (N, 1, 2)
        points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        
        # Apply transformation
        transformed = cv2.perspectiveTransform(points_reshaped, self.homography_matrix)
        
        # Reshape back to (N, 2)
        return transformed.reshape(-1, 2)
    
    def calculate_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """
        Calculate the real-world distance between two image points.
        
        Process:
        -------
        1. Transform both points to real-world coordinates
        2. Calculate Euclidean distance in meters
        
        Args:
            point1: First point in image coordinates (pixels)
            point2: Second point in image coordinates (pixels)
            
        Returns:
            Distance in meters
        """
        # Transform both points to real-world coordinates
        real_point1 = self.transform_point(point1)
        real_point2 = self.transform_point(point2)
        
        # Calculate Euclidean distance
        distance = np.sqrt(
            (real_point2[0] - real_point1[0])**2 + 
            (real_point2[1] - real_point1[1])**2
        )
        
        return distance
    
    def is_point_in_zone(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside the detection zone.
        
        Uses cv2.pointPolygonTest which returns:
        - Positive value: inside
        - Zero: on the edge
        - Negative: outside
        
        Args:
            point: (x, y) coordinates to check
            
        Returns:
            True if point is inside or on the edge of the zone
        """
        result = cv2.pointPolygonTest(
            self.source_points.astype(np.int32), 
            point, 
            measureDist=False
        )
        return result >= 0


class SpeedEstimator:
    """
    Estimates vehicle speeds using perspective transformation and tracking data.
    
    SPEED CALCULATION METHOD:
    ========================
    1. Track vehicle positions across frames
    2. Transform positions to real-world coordinates using homography
    3. Calculate distance traveled in meters
    4. Divide by time elapsed (frames / FPS) to get speed in m/s
    5. Convert to km/h: speed_kmh = speed_ms * 3.6
    
    SMOOTHING:
    =========
    Raw speed calculations can be noisy due to:
    - Detection jitter (bounding box wobble)
    - Tracking errors
    - Perspective inaccuracies
    
    We use a sliding window average to smooth the speed estimates.
    """
    
    def __init__(self, transformer: PerspectiveTransformer, fps: float):
        """
        Initialize the speed estimator.
        
        Args:
            transformer: PerspectiveTransformer instance
            fps: Video frames per second (for time calculation)
        """
        self.transformer = transformer
        self.fps = fps
        self.time_per_frame = 1.0 / fps  # seconds per frame
        
        # Track history for each vehicle ID
        # Format: {track_id: [(frame_num, center_x, center_y), ...]}
        self.track_history = defaultdict(list)
        
        # Calculated speeds for each vehicle
        # Format: {track_id: [speed1, speed2, ...]} for smoothing
        self.speed_history = defaultdict(list)
        
        # Current smoothed speed for each vehicle
        self.current_speeds = {}
        
        # Violation tracking
        self.speeding_violations = set()
        self.total_vehicles_counted = set()
        
    def update(self, track_id: int, center: Tuple[float, float], 
               frame_num: int) -> Optional[float]:
        """
        Update tracking data and calculate speed for a vehicle.
        
        ALGORITHM:
        =========
        1. Store current position with frame number
        2. If we have enough history (MIN_TRACKING_FRAMES):
           a. Get position from N frames ago
           b. Transform both positions to real-world coordinates
           c. Calculate distance traveled (meters)
           d. Calculate time elapsed (frames / FPS)
           e. Speed (m/s) = distance / time
           f. Speed (km/h) = speed * 3.6
        3. Apply smoothing using sliding window average
        4. Check for speed violations
        
        Args:
            track_id: Unique ID of the tracked vehicle
            center: (x, y) center of the bounding box in pixels
            frame_num: Current frame number
            
        Returns:
            Smoothed speed in km/h, or None if not enough data
        """
        # Add current position to history
        self.track_history[track_id].append((frame_num, center[0], center[1]))
        
        # Track this vehicle
        self.total_vehicles_counted.add(track_id)
        
        # Need minimum frames for reliable speed calculation
        if len(self.track_history[track_id]) < MIN_TRACKING_FRAMES:
            return None
        
        # Get current and previous positions
        history = self.track_history[track_id]
        current = history[-1]
        previous = history[-MIN_TRACKING_FRAMES]
        
        # Current and previous positions in image coordinates
        current_pos = (current[1], current[2])
        previous_pos = (previous[1], previous[2])
        
        # Calculate real-world distance using perspective transform
        distance_meters = self.transformer.calculate_distance(previous_pos, current_pos)
        
        # Calculate time elapsed
        frames_elapsed = current[0] - previous[0]
        time_elapsed = frames_elapsed * self.time_per_frame  # seconds
        
        # Avoid division by zero
        if time_elapsed <= 0:
            return None
        
        # Calculate speed
        speed_ms = distance_meters / time_elapsed  # meters per second
        speed_kmh = speed_ms * 3.6  # convert to km/h
        
        # Filter out unrealistic speeds (likely tracking errors)
        if speed_kmh > 300 or speed_kmh < 0:
            return self.current_speeds.get(track_id, None)
        
        # Add to speed history for smoothing
        self.speed_history[track_id].append(speed_kmh)
        
        # Keep only recent speeds for smoothing
        if len(self.speed_history[track_id]) > SPEED_SMOOTHING_WINDOW:
            self.speed_history[track_id] = self.speed_history[track_id][-SPEED_SMOOTHING_WINDOW:]
        
        # Calculate smoothed speed (average of recent speeds)
        smoothed_speed = np.mean(self.speed_history[track_id])
        self.current_speeds[track_id] = smoothed_speed
        
        # Check for speeding violation
        if smoothed_speed > SPEED_LIMIT_KMH:
            self.speeding_violations.add(track_id)
        
        return smoothed_speed
    
    def get_speed(self, track_id: int) -> Optional[float]:
        """Get the current smoothed speed for a vehicle."""
        return self.current_speeds.get(track_id, None)
    
    def is_speeding(self, track_id: int) -> bool:
        """Check if a vehicle is speeding."""
        speed = self.current_speeds.get(track_id, 0)
        return speed > SPEED_LIMIT_KMH
    
    def get_stats(self) -> Tuple[int, int]:
        """
        Get statistics for the dashboard.
        
        Returns:
            (total_vehicles, speeding_violations)
        """
        return len(self.total_vehicles_counted), len(self.speeding_violations)
    
    def cleanup_old_tracks(self, current_frame: int, max_age: int = 30):
        """
        Remove old tracks that haven't been updated recently.
        
        This prevents memory from growing unbounded during long videos.
        
        Args:
            current_frame: Current frame number
            max_age: Maximum number of frames a track can be inactive
        """
        tracks_to_remove = []
        
        for track_id, history in self.track_history.items():
            if history:
                last_frame = history[-1][0]
                if current_frame - last_frame > max_age:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_history[track_id]
            if track_id in self.speed_history:
                del self.speed_history[track_id]
            if track_id in self.current_speeds:
                del self.current_speeds[track_id]
