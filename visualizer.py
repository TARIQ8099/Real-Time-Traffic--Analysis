"""
Visualization Module
====================
Handles all visual annotations including:
- Detection zone drawing
- Vehicle bounding boxes with speed labels
- Speeding violation highlights
- Dashboard with statistics
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from config import (
    SOURCE_POLYGON,
    COLOR_NORMAL,
    COLOR_SPEEDING,
    COLOR_ZONE,
    COLOR_TEXT,
    FONT_SCALE,
    FONT_THICKNESS,
    ZONE_OPACITY,
    DASHBOARD_POSITION,
    SPEED_LIMIT_KMH
)

# Import tracker types
from tracker import TrackedVehicle


class TrafficVisualizer:
    """
    Handles all visualization for the traffic analytics system.
    
    Features:
    - Detection zone overlay
    - Vehicle bounding boxes with color coding
    - Speed labels on vehicles
    - Dashboard with real-time statistics
    """
    
    def __init__(self, zone_polygon: np.ndarray = None):
        """
        Initialize the visualizer.
        
        Args:
            zone_polygon: Detection zone vertices (default from config)
        """
        self.zone_polygon = zone_polygon if zone_polygon is not None else SOURCE_POLYGON
        
    def draw_zone(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the detection zone on the frame.
        
        Creates a semi-transparent overlay to show the speed measurement area.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Frame with zone overlay
        """
        # Create overlay for transparency effect
        overlay = frame.copy()
        
        # Draw filled polygon
        cv2.fillPoly(
            overlay,
            [self.zone_polygon.astype(np.int32)],
            COLOR_ZONE
        )
        
        # Blend with original frame
        frame = cv2.addWeighted(overlay, ZONE_OPACITY, frame, 1 - ZONE_OPACITY, 0)
        
        # Draw zone border
        cv2.polylines(
            frame,
            [self.zone_polygon.astype(np.int32)],
            isClosed=True,
            color=COLOR_ZONE,
            thickness=2
        )
        
        # Add label for the zone
        zone_label_pos = (int(self.zone_polygon[0][0]), int(self.zone_polygon[0][1]) - 10)
        cv2.putText(
            frame,
            "SPEED DETECTION ZONE",
            zone_label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_ZONE,
            2
        )
        
        return frame
    
    def draw_vehicle(self, frame: np.ndarray, vehicle: TrackedVehicle,
                     speed_kmh: Optional[float] = None,
                     is_speeding: bool = False) -> np.ndarray:
        """
        Draw a vehicle's bounding box and information.
        
        Args:
            frame: Input frame
            vehicle: TrackedVehicle object
            speed_kmh: Calculated speed (km/h)
            is_speeding: Whether the vehicle is over the speed limit
            
        Returns:
            Frame with vehicle annotation
        """
        x1, y1, x2, y2 = [int(v) for v in vehicle.bbox]
        
        # Choose color based on speeding status
        color = COLOR_SPEEDING if is_speeding else COLOR_NORMAL
        box_thickness = 3 if is_speeding else 2
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
        
        # Prepare label text
        if speed_kmh is not None:
            label = f"ID:{vehicle.track_id} {speed_kmh:.0f} km/h"
            if is_speeding:
                label += " SPEEDING!"
        else:
            label = f"ID:{vehicle.track_id} {vehicle.class_name}"
        
        # Calculate label background size
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            FONT_THICKNESS
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - label_height - 10),
            (x1 + label_width + 5, y1),
            color,
            -1  # Filled
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (0, 0, 0),  # Black text on colored background
            FONT_THICKNESS
        )
        
        return frame
    
    def draw_dashboard(self, frame: np.ndarray, 
                       total_vehicles: int,
                       speeding_count: int,
                       current_fps: float = 0) -> np.ndarray:
        """
        Draw the statistics dashboard in the corner.
        
        Args:
            frame: Input frame
            total_vehicles: Total vehicles detected
            speeding_count: Number of speeding violations
            current_fps: Current processing FPS
            
        Returns:
            Frame with dashboard overlay
        """
        # Dashboard background
        dash_x, dash_y = DASHBOARD_POSITION
        dash_width = 300
        dash_height = 120
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (dash_x, dash_y),
            (dash_x + dash_width, dash_y + dash_height),
            (0, 0, 0),
            -1
        )
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw border
        cv2.rectangle(
            frame,
            (dash_x, dash_y),
            (dash_x + dash_width, dash_y + dash_height),
            COLOR_ZONE,
            2
        )
        
        # Title
        cv2.putText(
            frame,
            "TRAFFIC ANALYTICS",
            (dash_x + 10, dash_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_ZONE,
            2
        )
        
        # Statistics
        stats = [
            f"Vehicles Detected: {total_vehicles}",
            f"Speeding Violations: {speeding_count}",
            f"Speed Limit: {SPEED_LIMIT_KMH:.0f} km/h",
            f"FPS: {current_fps:.1f}"
        ]
        
        for i, stat in enumerate(stats):
            color = COLOR_SPEEDING if i == 1 and speeding_count > 0 else COLOR_TEXT
            cv2.putText(
                frame,
                stat,
                (dash_x + 10, dash_y + 50 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        return frame
    
    def annotate_frame(self, frame: np.ndarray,
                       vehicles: List[TrackedVehicle],
                       speeds: dict,
                       stats: Tuple[int, int],
                       fps: float = 0) -> np.ndarray:
        """
        Apply all annotations to a frame.
        
        Args:
            frame: Input frame
            vehicles: List of tracked vehicles
            speeds: Dictionary of {track_id: speed_kmh}
            stats: Tuple of (total_vehicles, speeding_count)
            fps: Current processing FPS
            
        Returns:
            Fully annotated frame
        """
        # Make a copy to avoid modifying original
        annotated = frame.copy()
        
        # Draw detection zone
        annotated = self.draw_zone(annotated)
        
        # Draw each vehicle
        for vehicle in vehicles:
            speed = speeds.get(vehicle.track_id, None)
            is_speeding = speed is not None and speed > SPEED_LIMIT_KMH
            annotated = self.draw_vehicle(annotated, vehicle, speed, is_speeding)
        
        # Draw dashboard
        total_vehicles, speeding_count = stats
        annotated = self.draw_dashboard(annotated, total_vehicles, speeding_count, fps)
        
        return annotated


def create_zone_selector(frame: np.ndarray) -> np.ndarray:
    """
    Interactive tool to select zone points on a frame.
    
    This is a helper function for calibrating the detection zone.
    Click 4 points on the road to define the zone.
    
    Args:
        frame: Frame to select points on
        
    Returns:
        Array of 4 selected points
    """
    points = []
    display_frame = frame.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal display_frame
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(display_frame, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(display_frame, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
            cv2.imshow("Select Zone", display_frame)
    
    cv2.imshow("Select Zone", display_frame)
    cv2.setMouseCallback("Select Zone", mouse_callback)
    
    print("Click 4 points to define the detection zone (clockwise from top-left)")
    print("Press any key when done...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return np.array(points, dtype=np.float32)
