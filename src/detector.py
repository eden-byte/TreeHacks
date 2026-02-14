"""
YOLOv5 Detection Utilities
Additional helper functions for object detection
"""

import torch
import cv2
import numpy as np
from typing import List, Tuple, Dict


class DetectorUtils:
    """Utility functions for YOLOv5 detection"""
    
    @staticmethod
    def filter_overlapping_boxes(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Remove overlapping bounding boxes using Non-Maximum Suppression
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for considering boxes as overlapping
            
        Returns:
            Filtered list of detections
        """
        if len(detections) <= 1:
            return detections
        
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    @staticmethod
    def estimate_distance(bbox: Tuple[int, int, int, int], 
                         frame_shape: Tuple[int, int],
                         object_type: str = 'person',
                         camera_vertical_fov_deg: float = None) -> float:
        """
        Estimate distance (meters) using a pinhole-camera approximation when
        camera_vertical_fov_deg is provided; otherwise fall back to the
        original heuristic based on bbox height.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            frame_shape: Frame dimensions (height, width)
            object_type: Type of detected object
            camera_vertical_fov_deg: Optional camera vertical FOV in degrees

        Returns:
            Estimated distance in meters (approximate)
        """
        x1, y1, x2, y2 = bbox
        bbox_height = max(1, (y2 - y1))  # avoid div/0
        frame_height = frame_shape[0]

        # Typical object reference heights (meters)
        if object_type in ['person', 'bicycle']:
            reference_height = 1.7
        elif object_type in ['car', 'bus', 'truck']:
            reference_height = 1.5
        else:
            reference_height = 1.0

        if camera_vertical_fov_deg:
            # focal length (px) from vertical FOV: f = (H/2) / tan(vfov/2)
            vfov_rad = np.radians(camera_vertical_fov_deg)
            focal_px = (frame_height / 2.0) / np.tan(vfov_rad / 2.0)
            # pinhole camera model: object_height_pixels = reference_height * focal_px / distance
            distance = (reference_height * focal_px) / bbox_height
        else:
            # fallback heuristic (original approach)
            distance = (reference_height * frame_height) / (bbox_height * 2.0)

        return float(max(0.5, min(distance, 50.0)))

    @staticmethod
    def lateral_offset_m(bbox: Tuple[int, int, int, int],
                         frame_shape: Tuple[int, int],
                         distance_m: float,
                         camera_horizontal_fov_deg: float = None) -> float:
        """
        Compute lateral offset (meters) of bbox center from camera centerline.

        Returns positive value to the right, negative to the left.
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        frame_width = frame_shape[1]

        # Normalized offset [-1, 1]
        norm_off = (center_x - (frame_width / 2.0)) / (frame_width / 2.0)

        if camera_horizontal_fov_deg:
            angle_rad = np.radians(norm_off * (camera_horizontal_fov_deg / 2.0))
        else:
            # assume ~60deg HFOV as fallback
            angle_rad = np.radians(norm_off * 30.0)

        lateral = distance_m * np.tan(angle_rad)
        return float(lateral)

    @staticmethod
    def is_in_walking_path(bbox: Tuple[int, int, int, int],
                           frame_shape: Tuple[int, int],
                           distance_m: float,
                           camera_horizontal_fov_deg: float = None,
                           path_width_m: float = 0.6) -> bool:
        """
        Decide whether a detected object lies within the user's walking path.

        Uses lateral offset (meters) + configured physical path width.
        """
        if distance_m is None:
            return False

        lateral = DetectorUtils.lateral_offset_m(bbox, frame_shape, distance_m, camera_horizontal_fov_deg)
        return abs(lateral) <= (path_width_m / 2.0)
    
    @staticmethod
    def calculate_trajectory(detections_history: List[List[Dict]], 
                            object_id: int) -> str:
        """
        Calculate object trajectory (approaching, receding, stationary)
        
        Args:
            detections_history: List of detection lists over time
            object_id: ID of object to track
            
        Returns:
            Trajectory description: 'approaching', 'receding', or 'stationary'
        """
        if len(detections_history) < 2:
            return 'unknown'
        
        # Compare bbox sizes over time (larger = closer)
        # This is a simplified version - real tracking would need object tracking
        
        return 'stationary'  # Placeholder
    
    # Add this to the DetectorUtils class in detector.py

    @staticmethod
    def collision_probability(distance_m: float, 
                            in_path: bool,
                            distance_zone: str,
                            object_velocity: float = 0) -> float:
        """
        Estimate collision probability based on distance, path membership, and zone.
        
        Args:
            distance_m: Distance to object in meters
            in_path: Whether object is in walking path
            distance_zone: 'danger', 'warning', or 'info'
            object_velocity: Future feature for tracking (currently unused)
            
        Returns:
            Probability from 0.0 to 1.0
        """
        if not in_path:
            return 0.0
        
        # High probability if in danger zone
        if distance_zone == 'danger':
            if distance_m < 1.0:
                return 0.95
            elif distance_m < 2.0:
                return 0.80
            else:
                return 0.60
        
        # Medium probability if in warning zone
        elif distance_zone == 'warning':
            if distance_m < 2.0:
                return 0.50
            elif distance_m < 3.0:
                return 0.30
            else:
                return 0.15
        
        # Low probability if in info zone
        else:
            return 0.05

    @staticmethod
    def get_quadrant(bbox: Tuple[int, int, int, int],
                    frame_shape: Tuple[int, int]) -> str:
        """
        Divide frame into 5 quadrants and determine which one the object is in.
        
        Quadrants:
        [far-left] [left] [center] [right] [far-right]
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Quadrant name: 'far-left', 'left', 'center', 'right', 'far-right'
        """
        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2
        frame_width = frame_shape[1]
        
        # Divide into 5 equal parts
        fifth = frame_width / 5
        
        if bbox_center_x < fifth:
            return 'far-left'
        elif bbox_center_x < 2 * fifth:
            return 'left'
        elif bbox_center_x < 3 * fifth:
            return 'center'
        elif bbox_center_x < 4 * fifth:
            return 'right'
        else:
            return 'far-right'
        

    @staticmethod
    def get_quadrant_overlap(bbox: Tuple[int, int, int, int],
                        frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Calculate what percentage of the bounding box overlaps each of 5 quadrants.
        
        Quadrants (equal width):
        [far-left] [left] [center] [right] [far-right]
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Dictionary with quadrant names and overlap percentages (0.0 to 1.0)
            Example: {'far-left': 0.0, 'left': 0.3, 'center': 0.7, 'right': 0.0, 'far-right': 0.0}
        """
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame_shape[:2]
        
        # Define quadrant boundaries
        fifth = frame_width / 5
        quadrants = {
            'far-left': (0, fifth),
            'left': (fifth, 2 * fifth),
            'center': (2 * fifth, 3 * fifth),
            'right': (3 * fifth, 4 * fifth),
            'far-right': (4 * fifth, frame_width)
        }
        
        bbox_width = x2 - x1
        if bbox_width <= 0:
            return {q: 0.0 for q in quadrants.keys()}
        
        overlap_percentages = {}
        
        for quad_name, (quad_start, quad_end) in quadrants.items():
            # Calculate overlap between bbox and quadrant
            overlap_start = max(x1, quad_start)
            overlap_end = min(x2, quad_end)
            
            overlap_width = max(0, overlap_end - overlap_start)
            overlap_percentage = overlap_width / bbox_width
            
            overlap_percentages[quad_name] = float(overlap_percentage)
        
        return overlap_percentages

    @staticmethod
    def get_primary_quadrants(quadrant_overlap: Dict[str, float], 
                            threshold: float = 0.1) -> List[str]:
        """
        Get list of quadrants where object has significant presence.
        
        Args:
            quadrant_overlap: Dictionary from get_quadrant_overlap()
            threshold: Minimum overlap percentage to consider (default 10%)
            
        Returns:
            List of quadrant names sorted by overlap percentage (highest first)
        """
        significant_quadrants = [(name, pct) for name, pct in quadrant_overlap.items() 
                                if pct >= threshold]
        significant_quadrants.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in significant_quadrants]