"""
General utility functions for the blind navigation system
"""

import cv2
import numpy as np
from typing import Tuple, List
import yaml
from pathlib import Path


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str = 'config.yaml'):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def resize_frame(frame: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio
    
    Args:
        frame: Input frame
        target_size: Target size for the longer side
        
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    
    if max(h, w) == target_size:
        return frame
    
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    
    return cv2.resize(frame, (new_w, new_h))


def draw_fps(frame: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """Draw FPS counter on frame"""
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    return frame


def create_output_dir(base_dir: str = 'output') -> Path:
    """Create output directory for logs and videos"""
    output_path = Path(base_dir)
    output_path.mkdir(exist_ok=True)
    return output_path


def log_detection(detection: dict, log_file: str = 'detections.log'):
    """Log detection to file"""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (f"[{timestamp}] {detection['class_name']} - "
                f"Zone: {detection['distance_zone']}, "
                f"Position: {detection['position']}, "
                f"Confidence: {detection['confidence']:.2f}\n")
    
    with open(log_file, 'a') as f:
        f.write(log_entry)


def get_zone_color(zone: str) -> Tuple[int, int, int]:
    """Get BGR color for distance zone"""
    colors = {
        'danger': (0, 0, 255),    # Red
        'warning': (0, 165, 255), # Orange
        'info': (0, 255, 0)       # Green
    }
    return colors.get(zone, (255, 255, 255))


def calculate_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """Calculate bounding box area"""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def get_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Get center point of bounding box"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def format_detection_message(detection: dict) -> str:
    """Format detection for display or logging (includes distance/path if present)."""
    base = (f"[{detection['distance_zone'].upper()}] "
            f"{detection['class_name']} on {detection['position']} "
            f"({detection['confidence']:.2f})")
    if detection.get('distance_m') is not None:
        base += f" — {detection['distance_m']:.2f}m"
    if detection.get('in_path'):
        base += " [IN PATH]"
    return base