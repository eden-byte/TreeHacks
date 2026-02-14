"""
Main entry point for YOLOv5 Blind Navigation System
Optimized for Jetson Nano with camera glasses
"""

import cv2
import torch
import yaml
import argparse
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / "src"))
from detector import DetectorUtils


class BlindNavigationSystem:
    def __init__(self, config_path="config.yaml"):
        """Initialize the blind navigation system"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("Loading YOLOv5 model...")
        self.model = self.load_model()
        
        print("Initializing camera...")
        self.cap = self.init_camera()
        
        print("Setting up audio feedback...")
        self.audio = self.init_audio()
        
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
    def load_model(self):
        """Load YOLOv5 model"""
        weights = self.config['model']['weights']
        
        try:
            from ultralytics import YOLO
            model = YOLO(weights)
            print(f"Loaded {weights} successfully")
        except ImportError:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
            print(f"Loaded {weights} via torch.hub")
        
        return model
    
    def init_camera(self):
        """Initialize camera capture"""
        source = self.config['camera']['source']
        width = self.config['camera']['width']
        height = self.config['camera']['height']
        
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")
        
        return cap
    
    def init_audio(self):
        """Initialize audio feedback system"""
        if not self.config['audio']['enabled']:
            return None
        
        try:
            from audio_feedback import AudioFeedback
            return AudioFeedback(self.config)
        except ImportError:
            print("Warning: audio_feedback module not found. Running without audio.")
            return None
    
    def process_detections(self, results, frame_shape):
        """Process YOLO detections and compute distance + path membership + collision probability."""
        detections = []

        frame_height, frame_width = frame_shape[:2]
        frame_area = frame_width * frame_height

        # Camera parameters from config (optional)
        cam_vfov = self.config['camera'].get('vertical_fov_deg')
        cam_hfov = self.config['camera'].get('horizontal_fov_deg')
        path_width_m = self.config['detection'].get('path_width_m', 0.6)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]

                if class_name not in self.config['detection']['priority_classes']:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_area = (x2 - x1) * (y2 - y1)
                relative_size = bbox_area / frame_area

                zones = self.config['detection']['distance_zones']
                if relative_size > zones['danger']:
                    distance_zone = 'danger'
                elif relative_size > zones['warning']:
                    distance_zone = 'warning'
                else:
                    distance_zone = 'info'

                bbox_center_x = (x1 + x2) / 2
                if bbox_center_x < frame_width / 3:
                    position = 'left'
                elif bbox_center_x < 2 * frame_width / 3:
                    position = 'center'
                else:
                    position = 'right'

                # Estimate distance (meters) using detector utility
                try:
                    distance_m = DetectorUtils.estimate_distance((x1, y1, x2, y2),
                                                                (frame_height, frame_width),
                                                                object_type=class_name,
                                                                camera_vertical_fov_deg=cam_vfov)
                except Exception:
                    distance_m = None

                # Compute whether object lies within the user's walking path
                try:
                    in_path = DetectorUtils.is_in_walking_path((x1, y1, x2, y2),
                                                            (frame_height, frame_width),
                                                            distance_m,
                                                            camera_horizontal_fov_deg=cam_hfov,
                                                            path_width_m=path_width_m)
                except Exception:
                    in_path = False

                # Calculate collision probability
                try:
                    collision_prob = DetectorUtils.collision_probability(
                        distance_m if distance_m else 10.0,
                        in_path,
                        distance_zone
                    )
                except Exception:
                    collision_prob = 0.0

                # Get quadrant location
                try:
                    quadrant = DetectorUtils.get_quadrant((x1, y1, x2, y2),
                                                        (frame_height, frame_width))
                except Exception:
                    quadrant = 'unknown'

                # Get quadrant overlap percentages
                try:
                    quadrant_overlap = DetectorUtils.get_quadrant_overlap((x1, y1, x2, y2),
                                                                        (frame_height, frame_width))
                    primary_quadrants = DetectorUtils.get_primary_quadrants(quadrant_overlap, 
                                                                        threshold=0.1)
                except Exception:
                    quadrant_overlap = {}
                    primary_quadrants = []

                detections.append({
                    'class_name': class_name,
                    'confidence': conf,
                    'distance_zone': distance_zone,
                    'position': position,
                    'bbox': (x1, y1, x2, y2),
                    'distance_m': distance_m,
                    'in_path': in_path,
                    'collision_probability': collision_prob,
                    'quadrant': quadrant
                })

            detections.sort(key=lambda x: (
                0 if x['distance_zone'] == 'danger' else 1 if x['distance_zone'] == 'warning' else 2,
                -x['collision_probability'],  # Sort by collision probability too
                -x['confidence']
            ))

            return detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        colors = {
            'danger': (0, 0, 255),    # Red
            'warning': (0, 165, 255), # Orange
            'info': (0, 255, 0)       # Green
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = colors[det['distance_zone']]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show quadrant overlap info
            if det.get('primary_quadrants'):
                quad_label = "+".join([q[:3].upper() for q in det['primary_quadrants']])
                cv2.putText(frame, quad_label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def handle_collision_alerts(self, collision_alerts):
        """
        Handle collision alerts by logging and announcing them.
        
        Args:
            collision_alerts: List of detections with high collision probability
        """
        for alert in collision_alerts[:2]:  # Only alert for top 2 most critical
        # Build quadrant information string
            if alert.get('primary_quadrants'):
                quad_info = " + ".join(alert['primary_quadrants'])
                
                # Add percentages for detail
                overlap_details = []
                for quad in alert['primary_quadrants']:
                    pct = alert['quadrant_overlap'].get(quad, 0) * 100
                    if pct >= 10:  # Only show if >= 10%
                        overlap_details.append(f"{quad}:{pct:.0f}%")
                
                if overlap_details:
                    quad_info += f" ({', '.join(overlap_details)})"
            else:
                quad_info = alert['quadrant'].upper()

        for alert in collision_alerts[:2]:  # Only alert for top 2 most critical
            alert_msg = (
                f"COLLISION ALERT: {alert['class_name'].upper()} "
                f"in {alert['quadrant'].upper()} quadrant - "
                f"{alert['collision_probability']*100:.0f}% probability - "
                f"{alert['distance_m']:.1f}m away"
            )
            
            # Print to console
            print(alert_msg)
            
            # Log to file if enabled
            if self.config['output']['log_detections']:
                from utils import log_detection
                log_detection(alert)
            
            # Send to audio system if enabled
            if self.audio:
                self.audio.announce_custom(alert_msg)
    
    def run(self):
        """Main detection loop"""
        print("\nStarting blind navigation system...")
        print("Press 'q' to quit\n")

        # Collision alert threshold
        collision_threshold = self.config['detection'].get('collision_threshold', 0.5)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                results = self.model(frame, 
                                   conf=self.config['model']['confidence_threshold'],
                                   iou=self.config['model']['iou_threshold'],
                                   imgsz=self.config['model']['img_size'])
                
                detections = self.process_detections(results, frame.shape)
                
                # Check for collision alerts
                collision_alerts = [det for det in detections if det['collision_probability'] >= collision_threshold]

                if self.audio and detections:
                    self.audio.announce_detections(detections)
                
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed
                
                if self.config['output']['display']:
                    frame = self.draw_detections(frame, detections)
                    frame = self.draw_quadrant_lines(frame)
                    cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Blind Navigation System', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if self.config['output']['log_detections'] and detections:
                    for det in detections[:3]:
                        print(f"[{det['distance_zone'].upper()}] {det['class_name']} "
                              f"on {det['position']} - {det['confidence']:.2f}")
        
        except KeyboardInterrupt:
            print("\nStopping system...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        if self.audio:
            self.audio.stop()
        print("System stopped.")


def main():
    parser = argparse.ArgumentParser(description="YOLOv5 Blind Navigation System")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    system = BlindNavigationSystem(config_path=args.config)
    system.run()


if __name__ == "__main__":
    main()