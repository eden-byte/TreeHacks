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
from detector import DetectorUtils, DepthEstimator


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

        # Initialize depth estimator (optional)
        self.depth_model = None
        self.depth_map = None
        try:
            if self.config.get('depth', {}).get('enabled', False):
                self.depth_model = DepthEstimator(self.config['depth'])
                print(f"Depth model initialized: {self.depth_model}")
        except Exception as e:
            print(f"Warning: depth model failed to initialize: {e}")
            self.depth_model = None

        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
    def load_model(self):
        """Load YOLOv5 model and move to GPU when available / configured."""
        weights = self.config['model'].get('weights')

        # device selection: config['model'].device supports 'auto'|'cpu'|'cuda'|'cuda:0' etc.
        cfg_device = self.config.get('model', {}).get('device', 'auto')
        if cfg_device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = cfg_device
        self.device = device
        print(f"Model device: {device}")

        try:
            from ultralytics import YOLO
            model = YOLO(weights)
            # try to move the model to the selected device; if the wrapper doesn't support .to(),
            # we'll pass the device at inference time instead.
            try:
                if device != 'cpu':
                    model.to(device)
                    print(f"Moved YOLO model -> {device}")
            except Exception as ex:
                print(f"Warning: could not .to({device}) the YOLO wrapper: {ex}. Will pass device on inference.")
        except Exception:
            # fallback: older torch.hub loader (still attempt to move to device)
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
            try:
                model.to(device)
                print(f"Moved torch-hub model -> {device}")
            except Exception:
                print("Warning: failed to move torch-hub model to device; continuing (inference may run on CPU)")

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
    
    def process_detections(self, results, frame_shape, depth_map=None):
        """Process YOLO detections and compute distance + path membership + collision probability.

        If `depth_map` is provided (from a monocular depth model) per‑bbox depth sampling is used
        and a simple auto‑scale calibration may be applied. Otherwise the original bbox‑height
        heuristic is used.

        Returns:
            dict with keys:
              - 'detections': list of detection dicts (each includes 'quadrant_overlap' and 'primary_quadrants')
              - 'quadrant_presence': list[int] of length 5 with percentages [far-left, left, center, right, far-right]
        """
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

                # Prefer depth-map based estimate when available; otherwise fall back
                depth_val = None
                distance_m = None
                if self.depth_model is not None and depth_map is not None:
                    try:
                        depth_val = self.depth_model.sample_depth(depth_map, (x1, y1, x2, y2))
                        # Auto‑calibrate scale if configured and we see the reference class
                        if getattr(self.depth_model, 'auto_calibrate', False) and class_name == self.depth_model.reference_class:
                            self.depth_model.calibrate_from_depth(depth_val)
                        distance_m = self.depth_model.depth_to_meters(depth_val)
                    except Exception:
                        depth_val = None
                        distance_m = None

                if distance_m is None:
                    try:
                        distance_m = DetectorUtils.estimate_distance((x1, y1, x2, y2),
                                                                    (frame_height, frame_width),
                                                                    object_type=class_name,
                                                                    camera_vertical_fov_deg=cam_vfov)
                    except Exception:
                        distance_m = None

                # Determine close-person boolean (configurable)
                close_thresh = self.config['detection'].get('close_distance_m', 1.5)
                close_bbox_fraction = self.config['detection'].get('close_bbox_fraction', 0.45)

                is_close = False
                if class_name == 'person':
                    # 1) depth-based check (preferred)
                    if distance_m is not None and distance_m <= close_thresh:
                        is_close = True
                    else:
                        # 2) fallback: bounding-box height fraction
                        bbox_h = (y2 - y1)
                        if frame_height > 0 and (bbox_h / frame_height) >= close_bbox_fraction:
                            is_close = True

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
                    'depth_value': depth_val,
                    'in_path': in_path,
                    'collision_probability': collision_prob,
                    'quadrant': quadrant,
                    'quadrant_overlap': quadrant_overlap,
                    'primary_quadrants': primary_quadrants,
                    'is_close': is_close
                })

        # Sort detections by zone / probability / confidence
        detections.sort(key=lambda x: (
            0 if x['distance_zone'] == 'danger' else 1 if x['distance_zone'] == 'warning' else 2,
            -x['collision_probability'],
            -x['confidence']
        ))

        # Aggregate quadrant presence (0.0 - 1.0) using detection strength
        quadrant_names = ['far-left', 'left', 'center', 'right', 'far-right']
        presence_scores = {q: 0.0 for q in quadrant_names}

        for det in detections:
            overlap = det.get('quadrant_overlap', {})
            conf = det.get('confidence', 0.0)
            dist = det.get('distance_m')

            # proximity weight: closer -> stronger (clamp at 5m range)
            if dist is None:
                proximity = 0.3
            else:
                proximity = max(0.0, min(1.0, (5.0 - dist) / 5.0))

            # detection strength in [0,1]
            strength = conf * proximity

            for q in quadrant_names:
                presence_scores[q] += strength * overlap.get(q, 0.0)

        # Convert to 0-100 scale and clamp
        quadrant_presence = [int(min(1.0, presence_scores[q]) * 100) for q in quadrant_names]

        return {'detections': detections, 'quadrant_presence': quadrant_presence}
    
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
    
    def draw_quadrant_lines(self, frame):
        """Draw vertical lines showing the 5 quadrants"""
        height, width = frame.shape[:2]
        fifth = width // 5
        
        # Draw quadrant dividing lines
        for i in range(1, 5):
            x = i * fifth
            cv2.line(frame, (x, 0), (x, height), (128, 128, 128), 1)
        
        # Label quadrants at top
        quadrant_names = ['FAR-LEFT', 'LEFT', 'CENTER', 'RIGHT', 'FAR-RIGHT']
        for i, name in enumerate(quadrant_names):
            x_center = int((i + 0.5) * fifth)
            cv2.putText(frame, name, (x_center - 40, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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

                # Run depth model periodically (if enabled)
                if self.depth_model is not None:
                    run_every = self.config['depth'].get('run_every_n_frames', 2)
                    if (self.frame_count % run_every) == 0:
                        try:
                            self.depth_map = self.depth_model.infer_frame(frame)
                        except Exception as e:
                            print(f"Depth inference failed: {e}")
                            self.depth_map = None

                results = self.model(frame,
                                   device=getattr(self, 'device', 'cpu'),
                                   conf=self.config['model']['confidence_threshold'],
                                   iou=self.config['model']['iou_threshold'],
                                   imgsz=self.config['model']['img_size'])
                
                proc = self.process_detections(results, frame.shape, depth_map=self.depth_map)
                detections = proc['detections']
                quadrant_presence = proc.get('quadrant_presence', [0, 0, 0, 0, 0])
                
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
                        depth_raw = det.get('depth_value')
                        depth_raw_str = f"{depth_raw:.3f}" if depth_raw is not None else "N/A"
                        dist_str = f"{det['distance_m']:.2f}m" if det['distance_m'] is not None else "N/A"
                        close_str = 'YES' if det.get('is_close') else 'NO'
                        print(f"[{det['distance_zone'].upper()}] {det['class_name']} on {det['position']} - {det['confidence']:.2f} - {dist_str} - raw_depth:{depth_raw_str} - close:{close_str}")
                    # Print quadrant presence summary (far-left, left, center, right, far-right)
                    print(f"Quadrant presence: {quadrant_presence}")
        
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