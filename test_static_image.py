"""
Test the blind navigation system with static images
Perfect for testing without a camera!
"""

import cv2
import yaml
from pathlib import Path
import sys
import torch

sys.path.append(str(Path(__file__).parent / 'src'))
from detector import DetectorUtils, DepthEstimator

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_detections(results, frame_shape, config, depth_map=None, depth_model=None):
    """Process YOLO detections - same as main.py.

    Supports optional `depth_map` (and `depth_model`) so distance_m is computed from the
    monocular depth estimate when available; otherwise falls back to bbox heuristic.

    Returns a dict with:
      - 'detections': list of detection dicts (includes quadrant overlap info)
      - 'quadrant_presence': list[int] for [far-left, left, center, right, far-right] (0-100)
    """
    detections = []
    
    frame_height, frame_width = frame_shape[:2]
    frame_area = frame_width * frame_height
    
    cam_vfov = config['camera'].get('vertical_fov_deg')
    cam_hfov = config['camera'].get('horizontal_fov_deg')
    path_width_m = config['detection'].get('path_width_m', 0.6)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = result.names[cls]
            
            if class_name not in config['detection']['priority_classes']:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_area = (x2 - x1) * (y2 - y1)
            relative_size = bbox_area / frame_area
            
            zones = config['detection']['distance_zones']
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
            
            # Prefer depth map if available
            distance_m = None
            if depth_model is not None and depth_map is not None:
                try:
                    depth_val = depth_model.sample_depth(depth_map, (x1, y1, x2, y2))
                    if getattr(depth_model, 'auto_calibrate', False) and class_name == depth_model.reference_class:
                        depth_model.calibrate_from_depth(depth_val)
                    distance_m = depth_model.depth_to_meters(depth_val)
                except Exception:
                    distance_m = None

            if distance_m is None:
                try:
                    distance_m = DetectorUtils.estimate_distance(
                        (x1, y1, x2, y2),
                        (frame_height, frame_width),
                        object_type=class_name,
                        camera_vertical_fov_deg=cam_vfov
                    )
                except Exception:
                    distance_m = None

            # is_close only meaningful for persons; include bbox fallback when depth missing
            close_thresh = config['detection'].get('close_distance_m', 1.5)
            close_bbox_fraction = config['detection'].get('close_bbox_fraction', 0.45)

            is_close = False
            if class_name == 'person':
                if distance_m is not None and distance_m <= close_thresh:
                    is_close = True
                else:
                    bbox_h = (y2 - y1)
                    if frame_height > 0 and (bbox_h / frame_height) >= close_bbox_fraction:
                        is_close = True

            try:
                in_path = DetectorUtils.is_in_walking_path(
                    (x1, y1, x2, y2),
                    (frame_height, frame_width),
                    distance_m,
                    camera_horizontal_fov_deg=cam_hfov,
                    path_width_m=path_width_m
                )
            except Exception:
                in_path = False
            
            try:
                collision_prob = DetectorUtils.collision_probability(
                    distance_m if distance_m else 10.0,
                    in_path,
                    distance_zone
                )
            except Exception:
                collision_prob = 0.0
            
            try:
                quadrant = DetectorUtils.get_quadrant(
                    (x1, y1, x2, y2),
                    (frame_height, frame_width)
                )
            except Exception:
                quadrant = 'unknown'
            
            try:
                quadrant_overlap = DetectorUtils.get_quadrant_overlap(
                    (x1, y1, x2, y2),
                    (frame_height, frame_width)
                )
                primary_quadrants = DetectorUtils.get_primary_quadrants(
                    quadrant_overlap,
                    threshold=0.1
                )
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
    
    detections.sort(key=lambda x: (
        0 if x['distance_zone'] == 'danger' else 1 if x['distance_zone'] == 'warning' else 2,
        -x['collision_probability'],
        -x['confidence']
    ))

    # Aggregate quadrant presence (0-100) using a simple strength metric
    quadrant_names = ['far-left', 'left', 'center', 'right', 'far-right']
    presence_scores = {q: 0.0 for q in quadrant_names}

    for det in detections:
        overlap = det.get('quadrant_overlap', {})
        conf = det.get('confidence', 0.0)
        dist = det.get('distance_m')

        if dist is None:
            proximity = 0.3
        else:
            proximity = max(0.0, min(1.0, (5.0 - dist) / 5.0))

        strength = conf * proximity
        for q in quadrant_names:
            presence_scores[q] += strength * overlap.get(q, 0.0)

    quadrant_presence = [int(min(1.0, presence_scores[q]) * 100) for q in quadrant_names]

    return {'detections': detections, 'quadrant_presence': quadrant_presence}

def draw_quadrant_lines(frame):
    """Draw vertical lines showing the 5 quadrants"""
    height, width = frame.shape[:2]
    fifth = width // 5
    
    for i in range(1, 5):
        x = i * fifth
        cv2.line(frame, (x, 0), (x, height), (128, 128, 128), 2)
    
    quadrant_names = ['FAR-LEFT', 'LEFT', 'CENTER', 'RIGHT', 'FAR-RIGHT']
    for i, name in enumerate(quadrant_names):
        x_center = int((i + 0.5) * fifth)
        cv2.putText(frame, name, (x_center - 50, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

def draw_detections(frame, detections):
    """Draw bounding boxes and labels"""
    colors = {
        'danger': (0, 0, 255),
        'warning': (0, 165, 255),
        'info': (0, 255, 0)
    }
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = colors[det['distance_zone']]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        label = f"{det['class_name']} {det['confidence']:.2f}"
        
        if det.get('collision_probability', 0) >= 0.4:
            label += f" COLL:{det['collision_probability']*100:.0f}%"
        
        cv2.putText(frame, label, (x1, y1 - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if det.get('primary_quadrants'):
            quad_label = "+".join([q[:3].upper() for q in det['primary_quadrants']])
            cv2.putText(frame, quad_label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def main():
    print("=" * 60)
    print("Blind Navigation System - Image Testing Mode")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Optional: initialize depth model
    depth_model = None
    if config.get('depth', {}).get('enabled', False):
        try:
            depth_model = DepthEstimator(config['depth'])
            print(f"Depth model initialized: {depth_model}")
        except Exception as e:
            print(f"Warning: depth model failed to initialize: {e}")
            depth_model = None

    # Load YOLO model
    print("\n[1/4] Loading YOLOv5 model...")
    try:
        from ultralytics import YOLO
        # device selection via config (default: 'auto')
        cfg_device = config.get('model', {}).get('device', 'auto')
        if cfg_device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = cfg_device

        model = YOLO(config['model']['weights'])
        try:
            if device != 'cpu':
                model.to(device)
        except Exception:
            pass
        print(f"✅ Model loaded successfully (device={device})")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Find test images
    print("\n[2/4] Looking for test images...")
    test_dir = Path('test_images')
    image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
    
    if not image_files:
        print(f"❌ No images found in {test_dir}/")
        print("Please add some .jpg or .png images to the test_images folder")
        return
    
    print(f"✅ Found {len(image_files)} images")
    for img in image_files:
        print(f"   - {img.name}")
    
    # Process each image
    print("\n[3/4] Processing images...")
    collision_threshold = config['detection'].get('collision_threshold', 0.5)
    
    for img_path in image_files:
        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        print('='*60)
        
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"❌ Cannot read {img_path}")
            continue
        
        print(f"Image size: {frame.shape}")
        
        # Run detection (pass selected device for acceleration)
        results = model(frame,
                       device=device,
                       conf=config['model']['confidence_threshold'],
                       iou=config['model']['iou_threshold'],
                       imgsz=config['model']['img_size'])
        
        # Optional depth inference for this image
        depth_map = None
        if depth_model is not None:
            try:
                depth_map = depth_model.infer_frame(frame)
            except Exception as e:
                print(f"Depth inference failed for image {img_path.name}: {e}")
                depth_map = None

        # Process detections (use depth_map if available)
        proc = process_detections(results, frame.shape, config, depth_map=depth_map, depth_model=depth_model)
        detections = proc['detections']
        quadrant_presence = proc.get('quadrant_presence', [0, 0, 0, 0, 0])
        
        print(f"\n✅ Detected {len(detections)} objects:")
        print(f"Quadrant presence (far-left,left,center,right,far-right): {quadrant_presence}")
        
        for i, det in enumerate(detections, 1):
            print(f"\n  [{i}] {det['class_name'].upper()}")
            print(f"      Confidence: {det['confidence']:.2%}")
            print(f"      Zone: {det['distance_zone'].upper()}")
            print(f"      Position: {det['position']}")
            
            if det['distance_m']:
                print(f"      Distance: {det['distance_m']:.2f}m")
            if det.get('depth_value') is not None:
                print(f"      Raw depth (model): {det['depth_value']:.4f}")
            print(f"      In path: {'YES ⚠️' if det['in_path'] else 'NO'}")
            print(f"      Close to user: {'YES' if det.get('is_close') else 'NO'}")
            print(f"      Collision probability: {det['collision_probability']:.1%}")
            
            if det['quadrant_overlap']:
                print(f"      Quadrant overlap:")
                for quad, pct in det['quadrant_overlap'].items():
                    if pct > 0.05:
                        print(f"        - {quad}: {pct*100:.1f}%")
        
        # Check for collision alerts
        collision_alerts = [det for det in detections 
                          if det['collision_probability'] >= collision_threshold]
        
        if collision_alerts:
            print(f"\n{'!'*60}")
            print(f"⚠️  COLLISION ALERTS: {len(collision_alerts)}")
            print('!'*60)
            
            for alert in collision_alerts:
                quad_info = " + ".join(alert['primary_quadrants']) if alert['primary_quadrants'] else alert['quadrant']
                
                print(f"\n  🚨 {alert['class_name'].upper()} in {quad_info.upper()}")
                print(f"     Collision probability: {alert['collision_probability']*100:.0f}%")
                print(f"     Distance: {alert['distance_m']:.1f}m")
        
        # Draw visualization
        frame = draw_quadrant_lines(frame)
        frame = draw_detections(frame, detections)
        
        # Save result
        output_path = f"output/result_{img_path.stem}.jpg"
        cv2.imwrite(output_path, frame)
        print(f"\n💾 Saved visualization to: {output_path}")
        
        # Display
        cv2.imshow(f'Result: {img_path.name}', frame)
        print("\n👁️  Press any key to continue to next image...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("✅ All images processed!")
    print("="*60)
    print(f"\nResults saved in output/ folder")
    print("Check output/result_*.jpg files for visualizations")

if __name__ == "__main__":
    main()