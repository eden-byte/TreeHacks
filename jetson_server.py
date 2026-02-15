"""
Jetson Orin Nano — YOLO Object Detection HTTP Server
Receives frames from Raspberry Pi, runs YOLOv5/v8 + depth estimation,
returns detections JSON including quadrant_presence.

Usage:
    python jetson_server.py [--config config.yaml] [--port 5000]

Expects the following files on the Jetson:
    - src/detector.py  (DetectorUtils, DepthEstimator)
    - config.yaml      (detection parameters)
    - yolov5n.pt       (or whatever weights config specifies)
"""

import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from flask import Flask, request, jsonify
from ultralytics import YOLO

# Add src/ to path so we can import detector
sys.path.append(str(Path(__file__).parent / "src"))
from detector import DetectorUtils, DepthEstimator

app = Flask(__name__)

# Globals — set up at startup
model = None
config = None
depth_model = None
depth_map = None
device = "cpu"
frame_count = 0


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_yolo_model(cfg):
    global device
    cfg_device = cfg.get("model", {}).get("device", "auto")
    if cfg_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg_device
    print(f"[MODEL] Device: {device}")

    weights = cfg["model"].get("weights", "yolov5n.pt")
    m = YOLO(weights)
    if device != "cpu":
        try:
            m.to(device)
            print(f"[MODEL] Moved YOLO -> {device}")
        except Exception as ex:
            print(f"[MODEL WARNING] Could not .to({device}): {ex}. Will pass device at inference.")
    return m


def load_depth(cfg):
    try:
        if cfg.get("depth", {}).get("enabled", False):
            dm = DepthEstimator(cfg["depth"])
            print(f"[DEPTH] Model initialized: {dm}")
            return dm
    except Exception as e:
        print(f"[DEPTH WARNING] Failed to initialize: {e}")
    return None


def process_detections(results, frame_shape, depth_map_local=None):
    """Process ultralytics YOLO results into detection dicts + quadrant_presence."""
    detections = []
    frame_height, frame_width = frame_shape[:2]
    frame_area = frame_width * frame_height

    cam_vfov = config["camera"].get("vertical_fov_deg")
    cam_hfov = config["camera"].get("horizontal_fov_deg")
    path_width_m = config["detection"].get("path_width_m", 0.6)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = result.names[cls]

            if class_name not in config["detection"]["priority_classes"]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_area = (x2 - x1) * (y2 - y1)
            relative_size = bbox_area / frame_area if frame_area > 0 else 0

            zones = config["detection"]["distance_zones"]
            if relative_size > zones["danger"]:
                distance_zone = "danger"
            elif relative_size > zones["warning"]:
                distance_zone = "warning"
            else:
                distance_zone = "info"

            bbox_center_x = (x1 + x2) / 2
            if bbox_center_x < frame_width / 3:
                position = "left"
            elif bbox_center_x < 2 * frame_width / 3:
                position = "center"
            else:
                position = "right"

            # Distance estimation — prefer depth model, fall back to bbox heuristic
            depth_val = None
            distance_m = None
            if depth_model is not None and depth_map_local is not None:
                try:
                    depth_val = depth_model.sample_depth(depth_map_local, (x1, y1, x2, y2))
                    if getattr(depth_model, "auto_calibrate", False) and class_name == depth_model.reference_class:
                        depth_model.calibrate_from_depth(depth_val)
                    distance_m = depth_model.depth_to_meters(depth_val)
                except Exception:
                    depth_val = None
                    distance_m = None

            if distance_m is None:
                try:
                    distance_m = DetectorUtils.estimate_distance(
                        (x1, y1, x2, y2),
                        (frame_height, frame_width),
                        object_type=class_name,
                        camera_vertical_fov_deg=cam_vfov,
                    )
                except Exception:
                    distance_m = None

            # Close-person detection
            close_thresh = config["detection"].get("close_distance_m", 1.5)
            close_bbox_fraction = config["detection"].get("close_bbox_fraction", 0.45)
            is_close = False
            if class_name == "person":
                if distance_m is not None and distance_m <= close_thresh:
                    is_close = True
                else:
                    bbox_h = y2 - y1
                    if frame_height > 0 and (bbox_h / frame_height) >= close_bbox_fraction:
                        is_close = True

            # Walking path
            try:
                in_path = DetectorUtils.is_in_walking_path(
                    (x1, y1, x2, y2), (frame_height, frame_width),
                    distance_m, camera_horizontal_fov_deg=cam_hfov,
                    path_width_m=path_width_m,
                )
            except Exception:
                in_path = False

            # Collision probability
            try:
                collision_prob = DetectorUtils.collision_probability(
                    distance_m if distance_m else 10.0, in_path, distance_zone
                )
            except Exception:
                collision_prob = 0.0

            # Quadrant
            try:
                quadrant = DetectorUtils.get_quadrant((x1, y1, x2, y2), (frame_height, frame_width))
            except Exception:
                quadrant = "unknown"

            try:
                quadrant_overlap = DetectorUtils.get_quadrant_overlap((x1, y1, x2, y2), (frame_height, frame_width))
                primary_quadrants = DetectorUtils.get_primary_quadrants(quadrant_overlap, threshold=0.1)
            except Exception:
                quadrant_overlap = {}
                primary_quadrants = []

            detections.append({
                "class_name": class_name,
                "confidence": round(conf, 3),
                "distance_zone": distance_zone,
                "position": position,
                "bbox": [x1, y1, x2, y2],
                "distance_m": round(distance_m, 2) if distance_m is not None else None,
                "in_path": in_path,
                "collision_probability": round(collision_prob, 3),
                "quadrant": quadrant,
                "quadrant_overlap": {k: round(v, 3) for k, v in quadrant_overlap.items()} if quadrant_overlap else {},
                "primary_quadrants": primary_quadrants,
                "is_close": is_close,
            })

    # Sort by severity
    detections.sort(key=lambda x: (
        0 if x["distance_zone"] == "danger" else 1 if x["distance_zone"] == "warning" else 2,
        -x["collision_probability"],
        -x["confidence"],
    ))

    # Aggregate quadrant presence (0-100)
    quadrant_names = ["far-left", "left", "center", "right", "far-right"]
    presence_scores = {q: 0.0 for q in quadrant_names}
    for det in detections:
        overlap = det.get("quadrant_overlap", {})
        c = det.get("confidence", 0.0)
        dist = det.get("distance_m")
        proximity = max(0.0, min(1.0, (5.0 - dist) / 5.0)) if dist is not None else 0.3
        strength = c * proximity
        for q in quadrant_names:
            presence_scores[q] += strength * overlap.get(q, 0.0)

    quadrant_presence = [int(min(1.0, presence_scores[q]) * 100) for q in quadrant_names]

    return {"detections": detections, "quadrant_presence": quadrant_presence}


@app.route("/detect", methods=["POST"])
def detect():
    global depth_map, frame_count
    start = time.time()

    if "frame" not in request.files:
        return jsonify({"error": "No frame provided", "detections": [], "quadrant_presence": [0, 0, 0, 0, 0]}), 400

    file = request.files["frame"]
    jpg_bytes = file.read()
    nparr = np.frombuffer(jpg_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Could not decode frame", "detections": [], "quadrant_presence": [0, 0, 0, 0, 0]}), 400

    # Run depth model periodically
    if depth_model is not None:
        run_every = config.get("depth", {}).get("run_every_n_frames", 2)
        if frame_count % run_every == 0:
            try:
                depth_map = depth_model.infer_frame(frame)
            except Exception:
                depth_map = None

    # Run YOLO inference
    results = model(
        frame,
        device=device,
        conf=config["model"]["confidence_threshold"],
        iou=config["model"]["iou_threshold"],
        imgsz=config["model"]["img_size"],
    )

    proc = process_detections(results, frame.shape, depth_map)
    frame_count += 1

    elapsed_ms = (time.time() - start) * 1000
    proc["frame_time_ms"] = round(elapsed_ms, 1)

    return jsonify(proc)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": device,
        "model": config["model"].get("weights", "unknown"),
        "depth_enabled": depth_model is not None,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jetson Detection Server")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    config = load_config(args.config)
    print("[INIT] Loading YOLO model...")
    model = load_yolo_model(config)
    print("[INIT] Loading depth model...")
    depth_model = load_depth(config)
    print(f"[INIT] Server starting on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
