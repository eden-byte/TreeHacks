# Vera — Deployment Guide

## Architecture Overview

```
┌──────────────────────┐         HTTP (WiFi)        ┌──────────────────────┐
│   Raspberry Pi 5     │ ──── POST /detect ────────► │   Jetson Orin Nano   │
│                      │ ◄─── JSON detections ────── │                      │
│  - main.py           │                             │  - jetson_server.py  │
│  - Camera (USB/CSI)  │                             │  - YOLOv5/v8         │
│  - Microphone (USB)  │                             │  - Depth estimation  │
│  - Speaker / headset │                             │                      │
│  - Vibration motors  │                             └──────────────────────┘
│    (GPIO, optional)  │
│  - OpenAI API calls  │ ──────► Internet
│  - Perplexity API    │ ──────► Internet
└──────────────────────┘
```

The **Raspberry Pi** runs the main app (voice, camera, TTS, face recognition, LLM calls).
The **Jetson Nano** runs a detection server that receives camera frames and returns object detections + depth estimates.
They communicate over your local network via HTTP.

---

## 1. Network Setup

Both devices need to be on the same local network (WiFi or Ethernet).

### Option A: Same WiFi network
Connect both Pi and Jetson to the same WiFi. Find their IPs:
```bash
# On each device:
hostname -I
```

### Option B: Direct Ethernet (no router needed)
Connect Pi and Jetson with an Ethernet cable. Assign static IPs:
```bash
# On Pi — edit /etc/dhcpcd.conf (or use nmcli on Bookworm):
sudo nmcli con add type ethernet con-name jetson-link ifname eth0 \
  ip4 192.168.50.1/24

# On Jetson:
sudo nmcli con add type ethernet con-name pi-link ifname eth0 \
  ip4 192.168.50.2/24
```
In this case, `JETSON_URL=http://192.168.50.2:5000`.

### Verify connectivity
```bash
# From the Pi, ping the Jetson:
ping <jetson-ip>
```

---

## 2. Jetson Orin Nano Setup

### 2a. Clone the repo
```bash
git clone <your-repo-url> ~/vera
cd ~/vera
```

### 2b. Install dependencies
```bash
pip install -r requirements-jetson.txt
```

If using GPU-accelerated PyTorch on Jetson, follow NVIDIA's official PyTorch wheel instructions for your JetPack version instead of the pip torch package.

### 2c. Download YOLO weights
```bash
# YOLOv5 nano (fast, good for real-time):
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n.pt

# Or YOLOv8 nano:
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
```

### 2d. Create config.yaml

The Jetson server expects a `config.yaml`. Create one:

```yaml
model:
  weights: "yolov5n.pt"        # or yolov8n.pt
  device: "auto"                # "auto" picks CUDA if available
  confidence_threshold: 0.4
  iou_threshold: 0.45
  img_size: 640

camera:
  vertical_fov_deg: 55
  horizontal_fov_deg: 70

detection:
  path_width_m: 0.6
  close_distance_m: 1.5
  close_bbox_fraction: 0.45
  priority_classes:
    - person
    - car
    - bicycle
    - motorcycle
    - bus
    - truck
    - dog
    - cat
    - chair
    - bench
    - stop sign
    - fire hydrant
    - backpack
    - umbrella
    - bottle
    - cup
    - door               # custom class if your model supports it
  distance_zones:
    danger: 0.15          # bbox > 15% of frame area
    warning: 0.05         # bbox > 5% of frame area

depth:
  enabled: false          # set true if you have a depth model ONNX
  run_every_n_frames: 2
```

### 2e. Create src/detector.py

The Jetson server imports `DetectorUtils` and `DepthEstimator` from `src/detector.py`. If you don't have this file yet, you need it. At minimum it needs:
- `DetectorUtils.estimate_distance(bbox, frame_shape, ...)`
- `DetectorUtils.is_in_walking_path(...)`
- `DetectorUtils.collision_probability(...)`
- `DetectorUtils.get_quadrant(...)`
- `DetectorUtils.get_quadrant_overlap(...)`
- `DetectorUtils.get_primary_quadrants(...)`
- `DepthEstimator` class (only if `depth.enabled: true`)

### 2f. Start the detection server
```bash
python jetson_server.py --config config.yaml --port 5000 --host 0.0.0.0
```

Verify it's running:
```bash
# From Jetson itself:
curl http://localhost:5000/health
```
Expected response:
```json
{"status": "ok", "device": "cuda", "model": "yolov5n.pt", "depth_enabled": false}
```

---

## 3. Raspberry Pi Setup

### 3a. Clone the repo
```bash
git clone <your-repo-url> ~/vera
cd ~/vera
```

### 3b. Run the setup script
```bash
chmod +x setup_pi.sh
./setup_pi.sh
```

This installs system packages (opencv, portaudio, zbar, mpg123, espeak), creates a venv, installs Python deps, and downloads the MobileFaceNet ONNX model.

### 3c. Connect hardware

**Camera** — USB webcam or Pi Camera Module (CSI ribbon cable). Verify:
```bash
source venv/bin/activate
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL'); cap.release()"
```
If it says FAIL, try index 1 or check `ls /dev/video*`.

**Microphone** — USB microphone or USB sound card with mic input. Verify:
```bash
arecord -l   # should list your capture device
```

**Speaker/Headset** — 3.5mm jack, HDMI audio, or Bluetooth. Verify:
```bash
# Test with mpg123 or espeak:
espeak "hello"
```

**Vibration motors (optional)** — 5 motors connected via GPIO pins:
| Motor Zone   | GPIO Pin |
|-------------|----------|
| left        | 17       |
| left-center | 27       |
| center      | 22       |
| right-center| 23       |
| right       | 24       |

Each motor needs a transistor/MOSFET driver — do NOT drive motors directly from GPIO pins (they can only source ~16mA). A typical setup: GPIO → 1kΩ resistor → NPN base, motor between collector and 5V, emitter to GND, flyback diode across motor.

### 3d. Configure .env
```bash
cp .env.example .env   # or create from scratch
```

Edit `.env`:
```
OPENAI_API_KEY=sk-...your-key...
PERPLEXITY_API_KEY=pplx-...your-key...

# Point to the Jetson's IP and port:
JETSON_URL=http://<jetson-ip>:5000
JETSON_ENABLED=true

# Set to true if running without a monitor (headless Pi):
# VERA_HEADLESS=true
```

### 3e. Verify Jetson connection from the Pi
```bash
curl http://<jetson-ip>:5000/health
```
Should return `{"status": "ok", ...}`. If it times out, check your network setup (step 1).

### 3f. Start Vera
```bash
source venv/bin/activate
python3 main.py
```

You should see:
```
==================================================
  Vera — Vision Assistant (RAG-powered)
==================================================

[PLATFORM] Raspberry Pi | Headless | GPIO: Active
[JETSON] Enabled at http://<jetson-ip>:5000
[RAG] Memory has 0 stored interactions.
[CAMERA] Found working camera at index 0
[SPEAKING] Vera is ready. Just say Hey Vera whenever you need me.
[CALIBRATING] Adjusting for ambient noise...
[READY] Say "Hey Vera" to activate, "Bye" to sleep.
```

---

## 4. Startup Order

1. **Start Jetson server first** — `python jetson_server.py --config config.yaml`
2. **Wait for** `[INIT] Server starting on 0.0.0.0:5000`
3. **Then start Pi** — `python3 main.py`

The Pi's `jetson_worker` thread will immediately start sending frames to the Jetson. If the Jetson isn't reachable, detection silently returns empty results (no crash).

---

## 5. Running Without a Jetson

If you don't have a Jetson available (or it's not cooperating), just set:
```
JETSON_ENABLED=false
```
in your `.env`. Everything else (voice, camera, vision via GPT, face recognition, barcode scanning, TTS, deep research) works independently on the Pi.

---

## 6. Running on Desktop (macOS/Windows/Linux) for Development

The code auto-detects the platform. On non-Pi systems:
- GPIO is skipped (no vibration motors)
- `VERA_HEADLESS` defaults to `false` (shows a webcam preview window)
- TTS uses `afplay` on macOS, PowerShell on Windows, `mpg123`/`ffplay` on Linux

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys, set JETSON_ENABLED=false
python main.py
```

Press `q` in the webcam window to quit.

---

## 7. Quick Troubleshooting

| Problem | Fix |
|---------|-----|
| `[ERROR] Cannot open webcam` | Check `ls /dev/video*`, try different camera index |
| `[ERROR] Speech recognition service error` | Check internet — Google Speech Recognition needs it |
| Jetson returns empty detections | Check `curl http://<jetson-ip>:5000/health`, verify `config.yaml` has your YOLO weights path |
| No sound from TTS | Verify `mpg123` is installed (`sudo apt install mpg123`), check audio output device (`aplay -l`) |
| `[FACE WARNING] MobileFaceNet model not found` | Re-run `setup_pi.sh` or manually download `mobilefacenet.onnx` |
| Motors not vibrating | Check wiring, verify `HAS_GPIO` prints `Active`, test with `python3 -c "from gpiozero import PWMOutputDevice; m=PWMOutputDevice(17); m.value=0.5"` |
| Jetson CUDA out of memory | Use smaller weights (`yolov5n.pt`), reduce `img_size` to 320 in config |
