#!/usr/bin/env bash
# Setup script for Vera on Raspberry Pi 5 (Bookworm)
set -e

echo "=== Vera Pi 5 Setup ==="

# System packages
sudo apt update
sudo apt install -y \
    python3-opencv \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libzbar0 \
    mpg123 \
    espeak \
    flac \
    libopenblas-dev

# Create venv (--system-site-packages so apt-installed opencv is visible)
echo "=== Creating Python venv ==="
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install Python packages
echo "=== Installing Python packages ==="
pip install --upgrade pip
pip install -r requirements.txt

# Download MobileFaceNet ONNX model (~5MB)
MOBILEFACENET_URL="https://github.com/niceDev0908/face-recognition/raw/main/mobilefacenet.onnx"
if [ ! -f "mobilefacenet.onnx" ]; then
    echo "=== Downloading MobileFaceNet ONNX model ==="
    wget -O mobilefacenet.onnx "$MOBILEFACENET_URL" || \
    curl -L -o mobilefacenet.onnx "$MOBILEFACENET_URL" || \
    echo "[WARNING] Failed to download MobileFaceNet model. Face recognition will be disabled."
fi

# Verify installations
echo "=== Verifying installations ==="

python3 -c "import cv2; print(f'OpenCV {cv2.__version__} OK')"

python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, _ = cap.read()
    print('Camera OK' if ret else 'Camera found but cannot read')
    cap.release()
else:
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        print('Camera OK (index 1)')
        cap.release()
    else:
        print('[WARNING] No camera detected')
"

python3 -c "import onnxruntime; print(f'ONNX Runtime {onnxruntime.__version__} OK')"
python3 -c "from gpiozero import Device; print('GPIO OK')" 2>/dev/null || echo "[INFO] GPIO not available (expected on non-Pi)"
python3 -c "import speech_recognition as sr; print(f'SpeechRecognition {sr.__version__} OK')"

echo ""
echo "=== Setup complete ==="
echo "To run Vera: source venv/bin/activate && python3 main.py"
echo ""
echo "Don't forget to set up your .env file:"
echo "  OPENAI_API_KEY=sk-..."
echo "  PERPLEXITY_API_KEY=pplx-..."
echo "  JETSON_URL=http://192.168.50.2:5000"
echo "  JETSON_ENABLED=true"
