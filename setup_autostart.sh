#!/bin/bash
# Setup auto-start for Vera on boot (Jetson or Pi)
# Usage:
#   On Jetson:  sudo ./setup_autostart.sh jetson
#   On Pi:      sudo ./setup_autostart.sh pi

set -e

if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo ./setup_autostart.sh [jetson|pi]"
    exit 1
fi

DEVICE="${1:-}"

if [ "$DEVICE" = "jetson" ]; then
    SERVICE_FILE="vera-jetson.service"
    SERVICE_NAME="vera-jetson"
    echo "=== Setting up Vera auto-start on JETSON ==="
elif [ "$DEVICE" = "pi" ]; then
    SERVICE_FILE="vera-pi.service"
    SERVICE_NAME="vera-pi"
    echo "=== Setting up Vera auto-start on RASPBERRY PI ==="
else
    echo "Usage: sudo ./setup_autostart.sh [jetson|pi]"
    echo ""
    echo "  jetson  — Install detection server auto-start"
    echo "  pi      — Install Vera main app auto-start"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Copy service file
cp "$SCRIPT_DIR/$SERVICE_FILE" /etc/systemd/system/"$SERVICE_NAME".service
echo "[OK] Copied $SERVICE_FILE to /etc/systemd/system/"

# Reload systemd and enable
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
echo "[OK] Enabled $SERVICE_NAME to start on boot"

# Start it now
systemctl start "$SERVICE_NAME"
echo "[OK] Started $SERVICE_NAME"

echo ""
echo "=== Done! ==="
echo "  Status:   sudo systemctl status $SERVICE_NAME"
echo "  Logs:     sudo journalctl -u $SERVICE_NAME -f"
echo "  Stop:     sudo systemctl stop $SERVICE_NAME"
echo "  Disable:  sudo systemctl disable $SERVICE_NAME"
