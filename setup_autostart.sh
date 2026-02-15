#!/bin/bash
# Setup auto-start for Vera on boot (Jetson or Pi)
# Usage:
#   On Jetson:  sudo ./setup_autostart.sh jetson
#   On Pi:      ./setup_autostart.sh pi        (no sudo!)

set -e

DEVICE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$DEVICE" = "jetson" ]; then
    # Jetson: system service (no audio needed)
    if [ "$EUID" -ne 0 ]; then
        echo "Jetson setup requires sudo: sudo ./setup_autostart.sh jetson"
        exit 1
    fi

    echo "=== Setting up Vera auto-start on JETSON ==="
    cp "$SCRIPT_DIR/vera-jetson.service" /etc/systemd/system/vera-jetson.service
    systemctl daemon-reload
    systemctl enable vera-jetson
    systemctl start vera-jetson
    echo "[OK] Vera Jetson service installed and started."
    echo ""
    echo "  Status:   sudo systemctl status vera-jetson"
    echo "  Logs:     sudo journalctl -u vera-jetson -f"
    echo "  Stop:     sudo systemctl stop vera-jetson"
    echo "  Disable:  sudo systemctl disable vera-jetson"

elif [ "$DEVICE" = "pi" ]; then
    # Pi: user service (needs audio/Bluetooth access)
    if [ "$EUID" -eq 0 ]; then
        echo "Pi setup should NOT use sudo. Run: ./setup_autostart.sh pi"
        exit 1
    fi

    echo "=== Setting up Vera auto-start on RASPBERRY PI ==="

    # Remove old system service if it exists
    if systemctl list-unit-files vera-pi.service &>/dev/null; then
        echo "[CLEANUP] Removing old system service..."
        sudo systemctl stop vera-pi 2>/dev/null || true
        sudo systemctl disable vera-pi 2>/dev/null || true
        sudo rm -f /etc/systemd/system/vera-pi.service
        sudo systemctl daemon-reload
    fi

    # Install user service
    mkdir -p ~/.config/systemd/user
    cp "$SCRIPT_DIR/vera-pi.service" ~/.config/systemd/user/vera-pi.service
    echo "[OK] Copied vera-pi.service to ~/.config/systemd/user/"

    # Enable linger so user services start at boot without login
    sudo loginctl enable-linger "$USER"
    echo "[OK] Enabled linger for $USER (services start at boot)"

    # Reload and enable
    systemctl --user daemon-reload
    systemctl --user enable vera-pi
    systemctl --user start vera-pi
    echo "[OK] Vera Pi user service installed and started."
    echo ""
    echo "  Status:   systemctl --user status vera-pi"
    echo "  Logs:     journalctl --user -u vera-pi -f"
    echo "  Stop:     systemctl --user stop vera-pi"
    echo "  Disable:  systemctl --user disable vera-pi"

else
    echo "Usage:"
    echo "  On Jetson:  sudo ./setup_autostart.sh jetson"
    echo "  On Pi:      ./setup_autostart.sh pi        (no sudo!)"
    exit 1
fi

echo ""
echo "=== Done! Vera will auto-start on boot. ==="
