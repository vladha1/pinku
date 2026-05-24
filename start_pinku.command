#!/bin/bash
# Double-click this in Finder (VNC) to start Pinku inside Terminal.app.
# Terminal.app has camera permission → python3 inherits it as child process.
# Watchdog loop: auto-restarts Pinku if it crashes.

cd "$(dirname "$0")"
echo "============================================"
echo "  Pinku watchdog (camera via Terminal.app)  "
echo "============================================"
echo "  Close this window to stop Pinku."
echo ""

while true; do
    echo "$(date '+%H:%M:%S')  Starting Pinku..."
    .venv/bin/python3 pinku.py
    EXIT=$?
    echo ""
    echo "$(date '+%H:%M:%S')  Pinku exited (code $EXIT). Restarting in 5 s..."
    echo "  (Close this window to stop)"
    sleep 5
    echo ""
done
