#!/bin/bash
# One clean entry point for Pinku.
# Safe to run multiple times — kills any previous instance first.

cd "$(dirname "$0")"

# Kill any existing Pinku processes to avoid echoing/duplicate instances
pkill -f pinku.py 2>/dev/null && echo "Stopped previous Pinku instance." || true
sleep 1

echo "============================================"
echo "  Pinku starting (camera via Terminal.app)  "
echo "  Close this window to stop Pinku.          "
echo "============================================"

while true; do
    echo "$(date '+%H:%M:%S')  Starting Pinku..."
    .venv/bin/python3 pinku.py
    EXIT=$?
    echo ""
    echo "$(date '+%H:%M:%S')  Pinku exited (code $EXIT). Restarting in 5s..."
    sleep 5
done
