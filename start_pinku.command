#!/bin/bash
# One clean entry point for Pinku.
# Safe to run multiple times — kills any previous instance first.

cd "$(dirname "$0")"

# Kill ALL existing Pinku processes and whatever holds port 5100
pkill -f pinku.py 2>/dev/null && echo "Stopped previous Pinku instance." || true
kill -9 $(lsof -ti :5100) 2>/dev/null || true

# Wait until port 5100 is actually free (up to 10s)
echo "Waiting for port 5100 to be free..."
for i in $(seq 10); do
    lsof -ti :5100 > /dev/null 2>&1 || break
    sleep 1
done

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
    kill -9 $(lsof -ti :5100) 2>/dev/null || true
    sleep 5
done
