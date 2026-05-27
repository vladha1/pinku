#!/bin/bash
# One clean entry point for Pinku + Music app.
# Safe to run multiple times — kills any previous instances first.

cd "$(dirname "$0")"

# ── Kill existing instances and free ports ────────────────────────────────────
pkill -f pinku.py     2>/dev/null && echo "Stopped previous Pinku instance."     || true
pkill -f music_app.py 2>/dev/null && echo "Stopped previous Music app instance." || true
kill -9 $(lsof -ti :5100) 2>/dev/null || true
kill -9 $(lsof -ti :5101) 2>/dev/null || true

# Wait until both ports are actually free (up to 10s)
echo "Waiting for ports to free..."
for i in $(seq 10); do
    { lsof -ti :5100 >/dev/null 2>&1 || lsof -ti :5101 >/dev/null 2>&1; } || break
    sleep 1
done

echo "============================================"
echo "  Pinku starting"
echo "  Dashboard : http://localhost:5100"
echo "  Music app : http://localhost:5101"
echo "  Close this window to stop both."
echo "============================================"

export TF_CPP_MIN_LOG_LEVEL=3
export GRPC_VERBOSITY=ERROR

# ── Music app — background with auto-restart ──────────────────────────────────
(
  while true; do
    echo "$(date '+%H:%M:%S')  [Music] Starting..."
    .venv/bin/python3 music_app.py
    EXIT=$?
    echo "$(date '+%H:%M:%S')  [Music] Exited (code $EXIT). Restarting in 5s..."
    kill -9 $(lsof -ti :5101) 2>/dev/null || true
    sleep 5
  done
) &
MUSIC_PID=$!

# Kill music app when this terminal window closes
trap "kill $MUSIC_PID 2>/dev/null; pkill -f music_app.py 2>/dev/null" EXIT

# ── Pinku — foreground with auto-restart ─────────────────────────────────────
while true; do
    echo "$(date '+%H:%M:%S')  [Pinku] Starting..."
    .venv/bin/python3 pinku.py
    EXIT=$?
    echo ""
    echo "$(date '+%H:%M:%S')  [Pinku] Exited (code $EXIT). Restarting in 5s..."
    kill -9 $(lsof -ti :5100) 2>/dev/null || true
    sleep 5
done
