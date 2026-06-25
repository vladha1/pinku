#!/bin/bash
# One clean entry point for Pinku + Music app.
# Safe to run multiple times — kills any previous instances first.

cd "$(dirname "$0")"

# ── Kill existing instances and wait for them to fully exit ───────────────────
pkill -f pinku.py     2>/dev/null && echo "Stopped previous Pinku instance."     || true
pkill -f music_app.py 2>/dev/null && echo "Stopped previous Music app instance." || true

# Wait up to 8s for pinku.py to actually exit before proceeding.
# pkill sends SIGTERM which Pinku handles cleanly; the process needs a moment
# to flush threads and release the lockfile.  Without this wait, the new
# instance starts before the old one exits — both try to acquire the lock,
# the old one wins, the new one exits (code 2), and neither runs reliably.
echo "Waiting for previous Pinku to exit..."
for i in $(seq 8); do
    pgrep -f "python.*pinku\.py" > /dev/null 2>&1 || break
    sleep 1
done
# Force-kill anything still alive after the grace period
pkill -9 -f "python.*pinku\.py" 2>/dev/null || true
pkill -9 -f music_app.py 2>/dev/null || true
sleep 0.5   # brief pause so OS fully releases the lockfile

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
    if [ $EXIT -eq 2 ]; then
        echo "$(date '+%H:%M:%S')  [Pinku] Lock held by another instance — retrying in 10s..."
        sleep 10
    else
        echo "$(date '+%H:%M:%S')  [Pinku] Exited (code $EXIT). Restarting in 5s..."
        kill -9 $(lsof -ti :5100) 2>/dev/null || true
        sleep 5
    fi
done
