#!/bin/bash
# One clean entry point for Pinku + Music app.
# Safe to run multiple times — kills any previous instances first,
# including other terminals running this script.

cd "$(dirname "$0")"

# ── Kill other copies of this script running in other terminals ───────────────
MY_PID=$$
for pid in $(pgrep -f "start_pinku.command" 2>/dev/null); do
    [ "$pid" != "$MY_PID" ] && kill "$pid" 2>/dev/null
done
sleep 0.3   # let sibling scripts die before we start killing pinku.py

# ── Stop PM2-managed Pinku (prevents dual-instance with Terminal run) ─────────
pm2 stop pinku   2>/dev/null || true
pm2 delete pinku 2>/dev/null || true

# ── Kill existing Pinku + Music processes ─────────────────────────────────────
pkill -f "python.*pinku\.py"     2>/dev/null && echo "Stopped previous Pinku."     || true
pkill -f "python.*music_app\.py" 2>/dev/null && echo "Stopped previous Music app." || true

# Wait up to 8s for pinku.py to actually exit before proceeding.
# pkill sends SIGTERM; pinku.py handles it and exits cleanly within ~1s.
# Without this wait, the old instance holds the lockfile while the new one
# starts — new one exits immediately (code 2) and neither runs reliably.
echo "Waiting for previous Pinku to exit..."
for i in $(seq 8); do
    pgrep -f "python.*pinku\.py" > /dev/null 2>&1 || break
    sleep 1
done
# Force-kill anything still alive after the grace period
pkill -9 -f "python.*pinku\.py"     2>/dev/null || true
pkill -9 -f "python.*music_app\.py" 2>/dev/null || true
sleep 0.5   # brief pause so OS fully releases the lockfile + port

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
    # Free port 5101 BEFORE launching — evicts any stray/duplicate music_app
    # (e.g. left over from a prior run) that would otherwise cause an endless
    # "Address already in use" crash-loop.
    kill -9 $(lsof -ti :5101) 2>/dev/null || true
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
