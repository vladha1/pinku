#!/bin/bash
# Directly adds python3 to the macOS Camera TCC database.
# Run from Terminal on the Mac (GUI or SSH — no popup needed).

PYTHON="/Users/vivekladha/pinku/.venv/bin/python3"
TCC_DB="$HOME/Library/Application Support/com.apple.TCC/TCC.db"

echo "Target: $PYTHON"
echo "TCC DB: $TCC_DB"

# Detect macOS version for correct schema
MACOS_MAJOR=$(sw_vers -productVersion | cut -d. -f1)
echo "macOS major version: $MACOS_MAJOR"

if [ "$MACOS_MAJOR" -ge 13 ]; then
    # Ventura / Sonoma / Sequoia schema
    SQL="INSERT OR REPLACE INTO access(service,client,client_type,auth_value,auth_reason,auth_version,flags,last_modified) \
VALUES('kTCCServiceCamera','$PYTHON',1,2,4,1,0,CAST(strftime('%s','now') AS INTEGER));"
else
    # Monterey and earlier
    SQL="INSERT OR REPLACE INTO access(service,client,client_type,allowed,prompt_count,flags,last_modified) \
VALUES('kTCCServiceCamera','$PYTHON',1,1,0,0,CAST(strftime('%s','now') AS INTEGER));"
fi

sqlite3 "$TCC_DB" "$SQL"
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo ""
    echo "✅  Camera permission written for python3."
    echo "    Restarting tccd so it picks up the change..."
    killall -9 tccd 2>/dev/null && echo "    tccd restarted." || echo "    (tccd restart skipped — try: sudo killall -9 tccd)"
    echo ""
    echo "Now restart Pinku and the camera should work."
else
    echo ""
    echo "❌  sqlite3 failed (exit $STATUS)."
    echo "    Try:  sudo bash ~/pinku/fix_cam.sh"
fi
