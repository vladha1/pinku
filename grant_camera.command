#!/bin/bash
# Double-click this file in Finder to grant camera permission to Pinku's Python.
# It opens in Terminal (GUI session) so macOS shows the Allow/Deny popup.

cd "$(dirname "$0")"
echo "Requesting camera access for Pinku..."
.venv/bin/python3 - <<'PYEOF'
import cv2, time
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
ok = False
for i in range(40):
    ret, _ = cap.read()
    if ret:
        ok = True
        break
    time.sleep(0.2)
cap.release()
if ok:
    print("\n✅ Camera access GRANTED — you can close this window.")
else:
    print("\n❌ Camera access DENIED or no camera found.")
    print("   Go to System Settings → Privacy & Security → Camera")
    print("   and enable access for Python or Terminal.")
PYEOF
echo ""
echo "Press any key to close..."
read -n1
