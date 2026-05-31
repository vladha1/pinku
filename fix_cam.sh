#!/bin/bash
# Grants camera permission to Python by opening grant_camera.command in Terminal.app.
# Terminal.app becomes the "responsible process" — required for macOS to show the
# Allow/Deny dialog. Running Python directly from SSH won't trigger the dialog.
#
# WARNING: Do NOT run this from an SSH session and expect it to work silently.
# The camera permission dialog MUST be accepted by clicking Allow in the GUI.
#
# Run this if camera stops working:
#   bash ~/pinku/fix_cam.sh
# Then watch for the Allow dialog in VNC / physical screen and click it.

echo "Opening grant_camera.command in Terminal.app..."
echo "(Watch for the camera permission dialog and click Allow)"
open -a Terminal /Users/vivekladha/pinku/grant_camera.command
