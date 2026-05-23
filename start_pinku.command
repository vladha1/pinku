#!/bin/bash
# Double-click this in Finder (VNC) to start Pinku inside Terminal.app
# Terminal.app has camera permission → python3 inherits it as child process

cd "$(dirname "$0")"
echo "============================================"
echo "  Pinku starting (camera via Terminal.app)  "
echo "============================================"
exec .venv/bin/python3 pinku.py
