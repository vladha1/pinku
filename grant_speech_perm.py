#!/usr/bin/env python3
"""
Run this ONCE to add NSSpeechRecognitionUsageDescription to Python.app's
Info.plist. macOS requires this key before it will show the Speech
Recognition permission dialog to the user.

Usage:
  ~/pinku/.venv/bin/python3 grant_speech_perm.py
"""
import os
import subprocess
import sys

KEY = "NSSpeechRecognitionUsageDescription"
DESC = "Pinku voice assistant uses on-device speech recognition for hands-free control"


def main() -> None:
    real = os.path.realpath(sys.executable)
    print(f"Python binary: {real}")

    if "Python.app/Contents/MacOS" not in real:
        sys.exit(
            "ERROR: not running from Python.app binary.\n"
            f"Run with: ~/pinku/.venv/bin/python3 {os.path.basename(__file__)}"
        )

    # .../Python.app/Contents/MacOS/Python  →  .../Python.app/Contents/Info.plist
    plist = os.path.join(os.path.dirname(os.path.dirname(real)), "Info.plist")
    print(f"Plist:         {plist}")

    if not os.path.exists(plist):
        sys.exit(f"ERROR: Info.plist not found at {plist}")

    # Check if key already present
    check = subprocess.run(
        ["/usr/libexec/PlistBuddy", "-c", f"Print :{KEY}", plist],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        print(f"Key already set: {check.stdout.strip()}")
        print("Restart Pinku — the permission dialog should now appear.")
        return

    # Add the key
    add = subprocess.run(
        ["/usr/libexec/PlistBuddy", "-c", f"Add :{KEY} string {DESC}", plist],
        capture_output=True, text=True,
    )
    if add.returncode == 0:
        print(f"Added {KEY} to Info.plist")
        print("Now restart Pinku:  pkill -f pinku.py")
        print("When Pinku starts, click Allow on the Speech Recognition dialog.")
    else:
        print(f"ERROR adding key: {add.stderr.strip()}")
        print(f"Try: sudo ~/pinku/.venv/bin/python3 {os.path.basename(__file__)}")


if __name__ == "__main__":
    main()
