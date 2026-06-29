#!/usr/bin/env python3
"""
Run this from Terminal on the Mac to test Apple STT permission.
  ~/pinku/.venv/bin/python3 test_speech_perm.py
"""
import sys, time
print(f"Python binary: {sys.executable}")

try:
    import Speech, Foundation
except ImportError:
    print("ERROR: pyobjc-framework-Speech not installed")
    sys.exit(1)

status_names = ['NotDetermined', 'Denied', 'Restricted', 'Authorized']
before = Speech.SFSpeechRecognizer.authorizationStatus()
print(f"Current status: {status_names[before]} ({before})")

if before == 3:
    print("Already authorized — Apple STT should work.")
    sys.exit(0)

print("Requesting authorization — watch for a dialog on screen...")
done = [False]

def handler(s):
    print(f"Callback fired: {status_names[s]} ({s})")
    done[0] = True

Speech.SFSpeechRecognizer.requestAuthorization_(handler)

rl = Foundation.NSRunLoop.mainRunLoop()
deadline = time.time() + 20.0
while not done[0] and time.time() < deadline:
    rl.runUntilDate_(Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.1))

after = Speech.SFSpeechRecognizer.authorizationStatus()
print(f"Callback fired: {done[0]}")
print(f"Final status:   {status_names[after]} ({after})")

if not done[0]:
    print()
    print("No callback — the dialog likely never appeared.")
    print("Try running with the Python.app binary:")
    import os
    app_bin = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(sys.executable))),
        "Resources/Python.app/Contents/MacOS/Python"
    )
    print(f"  {app_bin} test_speech_perm.py")
