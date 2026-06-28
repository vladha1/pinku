"""
Apple on-device speech recognition via SFSpeechRecognizer.
~0.1-0.2s per utterance, fully on-device on M4, no network required.

Install: pip install pyobjc-framework-Speech
First run: macOS will prompt for Speech Recognition permission — grant it.
Falls back gracefully (returns "") if unavailable or permission denied.
"""
from __future__ import annotations
import os
import threading
import tempfile
import time as _time
import wave

_AVAILABLE  = False
_auth_ok    = None   # None = unchecked, True = granted, False = denied
_rec_en     = None
_rec_lock   = threading.Lock()


def _try_import() -> bool:
    global _AVAILABLE
    try:
        import Speech   # noqa
        import AppKit   # noqa
        _AVAILABLE = True
    except ImportError:
        print("[AppleSTT] pyobjc-framework-Speech not installed — "
              "run: pip install pyobjc-framework-Speech")
        _AVAILABLE = False
    return _AVAILABLE


_try_import()


def _get_recognizer():
    global _rec_en
    if not _AVAILABLE:
        return None
    with _rec_lock:
        if _rec_en is None:
            import Speech, AppKit
            locale = AppKit.NSLocale.localeWithLocaleIdentifier_("en-US")
            _rec_en = Speech.SFSpeechRecognizer.alloc().initWithLocale_(locale)
    return _rec_en


def request_authorization() -> bool:
    """Request permission at startup. Returns True if granted."""
    global _auth_ok
    if not _AVAILABLE:
        return False
    import Speech, Foundation
    status = Speech.SFSpeechRecognizer.authorizationStatus()
    Authorized = 3   # SFSpeechRecognizerAuthorizationStatusAuthorized

    if status == Authorized:
        _auth_ok = True
        print("[AppleSTT] Speech recognition authorized")
        return True

    if status == 0:  # NotDetermined — ask
        done = threading.Event()

        def _handler(new_status):
            global _auth_ok
            _auth_ok = (new_status == Authorized)
            done.set()

        Speech.SFSpeechRecognizer.requestAuthorization_(_handler)

        # Pump NSRunLoop — the authorization callback fires on the run loop
        # of the calling thread; without this done.wait() times out every time.
        rl       = Foundation.NSRunLoop.currentRunLoop()
        deadline = _time.time() + 20.0   # 20s for user to see and click dialog
        while not done.is_set() and _time.time() < deadline:
            rl.runUntilDate_(Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.05))

        if _auth_ok:
            print("[AppleSTT] Speech recognition authorized")
        else:
            print("[AppleSTT] Speech recognition permission denied or timed out — "
                  "grant in System Settings → Privacy & Security → Speech Recognition")
        return bool(_auth_ok)

    _auth_ok = False
    print("[AppleSTT] Speech recognition not available (status=%d)" % status)
    return False


def is_available() -> bool:
    if not _AVAILABLE:
        return False
    import Speech
    # Check TCC status directly — don't rely on _auth_ok which may not have
    # been set if the NSRunLoop callback fired after our wait timed out.
    if Speech.SFSpeechRecognizer.authorizationStatus() != 3:  # 3 = Authorized
        return False
    r = _get_recognizer()
    return r is not None and r.isAvailable()


def transcribe(pcm: bytes, sample_rate: int = 16000) -> str:
    """
    Transcribe raw 16-bit mono PCM. Returns transcript string, "" on failure.
    Caller should fall back to Gemini audio if this returns "".
    """
    if not is_available():
        return ""

    import Speech, AppKit

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)

        url     = AppKit.NSURL.fileURLWithPath_(tmp.name)
        request = Speech.SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
        request.setShouldReportPartialResults_(False)

        rec = _get_recognizer()
        if not rec or not rec.isAvailable():
            return ""

        result_box: list[str] = []
        done = threading.Event()

        def _handler(result, error):
            if result is not None:
                text = str(result.bestTranscription().formattedString())
                if text:
                    result_box.append(text)
            done.set()

        rec.recognitionTaskWithRequest_resultHandler_(request, _handler)

        # Pump the NSRunLoop while waiting — the SFSpeechRecognizer callback
        # fires on the run loop of the calling thread. Without pumping it here
        # the callback never arrives and done.wait() times out every time.
        import Foundation
        rl      = Foundation.NSRunLoop.currentRunLoop()
        deadline = _time.time() + 5.0
        while not done.is_set() and _time.time() < deadline:
            rl.runUntilDate_(Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.05))
        if not done.is_set():
            print("[AppleSTT] recognition timed out")
            return ""

        return result_box[0] if result_box else ""

    except Exception as e:
        print(f"[AppleSTT] error: {e}")
        return ""
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
