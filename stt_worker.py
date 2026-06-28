"""
Apple STT subprocess worker — run via Python.app binary which has the
NSSpeechRecognitionUsageDescription entitlement the CLI python3 lacks.

Protocol (stdin/stdout, binary):
  in:  4-byte LE uint32 (PCM byte length) + raw 16-bit mono PCM bytes
  out: transcript text + "\n"  (empty line = no speech detected)

First line on stdout: "READY\n" if authorized, "ERROR: ...\n" if not.
"""
from __future__ import annotations
import os, struct, sys, tempfile, threading, time, wave

# ── Bootstrap venv site-packages ─────────────────────────────────────────────
_site = os.environ.get("STT_VENV_SITE", "")
if _site and _site not in sys.path:
    sys.path.insert(0, _site)

try:
    import Speech, Foundation, AppKit
except ImportError as e:
    sys.stdout.write(f"ERROR: pyobjc import failed: {e}\n")
    sys.stdout.flush()
    sys.exit(1)

_Authorized = 3   # SFSpeechRecognizerAuthorizationStatus.authorized

# ── Authorization ─────────────────────────────────────────────────────────────

def _authorize() -> bool:
    status = Speech.SFSpeechRecognizer.authorizationStatus()
    if status == _Authorized:
        return True
    if status != 0:   # Denied or Restricted
        return False

    done = threading.Event()
    granted = [False]

    def _handler(s):
        granted[0] = (s == _Authorized)
        done.set()

    Speech.SFSpeechRecognizer.requestAuthorization_(_handler)

    rl = Foundation.NSRunLoop.mainRunLoop()
    deadline = time.time() + 20.0
    while not done.is_set() and time.time() < deadline:
        rl.runUntilDate_(Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.1))

    return granted[0]


# ── Recognizer (lazy singleton) ───────────────────────────────────────────────

_rec_en = None

def _get_rec():
    global _rec_en
    if _rec_en is None:
        locale = AppKit.NSLocale.localeWithLocaleIdentifier_("en-US")
        _rec_en = Speech.SFSpeechRecognizer.alloc().initWithLocale_(locale)
    return _rec_en


# ── Transcribe ────────────────────────────────────────────────────────────────

def _transcribe(pcm: bytes, sample_rate: int = 16000) -> str:
    rec = _get_rec()
    if not rec or not rec.isAvailable():
        return ""

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)

        url = AppKit.NSURL.fileURLWithPath_(tmp.name)
        req = Speech.SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
        req.setShouldReportPartialResults_(False)

        result_box: list[str] = []
        done = threading.Event()

        def _handler(result, error):
            if result is not None:
                t = str(result.bestTranscription().formattedString())
                if t:
                    result_box.append(t)
            done.set()

        rec.recognitionTaskWithRequest_resultHandler_(req, _handler)

        rl = Foundation.NSRunLoop.currentRunLoop()
        deadline = time.time() + 5.0
        while not done.is_set() and time.time() < deadline:
            rl.runUntilDate_(Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.05))

        return result_box[0] if result_box else ""
    except Exception as e:
        sys.stderr.write(f"[stt_worker] transcribe error: {e}\n")
        return ""
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ── Main loop ─────────────────────────────────────────────────────────────────

if not _authorize():
    sys.stdout.write("ERROR: Speech Recognition not authorized — "
                     "grant in System Settings → Privacy & Security → Speech Recognition\n")
    sys.stdout.flush()
    sys.exit(1)

sys.stdout.write("READY\n")
sys.stdout.flush()

while True:
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4:
        break   # EOF — parent died
    length = struct.unpack("<I", hdr)[0]
    pcm = sys.stdin.buffer.read(length)
    if len(pcm) < length:
        break
    transcript = _transcribe(pcm)
    sys.stdout.write(transcript + "\n")
    sys.stdout.flush()
