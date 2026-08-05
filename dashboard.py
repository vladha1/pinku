"""
Web dashboard — Pinku, using exact Pinky CSS face system (div-based, not SVG).
Face expressions, animations, sleeping state all identical to Pinky's index.html.
http://<mac-mini-ip>:5100
"""

import collections
import json
import os
import re as _re
import sys
import time
import threading
from flask import Flask, Response, render_template_string
from config import DASHBOARD_PORT, DASHBOARD_HOST, LOG_DIR

_app        = Flask(__name__)
_logger     = None
_start_time    = time.time()                        # unix ts — reset each restart
_start_time_str = time.strftime("%H:%M:%S")         # "07:16:42" — shown in header
_status = {
    "state": "idle", "muted": False, "model": "",
    "last_transcript": "", "last_reply": "", "speaking": False,
    "detections": [], "speaker": None, "spk_score": 0.0,
}

_clients_lock = threading.Lock()
_clients: list = []

# Server-side log ring-buffer — replayed to each new SSE client on connect
# so the log panel is populated immediately even when opened mid-session.
_log_ring: collections.deque = collections.deque(maxlen=150)
_log_ring_lock = threading.Lock()

# ── Conversation history ──────────────────────────────────────────────────────
import os as _os
_HISTORY_FILE = _os.path.join(_os.path.expanduser("~"), ".pinku_history.jsonl")
_history_lock = threading.Lock()

def record_conversation(transcript: str, reply: str,
                        lang: str = "en", source: str = ""):
    """Append a user↔Pinky exchange to the persistent history file."""
    if not transcript and not reply:
        return
    entry = {
        "ts":         time.strftime("%Y-%m-%d %H:%M:%S"),
        "transcript": transcript,
        "reply":      reply,
        "lang":       lang,
        "source":     source,
    }
    with _history_lock:
        try:
            with open(_HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Dashboard] history write error: {e}")
    _broadcast({"type": "history", **entry})

# ── SSE helpers ───────────────────────────────────────────────────────────────

def _broadcast(data: dict):
    msg = f"data: {json.dumps(data)}\n\n"
    with _clients_lock:
        dead = []
        for q in _clients:
            try:
                q.append(msg)
            except Exception:
                dead.append(q)
        for q in dead:
            _clients.remove(q)

def update_status(**kwargs):
    _status.update(kwargs)
    _broadcast({"type": "status", **_status, "start_ts": _start_time, "start_time_str": _start_time_str})

_log_file_lock = threading.Lock()
_log_file_path: str = ""
_log_fh = None

def _get_log_fh():
    """Return an open file handle for today's pinku log, rotating at midnight."""
    global _log_file_path, _log_fh
    today_path = os.path.join(LOG_DIR, f"pinku_{time.strftime('%Y-%m-%d')}.log")
    if _log_file_path != today_path:
        if _log_fh:
            _log_fh.close()
        os.makedirs(LOG_DIR, exist_ok=True)
        _log_fh = open(today_path, "a", buffering=1)
        _log_file_path = today_path
    return _log_fh


_last_log_key: tuple = ("", "")   # (level, normalised-msg) — consecutive-dedup

def _log_norm(msg: str) -> str:
    """Normalise numbers so repeated diagnostic lines with different values compare equal."""
    return _re.sub(r'[\d.]+', '#', msg[:80])

def log_message(level: str, msg: str):
    """Stream a log line to all dashboard clients and append to daily log file."""
    global _last_log_key
    key = (level, _log_norm(msg))
    if key == _last_log_key:
        return
    _last_log_key = key
    ts = time.strftime("%H:%M:%S")
    entry = {"type": "log", "level": level, "msg": msg, "ts": ts, "_t": time.time()}
    with _log_ring_lock:
        _log_ring.append(entry)
    _broadcast(entry)
    with _log_file_lock:
        try:
            _get_log_fh().write(f"{time.strftime('%Y-%m-%d')} {ts} [{level.upper():8s}] {msg}\n")
        except Exception:
            pass


_LEVEL_RE = _re.compile(r'^\[([A-Z]+)\]\s*(.*)', _re.DOTALL)

# Lines to suppress from the dashboard (still printed to the terminal).
# These are internal timing / raw-response lines that clutter the log panel.
_SUPPRESS_DASHBOARD = _re.compile(
    r'^\[LLM\] Gemini HTTP:'
    r'|^\[LLM\] Gemini audio raw:'
    r'|^\[LLM\] text.route raw:'
    r'|^\[LLM\] Gemini reply:'
    r'|^\[TTS\] known='
    r'|100%\|'                   # tqdm progress bars from mlx-whisper
)

class _StdoutTee:
    """Forward every print() line to the dashboard log stream in addition to real stdout."""
    def __init__(self, orig):
        self._orig = orig
        self._buf  = ""
        self._lock = threading.Lock()   # protects _buf against concurrent writers

    def write(self, text: str):
        self._orig.write(text)
        lines = []
        with self._lock:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if line:
                    lines.append(line)
        for line in lines:
            if _SUPPRESS_DASHBOARD.match(line):
                continue   # terminal already has it; keep dashboard clean
            try:
                m = _LEVEL_RE.match(line)
                if m:
                    log_message(m.group(1).lower(), m.group(2).strip())
                else:
                    log_message("info", line)
            except Exception:
                pass

    def flush(self):
        self._orig.flush()

    def fileno(self):
        return self._orig.fileno()

    def isatty(self):
        return False


def start_stdout_tee():
    """Call once from pinku.py to forward all print() output to the log panel."""
    if not isinstance(sys.stdout, _StdoutTee):
        sys.stdout = _StdoutTee(sys.stdout)

# ── Enrollment flag + live mic level during recording ─────────────────────────
_enrolling   = False
_enroll_rms  = 0.0   # smoothed RMS updated during enroll_record(), read by /api/enroll/level

def is_enrolling() -> bool:
    """Returns True while the Voices tab is actively recording for enrollment."""
    return _enrolling

# ── Action callbacks (registered by pinku.py) ─────────────────────────────────

_actions: dict = {}

def register_action(name: str, fn):
    """Register a callable that dashboard buttons can invoke."""
    _actions[name] = fn

# ── Routes ────────────────────────────────────────────────────────────────────

@_app.route("/")
def index():
    return render_template_string(HTML)

@_app.route("/api/stream")
def stream():
    q: list = []
    with _clients_lock:
        _clients.append(q)
    def gen():
        try:
            init = {"type": "status", **_status, "start_ts": _start_time, "start_time_str": _start_time_str}
            yield f"data: {json.dumps(init)}\n\n"
            # Replay: last 5 minutes OR last 30 entries, whichever gives more context.
            cutoff = time.time() - 300
            with _log_ring_lock:
                buf = list(_log_ring)
            recent = [e for e in buf if e.get("_t", 0) >= cutoff]
            history = recent if len(recent) >= 30 else buf[-30:]
            for entry in history:
                yield f"data: {json.dumps(entry)}\n\n"
            last_heartbeat = time.time()
            last_client_activity = time.time()
            while True:
                time.sleep(0.05)
                flushed = False
                while q:
                    yield q.pop(0)
                    flushed = True
                if flushed:
                    last_client_activity = time.time()
                now = time.time()
                # Heartbeat every 15s keeps the connection alive and detects dead clients:
                # when the write fails on a disconnected client, GeneratorExit is raised.
                if now - last_heartbeat > 15:
                    yield ": heartbeat\n\n"
                    last_heartbeat = now
                    last_client_activity = now
                # Hard timeout: if no successful write in 10 min, assume zombie and exit.
                # Prevents stale threads from accumulating and exhausting Flask's thread pool.
                if now - last_client_activity > 600:
                    break
        except GeneratorExit:
            pass
        finally:
            # Always clean up — catches both GeneratorExit and the zombie timeout break.
            with _clients_lock:
                try: _clients.remove(q)
                except ValueError: pass
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@_app.route("/api/status")
def api_status():
    from flask import jsonify
    return jsonify(_status)

@_app.route("/api/history")
def api_history():
    """Return last N conversation entries from the persistent history file."""
    from flask import request, jsonify
    limit = int(request.args.get("limit", 200))
    entries = []
    with _history_lock:
        try:
            with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            pass
        except FileNotFoundError:
            pass
    return jsonify(entries[-limit:])

@_app.route("/api/action", methods=["POST"])
def api_action():
    from flask import request, jsonify
    body = request.get_json(silent=True) or {}
    name = body.get("action", "")
    fn   = _actions.get(name)
    if fn:
        threading.Thread(target=fn, daemon=True).start()
        return jsonify({"ok": True, "action": name})
    return jsonify({"ok": False, "error": f"Unknown action: {name!r}"}), 404

@_app.route("/api/mic_level")
def mic_level_api():
    """Fast-poll endpoint for the dashboard mic meter (~12 fps)."""
    try:
        import stt as _stt
        level = _stt.get_mic_level()
    except Exception:
        level = 0.0
    return json.dumps({"level": round(level, 3)})

@_app.route("/restart")
def restart_page():
    """Simple tap-to-restart page — works from any browser including iPad Safari."""
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Restart Pinku</title>
<style>
body{background:#0a0a14;color:#e8e8f4;font-family:-apple-system,sans-serif;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0;}
button{padding:20px 40px;font-size:1.2rem;border-radius:16px;border:2px solid #fbbf24;
background:rgba(251,191,36,0.12);color:#fbbf24;cursor:pointer;}
button:disabled{opacity:0.5;}
#msg{margin-top:20px;font-size:0.95rem;color:#94a3b8;text-align:center;}
</style></head><body>
<div style="text-align:center">
  <button id="btn" onclick="doRestart()">↺ git pull + restart Pinku</button>
  <div id="msg">Pulls latest code from git, then restarts.</div>
</div>
<script>
function doRestart(){
  const btn=document.getElementById('btn');
  const msg=document.getElementById('msg');
  btn.disabled=true; btn.textContent='⏳ Restarting…';
  msg.textContent='Pulling from git…';
  fetch('/api/restart',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({token:'pinku'})})
  .then(r=>r.json()).then(d=>{
    msg.textContent=d.ok?'✓ Done — reloading in 6s…':'✗ '+d.error;
    if(d.ok) setTimeout(()=>location.href='/',6000);
    else { btn.disabled=false; btn.textContent='↺ git pull + restart Pinku'; }
  }).catch(()=>{
    msg.textContent='✓ Restarting… reloading in 6s';
    setTimeout(()=>location.href='/',6000);
  });
}
</script></body></html>"""


@_app.route("/api/restart", methods=["POST"])
def api_restart():
    """
    git pull + restart Pinku.  Kills the current process after a short
    delay so the HTTP response reaches the browser first.
    Pinku must be launched by a wrapper that auto-restarts on exit
    (start_pinku.command uses a simple loop for this).
    """
    import subprocess, sys, os, threading
    from flask import jsonify, request

    token = (request.get_json(silent=True) or {}).get("token", "")
    if token != "pinku":
        return jsonify({"ok": False, "error": "bad token"}), 403

    def _do_restart():
        import time
        time.sleep(1.0)   # let the HTTP response fly first
        try:
            subprocess.run(
                ["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                 "pull", "--ff-only"],
                timeout=30,
            )
        except Exception as e:
            print(f"[Restart] git pull failed: {e}")
        # Also kill music_app.py so start_pinku.command restarts it with new code
        subprocess.run(["pkill", "-f", "music_app.py"], capture_output=True)
        os.kill(os.getpid(), 15)   # SIGTERM → triggers the finally block in main()

    threading.Thread(target=_do_restart, daemon=True).start()
    return jsonify({"ok": True, "msg": "Pulling & restarting in ~1 s…"})


@_app.route("/api/enroll/pause", methods=["POST"])
def enroll_pause():
    global _enrolling
    _enrolling = True
    return json.dumps({"ok": True})

@_app.route("/api/enroll/resume", methods=["POST"])
def enroll_resume():
    global _enrolling
    _enrolling = False
    return json.dumps({"ok": True})

@_app.route("/api/enroll/level")
def enroll_level():
    return json.dumps({"level": round(_enroll_rms * 20.0, 3), "active": _enrolling})

@_app.route("/api/enroll/profiles")
def enroll_profiles():
    try:
        from config import SPEAKER_PROFILES_DIR
        profiles = []
        if os.path.isdir(SPEAKER_PROFILES_DIR):
            for fn in sorted(os.listdir(SPEAKER_PROFILES_DIR)):
                if fn.endswith(".npy"):
                    name = fn[:-4]
                    count_path = os.path.join(SPEAKER_PROFILES_DIR, f"{name}.clips")
                    clips = 1
                    try:
                        clips = int(open(count_path).read().strip())
                    except Exception:
                        pass
                    profiles.append({"name": name, "clips": clips})
        return json.dumps({"profiles": profiles})
    except Exception as e:
        return json.dumps({"profiles": [], "error": str(e)})

@_app.route("/api/enroll/record", methods=["POST"])
def enroll_record():
    """Record N seconds from Pinku's own mic, create embedding, save profile."""
    from flask import request as _req
    global _enrolling
    body    = _req.get_json(silent=True) or {}
    name    = body.get("name", "").strip()
    seconds = min(int(body.get("seconds", 15)), 30)
    if not name or not name.replace("_", "").replace("-", "").replace(" ", "").isalnum():
        return json.dumps({"ok": False, "error": "Invalid name (letters/numbers/_ only)"}), 400
    _enrolling = True
    try:
        import numpy as np, pyaudio
        from config import MIC_DEVICE_INDEX, SPEAKER_PROFILES_DIR
        RATE  = 16000
        CHUNK = 1024
        p = pyaudio.PyAudio()
        kw = dict(format=pyaudio.paInt16, channels=1, rate=RATE,
                  input=True, frames_per_buffer=CHUNK)
        if MIC_DEVICE_INDEX >= 0:
            kw["input_device_index"] = MIC_DEVICE_INDEX
        stream = p.open(**kw)
        global _enroll_rms
        frames = []
        for _ in range(int(RATE / CHUNK * seconds)):
            chunk = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(chunk)
            arr  = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            rms  = float(np.sqrt(np.mean(arr ** 2)))
            _enroll_rms = _enroll_rms * 0.6 + rms * 0.4   # smoothed EWA
        _enroll_rms = 0.0
        stream.stop_stream(); stream.close(); p.terminate()
        audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        from resemblyzer import VoiceEncoder, preprocess_wav
        enc       = VoiceEncoder()
        wav_proc  = preprocess_wav(audio, source_sr=RATE)
        new_emb   = enc.embed_utterance(wav_proc)
        new_emb   = new_emb / np.linalg.norm(new_emb)

        safe_name  = name.replace(" ", "_")
        os.makedirs(SPEAKER_PROFILES_DIR, exist_ok=True)
        profile_path = os.path.join(SPEAKER_PROFILES_DIR, f"{safe_name}.npy")
        count_path   = os.path.join(SPEAKER_PROFILES_DIR, f"{safe_name}.clips")

        # If a profile already exists, blend new clip into it (EWA, weight 30% new).
        # Each additional recording shifts the embedding toward the new sample,
        # building a more representative average without needing to store all clips.
        clip_count = 1
        if os.path.exists(profile_path):
            try:
                old_emb    = np.load(profile_path).astype(np.float32)
                old_count  = int(open(count_path).read().strip()) if os.path.exists(count_path) else 1
                blended    = 0.7 * old_emb + 0.3 * new_emb
                new_emb    = blended / np.linalg.norm(blended)
                clip_count = old_count + 1
            except Exception:
                clip_count = 1   # corrupt old profile — overwrite

        np.save(profile_path, new_emb)
        with open(count_path, "w") as _f:
            _f.write(str(clip_count))

        try:
            import speaker_id as _sid
            _sid.load()
        except Exception:
            pass
        msg = (f"SpeakerID: enrolled '{safe_name}' — clip {clip_count} blended in"
               if clip_count > 1 else f"SpeakerID: enrolled '{safe_name}' (clip 1)")
        log_message("info", msg)
        return json.dumps({"ok": True, "name": safe_name, "clips": clip_count})
    except Exception as e:
        log_message("error", f"Enroll record error: {e}")
        return json.dumps({"ok": False, "error": str(e)}), 500
    finally:
        _enrolling = False

@_app.route("/api/enroll/delete/<name>", methods=["DELETE"])
def enroll_delete(name):
    try:
        from config import SPEAKER_PROFILES_DIR
        path = os.path.join(SPEAKER_PROFILES_DIR, f"{name}.npy")
        if not os.path.exists(path):
            return json.dumps({"ok": False, "error": "Profile not found"}), 404
        os.remove(path)
        clips_path = os.path.join(SPEAKER_PROFILES_DIR, f"{name}.clips")
        if os.path.exists(clips_path):
            os.remove(clips_path)
        try:
            import speaker_id as _sid
            _sid.load()
        except Exception:
            pass
        log_message("info", f"SpeakerID: deleted profile '{name}'")
        return json.dumps({"ok": True})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}), 500

# ── Market + Weather data (cached, fetched from Yahoo Finance / wttr.in) ─────

_market_cache: dict = {"data": None, "ts": 0.0}
_market_lock  = threading.Lock()

@_app.route("/api/market")
def api_market():
    with _market_lock:
        if time.time() - _market_cache["ts"] < 30 and _market_cache["data"]:
            return json.dumps(_market_cache["data"]), 200, {"Content-Type": "application/json"}
    import urllib.request as _ur

    def _yf(symbol):
        try:
            url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                   f"?interval=1m&range=1d")
            req = _ur.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "Accept": "application/json",
            })
            with _ur.urlopen(req, timeout=8) as r:
                d = json.loads(r.read())
            meta  = d["chart"]["result"][0]["meta"]
            price = float(meta.get("regularMarketPrice") or 0)
            prev  = float(meta.get("previousClose") or meta.get("chartPreviousClose") or price)
            chg   = price - prev
            pct   = (chg / prev * 100) if prev else 0
            return {"price": round(price, 2), "change": round(chg, 2), "pct": round(pct, 2)}
        except Exception as e:
            return {"price": None, "change": 0, "pct": 0, "err": str(e)}

    data = {"nifty": _yf("%5ENSEI"), "btc": _yf("BTC-USD"), "ts": time.time()}
    with _market_lock:
        _market_cache["data"] = data
        _market_cache["ts"]   = time.time()
    return json.dumps(data), 200, {"Content-Type": "application/json", "Cache-Control": "no-cache"}


_weather_cache: dict = {"data": None, "ts": 0.0}

@_app.route("/api/weather")
def api_weather():
    if time.time() - _weather_cache["ts"] < 300 and _weather_cache["data"]:
        return json.dumps(_weather_cache["data"]), 200, {"Content-Type": "application/json"}
    import urllib.request as _ur
    try:
        req = _ur.Request("https://wttr.in/Gurugram?format=%t+%C&m",
                          headers={"User-Agent": "curl/7.79"})
        with _ur.urlopen(req, timeout=6) as r:
            text = r.read().decode().strip()
        parts = text.split(" ", 1)
        data = {"temp": parts[0], "cond": parts[1] if len(parts) > 1 else "", "ok": True}
    except Exception as e:
        data = {"temp": "—", "cond": "", "ok": False}
    _weather_cache["data"] = data
    _weather_cache["ts"]   = time.time()
    return json.dumps(data), 200, {"Content-Type": "application/json", "Cache-Control": "no-cache"}


# ── Start ─────────────────────────────────────────────────────────────────────

def start(logger=None, port=DASHBOARD_PORT):
    global _logger
    _logger = logger

    # Silence Flask/werkzeug per-request logs
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    t = threading.Thread(
        target=lambda: _app.run(host=DASHBOARD_HOST, port=port,
                                debug=False, use_reloader=False, threaded=True),
        daemon=True,
    )
    t.start()
    print(f"[Dashboard] http://{DASHBOARD_HOST}:{port}")

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black">
<title>Pinku</title>
<style>
/* iOS 9 compatible — no clamp(), no max(), no inset, no gap, no env(),
   all webkit-prefixed, fetch() replaced with XHR                        */

*, *::before, *::after {
  -webkit-box-sizing: border-box;
  box-sizing: border-box;
  margin: 0; padding: 0;
}

html, body {
  height: 100%; width: 100%; overflow: hidden;
  background: #05050d;
  color: #dde4f5;
  font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -webkit-text-size-adjust: 100%;
}

/* Two full-screen views — crossfade between them.
   Use top/left/right/bottom instead of inset shorthand (iOS 9 compat). */
.view {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  display: -webkit-flex; display: flex;
  -webkit-flex-direction: column; flex-direction: column;
  -webkit-align-items: center; align-items: center;
  -webkit-justify-content: center; justify-content: center;
  /* leave 120px at bottom for the fixed controls bar */
  padding: 28px 24px 120px;
  -webkit-transition: opacity 0.55s ease;
  transition: opacity 0.55s ease;
}

#clock-view  { opacity: 1; }
#think-view  { opacity: 0; pointer-events: none; }
#conv-view   { opacity: 0; pointer-events: none; }

/* thinking: hide clock, show ring */
body.is-thinking #clock-view { opacity: 0; pointer-events: none; }
body.is-thinking #think-view { opacity: 1; pointer-events: auto; }

/* conversation: takes priority over both */
body.has-conv #clock-view { opacity: 0; pointer-events: none; }
body.has-conv #think-view { opacity: 0; pointer-events: none; }
body.has-conv #conv-view  { opacity: 1; pointer-events: auto; }

/* ── Clock ── */
#time-row {
  display: -webkit-flex; display: flex;
  -webkit-align-items: flex-start; align-items: flex-start;
  line-height: 1;
}

/* vmin: min(vw,vh) — supported iOS 6+. On iPad mini 768px min: 22vmin = ~169px */
#time {
  font-size: 22vmin;
  font-weight: 100;
  letter-spacing: -0.025em;
  font-variant-numeric: tabular-nums;
  color: #f472b6;
  line-height: 1;
}

#ampm {
  font-size: 3.5vmin;
  font-weight: 300;
  color: #f59e0b;
  margin-left: 8px;
  margin-top: 6px;
  letter-spacing: 0.07em;
}

#date-str {
  font-size: 13vmin;
  font-weight: 100;
  color: #60a5fa;
  letter-spacing: 0.04em;
  margin-top: 6px;
  text-align: center;
}

/* ── Conversation ── */
#conv-view {
  -webkit-justify-content: space-between;
  justify-content: space-between;
  padding-top: 40px;
}

#spk-label {
  font-size: 2.5vmin;
  font-weight: 300;
  color: rgba(245,158,11,0.55);
  letter-spacing: 0.06em;
  text-align: right;
  padding: 3px 2px 10px;
  width: 100%;
  min-height: 1em;
}

#q-text {
  width: 100%;
  font-size: 5vmin;
  font-weight: 200;
  font-style: italic;
  color: rgba(226,232,248,0.62);
  line-height: 1.4;
  text-align: center;
  padding-bottom: 18px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}

#r-text {
  -webkit-flex: 1; flex: 1;
  width: 100%;
  display: -webkit-flex; display: flex;
  -webkit-align-items: center; align-items: center;
  -webkit-justify-content: center; justify-content: center;
  font-size: 6.2vmin;
  font-weight: 200;
  color: #e2e8f8;
  line-height: 1.55;
  text-align: center;
  white-space: pre-wrap;
  overflow-y: auto;
  padding: 16px 0;
}

/* ── Fixed bottom bar ── */
#bottom-bar {
  position: fixed;
  bottom: 0; left: 0; right: 0;
  display: -webkit-flex; display: flex;
  -webkit-flex-direction: column; flex-direction: column;
  -webkit-align-items: center; align-items: center;
  padding: 0 20px 18px;
  z-index: 20;
}

/* ── Mic level bar (always visible above conversation-progress bar) ── */
#mic-bar {
  width: 100%;
  height: 3px;
  background: rgba(255,255,255,0.04);
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 8px;
  -webkit-flex-shrink: 0; flex-shrink: 0;
}
#mic-fill {
  height: 100%;
  width: 0%;
  border-radius: 2px;
  background: rgba(255,255,255,0.12);
  -webkit-transition: width 0.13s linear, background 0.35s;
  transition: width 0.13s linear, background 0.35s;
}

/* READY-state dot: green when mic active (volatile), orange when silent */
body.mic-live:not(.s-awake):not(.s-thinking):not(.s-speaking):not(.s-muted) #sdot {
  background: rgba(74,222,128,0.9);
  box-shadow: 0 0 0 3px rgba(74,222,128,0.18);
}
body.mic-quiet:not(.s-awake):not(.s-thinking):not(.s-speaking):not(.s-muted) #sdot {
  background: rgba(251,146,60,0.85);
  box-shadow: 0 0 0 3px rgba(251,146,60,0.18);
}

#cbar {
  width: 100%;
  height: 2px;
  background: rgba(255,255,255,0.06);
  border-radius: 2px;
  overflow: hidden;
  opacity: 0;
  margin-bottom: 10px;
  -webkit-transition: opacity 0.4s;
  transition: opacity 0.4s;
}
body.has-conv #cbar { opacity: 1; }

#cbar-fill {
  height: 100%;
  width: 100%;
  background: rgba(226,232,248,0.22);
  border-radius: 2px;
  -webkit-transition: width linear;
  transition: width linear;
}

#status-pill {
  display: -webkit-inline-flex; display: inline-flex;
  -webkit-align-items: center; align-items: center;
  padding: 10px 24px;
  border-radius: 100px;
  border: 1px solid rgba(255,255,255,0.07);
  background: rgba(255,255,255,0.025);
  margin-bottom: 10px;
  -webkit-transition: border-color 0.4s;
  transition: border-color 0.4s;
}

#sdot {
  width: 11px; height: 11px;
  border-radius: 50%;
  background: rgba(50,55,90,0.9);
  -webkit-flex-shrink: 0; flex-shrink: 0;
  margin-right: 9px;
  -webkit-transition: background 0.4s, box-shadow 0.4s;
  transition: background 0.4s, box-shadow 0.4s;
}

#slabel {
  font-size: 2.2vmin;
  font-weight: 700;
  letter-spacing: 0.17em;
  color: rgba(226,232,248,0.28);
  -webkit-transition: color 0.4s;
  transition: color 0.4s;
}

#sse-ping {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: rgba(74,222,128,0.8);
  -webkit-flex-shrink: 0; flex-shrink: 0;
  margin-left: 10px;
  opacity: 0;
  -webkit-transition: opacity 0.35s;
  transition: opacity 0.35s;
}

/* status state classes on body */
body.s-awake    #sdot  { background: #22c55e; box-shadow: 0 0 0 4px rgba(34,197,94,0.2); }
body.s-awake    #slabel { color: #22c55e; }
body.s-awake    #status-pill { border-color: rgba(34,197,94,0.2); }

body.s-thinking #sdot  { background: #f59e0b; box-shadow: 0 0 0 4px rgba(245,158,11,0.2);
  -webkit-animation: blink 0.8s ease-in-out infinite;
  animation: blink 0.8s ease-in-out infinite; }
body.s-thinking #slabel { color: #f59e0b; }
body.s-thinking #status-pill { border-color: rgba(245,158,11,0.2); }

body.s-speaking #sdot  { background: #818cf8; box-shadow: 0 0 0 4px rgba(129,140,248,0.2);
  -webkit-animation: swell 1.7s ease-in-out infinite;
  animation: swell 1.7s ease-in-out infinite; }
body.s-speaking #slabel { color: #818cf8; }
body.s-speaking #status-pill { border-color: rgba(129,140,248,0.2); }

body.s-muted #sdot    { background: #f87171; box-shadow: 0 0 0 3px rgba(248,113,113,0.15); }
body.s-muted #slabel  { color: #f87171; }
body.s-muted #status-pill { border-color: rgba(248,113,113,0.2); }

@-webkit-keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
@keyframes         blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
@-webkit-keyframes swell {
  0%, 100% { box-shadow: 0 0 0 3px rgba(129,140,248,0.15); }
  50%       { box-shadow: 0 0 0 9px rgba(129,140,248,0.28); }
}
@keyframes swell {
  0%, 100% { box-shadow: 0 0 0 3px rgba(129,140,248,0.15); }
  50%       { box-shadow: 0 0 0 9px rgba(129,140,248,0.28); }
}

/* ── Working view (processing): live transcript + action indicator ── */
#think-view {
  -webkit-flex-direction: column; flex-direction: column;
  -webkit-justify-content: center; justify-content: center;
  -webkit-align-items: center; align-items: center;
  padding: 0 6vmin;
}

/* What Pinky heard — the star of the view, big and bright */
#think-heard {
  width: 100%;
  font-size: 6.4vmin;
  font-weight: 200;
  color: #e2e8f8;
  line-height: 1.38;
  text-align: center;
  max-height: 56vh;
  overflow-y: auto;
}
#think-heard:empty { display: none; }

/* Action indicator row — pulsing dot + label ("Thinking…") */
#think-status {
  display: -webkit-flex; display: flex;
  -webkit-align-items: center; align-items: center;
  gap: 2.2vmin;
  margin-top: 5.5vmin;
}
#think-dot {
  width: 3vmin; height: 3vmin;
  border-radius: 50%;
  background: #f59e0b;
  box-shadow: 0 0 3.4vmin rgba(245,158,11,0.7);
  -webkit-animation: tpulse 1.1s ease-in-out infinite;
  animation: tpulse 1.1s ease-in-out infinite;
}
@-webkit-keyframes tpulse { 0%,100% { opacity:1; -webkit-transform:scale(1); } 50% { opacity:0.3; -webkit-transform:scale(0.65); } }
@keyframes         tpulse { 0%,100% { opacity:1; transform:scale(1); }         50% { opacity:0.3; transform:scale(0.65); } }
#think-label {
  font-size: 4.6vmin;
  font-weight: 300;
  color: rgba(245,158,11,0.92);
  letter-spacing: 0.1em;
  text-align: center;
}

/* controls */
#controls {
  display: -webkit-flex; display: flex;
}

.cb {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  color: rgba(226,232,248,0.55);
  font-size: 4.5vmin;
  width: 12vmin;
  height: 12vmin;
  display: -webkit-inline-flex; display: inline-flex;
  -webkit-align-items: center; align-items: center;
  -webkit-justify-content: center; justify-content: center;
  border-radius: 50%;
  cursor: pointer;
  -webkit-tap-highlight-color: transparent;
  -webkit-transition: background 0.18s, color 0.18s, border-color 0.18s;
  transition: background 0.18s, color 0.18s, border-color 0.18s;
  margin-right: 6vmin;
}
.cb:last-child { margin-right: 0; }
.cb:active { background: rgba(255,255,255,0.12); color: rgba(226,232,248,0.9); }
.cb.on     { border-color: rgba(248,113,113,0.5); color: rgba(248,113,113,0.85); }

/* ── Voices slide-up ── */
#voices-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(5,5,13,0.88);
  display: -webkit-flex; display: flex;
  -webkit-align-items: flex-end; align-items: flex-end;
  opacity: 0; pointer-events: none;
  -webkit-transition: opacity 0.3s;
  transition: opacity 0.3s;
  z-index: 50;
}
#voices-overlay.open { opacity: 1; pointer-events: auto; }

#voices-panel {
  width: 100%;
  background: #0b0b1d;
  border-top: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px 20px 0 0;
  padding: 24px 24px 40px;
  -webkit-transform: translateY(100%);
  transform: translateY(100%);
  -webkit-transition: -webkit-transform 0.4s cubic-bezier(.4,0,.2,1);
  transition: transform 0.4s cubic-bezier(.4,0,.2,1);
  max-height: 80%;
  overflow-y: auto;
  -webkit-overflow-scrolling: touch;
}
#voices-overlay.open #voices-panel {
  -webkit-transform: none;
  transform: none;
}

#voices-panel h3 {
  font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
  color: rgba(226,232,248,0.38); margin-bottom: 18px;
}
.erow { display: -webkit-flex; display: flex; margin-bottom: 12px; }
.erow input {
  -webkit-flex: 1; flex: 1;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 10px; color: #dde4f5;
  font-size: 15px; padding: 11px 14px;
  font-family: inherit; outline: none;
  margin-right: 8px;
}
.erow input::-webkit-input-placeholder { color: rgba(226,232,248,0.25); }
.erow button {
  background: rgba(245,158,11,0.12);
  border: 1px solid rgba(245,158,11,0.28);
  color: #f59e0b; font-size: 13px; font-weight: 700;
  padding: 11px 16px; border-radius: 10px;
  cursor: pointer; font-family: inherit; white-space: nowrap;
}
#mtrack { height: 3px; background: rgba(255,255,255,0.08); border-radius: 3px; margin-bottom: 10px; overflow: hidden; }
#mfill  { height: 100%; width: 0; background: #22c55e; border-radius: 3px; -webkit-transition: width 0.08s; transition: width 0.08s; }
#emsg   { font-size: 12px; color: rgba(226,232,248,0.38); min-height: 16px; margin-bottom: 14px; }
#plist  { display: -webkit-flex; display: flex; -webkit-flex-wrap: wrap; flex-wrap: wrap; }
.chip {
  display: -webkit-inline-flex; display: inline-flex;
  -webkit-align-items: center; align-items: center;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 20px; padding: 5px 8px 5px 12px;
  font-size: 12px; color: rgba(226,232,248,0.62);
  margin-right: 8px; margin-bottom: 8px;
}
.chip button {
  background: none; border: none;
  color: rgba(226,232,248,0.28);
  font-size: 11px; cursor: pointer;
  padding: 0 2px; line-height: 1; margin-left: 4px;
}

/* ── Log slide-up ── */
#log-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(5,5,13,0.88);
  display: -webkit-flex; display: flex;
  -webkit-align-items: flex-end; align-items: flex-end;
  opacity: 0; pointer-events: none;
  -webkit-transition: opacity 0.3s;
  transition: opacity 0.3s;
  z-index: 50;
}
#log-overlay.open { opacity: 1; pointer-events: auto; }

#log-panel {
  width: 100%;
  background: #06060f;
  border-top: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px 20px 0 0;
  padding: 0 0 34px;
  -webkit-transform: translateY(100%);
  transform: translateY(100%);
  -webkit-transition: -webkit-transform 0.4s cubic-bezier(.4,0,.2,1);
  transition: transform 0.4s cubic-bezier(.4,0,.2,1);
  height: 68%;
  display: -webkit-flex; display: flex;
  -webkit-flex-direction: column; flex-direction: column;
}
#log-overlay.open #log-panel {
  -webkit-transform: none;
  transform: none;
}

#log-header {
  display: -webkit-flex; display: flex;
  -webkit-align-items: center; align-items: center;
  -webkit-justify-content: space-between; justify-content: space-between;
  padding: 16px 20px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  -webkit-flex-shrink: 0; flex-shrink: 0;
}
#log-header h3 {
  font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
  color: rgba(226,232,248,0.38); margin: 0;
}
#log-close {
  background: none; border: none;
  color: rgba(226,232,248,0.38); font-size: 20px;
  cursor: pointer; padding: 2px 4px; line-height: 1;
  font-family: inherit;
}

#log-list {
  -webkit-flex: 1; flex: 1;
  overflow-y: auto;
  -webkit-overflow-scrolling: touch;
  padding: 10px 16px 8px;
  font-family: 'SF Mono', 'Menlo', 'Monaco', Consolas, monospace;
  font-size: 11px;
  line-height: 1.7;
}
.log-line { white-space: pre-wrap; word-break: break-all; }
.log-info   { color: rgba(226,232,248,0.35); }
.log-stt    { color: rgba(226,232,248,0.35); }
.log-tts    { color: rgba(226,232,248,0.35); }
.log-llm    { color: rgba(226,232,248,0.35); }
.log-source { color: rgba(134,239,172,0.6); }
.log-wake   { color: #fbbf24; font-weight: 600; }
.log-user   { color: #93c5fd; }
.log-pinku  { color: #6ee7b7; }
.log-warn   { color: #fb923c; }
.log-error  { color: #f87171; font-weight: 600; }

/* ── Detection feed (inline, above bottom bar) ──
   Live "what did the mic hear and how was it handled" stream: raw transcripts
   and pre-processing verdicts (dropped / no wake word / wake / unknown voice). */
#feed {
  position: fixed;
  left: 0; right: 0;
  /* Sit clear above the whole bottom bar (controls are 12vmin tall). */
  bottom: calc(12vmin + 66px);
  max-height: 132px;
  z-index: 15;
  pointer-events: none;
  padding: 0 16px;
  display: -webkit-flex; display: flex;
  -webkit-flex-direction: column; flex-direction: column;
  -webkit-justify-content: flex-end; justify-content: flex-end;
  overflow: hidden;
  -webkit-mask-image: linear-gradient(to bottom, transparent 0, #000 30px);
          mask-image: linear-gradient(to bottom, transparent 0, #000 30px);
}
.fd {
  font-family: 'SF Mono', 'Menlo', 'Monaco', Consolas, monospace;
  font-size: 13px; line-height: 1.5;
  padding: 3px 11px; margin-top: 4px;
  border-radius: 7px;
  background: rgba(10,12,24,0.55);
  border-left: 2px solid rgba(148,163,184,0.35);
  color: rgba(226,232,248,0.62);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  -webkit-animation: fdin 0.22s ease-out; animation: fdin 0.22s ease-out;
}
@-webkit-keyframes fdin { from { opacity: 0; -webkit-transform: translateY(6px);} to { opacity: 1; -webkit-transform: none;} }
@keyframes fdin { from { opacity: 0; transform: translateY(6px);} to { opacity: 1; transform: none;} }
.fd .fq { color: rgba(226,232,248,0.92); }   /* the transcript text itself */
.fd .fv { opacity: 0.75; }                    /* verdict / how-handled clause */
/* verdict-coloured left border + accent */
.fd-heard { border-left-color: rgba(147,197,253,0.75); }   /* transcript captured */
.fd-wake  { border-left-color: #fbbf24; color: rgba(251,191,36,0.9); }
.fd-drop  { border-left-color: rgba(148,163,184,0.4); color: rgba(226,232,248,0.4); }
.fd-act   { border-left-color: rgba(110,231,183,0.8); color: rgba(167,243,208,0.85); }
.fd-spk   { border-left-color: rgba(196,181,253,0.7); }
</style>
</head>
<body>

<!-- clock -->
<div class="view" id="clock-view">
  <div id="time-row">
    <span id="time">12:00</span>
    <span id="ampm">AM</span>
  </div>
  <div id="date-str">1 January, Mon</div>
</div>

<!-- working view (shown during processing) — live transcript + action indicator -->
<div class="view" id="think-view">
  <div id="think-heard"></div>
  <div id="think-status">
    <span id="think-dot"></span>
    <span id="think-label"></span>
  </div>
</div>

<!-- conversation (crossfades over clock) -->
<div class="view" id="conv-view">
  <div id="q-text"></div>
  <div id="spk-label"></div>
  <div id="r-text"></div>
</div>

<!-- always-on-top bottom bar -->
<div id="bottom-bar">
  <div id="mic-bar"><div id="mic-fill"></div></div>
  <div id="cbar"><div id="cbar-fill"></div></div>
  <div id="status-pill">
    <div id="sdot"></div>
    <span id="slabel">READY</span>
    <span id="sse-ping"></span>
  </div>
  <div id="controls">
    <button class="cb" onclick="act('wake')">☀️</button>
    <button class="cb" onclick="act('sleep')">🌙</button>
    <button class="cb" id="mute-btn" onclick="act('mute_toggle')">🔊</button>
    <button class="cb" onclick="openVoices()">🎙</button>
    <button class="cb" onclick="openLog()">📋</button>
  </div>
</div>

<!-- detection feed — live transcripts + how each was handled -->
<div id="feed"></div>

<!-- log panel -->
<div id="log-overlay" onclick="if(event.target===this)closeLog()">
  <div id="log-panel">
    <div id="log-header">
      <h3>LOGS</h3>
      <button id="log-close" onclick="closeLog()">✕</button>
    </div>
    <div id="log-list"></div>
  </div>
</div>

<!-- voices panel -->
<div id="voices-overlay" onclick="if(event.target===this)closeVoices()">
  <div id="voices-panel">
    <h3>VOICE ENROLLMENT</h3>
    <div class="erow">
      <input id="ename" type="text" placeholder="Name" autocomplete="off" autocorrect="off" autocapitalize="words">
      <button id="ebtn" onclick="startEnroll()">Record 15s</button>
    </div>
    <div id="mtrack"><div id="mfill"></div></div>
    <div id="emsg"></div>
    <div id="plist"></div>
  </div>
</div>

<script>
// ── XHR helper — replaces fetch() which is absent on iOS 9 ───────────────────
function xhr(method, url, body, cb) {
  var req = new XMLHttpRequest();
  req.open(method, url, true);
  if (body !== null) req.setRequestHeader('Content-Type', 'application/json');
  req.onreadystatechange = function() {
    if (req.readyState === 4 && cb) {
      try { cb(null, JSON.parse(req.responseText)); }
      catch(e) { cb(null, {}); }
    }
  };
  req.onerror = function() { if (cb) cb(new Error('xhr error'), null); };
  req.send(body ? JSON.stringify(body) : null);
}

// ── Clock ─────────────────────────────────────────────────────────────────────
var _lastMin = -1;
var DAYS   = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
var MONTHS = ['January','February','March','April','May','June','July',
              'August','September','October','November','December'];
function tick() {
  var n = new Date(), h = n.getHours(), m = n.getMinutes();
  if (m === _lastMin) return;
  _lastMin = m;
  document.getElementById('ampm').textContent = h >= 12 ? 'PM' : 'AM';
  h = h % 12 || 12;
  document.getElementById('time').textContent = h + ':' + (m < 10 ? '0' : '') + m;
  document.getElementById('date-str').textContent = n.getDate() + ' ' + MONTHS[n.getMonth()] + ', ' + DAYS[n.getDay()].slice(0,3);
}
tick();
setInterval(tick, 4000);

// ── Status ────────────────────────────────────────────────────────────────────
var _THINK_PHRASES = ['On it', 'Got it', 'One moment', 'Thinking'];
var _prevProcessing = false;

function setStatus(s) {
  var cls, lbl;
  if (s.muted)                       { cls = 's-muted';   lbl = 'MUTED'; }
  else if (s.speaking)               { cls = 's-speaking'; lbl = 'SPEAKING'; }
  else if (s.state === 'processing') { cls = 's-thinking'; lbl = 'THINKING'; }
  else if (s.state === 'awake')      { cls = 's-awake';   lbl = 'LISTENING'; }
  else                               { cls = '';           lbl = 'READY'; }
  // Add is-thinking view class only during processing
  var viewCls = (s.state === 'processing') ? ' is-thinking' : '';
  // Preserve has-conv (showConv/clearTimer) and mic-live/mic-quiet (_pollMic)
  var bc = document.body.className;
  var hasConv  = bc.indexOf('has-conv')  >= 0 ? ' has-conv'  : '';
  var micLive  = bc.indexOf('mic-live')  >= 0 ? ' mic-live'  : '';
  var micQuiet = bc.indexOf('mic-quiet') >= 0 ? ' mic-quiet' : '';
  document.body.className = (cls + viewCls + hasConv + micLive + micQuiet).replace(/\s+/g,' ').trim();
  document.getElementById('slabel').textContent = lbl;
  var muteBtn = document.getElementById('mute-btn');
  muteBtn.className = 'cb' + (s.muted ? ' on' : '');
  muteBtn.textContent = s.muted ? '🔇' : '🔊';

  // Track speaker identity for conv-view attribution
  if (s.state === 'processing') {
    if (s.speaker !== undefined) _lastSpeaker = s.speaker;
    if (s.spk_score !== undefined && s.spk_score > 0) _lastSpkScore = s.spk_score;
  }

  // Working view — show what Pinky heard live, plus a stable action label.
  var nowProcessing = (s.state === 'processing');
  if (nowProcessing) {
    var heardEl = document.getElementById('think-heard');
    if (heardEl) {
      // Clear stale text on entry; fill as soon as the transcript arrives.
      if (!_prevProcessing) heardEl.textContent = '';
      if (s.last_transcript) heardEl.textContent = '“' + s.last_transcript + '”';
    }
    // Set the action label once per processing burst to avoid phrase flicker.
    if (!_prevProcessing) {
      var phrase = _THINK_PHRASES[Math.floor(Math.random() * _THINK_PHRASES.length)];
      document.getElementById('think-label').textContent =
        s.speaker ? (phrase + ', ' + s.speaker) : (phrase + '…');
    }
  }
  _prevProcessing = nowProcessing;
}

// ── Conversation ──────────────────────────────────────────────────────────────
var _clearTimer = null;
var _isSpeaking = false;
var AFTER_SPEECH_MS = 5000;   // linger after TTS ends before clearing
var _lastSpeaker = null;
var _lastSpkScore = 0;

function _clearConv() {
  _clearTimer = null;
  document.body.className = document.body.className.replace(/\bhas-conv\b/g, '').replace(/\s+/g,' ').trim();
}

function showConv(q, r) {
  if (!r && !q) return;
  if (_clearTimer) { clearTimeout(_clearTimer); _clearTimer = null; }
  document.getElementById('q-text').textContent = q ? '”' + q + '”' : '';
  document.getElementById('r-text').textContent = r || '';
  var spkEl = document.getElementById('spk-label');
  if (spkEl) {
    var pct = _lastSpkScore > 0 ? Math.round(_lastSpkScore * 100) + '%' : '';
    var name = _lastSpeaker ? _lastSpeaker : (_lastSpkScore > 0 ? '?' : '');
    spkEl.textContent = (name && pct) ? name + ' · ' + pct : (name || pct);
  }
  document.body.className = (document.body.className + ' has-conv').replace(/\s+/g,' ').trim();

  var fill = document.getElementById('cbar-fill');
  if (fill) fill.style.cssText = 'transition:none;width:100%';
  // Don't start drain animation here — it fires when speaking=False arrives.
  // 90s safety-net clears if the server never sends speaking=False.
  _clearTimer = setTimeout(_clearConv, 90000);
}

// ── SSE ───────────────────────────────────────────────────────────────────────
var _serverTs = null;
var _sseCount = 0;

function _sseFlash() {
  _sseCount++;
  var p = document.getElementById('sse-ping');
  if (!p) return;
  p.style.opacity = '1';
  setTimeout(function() { p.style.opacity = '0'; }, 350);
}

function connect() {
  var es = new EventSource('/api/stream');
  es.onmessage = function(evt) {
    _sseFlash();
    var d;
    try { d = JSON.parse(evt.data); } catch(e) { return; }

    if (d.type === 'status') {
      // Auto-reload when server restarts after a deploy (start_ts changes)
      if (_serverTs === null) {
        _serverTs = d.start_ts;
      } else if (d.start_ts !== _serverTs) {
        location.reload();
        return;
      }
      var wasSpeaking = _isSpeaking;
      _isSpeaking = !!d.speaking;
      // Isolated try so a setStatus error never blocks showConv
      try { setStatus(d); } catch(e) { if (window.console) console.error('Pinku setStatus:', e); }
      // Show Q&A when speaking starts — cancel the 90s safety-net, keep visible
      if (d.speaking && d.last_transcript && d.last_reply) {
        try {
          showConv(d.last_transcript, d.last_reply);
          if (_clearTimer) { clearTimeout(_clearTimer); _clearTimer = null; }
          var fillEl = document.getElementById('cbar-fill');
          if (fillEl) fillEl.style.cssText = 'transition:none;width:100%';
        } catch(e) { if (window.console) console.error('Pinku showConv:', e); }
      }
      // When TTS finishes, linger a few seconds then drain the progress bar and clear
      if (wasSpeaking && !d.speaking) {
        var hasConvNow = document.body.className.indexOf('has-conv') >= 0;
        if (hasConvNow) {
          if (_clearTimer) { clearTimeout(_clearTimer); _clearTimer = null; }
          (function() {
            var f = document.getElementById('cbar-fill');
            if (f) f.style.cssText = 'transition:none;width:100%';
            setTimeout(function() {
              var f2 = document.getElementById('cbar-fill');
              if (f2) f2.style.cssText = 'transition:width ' + (AFTER_SPEECH_MS/1000) + 's linear;width:0%';
            }, 30);
          })();
          _clearTimer = setTimeout(_clearConv, AFTER_SPEECH_MS);
        }
      }
    }
    if (d.type === 'history') {
      try { showConv(d.transcript, d.reply); } catch(e) { if (window.console) console.error('Pinku showConv (history):', e); }
    }
    if (d.type === 'log') {
      _pushLog(d);
      try { _feedFromLog(d); } catch(e) { if (window.console) console.error('Pinku feed:', e); }
    }
  };
  es.onerror = function() {
    es.close();
    setTimeout(connect, 3000);
  };
}
connect();

// ── Mic level meter ───────────────────────────────────────────────────────────
// Polls /api/mic_level every 150 ms (~6 fps).  Updates the thin bar and
// flips body to mic-live (green dot) or mic-quiet (orange dot) in READY state.
var _MIC_THRESHOLD = 0.08;  // 0–1 normalised; 0.08 ≈ quiet room at 2-3 m
var _micLiveUntil  = 0;     // keep "live" for 2 s after last peak

function _pollMic() {
  xhr('GET', '/api/mic_level', null, function(err, r) {
    var lvl = (r && typeof r.level === 'number') ? r.level : 0;
    var fill = document.getElementById('mic-fill');
    if (fill) {
      fill.style.width = Math.min(100, Math.round(lvl * 100)) + '%';
      fill.style.background = lvl >= _MIC_THRESHOLD
        ? 'rgba(74,222,128,0.72)' : 'rgba(255,255,255,0.12)';
    }
    if (lvl >= _MIC_THRESHOLD) _micLiveUntil = Date.now() + 2000;
    var isLive = Date.now() < _micLiveUntil;
    var b = document.body;
    var has = function(c) { return b.className.indexOf(c) >= 0; };
    var set = function(add, rem) {
      b.className = b.className.replace(new RegExp('\\b' + rem + '\\b', 'g'), '')
                               .replace(/\s+/g,' ').trim();
      if (!has(add)) b.className = (b.className + ' ' + add).replace(/\s+/g,' ').trim();
    };
    if (isLive) { set('mic-live',  'mic-quiet'); }
    else        { set('mic-quiet', 'mic-live');  }
  });
  setTimeout(_pollMic, 150);
}
_pollMic();

// ── Actions ───────────────────────────────────────────────────────────────────
function act(name) {
  xhr('POST', '/api/action', {action: name}, null);
}

// ── Log panel ─────────────────────────────────────────────────────────────────
var _logBuf = [];
var _LOG_MAX = 120;
var _logOpen = false;

function _makeLogEl(e) {
  var d = document.createElement('div');
  d.className = 'log-line log-' + (e.level || 'info');
  d.textContent = (e.ts || '') + ' [' + (e.level || '').toUpperCase() + '] ' + (e.msg || '');
  return d;
}

function _pushLog(e) {
  _logBuf.push(e);
  if (_logBuf.length > _LOG_MAX) _logBuf.shift();
  if (!_logOpen) return;
  var list = document.getElementById('log-list');
  if (!list) return;
  var atBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 60;
  list.appendChild(_makeLogEl(e));
  while (list.children.length > _LOG_MAX) list.removeChild(list.firstChild);
  if (atBottom) list.scrollTop = list.scrollHeight;
}

function openLog() {
  _logOpen = true;
  document.getElementById('log-overlay').className = 'open';
  var list = document.getElementById('log-list');
  list.innerHTML = '';
  for (var i = 0; i < _logBuf.length; i++) list.appendChild(_makeLogEl(_logBuf[i]));
  list.scrollTop = list.scrollHeight;
}

function closeLog() {
  _logOpen = false;
  document.getElementById('log-overlay').className = '';
}

// ── Voices ────────────────────────────────────────────────────────────────────
function openVoices()  {
  document.getElementById('voices-overlay').className = 'open';
  loadProfiles();
}
function closeVoices() {
  document.getElementById('voices-overlay').className = '';
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function loadProfiles() {
  xhr('GET', '/api/enroll/profiles', null, function(err, r) {
    var el = document.getElementById('plist');
    if (!r || !r.profiles || !r.profiles.length) {
      el.innerHTML = '<span style="font-size:12px;color:rgba(226,232,248,0.28)">No voices enrolled yet.</span>';
      return;
    }
    el.innerHTML = r.profiles.map(function(p) {
      return '<div class="chip"><span>👤 ' + escHtml(p.name) + '</span>'
        + '<span style="color:rgba(226,232,248,0.3);font-size:10px;margin-left:3px">'
        + p.clips + ' clip' + (p.clips !== 1 ? 's' : '') + '</span>'
        + '<button onclick="delProfile(\'' + escHtml(p.name) + '\')">✕</button></div>';
    }).join('');
  });
}

function delProfile(name) {
  if (!confirm('Delete profile for "' + name + '"?')) return;
  xhr('DELETE', '/api/enroll/delete/' + encodeURIComponent(name), null, function() {
    loadProfiles();
  });
}

// ── Detection feed ────────────────────────────────────────────────────────────
// Live view of what the mic picked up and how the pipeline handled it — raw
// transcripts plus every pre-processing verdict (unknown voice, no wake word,
// wake confirmed, routed, handled).  Fed straight off the SSE log stream so it
// mirrors exactly what the system is doing, in order.
var _FEED_MAX = 5;

function _esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Pull the first quoted span out of a message — the transcript text.
function _firstQuote(s) {
  var m = s.match(/["'“](.*?)["'”]/);
  return m ? m[1] : null;
}

// Turn a raw log line into {kind, html} for the feed, or null to skip it.
function _parseDetect(level, msg) {
  var tr = _firstQuote(msg);
  var q  = tr ? '<span class="fq">' + _esc(tr) + '</span>' : '';

  // Speaker gate — only surface named recognition; drop the anonymous
  // "unidentified voice — passed" and bare "unrecognised — dropped" chatter
  // that carries no transcript (pure ambient noise, floods the feed).
  if (level === 'spk') {
    if (/uncertain|anonymously|unknown|dropped/i.test(msg)) return null;
    var sm = msg.match(/\(([\d.]+)\)/);
    var sc = sm ? ' <span style="opacity:.5">' + sm[1] + '</span>' : '';
    var nm = msg.match(/SpeakerID:\s*([^\(]+?)\s*\(/);
    return {kind:'spk', html:'<span class="fq">' + _esc(nm?nm[1]:'known') + '</span> recognised' + sc};
  }

  // Wake / session
  if (level === 'wake')
    return {kind:'wake', html:'wake word — session open'};

  // How it was ultimately handled
  if (level === 'source')
    return {kind:'act', html:'→ handled <span class="fv">via ' + _esc(msg) + '</span>'};

  // STT stream (level 'stt') — the [STT] prints
  if (level === 'stt') {
    if (/^Captured /.test(msg)) return null;   // raw audio-capture ticks — too noisy

    var eng = msg.match(/^(Apple|MLX):\s*(.*)$/);
    if (eng) {
      var rest = eng[2];
      if (/^empty/i.test(rest)) return null;          // no text — skip the noise
      var isWake = /\(wake\)/i.test(rest);
      return {kind: isWake ? 'wake' : 'heard',
              html: '<span style="opacity:.5">' + eng[1] + '</span> ' +
                    (q || _esc(rest)) + (isWake ? ' <span class="fv">wake ✓</span>' : '')};
    }
    if (/no wake word/i.test(msg))
      return {kind:'drop', html:(q ? q + ' ' : '') + '<span class="fv">no wake word — dropped</span>'};
    if (/no speech/i.test(msg))
      return {kind:'drop', html:'<span class="fv">no speech — dropped</span>'};
    if (/skipped long audio/i.test(msg))
      return {kind:'drop', html:'<span class="fv">too long — skipped</span>'};
    if (/text.route ignored|both empty|Gemini audio/i.test(msg))
      return {kind:'drop', html:'<span class="fv">routed to cloud</span>'};
    if (/hallucinat/i.test(msg))
      return {kind:'drop', html:(q ? q + ' ' : '') + '<span class="fv">phantom wake word — ignored</span>'};
    if (/^Ignored/i.test(msg))
      return {kind:'drop', html:(q ? q + ' ' : '') + '<span class="fv">not for me — ignored</span>'};
    return null;   // other stt chatter — keep the feed clean
  }
  return null;
}

function _feedFromLog(e) {
  var lvl = e.level || '';
  if (lvl !== 'stt' && lvl !== 'spk' && lvl !== 'wake' && lvl !== 'source') return;
  var p;
  try { p = _parseDetect(lvl, e.msg || ''); } catch (err) { p = null; }
  if (!p) return;
  var feed = document.getElementById('feed');
  if (!feed) return;
  var row = document.createElement('div');
  row.className = 'fd fd-' + p.kind;
  row.innerHTML = (e.ts ? '<span style="opacity:.35">' + e.ts + '</span> ' : '') + p.html;
  feed.appendChild(row);
  while (feed.children.length > _FEED_MAX) feed.removeChild(feed.firstChild);
}

var enrolling = false, micIv = null;

function startEnroll() {
  var name = document.getElementById('ename').value.replace(/^\s+|\s+$/g, '');
  if (!name) { document.getElementById('emsg').textContent = 'Enter a name first.'; return; }
  if (enrolling) return;
  enrolling = true;
  var btn = document.getElementById('ebtn');
  btn.disabled = true; btn.textContent = 'Recording…';
  document.getElementById('emsg').textContent = 'Listening — speak naturally for 15 seconds…';

  micIv = setInterval(function() {
    xhr('GET', '/api/enroll/level', null, function(err, d) {
      if (d && d.level !== undefined) {
        document.getElementById('mfill').style.width = Math.min(d.level * 100, 100) + '%';
      }
    });
  }, 200);

  xhr('POST', '/api/enroll/record', {name: name, seconds: 15}, function(err, d) {
    clearInterval(micIv);
    document.getElementById('mfill').style.width = '0%';
    if (d && d.ok) {
      document.getElementById('emsg').textContent =
        '✓ Enrolled as "' + d.name + '" — ' + d.clips + ' clip' + (d.clips !== 1 ? 's' : '');
      document.getElementById('ename').value = '';
      loadProfiles();
    } else {
      document.getElementById('emsg').textContent = '✗ ' + ((d && d.error) || 'Failed');
    }
    btn.disabled = false; btn.textContent = 'Record 15s';
    enrolling = false;
  });
}
</script>
</body>
</html>
"""
