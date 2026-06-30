"""
Web dashboard — Pinku, using exact Pinky CSS face system (div-based, not SVG).
Face expressions, animations, sleeping state all identical to Pinky's index.html.
http://<mac-mini-ip>:5100
"""

import json
import os
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
    "detections": [],
}

_clients_lock = threading.Lock()
_clients: list = []

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


def log_message(level: str, msg: str):
    """Stream a log line to all dashboard clients and append to daily log file."""
    ts = time.strftime("%H:%M:%S")
    _broadcast({
        "type":  "log",
        "level": level,
        "msg":   msg,
        "ts":    ts,
    })
    with _log_file_lock:
        try:
            _get_log_fh().write(f"{time.strftime('%Y-%m-%d')} {ts} [{level.upper():8s}] {msg}\n")
        except Exception:
            pass

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
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Pinku</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  height: 100%; width: 100%; overflow: hidden;
  background: #05050e;
  color: #dde3f4;
  font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -webkit-text-size-adjust: 100%;
  touch-action: manipulation;
}

/* subtle depth in background */
body::before {
  content: '';
  position: fixed; inset: 0;
  background: radial-gradient(ellipse 80% 60% at 50% 20%, #0c0c22 0%, transparent 70%);
  pointer-events: none;
}

/* ── Main layout ── */
#screen {
  position: relative;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  /* top  middle  bottom */
  padding: max(28px,4vmin) max(20px,3vmin) max(20px,3vmin);
}

/* ── Clock ── */
#clock-section {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  transition: flex-grow 0.55s cubic-bezier(.4,0,.2,1),
              margin-bottom 0.55s cubic-bezier(.4,0,.2,1);
}

/* when conversation is showing, clock shrinks upward */
.has-conv #clock-section {
  flex-grow: 0;
  margin-bottom: 0;
}

#time-row {
  display: flex;
  align-items: flex-start;
  line-height: 1;
}

#time {
  font-size: clamp(64px, 22vmin, 152px);
  font-weight: 100;
  letter-spacing: -0.025em;
  font-variant-numeric: tabular-nums;
  color: #e0e6f7;
  line-height: 1;
  transition: font-size 0.55s cubic-bezier(.4,0,.2,1);
}

.has-conv #time {
  font-size: clamp(34px, 9vmin, 70px);
}

#ampm {
  font-size: clamp(13px, 3.8vmin, 26px);
  font-weight: 400;
  color: #f59e0b;
  margin-left: max(6px,1.2vmin);
  margin-top: max(4px,0.6vmin);
  letter-spacing: 0.07em;
  transition: font-size 0.55s cubic-bezier(.4,0,.2,1),
              opacity 0.55s ease;
}

.has-conv #ampm {
  font-size: clamp(9px, 2vmin, 16px);
}

#day-date {
  text-align: center;
  margin-top: max(6px,1.2vmin);
  transition: opacity 0.4s ease, max-height 0.4s ease;
  max-height: 80px;
  overflow: hidden;
}

.has-conv #day-date {
  opacity: 0;
  max-height: 0;
  margin-top: 0;
}

#day-name {
  font-size: clamp(17px, 5vmin, 38px);
  font-weight: 300;
  color: rgba(224, 230, 247, 0.62);
  letter-spacing: 0.03em;
}

#date-str {
  font-size: clamp(12px, 3.2vmin, 24px);
  font-weight: 300;
  color: rgba(224, 230, 247, 0.36);
  margin-top: max(3px,0.5vmin);
  letter-spacing: 0.04em;
}

/* ── Conversation ── */
#conv-section {
  width: 100%;
  max-width: 720px;
  flex: 0 0 auto;
  max-height: 0;
  overflow: hidden;
  opacity: 0;
  transform: translateY(12px);
  transition: max-height 0.5s cubic-bezier(.4,0,.2,1),
              opacity 0.45s ease,
              transform 0.45s ease;
}

.has-conv #conv-section {
  max-height: 50vh;
  opacity: 1;
  transform: translateY(0);
}

#q-text {
  font-size: clamp(13px, 3.5vmin, 24px);
  font-weight: 300;
  color: rgba(224, 230, 247, 0.4);
  font-style: italic;
  line-height: 1.45;
  padding: 0 0 max(10px,2.2vmin);
  border-bottom: 1px solid rgba(255,255,255,0.06);
  margin-bottom: max(12px,2.5vmin);
}

#r-text {
  font-size: clamp(18px, 5.2vmin, 40px);
  font-weight: 300;
  color: #e0e6f7;
  line-height: 1.55;
}

/* countdown bar */
#cbar {
  height: 2px;
  background: rgba(255,255,255,0.07);
  border-radius: 2px;
  margin-top: max(14px,3vmin);
  overflow: hidden;
}

#cbar-fill {
  height: 100%;
  width: 100%;
  background: rgba(224,230,247,0.22);
  border-radius: 2px;
  transition: width linear;
}

/* ── Status ── */
#status-section {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: max(16px,3vmin) 0 max(10px,2vmin);
}

#status-pill {
  display: inline-flex;
  align-items: center;
  gap: max(7px,1.4vmin);
  padding: max(9px,1.8vmin) max(22px,4.5vmin);
  border-radius: 100px;
  border: 1px solid rgba(255,255,255,0.07);
  background: rgba(255,255,255,0.03);
  transition: border-color 0.4s, background 0.4s;
}

#sdot {
  width: max(10px,2vmin);
  height: max(10px,2vmin);
  border-radius: 50%;
  flex-shrink: 0;
  background: rgba(60,65,100,0.7);
  transition: background 0.4s, box-shadow 0.4s;
}

#slabel {
  font-size: clamp(12px, 2.8vmin, 20px);
  font-weight: 600;
  letter-spacing: 0.16em;
  color: rgba(224,230,247,0.32);
  transition: color 0.4s;
}

/* state classes */
body.s-awake #sdot     { background: #22c55e; box-shadow: 0 0 0 4px rgba(34,197,94,0.18); }
body.s-awake #slabel   { color: #22c55e; }
body.s-awake #status-pill { border-color: rgba(34,197,94,0.2); }

body.s-thinking #sdot  { background: #f59e0b; box-shadow: 0 0 0 4px rgba(245,158,11,0.2); animation: blink 0.85s ease-in-out infinite; }
body.s-thinking #slabel { color: #f59e0b; }
body.s-thinking #status-pill { border-color: rgba(245,158,11,0.2); }

body.s-speaking #sdot  { background: #818cf8; box-shadow: 0 0 0 4px rgba(129,140,248,0.2); animation: swell 1.7s ease-in-out infinite; }
body.s-speaking #slabel { color: #818cf8; }
body.s-speaking #status-pill { border-color: rgba(129,140,248,0.2); }

body.s-muted #sdot     { background: #f87171; box-shadow: 0 0 0 3px rgba(248,113,113,0.14); }
body.s-muted #slabel   { color: #f87171; }
body.s-muted #status-pill { border-color: rgba(248,113,113,0.2); }

@keyframes blink {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.45; transform: scale(0.8); }
}
@keyframes swell {
  0%, 100% { box-shadow: 0 0 0 3px rgba(129,140,248,0.15); }
  50%       { box-shadow: 0 0 0 8px rgba(129,140,248,0.25); }
}

/* ── Controls ── */
#controls {
  flex-shrink: 0;
  display: flex;
  gap: max(8px,1.8vmin);
  padding-bottom: env(safe-area-inset-bottom, 0px);
}

.cb {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.09);
  color: rgba(224,230,247,0.38);
  font-size: clamp(9px, 2vmin, 13px);
  font-weight: 700;
  letter-spacing: 0.1em;
  padding: max(8px,1.5vmin) max(18px,3.5vmin);
  border-radius: 100px;
  cursor: pointer;
  -webkit-tap-highlight-color: transparent;
  transition: background 0.18s, color 0.18s, border-color 0.18s;
  font-family: inherit;
  white-space: nowrap;
}
.cb:active {
  background: rgba(255,255,255,0.1);
  color: rgba(224,230,247,0.85);
  border-color: rgba(255,255,255,0.18);
}
.cb.on {
  border-color: rgba(248,113,113,0.4);
  color: rgba(248,113,113,0.75);
}

/* ── Voices slide-up ── */
#voices-overlay {
  position: fixed; inset: 0;
  background: rgba(5,5,14,0.82);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  display: flex; align-items: flex-end;
  opacity: 0; pointer-events: none;
  transition: opacity 0.3s ease;
  z-index: 50;
}
#voices-overlay.open { opacity: 1; pointer-events: auto; }

#voices-panel {
  width: 100%;
  background: #0b0b1c;
  border-top: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px 20px 0 0;
  padding: 24px 24px calc(32px + env(safe-area-inset-bottom,0px));
  transform: translateY(100%);
  transition: transform 0.4s cubic-bezier(.4,0,.2,1);
  max-height: 80vh;
  overflow-y: auto;
}
#voices-overlay.open #voices-panel { transform: none; }

#voices-panel h3 {
  font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
  color: rgba(224,230,247,0.4); margin-bottom: 18px;
}
.erow { display: flex; gap: 8px; margin-bottom: 12px; }
.erow input {
  flex: 1; background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.1); border-radius: 10px;
  color: #dde3f4; font-size: 15px; padding: 11px 14px;
  font-family: inherit; outline: none;
}
.erow input::placeholder { color: rgba(224,230,247,0.25); }
.erow input:focus { border-color: rgba(245,158,11,0.4); }
.erow button {
  background: rgba(245,158,11,0.12);
  border: 1px solid rgba(245,158,11,0.28);
  color: #f59e0b; font-size: 13px; font-weight: 700;
  padding: 11px 16px; border-radius: 10px; cursor: pointer;
  font-family: inherit; white-space: nowrap;
}
#mtrack { height: 3px; background: rgba(255,255,255,0.08); border-radius: 3px; margin-bottom: 10px; overflow: hidden; }
#mfill  { height: 100%; width: 0; background: #22c55e; border-radius: 3px; transition: width 0.08s; }
#emsg   { font-size: 12px; color: rgba(224,230,247,0.38); min-height: 16px; margin-bottom: 14px; }
#plist  { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  display: flex; align-items: center; gap: 4px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 20px; padding: 5px 8px 5px 12px;
  font-size: 12px; color: rgba(224,230,247,0.65);
}
.chip button {
  background: none; border: none; color: rgba(224,230,247,0.28);
  font-size: 11px; cursor: pointer; padding: 0 2px; line-height: 1;
}
</style>
</head>
<body>

<div id="screen">

  <!-- clock -->
  <div id="clock-section">
    <div id="time-row">
      <span id="time">12:00</span>
      <span id="ampm">AM</span>
    </div>
    <div id="day-date">
      <div id="day-name">Monday</div>
      <div id="date-str">1 January</div>
    </div>
  </div>

  <!-- conversation (hidden until spoken) -->
  <div id="conv-section">
    <div id="q-text"></div>
    <div id="r-text"></div>
    <div id="cbar"><div id="cbar-fill"></div></div>
  </div>

  <!-- status -->
  <div id="status-section">
    <div id="status-pill">
      <div id="sdot"></div>
      <span id="slabel">READY</span>
    </div>
  </div>

  <!-- controls -->
  <div id="controls">
    <button class="cb" onclick="act('wake')">WAKE</button>
    <button class="cb" onclick="act('sleep')">SLEEP</button>
    <button class="cb" id="mute-btn" onclick="act('mute_toggle')">MUTE</button>
    <button class="cb" onclick="openVoices()">VOICES</button>
  </div>

</div>

<!-- voices overlay -->
<div id="voices-overlay" onclick="if(event.target===this)closeVoices()">
  <div id="voices-panel">
    <h3>VOICE ENROLLMENT</h3>
    <div class="erow">
      <input id="ename" type="text" placeholder="Name" autocomplete="off" autocorrect="off" autocapitalize="words">
      <button id="ebtn" onclick="startEnroll()">&#9210; Record 15s</button>
    </div>
    <div id="mtrack"><div id="mfill"></div></div>
    <div id="emsg"></div>
    <div id="plist"></div>
  </div>
</div>

<script>
// ── Clock ────────────────────────────────────────────────────────────────────
var _lastMin = -1;
var DAYS   = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
var MONTHS = ['January','February','March','April','May','June','July',
              'August','September','October','November','December'];

function tick() {
  var n = new Date(), h = n.getHours(), m = n.getMinutes();
  if (m === _lastMin) return;
  _lastMin = m;
  var ap = h >= 12 ? 'PM' : 'AM';
  h = h % 12 || 12;
  document.getElementById('time').textContent = h + ':' + (m < 10 ? '0' : '') + m;
  document.getElementById('ampm').textContent = ap;
  document.getElementById('day-name').textContent = DAYS[n.getDay()];
  document.getElementById('date-str').textContent = n.getDate() + ' ' + MONTHS[n.getMonth()];
}
tick();
setInterval(tick, 4000);

// ── Status ────────────────────────────────────────────────────────────────────
var STATE_CLASS = { idle:'', awake:'s-awake', processing:'s-thinking', speaking:'s-speaking', muted:'s-muted' };
var STATE_LABEL = { idle:'READY', awake:'LISTENING', processing:'THINKING', speaking:'SPEAKING', muted:'MUTED' };

function setStatus(s) {
  var cls, lbl;
  if (s.muted)                     { cls = 's-muted';   lbl = 'MUTED'; }
  else if (s.speaking)             { cls = 's-speaking'; lbl = 'SPEAKING'; }
  else if (s.state === 'processing') { cls = 's-thinking'; lbl = 'THINKING'; }
  else if (s.state === 'awake')      { cls = 's-awake';   lbl = 'LISTENING'; }
  else                              { cls = '';           lbl = 'READY'; }
  document.body.className = cls;
  document.getElementById('slabel').textContent = lbl;
  document.getElementById('mute-btn').classList.toggle('on', !!s.muted);
}

// ── Conversation ──────────────────────────────────────────────────────────────
var _clearTimer = null;
var SHOW_MS = 13000;

function showConv(q, r) {
  if (!r && !q) return;
  if (_clearTimer) { clearTimeout(_clearTimer); _clearTimer = null; }

  var qt = document.getElementById('q-text');
  var rt = document.getElementById('r-text');
  qt.textContent = q ? '“' + q + '”' : '';
  rt.textContent = r || '';

  document.getElementById('screen').classList.add('has-conv');

  // restart countdown bar
  var fill = document.getElementById('cbar-fill');
  fill.style.transition = 'none';
  fill.style.width = '100%';
  requestAnimationFrame(function() {
    requestAnimationFrame(function() {
      fill.style.transition = 'width ' + (SHOW_MS / 1000) + 's linear';
      fill.style.width = '0%';
    });
  });

  _clearTimer = setTimeout(function() {
    document.getElementById('screen').classList.remove('has-conv');
    _clearTimer = null;
  }, SHOW_MS);
}

// ── SSE ───────────────────────────────────────────────────────────────────────
function connect() {
  var es = new EventSource('/api/stream');
  es.onmessage = function(e) {
    try {
      var d = JSON.parse(e.data);
      if (d.type === 'status') setStatus(d);
      if (d.type === 'history') showConv(d.transcript, d.reply);
    } catch(err) {}
  };
  es.onerror = function() { es.close(); setTimeout(connect, 3000); };
}
connect();

// ── Actions ───────────────────────────────────────────────────────────────────
function act(name) {
  fetch('/api/action', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: name })
  });
}

// ── Voices ────────────────────────────────────────────────────────────────────
function openVoices()  { document.getElementById('voices-overlay').classList.add('open'); loadProfiles(); }
function closeVoices() { document.getElementById('voices-overlay').classList.remove('open'); }

function esc(s) {
  return String(s).replace(/[&<>"]/g, function(c) {
    return { '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;' }[c];
  });
}

async function loadProfiles() {
  var r = await fetch('/api/enroll/profiles').then(function(r){ return r.json(); }).catch(function(){ return { profiles:[] }; });
  var el = document.getElementById('plist');
  if (!r.profiles || !r.profiles.length) {
    el.innerHTML = '<span style="font-size:12px;color:rgba(224,230,247,0.28)">No voices enrolled yet.</span>';
    return;
  }
  el.innerHTML = r.profiles.map(function(p) {
    return '<div class="chip"><span>👤 ' + esc(p.name) + '</span>'
      + '<span style="color:rgba(224,230,247,0.3);font-size:10px;margin-left:2px">' + p.clips + ' clip' + (p.clips !== 1 ? 's' : '') + '</span>'
      + '<button onclick="delProfile(\'' + esc(p.name) + '\')" title="Delete">&#x2715;</button></div>';
  }).join('');
}

async function delProfile(name) {
  if (!confirm('Delete profile for "' + name + '"?')) return;
  await fetch('/api/enroll/delete/' + encodeURIComponent(name), { method: 'DELETE' });
  loadProfiles();
}

var enrolling = false, micIv;

async function startEnroll() {
  var name = document.getElementById('ename').value.trim();
  if (!name) { document.getElementById('emsg').textContent = 'Enter a name first.'; return; }
  if (enrolling) return;
  enrolling = true;
  var btn = document.getElementById('ebtn');
  btn.disabled = true; btn.textContent = '&#9210; Recording…';
  document.getElementById('emsg').textContent = 'Listening — speak naturally for 15 seconds…';

  micIv = setInterval(async function() {
    try {
      var d = await fetch('/api/enroll/level').then(function(r){ return r.json(); });
      document.getElementById('mfill').style.width = Math.min(d.level * 100, 100) + '%';
    } catch(e) {}
  }, 80);

  try {
    var d = await fetch('/api/enroll/record', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name, seconds: 15 })
    }).then(function(r){ return r.json(); });
    document.getElementById('emsg').textContent = d.ok
      ? '✓ Enrolled as “' + d.name + '” — ' + d.clips + ' clip' + (d.clips !== 1 ? 's' : '')
      : '✗ ' + (d.error || 'Failed');
    if (d.ok) { document.getElementById('ename').value = ''; loadProfiles(); }
  } catch(e) {
    document.getElementById('emsg').textContent = '✗ ' + e.message;
  }
  clearInterval(micIv);
  document.getElementById('mfill').style.width = '0%';
  btn.disabled = false; btn.textContent = '&#9210; Record 15s';
  enrolling = false;
}
</script>
</body>
</html>
"""
