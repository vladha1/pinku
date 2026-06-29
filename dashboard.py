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
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Pinku</title>
<style>
:root {
  --bg:       #06060f;
  --bg2:      #0d0d1c;
  --surface:  rgba(255,255,255,0.028);
  --surface2: rgba(255,255,255,0.052);
  --border:   rgba(255,255,255,0.065);
  --text:     #dde2f0;
  --muted:    rgba(221,226,240,0.38);
  --accent:   #f59e0b;
  --adim:     rgba(245,158,11,0.14);
  --green:    #34d399;
  --red:      #f87171;
  --blue:     #60a5fa;
  --purple:   #a78bfa;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Inter, sans-serif;
  -webkit-font-smoothing: antialiased;
  display: flex; flex-direction: column;
}

/* ──── TOP BAR ──── */
#topbar {
  height: 60px; flex-shrink: 0;
  display: flex; align-items: center;
  padding: 0 28px; gap: 16px;
  border-bottom: 1px solid var(--border);
  background: rgba(6,6,15,0.9);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
}
.status-cluster {
  display: flex; align-items: center; gap: 9px; min-width: 130px;
}
.dot {
  width: 8px; height: 8px; border-radius: 50%;
  flex-shrink: 0; transition: background 0.5s, box-shadow 0.5s;
}
.dot.idle      { background: rgba(255,255,255,0.18); }
.dot.awake     { background: var(--green); box-shadow: 0 0 10px var(--green); animation: throb 2s ease-in-out infinite; }
.dot.thinking  { background: var(--purple); box-shadow: 0 0 10px var(--purple); animation: throb 0.7s ease-in-out infinite; }
.dot.speaking  { background: var(--blue);   box-shadow: 0 0 10px var(--blue);   animation: throb 1.1s ease-in-out infinite; }
.dot.muted     { background: var(--red); }
@keyframes throb { 0%,100%{opacity:1} 50%{opacity:0.35} }
#state-lbl {
  font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted);
}
.brand {
  flex: 1; text-align: center;
  font-size: 13px; font-weight: 700; letter-spacing: 0.28em;
  text-transform: uppercase; color: var(--accent);
}
.ctrls { display: flex; align-items: center; gap: 7px; min-width: 130px; justify-content: flex-end; }
.btn {
  height: 32px; padding: 0 13px; border-radius: 8px;
  border: 1px solid var(--border); background: var(--surface);
  color: var(--text); font-size: 12px; font-weight: 500;
  cursor: pointer; transition: all 0.14s; white-space: nowrap;
  display: inline-flex; align-items: center; gap: 5px;
  -webkit-user-select: none; user-select: none;
}
.btn:hover  { background: var(--surface2); border-color: rgba(255,255,255,0.12); }
.btn:active { transform: scale(0.95); }
.btn.g { border-color: rgba(52,211,153,0.32); color: var(--green); }
.btn.g:hover { background: rgba(52,211,153,0.09); }
.btn.b { border-color: rgba(96,165,250,0.32); color: var(--blue); }
.btn.b:hover { background: rgba(96,165,250,0.09); }
.btn.r { border-color: rgba(248,113,113,0.32); color: var(--red); }
.btn.r:hover { background: rgba(248,113,113,0.09); }
.btn.a { border-color: rgba(245,158,11,0.32); color: var(--accent); }
.btn.a:hover { background: var(--adim); }

/* ──── MAIN ──── */
main {
  flex: 1; overflow: hidden;
  display: grid;
  grid-template-rows: auto auto 1fr;
  padding: 28px 40px 20px;
  gap: 22px;
}

/* ──── CLOCK ──── */
#hero { text-align: center; }
#clock {
  font-size: 88px; font-weight: 200; letter-spacing: -4px;
  font-variant-numeric: tabular-nums; line-height: 1;
  background: linear-gradient(135deg, #f0f2ff 60%, rgba(240,242,255,0.55));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
#datestr {
  margin-top: 8px; font-size: 12px; font-weight: 500;
  letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted);
}

/* ──── MARKET CARDS ──── */
#cards {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
}
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 18px 22px;
  transition: background 0.2s, border-color 0.2s;
  cursor: default;
}
.card:hover { background: var(--surface2); border-color: rgba(255,255,255,0.1); }
.card-lbl {
  font-size: 9px; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 10px;
}
.card-val {
  font-size: 30px; font-weight: 300; letter-spacing: -0.5px;
  font-variant-numeric: tabular-nums; color: #f0f2ff; line-height: 1.1;
}
.card-sub {
  margin-top: 6px; font-size: 11px; font-weight: 500;
  color: var(--muted); display: flex; align-items: center; gap: 4px;
}
.card-sub.up   { color: var(--green); }
.card-sub.dn   { color: var(--red); }

/* ──── FEED ──── */
#feed {
  display: flex; flex-direction: column; min-height: 0;
}
.feed-hdr {
  font-size: 9px; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--muted);
  padding-bottom: 10px; border-bottom: 1px solid var(--border);
  flex-shrink: 0; margin-bottom: 2px;
}
#feed-list {
  flex: 1; overflow-y: auto; overflow-x: hidden;
}
#feed-list::-webkit-scrollbar { width: 3px; }
#feed-list::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
.exchange {
  display: grid; grid-template-columns: 1fr auto;
  grid-template-rows: auto auto;
  column-gap: 16px; padding: 10px 0;
  border-bottom: 1px solid rgba(255,255,255,0.035);
}
.exchange:last-child { border-bottom: none; }
.ex-user {
  grid-column: 1; grid-row: 1;
  font-size: 12px; color: var(--muted); margin-bottom: 3px;
}
.ex-pinku {
  grid-column: 1; grid-row: 2;
  font-size: 13px; color: var(--text);
  display: flex; align-items: baseline; gap: 7px;
}
.ex-pinku::before {
  content: '◆'; color: var(--accent); font-size: 6px;
  flex-shrink: 0; position: relative; top: -1px;
}
.ex-ts {
  grid-column: 2; grid-row: 1 / 3;
  font-size: 10px; color: var(--muted);
  align-self: center; white-space: nowrap;
}
.feed-empty {
  padding: 20px 0; text-align: center;
  font-size: 13px; color: var(--muted);
}

/* ──── VOICES OVERLAY ──── */
#voices-overlay {
  position: fixed; inset: 0; z-index: 200;
  background: rgba(0,0,0,0.55);
  backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
  display: flex; align-items: flex-end;
  opacity: 0; pointer-events: none;
  transition: opacity 0.25s;
}
#voices-overlay.open { opacity: 1; pointer-events: auto; }
#voices-panel {
  width: 100%; max-height: 55vh;
  background: var(--bg2); border-top: 1px solid var(--border);
  border-radius: 20px 20px 0 0; padding: 24px 32px 40px;
  overflow-y: auto;
  transform: translateY(100%);
  transition: transform 0.3s cubic-bezier(0.32,0.72,0,1);
}
#voices-overlay.open #voices-panel { transform: none; }
.panel-hdr {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 18px;
}
.panel-ttl {
  font-size: 10px; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--muted);
}
.panel-x {
  background: none; border: none; color: var(--muted);
  font-size: 20px; line-height: 1; cursor: pointer; padding: 0 4px;
}
.panel-x:hover { color: var(--text); }
.enroll-row { display: flex; gap: 9px; margin-bottom: 10px; }
.enroll-row input {
  flex: 1; height: 38px; padding: 0 13px;
  border-radius: 8px; border: 1px solid var(--border);
  background: var(--surface); color: var(--text);
  font-size: 13px; outline: none;
}
.enroll-row input:focus { border-color: var(--accent); }
.enroll-row input::placeholder { color: var(--muted); }
#enroll-btn {
  height: 38px; padding: 0 18px; border-radius: 8px;
  background: var(--adim); border: 1px solid rgba(245,158,11,0.3);
  color: var(--accent); font-size: 12px; font-weight: 600;
  cursor: pointer; white-space: nowrap;
}
#enroll-btn:disabled { opacity: 0.45; cursor: not-allowed; }
#mic-wrap { height: 2px; background: var(--border); border-radius: 2px; margin-bottom: 10px; overflow: hidden; }
#mic-bar  { height: 100%; width: 0%; background: var(--green); border-radius: 2px; transition: width 0.08s; }
#enroll-msg { font-size: 12px; color: var(--muted); min-height: 14px; margin-bottom: 14px; }
#profile-list { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  display: flex; align-items: center; gap: 6px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 20px; padding: 5px 10px 5px 12px; font-size: 12px;
}
.chip-del {
  background: none; border: none; color: var(--muted);
  font-size: 15px; cursor: pointer; line-height: 1; padding: 0 2px;
}
.chip-del:hover { color: var(--red); }
</style>
</head>
<body>

<header id="topbar">
  <div class="status-cluster">
    <span class="dot idle" id="dot"></span>
    <span id="state-lbl">IDLE</span>
  </div>
  <div class="brand">Pinku</div>
  <div class="ctrls">
    <button class="btn g" onclick="act('wake')">▶ Wake</button>
    <button class="btn b" onclick="act('sleep')">⏸ Sleep</button>
    <button class="btn" id="mute-btn" onclick="act('mute_toggle')">Mute</button>
    <button class="btn a" onclick="toggleVoices()">Voices</button>
  </div>
</header>

<main>
  <section id="hero">
    <div id="clock">—</div>
    <div id="datestr">—</div>
  </section>

  <section id="cards">
    <div class="card">
      <div class="card-lbl">Nifty 50</div>
      <div class="card-val" id="n-val">—</div>
      <div class="card-sub" id="n-chg">—</div>
    </div>
    <div class="card">
      <div class="card-lbl">Bitcoin</div>
      <div class="card-val" id="b-val">—</div>
      <div class="card-sub" id="b-chg">—</div>
    </div>
    <div class="card">
      <div class="card-lbl">Gurugram</div>
      <div class="card-val" id="w-temp">—</div>
      <div class="card-sub" id="w-cond">—</div>
    </div>
  </section>

  <section id="feed">
    <div class="feed-hdr">Recent</div>
    <div id="feed-list"><div class="feed-empty">Waiting for conversations…</div></div>
  </section>
</main>

<!-- Voices slide-up -->
<div id="voices-overlay" onclick="overlayClick(event)">
  <div id="voices-panel">
    <div class="panel-hdr">
      <span class="panel-ttl">Voice Profiles</span>
      <button class="panel-x" onclick="toggleVoices()">✕</button>
    </div>
    <div class="enroll-row">
      <input id="enroll-name" type="text" placeholder="Name (e.g. Vivek)" maxlength="30"
             onkeydown="if(event.key==='Enter')startEnroll()">
      <button id="enroll-btn" onclick="startEnroll()">⏺ Record 15s</button>
    </div>
    <div id="mic-wrap"><div id="mic-bar"></div></div>
    <div id="enroll-msg"></div>
    <div id="profile-list"></div>
  </div>
</div>

<script>
// ── Clock ────────────────────────────────────────────────────────────────────
const DAYS   = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
const MONTHS = ['January','February','March','April','May','June',
                'July','August','September','October','November','December'];

function tick() {
  const n = new Date();
  let h = n.getHours(), m = n.getMinutes(), s = n.getSeconds();
  const ap = h >= 12 ? 'PM' : 'AM';
  h = h % 12 || 12;
  document.getElementById('clock').textContent =
    `${h}:${pad(m)}:${pad(s)} ${ap}`;
  document.getElementById('datestr').textContent =
    `${DAYS[n.getDay()]}  ·  ${n.getDate()} ${MONTHS[n.getMonth()]} ${n.getFullYear()}`;
}
function pad(x) { return String(x).padStart(2,'0'); }
tick(); setInterval(tick, 1000);

// ── SSE Status ───────────────────────────────────────────────────────────────
const dot     = document.getElementById('dot');
const stateLbl = document.getElementById('state-lbl');
const muteBtn  = document.getElementById('mute-btn');

function applyStatus(s) {
  let cls = 'idle', lbl = 'IDLE';
  if (s.muted)               { cls='muted';   lbl='MUTED'; }
  else if (s.speaking)       { cls='speaking'; lbl='SPEAKING'; }
  else if (s.state==='processing') { cls='thinking'; lbl='THINKING'; }
  else if (s.state==='awake')      { cls='awake';   lbl='LISTENING'; }
  dot.className = 'dot ' + cls;
  stateLbl.textContent = lbl;
  muteBtn.textContent = s.muted ? '🔇 Unmute' : '🔇 Mute';
  muteBtn.className = 'btn' + (s.muted ? ' r' : '');
}

const es = new EventSource('/api/stream');
es.onmessage = e => {
  try {
    const d = JSON.parse(e.data);
    if (d.type === 'status')  applyStatus(d);
    if (d.type === 'history') pushExchange(d);
  } catch {}
};

// ── Actions ───────────────────────────────────────────────────────────────────
function act(name) {
  fetch('/api/action', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({action: name})
  });
}

// ── Market data ───────────────────────────────────────────────────────────────
function fmtINR(n) {
  if (n == null) return '—';
  return n.toLocaleString('en-IN', {maximumFractionDigits: 0});
}
function fmtUSD(n) {
  if (n == null) return '—';
  return '$' + n.toLocaleString('en-US', {maximumFractionDigits: 0});
}
function setCard(valId, chgId, price, change, pct, fmt) {
  document.getElementById(valId).textContent = fmt(price);
  const el = document.getElementById(chgId);
  if (price == null) { el.textContent = 'Unavailable'; el.className = 'card-sub'; return; }
  const up = change >= 0;
  el.className = 'card-sub ' + (up ? 'up' : 'dn');
  const arrow = up ? '▲' : '▼';
  const ac = Math.abs(change), ap = Math.abs(pct);
  el.textContent = `${arrow} ${fmt === fmtINR
    ? ac.toFixed(0)
    : '$'+ac.toLocaleString('en-US',{maximumFractionDigits:0})} (${ap.toFixed(2)}%)`;
}

async function fetchMarket() {
  try {
    const d = await fetch('/api/market').then(r=>r.json());
    if (d.nifty) setCard('n-val','n-chg', d.nifty.price, d.nifty.change, d.nifty.pct, fmtINR);
    if (d.btc)   setCard('b-val','b-chg', d.btc.price,   d.btc.change,   d.btc.pct,   fmtUSD);
  } catch {}
}

async function fetchWeather() {
  try {
    const d = await fetch('/api/weather').then(r=>r.json());
    document.getElementById('w-temp').textContent = d.temp || '—';
    const el = document.getElementById('w-cond');
    el.textContent = d.cond || '—'; el.className = 'card-sub';
  } catch {}
}

fetchMarket(); fetchWeather();
setInterval(fetchMarket, 60000);
setInterval(fetchWeather, 600000);

// ── Conversation Feed ─────────────────────────────────────────────────────────
const MAX = 8;
let convs = [];
const feedList = document.getElementById('feed-list');

function esc(s) {
  return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderFeed() {
  if (!convs.length) {
    feedList.innerHTML = '<div class="feed-empty">Waiting for conversations…</div>';
    return;
  }
  feedList.innerHTML = convs.slice(-MAX).reverse().map(c => {
    const ts = (c.ts||'').split(' ')[1]||'';
    return `<div class="exchange">
      <div class="ex-user">${esc(c.transcript)}</div>
      <div class="ex-pinku">${esc(c.reply)}</div>
      <div class="ex-ts">${ts.substring(0,5)}</div>
    </div>`;
  }).join('');
}

function pushExchange(e) {
  if (!e.transcript && !e.reply) return;
  convs.push(e);
  if (convs.length > 50) convs = convs.slice(-50);
  renderFeed();
}

fetch('/api/history?limit=20').then(r=>r.json()).then(data => {
  convs = data; renderFeed();
}).catch(()=>{});

// ── Voices ────────────────────────────────────────────────────────────────────
const overlay = document.getElementById('voices-overlay');
let voicesOpen = false;

function toggleVoices() {
  voicesOpen = !voicesOpen;
  overlay.classList.toggle('open', voicesOpen);
  if (voicesOpen) loadProfiles();
}
function overlayClick(e) { if (e.target === overlay) toggleVoices(); }

async function loadProfiles() {
  const r = await fetch('/api/enroll/profiles').then(r=>r.json());
  const el = document.getElementById('profile-list');
  if (!r.profiles || !r.profiles.length) {
    el.innerHTML = '<span style="font-size:12px;color:var(--muted)">No voices enrolled yet.</span>';
    return;
  }
  el.innerHTML = r.profiles.map(p =>
    `<div class="chip">
      <span>👤 ${esc(p.name)}</span>
      <span style="color:var(--muted);font-size:10px;margin-left:2px">${p.clips} clip${p.clips!==1?'s':''}</span>
      <button class="chip-del" onclick="delProfile('${esc(p.name)}')" title="Delete">✕</button>
    </div>`
  ).join('');
}

async function delProfile(name) {
  if (!confirm(`Delete profile for "${name}"?`)) return;
  await fetch(`/api/enroll/delete/${encodeURIComponent(name)}`, {method:'DELETE'});
  loadProfiles();
}

let enrolling = false, micInterval;

async function startEnroll() {
  const name = document.getElementById('enroll-name').value.trim();
  if (!name) { document.getElementById('enroll-msg').textContent = 'Enter a name first.'; return; }
  if (enrolling) return;
  enrolling = true;
  const btn = document.getElementById('enroll-btn');
  btn.disabled = true; btn.textContent = '⏺ Recording…';
  document.getElementById('enroll-msg').textContent = 'Listening — speak naturally for 15 seconds…';
  document.getElementById('mic-bar').style.width = '0%';

  micInterval = setInterval(async () => {
    try {
      const d = await fetch('/api/enroll/level').then(r=>r.json());
      document.getElementById('mic-bar').style.width = Math.min(d.level*100,100)+'%';
    } catch {}
  }, 80);

  try {
    const d = await fetch('/api/enroll/record', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name, seconds:15})
    }).then(r=>r.json());
    document.getElementById('enroll-msg').textContent = d.ok
      ? `✓ Enrolled as "${d.name}" — ${d.clips} clip${d.clips!==1?'s':''}`
      : '✗ ' + (d.error||'Failed');
    if (d.ok) { document.getElementById('enroll-name').value = ''; loadProfiles(); }
  } catch(e) {
    document.getElementById('enroll-msg').textContent = '✗ ' + e.message;
  }

  clearInterval(micInterval);
  document.getElementById('mic-bar').style.width = '0%';
  btn.disabled = false; btn.textContent = '⏺ Record 15s';
  enrolling = false;
}
</script>
</body>
</html>
"""
