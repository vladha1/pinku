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

_app    = Flask(__name__)
_logger = None
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

# ── Dart hit state ───────────────────────────────────────────────────────────
_dart_hits: list = []   # [{x, y, score, callout}] — last 20 throws (mirrored from pinku.py)
_dart_lock = threading.Lock()

def push_dart_hit(hit: dict):
    """Record a dart hit and broadcast it to all dashboard clients."""
    with _dart_lock:
        _dart_hits.append(hit)
        if len(_dart_hits) > 20:
            _dart_hits[:] = _dart_hits[-20:]
    _broadcast({"type": "dart_hit", **hit})

def dart_reset():
    """Clear all dart hits and notify clients."""
    with _dart_lock:
        _dart_hits.clear()
    _broadcast({"type": "dart_reset"})

# ── Dart game state ───────────────────────────────────────────────────────────
_game_state: dict = {
    "player": 1, "shots_done": 0, "shots_total": 5,
    "turn_total": 0, "awaiting": False, "rounds": [],
}
_game_lock = threading.Lock()

def push_game_update(state: dict):
    with _game_lock:
        _game_state.update(state)
    _broadcast({"type": "dart_game", **dict(_game_state)})

def push_dart_hits_clear():
    """Clear hit-dot overlay only (not scoreboard) — used on Next Player."""
    _broadcast({"type": "dart_hits_clear"})

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
    _broadcast({"type": "status", **_status})

def push_camera_status(camera: str, detection: str):
    """Called by camera.py whenever its status changes (ok / error / starting)."""
    _status["cam_status"]   = camera
    _status["det_status"]   = detection
    _broadcast({"type": "status", **_status})

def push_detection(event: dict):
    _status["detections"].append(event)
    if len(_status["detections"]) > 50:
        _status["detections"] = _status["detections"][-50:]
    _broadcast({"type": "detection", **event})

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

# ── Laser-active flag (set by browser when darts tab is open) ────────────────
_laser_active = False

def laser_enabled() -> bool:
    """Returns True only while the darts tab is open in the browser."""
    return _laser_active

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
            # Merge live camera status into initial push so reconnecting clients
            # immediately see the real camera state without waiting for a tab switch.
            try:
                from camera import get_status as _cam_get_status
                cs = _cam_get_status()
                init = {"type": "status", **_status,
                        "cam_status": cs["camera"], "det_status": cs["detection"]}
            except Exception:
                init = {"type": "status", **_status}
            yield f"data: {json.dumps(init)}\n\n"
            last_heartbeat = time.time()
            while True:
                time.sleep(0.05)
                while q:
                    yield q.pop(0)
                # Heartbeat every 15s — causes GeneratorExit on disconnected clients
                # so they are cleaned up from _clients rather than accumulating
                if time.time() - last_heartbeat > 15:
                    yield ": heartbeat\n\n"
                    last_heartbeat = time.time()
        except GeneratorExit:
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

@_app.route("/api/camera.jpg")
def camera_snapshot():
    """Current camera frame as JPEG. Returns 503 if camera not active.
    Pass ?annotated=1 to get dart rings + laser dot overlay (darts tab only)."""
    try:
        from camera import get_frame
        from flask import request as _req
        import cv2
        annotated = _req.args.get("annotated", "0") == "1"
        frame = get_frame(annotated=annotated)
        if frame is None:
            return ("No camera frame — camera may still be starting up", 503)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        from flask import make_response
        r = make_response(buf.tobytes())
        r.headers["Content-Type"]  = "image/jpeg"
        r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return r
    except Exception as e:
        return (str(e), 503)

@_app.route("/api/dart/hits")
def dart_hits():
    from flask import jsonify
    with _dart_lock:
        return jsonify(list(_dart_hits))

@_app.route("/api/dart/game")
def dart_game():
    from flask import jsonify
    with _game_lock:
        return jsonify(dict(_game_state))


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


@_app.route("/api/camera/status")
def camera_status():
    from flask import jsonify
    try:
        from camera import get_status
        return jsonify(get_status())
    except Exception as e:
        return jsonify({"camera": f"error: {e}", "detection": "unknown"})

@_app.route("/api/laser/debug")
def laser_debug():
    """
    Capture a frame, run laser detection with verbose logging, and return
    a JPEG showing the HSV mask so you can see exactly what colours are matched.
    Visit http://mac-ip:5100/api/laser/debug in a browser.
    """
    try:
        from camera import get_frame, _detect_laser_dots
        import cv2, numpy as np
        frame = get_frame()
        if frame is None:
            return ("No frame", 503)

        # Run with debug output
        dots = _detect_laser_dots(frame, debug=True)

        # Build HSV mask image for visual inspection
        import config as _cfg
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([_cfg.LASER_HUE_LO, _cfg.LASER_SAT_MIN, _cfg.LASER_VAL_MIN]),
            np.array([_cfg.LASER_HUE_HI, 255, 255]),
        )
        # Stack: original | mask (green channel brightened)
        mask_bgr  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # Mark detected dots on original
        for d in dots:
            cv2.circle(frame, (d["px"], d["py"]), max(8, int(d["r"]*2)), (0,255,0), 2)
        combined = np.hstack([frame, mask_bgr])
        _, buf = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
        from flask import make_response
        r = make_response(buf.tobytes())
        r.headers["Content-Type"]  = "image/jpeg"
        r.headers["Cache-Control"] = "no-store"
        # Also log summary
        print(f"[Laser debug] dots found: {len(dots)} — {dots}")
        return r
    except Exception as e:
        import traceback
        return (traceback.format_exc(), 500)


@_app.route("/api/laser/calibrate", methods=["POST"])
def laser_calibrate():
    """
    Set bullseye = current laser dot position.
    Body: {} (empty) — reads current dot from camera overlay.
    Returns {"ok": bool, "bull_x"?, "bull_y"?, "error"?}
    """
    from flask import jsonify
    fn = _actions.get("laser_calibrate")
    if fn is None:
        return jsonify({"ok": False, "error": "Calibrate action not registered"}), 503
    result = fn()
    if result is None:
        result = {"ok": True}
    code = 200 if result.get("ok") else 400
    return jsonify(result), code


@_app.route("/api/laser/active", methods=["POST"])
def laser_active_set():
    """Browser calls this when entering/leaving the darts tab.
    Body: {"active": true|false}
    """
    from flask import jsonify, request as _req
    global _laser_active
    data = _req.get_json(silent=True) or {}
    _laser_active = bool(data.get("active", False))
    return jsonify({"ok": True, "laser_active": _laser_active})


@_app.route("/api/camera/request-permission")
def camera_request_permission():
    """
    Trigger macOS camera permission dialog by running camera access
    inside the user's GUI login session (via launchctl asuser),
    so the permission popup appears on screen / VNC.
    """
    from flask import jsonify
    import subprocess, sys, os, tempfile

    script = (
        "import cv2, time\n"
        "cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)\n"
        "ok = False\n"
        "for _ in range(30):\n"
        "    ret, _ = cap.read()\n"
        "    if ret: ok = True; break\n"
        "    time.sleep(0.2)\n"
        "cap.release()\n"
        "print('OK' if ok else 'DENIED')\n"
    )
    # Write to temp file (launchctl asuser needs a real path)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write(script); tmp.flush(); tmp.close()

    try:
        uid = str(os.getuid())
        result = subprocess.run(
            ["launchctl", "asuser", uid, sys.executable, tmp.name],
            capture_output=True, text=True, timeout=15,
        )
        out = (result.stdout + result.stderr).strip()
        ok  = "OK" in out
        return jsonify({"ok": ok, "output": out or "(no output — check VNC for popup)"})
    except Exception as e:
        return jsonify({"ok": False, "output": str(e)})
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

# ── Start ─────────────────────────────────────────────────────────────────────

def start(logger=None, port=DASHBOARD_PORT):
    global _logger
    _logger = logger

    # Silence Flask/werkzeug per-request logs (camera.jpg polls at 5 fps = noise)
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
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Pinku</title>
<!-- React 17 + Babel for iOS 9 Safari compatibility -->
<script src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
<script src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<style>
/* ── Reset & Base ── */
*, *:before, *:after { -webkit-box-sizing: border-box; box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  background: #0a0a14;
  color: #e8e8f4;
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
  display: -webkit-flex; display: flex;
  -webkit-flex-direction: column; flex-direction: column;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
  -webkit-user-select: none; user-select: none;
}
#root {
  display: -webkit-flex; display: flex;
  -webkit-flex-direction: column; flex-direction: column;
  overflow: hidden;
}

/* ── Header ── */
.header {
  display: -webkit-flex; display: flex;
  -webkit-align-items: center; align-items: center;
  -webkit-justify-content: space-between; justify-content: space-between;
  padding: 12px 16px 10px;
  border-bottom: 1px solid #24243a;
  -webkit-flex-shrink: 0; flex-shrink: 0;
  background: -webkit-linear-gradient(top, rgba(124,92,191,0.07) 0%, transparent 100%);
  background: linear-gradient(180deg, rgba(124,92,191,0.07) 0%, transparent 100%);
}
.header-left { display: -webkit-flex; display: flex; -webkit-align-items: center; align-items: center; gap: 10px; }
.dot-wrap { position: relative; }
.avatar-ring {
  width: 36px; height: 36px; border-radius: 50%;
  background: -webkit-linear-gradient(135deg, #7c3aed, #4c1d95);
  background: linear-gradient(135deg, #7c3aed, #4c1d95);
  display: -webkit-flex; display: flex;
  -webkit-align-items: center; align-items: center;
  -webkit-justify-content: center; justify-content: center;
  font-weight: 700; font-size: 0.95rem; color: #fff;
  -webkit-box-shadow: 0 0 12px rgba(124,58,237,0.45); box-shadow: 0 0 12px rgba(124,58,237,0.45);
}
.status-dot {
  position: absolute; bottom: -1px; right: -1px;
  width: 11px; height: 11px; border-radius: 50%;
  border: 2px solid #0a0a14;
  background: #4ade80;
  -webkit-transition: background 0.5s ease; transition: background 0.5s ease;
}
.status-dot.sleeping  { background: #2a2a42; }
.status-dot.paused    { background: #f87171; }
.status-dot.thinking,
.status-dot.triggered { background: #c084fc; }
.status-dot.speaking  { background: #f472b6; }
.app-name {
  font-size: 1.05rem; font-weight: 700;
  background: -webkit-linear-gradient(left, #c9b1ff, #f472b6);
  background: linear-gradient(90deg, #c9b1ff, #f472b6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.status-pills { display: -webkit-flex; display: flex; gap: 5px; margin-top: 2px; }
.spill {
  font-size: 0.72rem; padding: 1px 6px; border-radius: 8px;
  background: rgba(255,255,255,0.05); border: 1px solid #24243a;
  color: #666683; -webkit-transition: background 0.3s, color 0.3s; transition: background 0.3s, color 0.3s;
}
.spill.on { background: rgba(74,222,128,0.12); color: #4ade80; border-color: rgba(74,222,128,0.3); }
.header-btns { display: -webkit-flex; display: flex; gap: 8px; }
.hdr-btn {
  padding: 6px 12px; border-radius: 10px; font-size: 0.85rem;
  border: 1px solid #24243a; background: rgba(255,255,255,0.05);
  color: #e8e8f4; cursor: pointer; -webkit-transition: background 0.18s; transition: background 0.18s;
  display: -webkit-flex; display: flex;
  -webkit-align-items: center; align-items: center;
  -webkit-justify-content: center; justify-content: center;
  min-width: 44px; min-height: 44px;
}
.hdr-btn:hover { background: rgba(255,255,255,0.1); }
.hdr-btn.stop-btn  { border-color: rgba(248,113,113,0.4); color: #f87171; }
.hdr-btn.mute-btn  { border-color: rgba(248,113,113,0.35); }
.hdr-btn.mute-btn.active { background: rgba(192,132,252,0.18); border-color: rgba(192,132,252,0.5); }
.hdr-btn.resume-btn{ border-color: rgba(74,222,128,0.4); color: #4ade80; }
.hdr-btn.restart-btn { border-color: rgba(251,191,36,0.4); color: #fbbf24; font-size: 0.8rem; }

/* ── Main area ── */
.main-area {
  -webkit-flex: 1; flex: 1; display: -webkit-flex; display: flex; -webkit-flex-direction: column; flex-direction: column;
  -webkit-align-items: center; align-items: center; -webkit-justify-content: center; justify-content: center;
  overflow: hidden; gap: 0;
  padding: 8px 0 4px;
}

/* ── Thought bubble ── */
.face-bubble {
  position: absolute; top: 0%; right: -4%;
  background: rgba(14, 12, 28, 0.92);
  border: 1.5px solid #c084fc;
  border-radius: 999px;
  padding: 5px 11px;
  display: -webkit-flex; display: flex;
  -webkit-align-items: center; align-items: center; gap: 4px;
  z-index: 8; pointer-events: none;
  -webkit-box-shadow: 0 4px 16px rgba(0,0,0,0.5); box-shadow: 0 4px 16px rgba(0,0,0,0.5);
  opacity: 0;
  -webkit-transform: translateY(8px) scale(0.86); transform: translateY(8px) scale(0.86);
  -webkit-transition: opacity 0.32s, -webkit-transform 0.34s; transition: opacity 0.32s, transform 0.34s;
}
.face-bubble.triggered { border-color: #fb923c; }
.face-bubble.visible { opacity: 1; -webkit-transform: translateY(0) scale(1); transform: translateY(0) scale(1); }
.bubble-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: #c084fc; display: inline-block;
  -webkit-animation: dot-bounce 1.22s ease-in-out infinite; animation: dot-bounce 1.22s ease-in-out infinite;
}
.face-bubble.triggered .bubble-dot { background: #fb923c; }
.bubble-dot.d2 { -webkit-animation-delay: 0.16s; animation-delay: 0.16s; }
.bubble-dot.d3 { -webkit-animation-delay: 0.32s; animation-delay: 0.32s; }
@-webkit-keyframes dot-bounce {
  0%,100% { -webkit-transform: translateY(0) scale(1); opacity: 0.82; }
  22%     { -webkit-transform: translateY(0) scale(1.05); opacity: 1; }
  48%     { -webkit-transform: translateY(-6px) scale(1.1); opacity: 1; }
  72%     { -webkit-transform: translateY(-1px) scale(1); opacity: 0.9; }
}
@keyframes dot-bounce {
  0%,100% { transform: translateY(0) scale(1); opacity: 0.82; }
  22%     { transform: translateY(0) scale(1.05); opacity: 1; }
  48%     { transform: translateY(-6px) scale(1.1); opacity: 1; }
  72%     { transform: translateY(-1px) scale(1); opacity: 0.9; }
}

/* ── Stage & layout ── */
.pinky-stage {
  display: -webkit-flex; display: flex; -webkit-flex-direction: column; flex-direction: column;
  -webkit-align-items: center; align-items: center; -webkit-justify-content: center; justify-content: center;
  width: 100%; max-width: 720px; margin: 0 auto;
}
.face-with-actions {
  display: -webkit-flex; display: flex; -webkit-flex-direction: column; flex-direction: column;
  -webkit-align-items: center; align-items: center; -webkit-justify-content: center; justify-content: center;
  width: 100%; max-width: 94vw;
  padding: 2px 4px 10px;
}
.face-with-actions .face-wrap {
  -webkit-flex: 0 1 auto; flex: 0 1 auto; width: 100%;
  max-width: 440px;
  margin-left: auto; margin-right: auto;
}
@media (min-width: 500px) {
  .face-with-actions .face-wrap { max-width: 460px; }
}
@media (min-width: 900px) {
  .face-with-actions .face-wrap { max-width: 520px; }
}
.wake-btn-top {
  -webkit-flex: 0 0 auto; flex: 0 0 auto; width: 56px; height: 56px;
  border-radius: 50%;
  border: 2px solid rgba(216,180,254,0.9);
  background: -webkit-linear-gradient(top, #7c3aed 0%, #5b21b6 52%, #4c1d95 100%);
  background: linear-gradient(180deg, #7c3aed 0%, #5b21b6 52%, #4c1d95 100%);
  color: #fff; cursor: pointer;
  display: -webkit-flex; display: flex;
  -webkit-align-items: center; align-items: center; -webkit-justify-content: center; justify-content: center;
  -webkit-box-shadow: 0 4px 0 rgba(49,10,90,0.75), 0 8px 18px rgba(91,33,182,0.38);
  box-shadow: 0 4px 0 rgba(49,10,90,0.75), 0 8px 18px rgba(91,33,182,0.38);
  -webkit-transition: -webkit-transform 0.08s ease; transition: transform 0.08s ease;
  font-size: 1.55rem; line-height: 1; margin-bottom: 12px;
}
.wake-btn-top:active { -webkit-transform: translateY(2px); transform: translateY(2px); }

/* ── Question & reply clouds ── */
.question-cloud {
  margin-bottom: 10px; width: 100%; max-width: 450px;
  padding: 10px 14px 12px; font-size: 1.05rem; line-height: 1.5;
  color: #0c1e2e; text-align: left;
  background: -webkit-linear-gradient(top, #f8fcff 0%, #e8f4fc 45%, #dff0fb 100%);
  background: linear-gradient(165deg,#f8fcff 0%,#e8f4fc 45%,#dff0fb 100%);
  border: 2px solid rgba(56,189,248,0.55);
  border-radius: 26px; border-bottom-right-radius: 8px;
  -webkit-box-shadow: 0 8px 22px rgba(0,0,0,0.18); box-shadow: 0 8px 22px rgba(0,0,0,0.18);
  display: none;
}
.question-cloud.has-text { display: block; }
.question-cloud-label {
  display: block; font-size: 0.58rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: #0369a1; margin-bottom: 6px;
}
.reply-cloud {
  margin-top: 10px; width: 100%; max-width: 450px;
  padding: 10px 14px 12px; font-size: 1.05rem; line-height: 1.5;
  color: #241830; text-align: left;
  background: -webkit-linear-gradient(top, #fdfcff 0%, #f3e8ff 45%, #ede9fe 100%);
  background: linear-gradient(165deg,#fdfcff 0%,#f3e8ff 45%,#ede9fe 100%);
  border: 2px solid rgba(167,139,250,0.55);
  border-radius: 26px; border-bottom-left-radius: 8px;
  -webkit-box-shadow: 0 10px 28px rgba(0,0,0,0.22); box-shadow: 0 10px 28px rgba(0,0,0,0.22);
  cursor: pointer; display: none;
}
.reply-cloud.has-text { display: block; }
.reply-cloud-label {
  display: block; font-size: 0.58rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: #7c3aed; margin-bottom: 6px;
}
.reply-cloud-body { display: block; word-break: break-word; }

/* ── Person badge ── */
.person-badge {
  margin-top: 6px;
  display: -webkit-flex; display: flex;
  -webkit-align-items: center; align-items: center; gap: 6px;
  background: rgba(74,222,128,0.10);
  border: 1px solid rgba(74,222,128,0.28);
  border-radius: 20px; padding: 4px 12px;
  font-size: 0.72rem; color: #4ade80;
  opacity: 0; -webkit-transition: opacity 0.4s; transition: opacity 0.4s; pointer-events: none;
}
.person-badge.visible { opacity: 1; }
.person-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: #4ade80;
  -webkit-animation: live-blink 1.6s ease-in-out infinite; animation: live-blink 1.6s ease-in-out infinite;
}
@-webkit-keyframes live-blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
@keyframes live-blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Status bar ── */
.status-bar {
  -webkit-flex-shrink: 0; flex-shrink: 0; margin-top: 10px;
  padding: 9px 14px 9px 12px; border-radius: 999px;
  background: -webkit-linear-gradient(top, rgba(22,24,40,0.94), rgba(14,16,28,0.96));
  background: linear-gradient(180deg, rgba(22,24,40,0.94), rgba(14,16,28,0.96));
  border: 1px solid rgba(100,116,139,0.4);
  -webkit-box-shadow: 0 8px 24px rgba(0,0,0,0.4); box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  max-width: 460px; width: 96vw;
  display: -webkit-flex; display: flex;
  -webkit-align-items: center; align-items: center; gap: 10px;
  -webkit-transition: border-color 0.6s ease; transition: border-color 0.6s ease;
}
.status-bar.sleeping  { border-color: rgba(100,116,139,0.4); }
.status-bar.awake     { border-color: rgba(192,132,252,0.5); }
.status-bar.thinking,
.status-bar.triggered { border-color: rgba(232,121,249,0.48); }
.status-bar.speaking  { border-color: rgba(192,132,252,0.42); }
.status-bar.paused    { border-color: rgba(248,113,113,0.42); }
.status-bar-icon { font-size: 1.65rem; line-height: 1; -webkit-flex-shrink: 0; flex-shrink: 0; }
.status-bar-icon.thinking,
.status-bar-icon.triggered { -webkit-animation: notify-pulse 1.12s ease-in-out infinite; animation: notify-pulse 1.12s ease-in-out infinite; }
@-webkit-keyframes notify-pulse { 0%,100%{opacity:1;-webkit-transform:scale(1)} 50%{opacity:0.7;-webkit-transform:scale(0.9)} }
@keyframes notify-pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.7;transform:scale(0.9)} }
.status-bar-text { -webkit-flex: 1; flex: 1; min-width: 0; display: -webkit-flex; display: flex; -webkit-flex-direction: column; flex-direction: column; }
.status-bar-label { font-size: 1.15rem; font-weight: 700; color: #e8e8f4; white-space: nowrap; }
.status-bar-sub   { font-size: 0.98rem; color: #8a8aaa; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 1px; }
.status-bar-right { display: -webkit-flex; display: flex; -webkit-align-items: center; align-items: center; gap: 8px; -webkit-flex-shrink: 0; flex-shrink: 0; }
.ws-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.ws-dot.on  { background: #4ade80; -webkit-box-shadow: 0 0 5px rgba(74,222,128,0.6); box-shadow: 0 0 5px rgba(74,222,128,0.6); }
.ws-dot.off { background: #64748b; }

/* ── Face wrap ── */
.face-wrap {
  position: relative; width: 100%; max-width: 440px;
  margin: 0 auto; -webkit-flex-shrink: 0; flex-shrink: 0;
  border-radius: 48% 48% 46% 46% / 44% 44% 56% 56%;
  border: 1px solid rgba(255,182,220,0.22);
  background:
    -webkit-radial-gradient(50% 36% ellipse, rgba(255,210,232,0.22) 0%, transparent 58%),
    -webkit-radial-gradient(32% 20%, #4a4478 0%, #322d58 38%, #1c1a34 100%);
  background:
    radial-gradient(ellipse 72% 58% at 50% 36%, rgba(255,210,232,0.22) 0%, transparent 58%),
    radial-gradient(circle at 32% 20%, #4a4478 0%, #322d58 38%, #1c1a34 100%);
  -webkit-box-shadow: 0 24px 48px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.08) inset;
  box-shadow: 0 24px 48px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.08) inset;
  overflow: visible;
}
.face-wrap:before { content:''; display:block; padding-top:100%; }
.face-wrap:after {
  content:''; position:absolute; left:-8%; right:-8%; top:-8%; bottom:-8%;
  border-radius:50%;
  border: 1px solid rgba(96,165,250,0.36);
  -webkit-box-shadow: 0 0 24px rgba(59,130,246,0.28), inset 0 0 12px rgba(147,51,234,0.25);
  box-shadow: 0 0 24px rgba(59,130,246,0.28), inset 0 0 12px rgba(147,51,234,0.25);
  pointer-events:none; opacity:0.52;
  -webkit-transition: opacity 0.5s ease; transition: opacity 0.5s ease;
}
@-webkit-keyframes face-ambient-ring { 0%,100%{opacity:0.38} 50%{opacity:0.72} }
@keyframes face-ambient-ring { 0%,100%{opacity:0.38} 50%{opacity:0.72} }
.face-wrap.awake:after, .face-wrap.listening:after { -webkit-animation: face-ambient-ring 3.5s ease-in-out infinite; animation: face-ambient-ring 3.5s ease-in-out infinite; }
.face-wrap.thinking:after, .face-wrap.triggered:after { -webkit-animation: face-ambient-ring 2.35s ease-in-out infinite; animation: face-ambient-ring 2.35s ease-in-out infinite; }
.face-wrap.paused:after, .face-wrap.sleeping:after { -webkit-animation: none; animation: none; opacity: 0.1; }

.face-wrap .face {
  position: absolute; left:0; right:0; top:0; bottom:0;
  width:71%; height:86%; margin:auto; -webkit-transform:none; transform:none;
  display:-webkit-flex; display:flex; -webkit-flex-direction:column; flex-direction:column;
  -webkit-align-items:center; align-items:center; -webkit-justify-content:flex-start; justify-content:flex-start;
}

/* ── Head & features ── */
.head {
  position: relative; width: 100%; height: 88%;
  border-radius: 49% 49% 47% 47% / 47% 47% 55% 55%;
  background:
    -webkit-radial-gradient(50% 22% ellipse, #fffefb 0%, #fff5f0 18%, #ffe8dc 38%, #ffd0c0 62%, #f5c4b0 92%);
  background:
    radial-gradient(ellipse 120% 88% at 50% 22%,
      #fffefb 0%, #fff5f0 18%, #ffe8dc 38%, #ffd0c0 62%, #f5c4b0 92%);
  border: 1px solid rgba(240,180,165,0.35);
  -webkit-box-shadow: inset 0 -10px 22px rgba(200,120,110,0.1), inset 12px 18px 32px rgba(255,255,255,0.55);
  box-shadow: inset 0 -10px 22px rgba(200,120,110,0.1), inset 12px 18px 32px rgba(255,255,255,0.55);
  -webkit-transition: filter 0.45s ease, -webkit-box-shadow 0.45s ease; transition: filter 0.45s ease, box-shadow 0.45s ease;
}
.hair-side {
  position: absolute; top: 16%; width: 26%; height: 86%;
  z-index: 1; pointer-events: none;
  background: -webkit-linear-gradient(left, rgba(255,255,255,0.12) 0%, #ffc8dd 22%, #f9a8d4 50%, #f472b6 78%, #ec4899 100%);
  background: linear-gradient(90deg, rgba(255,255,255,0.12) 0%, #ffc8dd 22%, #f9a8d4 50%, #f472b6 78%, #ec4899 100%);
  border-radius: 58% 42% 55% 45% / 22% 30% 88% 72%;
  -webkit-box-shadow: inset -3px 0 8px rgba(255,255,255,0.25); box-shadow: inset -3px 0 8px rgba(255,255,255,0.25);
}
.hair-side.left  { left: -14%; -webkit-transform: rotate(-5deg); transform: rotate(-5deg); }
.hair-side.right { right: -14%; -webkit-transform: rotate(5deg) scaleX(-1); transform: rotate(5deg) scaleX(-1); }
.hair-fringe {
  position: absolute; top: -4%; left: 10%; width: 80%; height: 18%;
  z-index: 2; pointer-events: none;
  background: -webkit-linear-gradient(top, #ffe4f0 0%, #fda4d0 45%, #f472b6 100%);
  background: linear-gradient(180deg, #ffe4f0 0%, #fda4d0 45%, #f472b6 100%);
  border-radius: 50% 50% 45% 45% / 85% 85% 28% 28%;
  -webkit-clip-path: ellipse(92% 100% at 50% 0%); clip-path: ellipse(92% 100% at 50% 0%);
  -webkit-box-shadow: 0 2px 6px rgba(236,72,153,0.2); box-shadow: 0 2px 6px rgba(236,72,153,0.2);
}
.brows {
  position: absolute; top: 30.5%; left: 50%; -webkit-transform: translateX(-50%); transform: translateX(-50%);
  width: 58%; display: -webkit-flex; display: flex; -webkit-justify-content: space-between; justify-content: space-between; z-index: 4;
}
.brow {
  width: 32%; height: 3px; border-radius: 999px;
  background: -webkit-linear-gradient(left, #b0887a, #8a5f52); background: linear-gradient(90deg, #b0887a, #8a5f52);
  opacity: 0.9; -webkit-transform-origin: 50% 50%; transform-origin: 50% 50%;
}
.brow.left  { -webkit-transform: rotate(-5deg) translateY(1px); transform: rotate(-5deg) translateY(1px); border-radius: 55% 45% 50% 50%; }
.brow.right { -webkit-transform: rotate(5deg) translateY(1px); transform: rotate(5deg) translateY(1px); border-radius: 45% 55% 50% 50%; }
.eyes {
  position: absolute; top: 39.5%; left: 18%; width: 64%;
  display: -webkit-flex; display: flex; -webkit-justify-content: space-between; justify-content: space-between; -webkit-align-items: center; align-items: center;
  z-index: 4; -webkit-transform-origin: 50% 45%; transform-origin: 50% 45%;
}
.eye {
  width: 40px; height: 40px;
  border-radius: 50%;
  border: 2px solid rgba(55,65,81,0.22);
  background: -webkit-radial-gradient(40% 28%, #ffffff 0%, #f5f9ff 62%, #dbeafe 100%);
  background: radial-gradient(circle at 40% 28%, #ffffff 0%, #f5f9ff 62%, #dbeafe 100%);
  -webkit-box-shadow: 0 0 16px rgba(255,255,255,0.5), 0 5px 16px rgba(59,130,246,0.2), inset 0 -4px 7px rgba(15,23,42,0.12);
  box-shadow: 0 0 16px rgba(255,255,255,0.5), 0 5px 16px rgba(59,130,246,0.2), inset 0 -4px 7px rgba(15,23,42,0.12);
  position: relative; -webkit-transform-origin: center center; transform-origin: center center;
}
@media (min-width: 400px) { .eye { width: 46px; height: 46px; } }
@media (min-width: 600px) { .eye { width: 54px; height: 54px; } }
.eye-lashes {
  position: absolute; left:-6%; right:-6%; top:-16%; height:42%;
  display: -webkit-flex; display: flex; -webkit-justify-content: center; justify-content: center; -webkit-align-items: flex-end; align-items: flex-end;
  padding-bottom: 2px; z-index: 6; pointer-events: none;
}
.eye-lashes .lash {
  -webkit-flex: 0 0 auto; flex: 0 0 auto; width: 2px;
  height: 8px;
  margin: 0 1px; border-radius: 2px;
  background: -webkit-linear-gradient(top, rgba(28,24,36,0.95) 0%, rgba(28,24,36,0.2) 100%);
  background: linear-gradient(180deg, rgba(28,24,36,0.95) 0%, rgba(28,24,36,0.2) 100%);
  -webkit-transform-origin: 50% 100%; transform-origin: 50% 100%; opacity: 0.9;
  -webkit-box-shadow: 0 0 1px rgba(0,0,0,0.25); box-shadow: 0 0 1px rgba(0,0,0,0.25);
}
.eye .lash:nth-child(1) { -webkit-transform: rotate(-22deg); transform: rotate(-22deg); height: 6px; }
.eye .lash:nth-child(2) { -webkit-transform: rotate(-10deg); transform: rotate(-10deg); }
.eye .lash:nth-child(3) { -webkit-transform: rotate(0deg); height: 9px; }
.eye .lash:nth-child(4) { -webkit-transform: rotate(10deg); transform: rotate(10deg); }
.eye .lash:nth-child(5) { -webkit-transform: rotate(22deg); transform: rotate(22deg); height: 6px; }
.eye:after {
  content: ""; position: absolute;
  left: 50%; top: 56%; -webkit-transform: translate(-50%,-50%); transform: translate(-50%,-50%);
  width: 42%; height: 48%; border-radius: 50%;
  background: -webkit-radial-gradient(40% 34%, #7c3aed 0%, #4338ca 45%, #0f172a 84%, #020617 100%);
  background: radial-gradient(circle at 40% 34%, #7c3aed 0%, #4338ca 45%, #0f172a 84%, #020617 100%);
  -webkit-box-shadow: 0 2px 10px rgba(76,29,149,0.45), inset 0 -3px 7px rgba(0,0,0,0.45);
  box-shadow: 0 2px 10px rgba(76,29,149,0.45), inset 0 -3px 7px rgba(0,0,0,0.45);
}
.eye:before {
  content: ""; position: absolute;
  left: 58%; top: 30%; width: 17%; height: 19%;
  border-radius: 50%; background: rgba(255,255,255,0.98);
  z-index: 1; -webkit-box-shadow: 0 0 6px rgba(255,255,255,0.9); box-shadow: 0 0 6px rgba(255,255,255,0.9);
}
.nose-bridge {
  position: absolute; top: 54%; left: 50%; -webkit-transform: translateX(-50%); transform: translateX(-50%);
  width: 12%; height: 14%;
  background: -webkit-linear-gradient(left, transparent 0%, rgba(240,160,150,0.14) 35%, rgba(240,160,150,0.2) 50%, rgba(240,160,150,0.14) 65%, transparent 100%);
  background: linear-gradient(90deg, transparent 0%, rgba(240,160,150,0.14) 35%, rgba(240,160,150,0.2) 50%, rgba(240,160,150,0.14) 65%, transparent 100%);
  border-radius: 40%; pointer-events: none; z-index: 2;
}
.mouth {
  position: absolute; left: 50%; -webkit-transform: translateX(-50%); transform: translateX(-50%);
  -webkit-transform-origin: 50% 0%; transform-origin: 50% 0%;
  top: 70.5%; width: 38%; max-width: 200px; height: 22px;
  border-radius: 0 0 52% 52%; border: none;
  background:
    -webkit-radial-gradient(50% 0% ellipse, rgba(255,255,255,0.62) 0%, transparent 58%),
    -webkit-linear-gradient(top, rgba(255,245,250,0.65) 0%, rgba(251,146,170,0.45) 38%, rgba(236,72,120,0.62) 100%);
  background:
    radial-gradient(ellipse 95% 90% at 50% 0%, rgba(255,255,255,0.62) 0%, transparent 58%),
    linear-gradient(180deg, rgba(255,245,250,0.65) 0%, rgba(251,146,170,0.45) 38%, rgba(236,72,120,0.62) 100%);
  -webkit-box-shadow: 0 3px 0 0 #db2777, 0 8px 14px rgba(219,39,119,0.32), inset 0 2px 8px rgba(255,255,255,0.45);
  box-shadow: 0 3px 0 0 #db2777, 0 8px 14px rgba(219,39,119,0.32), inset 0 2px 8px rgba(255,255,255,0.45);
  z-index: 3;
}
.cheek {
  position: absolute; top: 56%; width: 21%; height: 11%;
  border-radius: 50%;
  background: -webkit-radial-gradient(circle, rgba(255,170,190,0.55) 0%, rgba(255,200,210,0.22) 42%, transparent 70%);
  background: radial-gradient(circle, rgba(255,170,190,0.55) 0%, rgba(255,200,210,0.22) 42%, transparent 70%);
  z-index: 2;
}
.cheek.left { left: 6%; }
.cheek.right { right: 6%; }

/* ── Voice wave ── */
.voice-wave {
  display: -webkit-flex; display: flex;
  -webkit-align-items: flex-end; align-items: flex-end; -webkit-justify-content: center; justify-content: center;
  width: 56%; height: 12%; margin-top: auto; padding-bottom: 2%;
  opacity: 0.18; position: relative; -webkit-flex-shrink: 0; flex-shrink: 0;
}
.voice-wave span {
  width: 10%; margin: 0 1.2%; min-height: 16%; border-radius: 999px;
  background: #8fd5ff; -webkit-box-shadow: 0 0 8px rgba(125,211,252,0.45); box-shadow: 0 0 8px rgba(125,211,252,0.45);
  -webkit-transform-origin: center bottom; transform-origin: center bottom;
  -webkit-animation: wave 1.2s ease-in-out infinite; animation: wave 1.2s ease-in-out infinite;
  -webkit-animation-play-state: paused; animation-play-state: paused;
}
.voice-wave span:nth-child(1) { -webkit-animation-delay: 0s; animation-delay: 0s; }
.voice-wave span:nth-child(2) { -webkit-animation-delay: 0.1s; animation-delay: 0.1s; }
.voice-wave span:nth-child(3) { -webkit-animation-delay: 0.2s; animation-delay: 0.2s; }
.voice-wave span:nth-child(4) { -webkit-animation-delay: 0.32s; animation-delay: 0.32s; }
.voice-wave span:nth-child(5) { -webkit-animation-delay: 0.44s; animation-delay: 0.44s; }
@-webkit-keyframes wave {
  0%,100% { -webkit-transform: scaleY(0.28); }
  18%     { -webkit-transform: scaleY(0.55); }
  38%     { -webkit-transform: scaleY(1.58); }
  58%     { -webkit-transform: scaleY(0.72); }
  78%     { -webkit-transform: scaleY(1.12); }
}
@keyframes wave {
  0%,100% { transform: scaleY(0.28); }
  18%     { transform: scaleY(0.55); }
  38%     { transform: scaleY(1.58); }
  58%     { transform: scaleY(0.72); }
  78%     { transform: scaleY(1.12); }
}
.face.awake .voice-wave,
.face.listening .voice-wave,
.face.thinking .voice-wave,
.face.triggered .voice-wave { opacity: 1; }
.face.awake .voice-wave span,
.face.listening .voice-wave span,
.face.thinking .voice-wave span,
.face.triggered .voice-wave span { -webkit-animation-play-state: running; animation-play-state: running; }
.face.thinking .voice-wave span { -webkit-animation-duration: 1.45s; animation-duration: 1.45s; }
.face.speaking .voice-wave span { -webkit-animation-play-state: running; animation-play-state: running; -webkit-animation-duration: 0.55s; animation-duration: 0.55s; }
.face.speaking .voice-wave { opacity: 1; }
.face.paused .voice-wave span { background: #fca5a5; -webkit-box-shadow: none; box-shadow: none; }

/* ── Sparkles ── */
.face-sparkles { position:absolute; top:0;left:0;right:0;bottom:0; pointer-events:none; z-index:10; overflow:visible; }
.sp {
  position: absolute; font-size: 0.6rem; line-height: 1;
  color: #f5d0fe; opacity: 0;
  text-shadow: 0 0 8px #c084fc, 0 0 18px rgba(192,132,252,0.45);
  -webkit-animation: sparkle-float 4.2s ease-in-out infinite; animation: sparkle-float 4.2s ease-in-out infinite;
}
.sp1 { top:3%;  left:9%;  -webkit-animation-delay:0s; animation-delay:0s; }
.sp2 { top:7%;  right:11%; font-size:0.45rem; -webkit-animation-delay:1.05s; animation-delay:1.05s; }
.sp3 { top:1%;  left:52%; -webkit-animation-delay:2.1s; animation-delay:2.1s; }
.sp4 { top:11%; right:5%; font-size:0.42rem; -webkit-animation-delay:0.62s; animation-delay:0.62s; }
@-webkit-keyframes sparkle-float {
  0%,100% { opacity:0;    -webkit-transform: translateY(0)    rotate(0deg)  scale(0.5); }
  25%,75% { opacity:0.92; -webkit-transform: translateY(-9px)  rotate(22deg) scale(1); }
  50%     { opacity:0.6;  -webkit-transform: translateY(-16px) rotate(44deg) scale(0.82); }
}
@keyframes sparkle-float {
  0%,100% { opacity:0;    transform: translateY(0)    rotate(0deg)  scale(0.5); }
  25%,75% { opacity:0.92; transform: translateY(-9px)  rotate(22deg) scale(1); }
  50%     { opacity:0.6;  transform: translateY(-16px) rotate(44deg) scale(0.82); }
}
.face.awake    ~ .face-sparkles .sp { color:#fde68a; text-shadow:0 0 8px #fb923c,0 0 18px rgba(251,146,60,0.4); -webkit-animation-duration:2.2s; animation-duration:2.2s; }
.face.thinking ~ .face-sparkles .sp { color:#e879f9; text-shadow:0 0 8px #c084fc,0 0 18px rgba(232,121,249,0.45); -webkit-animation-duration:3s; animation-duration:3s; }
.face.triggered ~ .face-sparkles .sp { color:#fb923c; text-shadow:0 0 8px #f97316,0 0 18px rgba(249,115,22,0.4); -webkit-animation-duration:1.6s; animation-duration:1.6s; }
.face.paused ~ .face-sparkles .sp,
.face.sleeping ~ .face-sparkles .sp { -webkit-animation:none!important; animation:none!important; opacity:0!important; }

/* ── State expressions ── */
@-webkit-keyframes lip-soft {
  0%,100% { -webkit-transform: translateX(-50%) scale(1,1); }
  28%     { -webkit-transform: translateX(-50%) scale(1.03,1.05); }
  48%     { -webkit-transform: translateX(-50%) scale(1.08,1.14); }
  72%     { -webkit-transform: translateX(-50%) scale(1.02,1.04); }
}
@keyframes lip-soft {
  0%,100% { transform: translateX(-50%) scale(1,1); }
  28%     { transform: translateX(-50%) scale(1.03,1.05); }
  48%     { transform: translateX(-50%) scale(1.08,1.14); }
  72%     { transform: translateX(-50%) scale(1.02,1.04); }
}
@-webkit-keyframes lip-think {
  0%,100% { -webkit-transform: translateX(-50%) scale(1,1); opacity:0.9; }
  45%     { -webkit-transform: translateX(-50%) scale(0.9,0.93); opacity:1; }
  70%     { -webkit-transform: translateX(-50%) scale(0.95,0.98); opacity:0.94; }
}
@keyframes lip-think {
  0%,100% { transform: translateX(-50%) scale(1,1); opacity:0.9; }
  45%     { transform: translateX(-50%) scale(0.9,0.93); opacity:1; }
  70%     { transform: translateX(-50%) scale(0.95,0.98); opacity:0.94; }
}
@-webkit-keyframes eye-glow {
  0%,100% { opacity:1;    -webkit-transform:translate(-50%,-50%) scale(1); }
  32%     { opacity:0.78; -webkit-transform:translate(-50%,-50%) scale(0.91); }
  55%     { opacity:0.68; -webkit-transform:translate(-50%,-50%) scale(0.85); }
  82%     { opacity:0.92; -webkit-transform:translate(-50%,-50%) scale(0.96); }
}
@keyframes eye-glow {
  0%,100% { opacity:1;    transform:translate(-50%,-50%) scale(1); }
  32%     { opacity:0.78; transform:translate(-50%,-50%) scale(0.91); }
  55%     { opacity:0.68; transform:translate(-50%,-50%) scale(0.85); }
  82%     { opacity:0.92; transform:translate(-50%,-50%) scale(0.96); }
}
@-webkit-keyframes eye-glow-think {
  0%,100% { opacity:1;    -webkit-transform:translate(-46%,-50%) scale(1); }
  35%     { opacity:0.8;  -webkit-transform:translate(-46%,-50%) scale(0.9); }
  60%     { opacity:0.72; -webkit-transform:translate(-46%,-50%) scale(0.84); }
  85%     { opacity:0.88; -webkit-transform:translate(-46%,-50%) scale(0.94); }
}
@keyframes eye-glow-think {
  0%,100% { opacity:1;    transform:translate(-46%,-50%) scale(1); }
  35%     { opacity:0.8;  transform:translate(-46%,-50%) scale(0.9); }
  60%     { opacity:0.72; transform:translate(-46%,-50%) scale(0.84); }
  85%     { opacity:0.88; transform:translate(-46%,-50%) scale(0.94); }
}
@-webkit-keyframes eye-sparkle {
  0%,100% { -webkit-transform:scale(1); opacity:1; }
  38%     { -webkit-transform:scale(1.12); opacity:0.88; }
  62%     { -webkit-transform:scale(1.04); opacity:0.95; }
}
@keyframes eye-sparkle {
  0%,100% { transform:scale(1); opacity:1; }
  38%     { transform:scale(1.12); opacity:0.88; }
  62%     { transform:scale(1.04); opacity:0.95; }
}
@-webkit-keyframes eye-scan {
  0%,100% { -webkit-transform:translate(-50%,-50%); }
  22%     { -webkit-transform:translate(-55%,-51%); }
  48%     { -webkit-transform:translate(-42%,-49%); }
  72%     { -webkit-transform:translate(-56%,-50%); }
}
@keyframes eye-scan {
  0%,100% { transform:translate(-50%,-50%); }
  22%     { transform:translate(-55%,-51%); }
  48%     { transform:translate(-42%,-49%); }
  72%     { transform:translate(-56%,-50%); }
}
@-webkit-keyframes cheek-pulse { 0%,100%{opacity:0.65} 50%{opacity:1} }
@keyframes cheek-pulse { 0%,100%{opacity:0.65} 50%{opacity:1} }

.face.listening .mouth, .face.awake .mouth {
  width: 36%; top: 70.5%; height: 21px;
  /* no lip animation when listening — lips only move when actually speaking */
}
.face.listening .eye:after { -webkit-animation: eye-glow 1.4s ease-in-out infinite; animation: eye-glow 1.4s ease-in-out infinite; }
.face.listening .eye:before { -webkit-animation: eye-sparkle 2.4s ease-in-out infinite; animation: eye-sparkle 2.4s ease-in-out infinite; }
.face.listening .cheek { -webkit-animation: cheek-pulse 3.8s ease-in-out infinite; animation: cheek-pulse 3.8s ease-in-out infinite; }
.face.awake .brow.left  { -webkit-transform: rotate(-8deg) translateY(0); transform: rotate(-8deg) translateY(0); }
.face.awake .brow.right { -webkit-transform: rotate(8deg) translateY(0); transform: rotate(8deg) translateY(0); }
.face.awake .eye:after { -webkit-animation: eye-glow 1.05s ease-in-out infinite; animation: eye-glow 1.05s ease-in-out infinite; }
.face.awake .eye:before { -webkit-animation: eye-sparkle 1.6s ease-in-out infinite; animation: eye-sparkle 1.6s ease-in-out infinite; }
.face.awake .cheek { -webkit-animation: cheek-pulse 1.5s ease-in-out infinite; animation: cheek-pulse 1.5s ease-in-out infinite; }
/* .face.awake .mouth — no lip animation; overridden above */
.face.thinking .eye:after { top:46%; left:52%; -webkit-animation: eye-glow-think 2.2s ease-in-out infinite; animation: eye-glow-think 2.2s ease-in-out infinite; }
.face.thinking .brow.left  { -webkit-transform: rotate(-5deg) translateY(0); transform: rotate(-5deg) translateY(0); }
.face.thinking .brow.right { -webkit-transform: rotate(5deg) translateY(0); transform: rotate(5deg) translateY(0); }
.face.thinking .mouth {
  width: 32%; height: 13px; top: 71.5%; border-radius: 999px;
  background: -webkit-linear-gradient(top, rgba(253,242,248,0.5), rgba(236,72,153,0.48));
  background: linear-gradient(180deg, rgba(253,242,248,0.5), rgba(236,72,153,0.48));
  -webkit-box-shadow: 0 3px 0 0 #9d174d, inset 0 1px 4px rgba(255,255,255,0.35);
  box-shadow: 0 3px 0 0 #9d174d, inset 0 1px 4px rgba(255,255,255,0.35);
  /* no lip animation while thinking */
}
.face.thinking .eye:before { -webkit-animation: eye-sparkle 3.4s ease-in-out infinite; animation: eye-sparkle 3.4s ease-in-out infinite; }
.face.thinking .cheek { -webkit-animation: cheek-pulse 3s ease-in-out infinite; animation: cheek-pulse 3s ease-in-out infinite; }
.face.triggered .mouth {
  width: 20%; height: 18%; border-radius: 50%; top: 69%;
  border: 3px solid #f472b6;
  background: -webkit-radial-gradient(40% 35%, rgba(255,255,255,0.35), rgba(251,113,133,0.45));
  background: radial-gradient(circle at 40% 35%, rgba(255,255,255,0.35), rgba(251,113,133,0.45));
  -webkit-box-shadow: 0 4px 0 0 #be185d, inset 0 2px 8px rgba(255,255,255,0.25);
  box-shadow: 0 4px 0 0 #be185d, inset 0 2px 8px rgba(255,255,255,0.25);
  /* no lip animation for triggered */
}
.face.triggered .eye:after { -webkit-animation: eye-glow 0.6s ease-in-out infinite; animation: eye-glow 0.6s ease-in-out infinite; }
.face.triggered .eye:before { -webkit-animation: eye-sparkle 1.1s ease-in-out infinite; animation: eye-sparkle 1.1s ease-in-out infinite; }
.face.speaking .mouth { -webkit-animation: lip-soft 0.32s ease-in-out infinite; animation: lip-soft 0.32s ease-in-out infinite; }
.face.speaking .eye { -webkit-animation: none; animation: none; -webkit-transform: scaleY(1); transform: scaleY(1); }
.face.speaking .eye:after { -webkit-animation: eye-glow 1.8s ease-in-out infinite; animation: eye-glow 1.8s ease-in-out infinite; }
.face.paused .eye { -webkit-transform: none; transform: none; -webkit-animation: none; animation: none; }
.face.paused .eye:before, .face.paused .eye:after { -webkit-animation: none !important; animation: none !important; }
.face.paused .eye-lashes { opacity: 0.38; -webkit-filter: saturate(0.85); filter: saturate(0.85); }
.face.paused .head { -webkit-filter: saturate(0.82) brightness(0.96); filter: saturate(0.82) brightness(0.96); }
.face.paused .eye:after { background: #5a3a40; }
.face.paused .mouth {
  width: 30%; height: 12px; border-radius: 999px; top: 74%;
  background: -webkit-linear-gradient(top, rgba(80,50,58,0.5), rgba(120,60,72,0.65));
  background: linear-gradient(180deg, rgba(80,50,58,0.5), rgba(120,60,72,0.65));
  -webkit-box-shadow: 0 -3px 0 0 rgba(120,60,72,0.95), inset 0 2px 4px rgba(0,0,0,0.15);
  box-shadow: 0 -3px 0 0 rgba(120,60,72,0.95), inset 0 2px 4px rgba(0,0,0,0.15);
  -webkit-animation: none !important; animation: none !important;
}
.face.paused .cheek { opacity: 0.3; }
@-webkit-keyframes sleep-breathe { 0%,100%{opacity:1} 50%{opacity:0.7} }
@keyframes sleep-breathe { 0%,100%{opacity:1} 50%{opacity:0.7} }
.face.sleeping {
  -webkit-filter: saturate(0.2) brightness(0.55); filter: saturate(0.2) brightness(0.55);
  -webkit-transition: filter 0.8s ease; transition: filter 0.8s ease;
}
.face.sleeping .eye {
  -webkit-transform: scaleY(0.18); transform: scaleY(0.18);
  -webkit-transition: transform 0.6s ease; transition: transform 0.6s ease;
  -webkit-animation: none !important; animation: none !important;
}
.face.sleeping .eye:before, .face.sleeping .eye:after { -webkit-animation: none !important; animation: none !important; opacity: 0 !important; }
.face.sleeping .mouth { width:16%; height:7px; border-radius:2px; top:69.5%; opacity:0.35; -webkit-animation:none!important; animation:none!important; }
.face.sleeping .cheek { opacity:0.15; -webkit-animation:none!important; animation:none!important; }
.face.sleeping .voice-wave { opacity:0!important; }
.face.awake, .face.thinking, .face.triggered, .face.listening { -webkit-transition: filter 0.6s ease; transition: filter 0.6s ease; }
.face.awake .eye, .face.thinking .eye, .face.triggered .eye { -webkit-transition: transform 0.5s ease; transition: transform 0.5s ease; }

/* ── Bottom nav ── */
.pinky-nav {
  display: -webkit-flex; display: flex; border-top: 1px solid #24243a;
  -webkit-flex-shrink: 0; flex-shrink: 0; background: rgba(0,0,0,0.35);
}
.pinky-nav-item {
  -webkit-flex: 1; flex: 1; padding: 10px 0 14px;
  display: -webkit-flex; display: flex; -webkit-flex-direction: column; flex-direction: column;
  -webkit-align-items: center; align-items: center; gap: 3px;
  cursor: pointer; opacity: 0.45; -webkit-transition: opacity 0.2s; transition: opacity 0.2s;
  border: none; background: none; color: #e8e8f4;
  font-size: 0.62rem; letter-spacing: 0.3px;
  min-height: 56px; -webkit-justify-content: center; justify-content: center;
}
.pinky-nav-item:hover { opacity: 0.75; }
.pinky-nav-item.active { opacity: 1; color: #c084fc; }
.pinky-nav-ic { font-size: 1.2rem; line-height: 1; }

/* ── Camera panel ── */
.camera-area {
  display: none; -webkit-flex-direction: column; flex-direction: column;
  -webkit-align-items: center; align-items: center; overflow: hidden; gap: 0;
}
.camera-area.open {
  display: -webkit-flex; display: flex;
  -webkit-flex: 1; flex: 1;
}
.cam-feed-wrap {
  position: relative; width: 100%; max-width: 640px; -webkit-flex-shrink: 0; flex-shrink: 0;
  background: #000; overflow: hidden;
}
.cam-feed-wrap img { width: 100%; height: auto; display: none; }
.cam-feed-wrap img.visible { display: block; }
.cam-offline {
  min-height: 180px;
  display: -webkit-flex; display: flex;
  -webkit-align-items: center; align-items: center; -webkit-justify-content: center; justify-content: center;
  font-size: 0.88rem; color: #666683; background: rgba(0,0,0,0.6);
}
.cam-toolbar {
  width: 100%; max-width: 640px; padding: 5px 10px;
  display: -webkit-flex; display: flex;
  -webkit-justify-content: space-between; justify-content: space-between;
  -webkit-align-items: center; align-items: center;
  border-bottom: 1px solid #24243a; background: rgba(0,0,0,0.2);
}
.cam-detect-feed {
  -webkit-flex: 1; flex: 1; overflow-y: auto; width: 100%; max-width: 640px; padding: 8px 12px;
}
.cam-detect-feed::-webkit-scrollbar { width: 3px; }
.cam-detect-feed::-webkit-scrollbar-thumb { background: #24243a; border-radius: 2px; }
.det-row {
  display: -webkit-flex; display: flex; -webkit-align-items: baseline; align-items: baseline; gap: 8px;
  padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
  font-size: 0.78rem; color: #666683;
}
.det-row .det-time { -webkit-flex-shrink: 0; flex-shrink: 0; font-size: 0.68rem; color: #444466; }
.det-row .det-body { -webkit-flex: 1; flex: 1; color: #e8e8f4; }
.det-chip {
  display: inline-block; padding: 1px 6px; border-radius: 6px; font-size: 0.7rem; margin: 0 2px;
}
.det-chip.person  { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.det-chip.gesture { background: rgba(192,132,252,0.15); color: #c084fc; border: 1px solid rgba(192,132,252,0.3); }
.det-chip.object  { background: rgba(251,191,36,0.13); color: #fbbf24; border: 1px solid rgba(251,191,36,0.28); }

/* ── Darts panel ── */
.darts-area {
  display: none; -webkit-flex-direction: column; flex-direction: column;
  -webkit-align-items: center; align-items: center; overflow: hidden; gap: 0;
}
.darts-area.open {
  display: -webkit-flex; display: flex;
  -webkit-flex: 1; flex: 1;
}
.darts-toolbar {
  width: 100%; max-width: 640px; padding: 6px 10px;
  display: -webkit-flex; display: flex;
  -webkit-justify-content: center; justify-content: center; -webkit-align-items: center; align-items: center; gap: 8px;
  border-bottom: 1px solid #24243a; background: rgba(0,0,0,0.2); -webkit-flex-shrink: 0; flex-shrink: 0;
}
.darts-btn {
  padding: 8px 12px; border-radius: 8px; font-size: 0.8rem; cursor: pointer;
  border: 1px solid #24243a; background: rgba(255,255,255,0.05); color: #e8e8f4;
  -webkit-transition: background 0.15s; transition: background 0.15s;
  min-height: 44px;
}
.darts-btn:hover  { background: rgba(255,255,255,0.1); }
.darts-btn:disabled { opacity: 0.5; cursor: default; }
.darts-btn.speak-btn { border-color: rgba(192,132,252,0.45); color: #c084fc; }
.darts-btn.cal-btn   { border-color: rgba(251,191,36,0.45); color: #fbbf24; }
.darts-btn.next-btn  { border-color: rgba(74,222,128,0.45); color: #4ade80; }
.darts-btn.new-btn   { border-color: rgba(248,113,113,0.4); color: #f87171; }
.darts-btn.next-btn.awaiting {
  background: rgba(74,222,128,0.18); font-weight: 700;
  -webkit-animation: pulse-green 1.1s ease-in-out infinite; animation: pulse-green 1.1s ease-in-out infinite;
}
@-webkit-keyframes pulse-green {
  0%,100% { -webkit-box-shadow: 0 0 6px rgba(74,222,128,0.25); }
  50%     { -webkit-box-shadow: 0 0 18px rgba(74,222,128,0.55); }
}
@keyframes pulse-green {
  0%,100% { box-shadow: 0 0 6px rgba(74,222,128,0.25); }
  50%     { box-shadow: 0 0 18px rgba(74,222,128,0.55); }
}
.darts-gamebar {
  width: 100%; max-width: 640px; -webkit-flex-shrink: 0; flex-shrink: 0;
  display: -webkit-flex; display: flex; -webkit-align-items: center; align-items: center; gap: 10px;
  padding: 7px 14px; background: rgba(0,0,0,0.3); border-bottom: 1px solid #24243a;
}
.dgb-player { font-size: 1rem; font-weight: 800; color: #c084fc; min-width: 26px; }
.dgb-dots   { -webkit-flex: 1; flex: 1; font-size: 1.15rem; letter-spacing: 3px; }
.dgb-score  { font-size: 1rem; font-weight: 700; color: #fbbf24; }
.darts-top-row {
  display: -webkit-flex; display: flex; -webkit-flex-direction: row; flex-direction: row; -webkit-align-items: flex-start; align-items: flex-start;
  width: 100%; max-width: 860px; gap: 0; -webkit-flex-shrink: 0; flex-shrink: 0;
}
.darts-leaderboard {
  -webkit-flex: 1; flex: 1; min-width: 130px; max-width: 190px; -webkit-flex-shrink: 0; flex-shrink: 0;
  padding: 6px 8px; -webkit-align-self: stretch; align-self: stretch;
  background: rgba(255,255,255,0.03);
  border-left: 1px solid rgba(255,255,255,0.07);
  border-radius: 0 8px 8px 0; overflow-y: auto;
}
.dart-lb-title { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.07em; color: #64748b; padding: 4px 0 3px; }
.dart-lb-empty { font-size: 0.7rem; color: #475569; padding: 6px 2px; text-align: center; }
.dart-lb-row {
  display: -webkit-flex; display: flex; -webkit-align-items: center; align-items: center; gap: 6px;
  padding: 5px 6px; border-radius: 8px; margin-bottom: 4px;
  background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.05);
  -webkit-flex-wrap: wrap; flex-wrap: wrap;
}
.dart-lb-row.lb-active { border-color: rgba(192,132,252,0.35); background: rgba(192,132,252,0.06); }
.dart-lb-player { font-size: 0.85rem; font-weight: 800; color: #c084fc; min-width: 22px; }
.dart-lb-turns  { font-size: 0.65rem; color: #64748b; -webkit-flex: 1; flex: 1; }
.dart-lb-shots  { font-size: 0.65rem; color: #94a3b8; width: 100%; padding-left: 2px; }
.dart-lb-total  { font-size: 1rem; font-weight: 800; text-align: right; }
.darts-hit-layer { position: absolute; top: 0; left: 0; right: 0; bottom: 0; pointer-events: none; }
.dart-hit-dot {
  position: absolute;
  width: 28px; height: 28px; border-radius: 50%;
  -webkit-transform: translate(-50%,-50%); transform: translate(-50%,-50%);
  display: -webkit-flex; display: flex; -webkit-align-items: center; align-items: center; -webkit-justify-content: center; justify-content: center;
  font-size: 0.6rem; font-weight: 800; color: #fff;
  border: 2px solid rgba(255,255,255,0.8);
  -webkit-box-shadow: 0 0 10px rgba(0,0,0,0.7); box-shadow: 0 0 10px rgba(0,0,0,0.7);
  -webkit-animation: dart-land 0.28s ease; animation: dart-land 0.28s ease;
  pointer-events: none;
}
@-webkit-keyframes dart-land {
  0%   { -webkit-transform: translate(-50%,-50%) scale(0.2); opacity: 0.5; }
  70%  { -webkit-transform: translate(-50%,-50%) scale(1.18); }
  100% { -webkit-transform: translate(-50%,-50%) scale(1); opacity: 1; }
}
@keyframes dart-land {
  0%   { transform: translate(-50%,-50%) scale(0.2); opacity: 0.5; }
  70%  { transform: translate(-50%,-50%) scale(1.18); }
  100% { transform: translate(-50%,-50%) scale(1); opacity: 1; }
}
.dart-score-badge {
  position: absolute; top: -16px; left: 50%; -webkit-transform: translateX(-50%); transform: translateX(-50%);
  background: rgba(0,0,0,0.75); color: #fff; font-size: 0.58rem; font-weight: 700;
  padding: 1px 5px; border-radius: 4px; white-space: nowrap;
  border: 1px solid rgba(255,255,255,0.2);
}
.darts-score-list {
  width: 100%; max-width: 640px; -webkit-flex: 1; flex: 1; overflow-y: auto;
  padding: 8px 14px; display: -webkit-flex; display: flex; -webkit-flex-direction: column; flex-direction: column; gap: 4px;
}
.darts-score-list::-webkit-scrollbar { width: 3px; }
.darts-score-list::-webkit-scrollbar-thumb { background: #24243a; border-radius: 2px; }
.dart-score-row {
  display: -webkit-flex; display: flex; -webkit-align-items: center; align-items: center; gap: 10px;
  padding: 5px 8px; border-radius: 8px;
  background: rgba(255,255,255,0.04); border: 1px solid #24243a;
  font-size: 0.82rem;
}
.dart-score-num   { font-size: 1.1rem; font-weight: 800; min-width: 38px; text-align: center; }
.dart-score-label { -webkit-flex: 1; flex: 1; color: #666683; }
.dart-score-pos   { font-size: 0.7rem; color: #64748b; font-family: monospace; }

/* ── Log panel ── */
.log-area {
  overflow-y: auto; padding: 0; display: none; -webkit-flex-direction: column; flex-direction: column; width: 100%;
}
.log-area.open {
  display: -webkit-flex; display: flex;
  -webkit-flex: 1; flex: 1;
}
.log-area::-webkit-scrollbar { width: 4px; }
.log-area::-webkit-scrollbar-thumb { background: #24243a; border-radius: 2px; }
.log-toolbar {
  -webkit-flex-shrink: 0; flex-shrink: 0; padding: 8px 20px; border-bottom: 1px solid #24243a;
  display: -webkit-flex; display: flex; -webkit-justify-content: space-between; justify-content: space-between; -webkit-align-items: center; align-items: center;
  background: rgba(0,0,0,0.25); font-size: 0.8rem; color: #666683;
}
.log-clear-btn {
  font-size: 0.75rem; padding: 6px 12px; border-radius: 6px; cursor: pointer;
  background: rgba(255,255,255,0.05); border: 1px solid #24243a; color: #666683;
  min-height: 36px;
}
.log-clear-btn:hover { background: rgba(255,255,255,0.1); }
.log-entries {
  -webkit-flex: 1; flex: 1; overflow-y: auto; padding: 0;
  display: -webkit-flex; display: flex; -webkit-flex-direction: column; flex-direction: column; gap: 0;
  width: 100%;
}
.log-entries::-webkit-scrollbar { width: 4px; }
.log-entries::-webkit-scrollbar-thumb { background: #24243a; border-radius: 2px; }
.log-entry {
  padding: 10px 16px 8px 14px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  display: -webkit-flex; display: flex; gap: 12px; -webkit-align-items: flex-start; align-items: flex-start;
  border-left: 3px solid transparent;
  width: 100%;
}
.log-entry:hover { background: rgba(255,255,255,0.02); }
.log-entry.log-stt   { background: rgba(147,197,253,0.04); }
.log-entry.log-pinku { background: rgba(192,132,252,0.04); }
.log-entry.log-error { background: rgba(248,113,113,0.05); }
.log-icon { font-size: 1.15rem; -webkit-flex-shrink: 0; flex-shrink: 0; width: 26px; text-align: center; line-height: 1.5; }
.log-body { -webkit-flex: 1; flex: 1; min-width: 0; }
.log-msg  { display: block; font-size: 1rem; line-height: 1.5; word-break: break-word; color: #c8ccdc; }
.log-entry.log-stt   .log-msg { color: #bfdbfe; font-weight: 500; font-size: 1.05rem; }
.log-entry.log-pinku .log-msg { color: #e9d5ff; font-weight: 500; font-size: 1.05rem; }
.log-entry.log-error .log-msg { color: #fca5a5; }
.log-entry.log-warn  .log-msg { color: #fde68a; }
.log-entry.log-wake  .log-msg { color: #fde68a; }
.log-ts { display: block; font-size: 0.7rem; color: #44475a; margin-top: 3px; font-family: 'SF Mono','Menlo','Consolas',monospace; }
</style>
</head>
<body>
<div id="root"></div>

<script type="text/babel">
// ── Constants ──────────────────────────────────────────────────────────────────
const STATE_UI = {
  sleeping:  { icon:'💤', label:'Sleeping',    sub:'Face or "Pinku" to wake' },
  awake:     { icon:'👂', label:'Listening',   sub:'Go ahead, I\'m listening…' },
  thinking:  { icon:'⚡', label:'Thinking…',   sub:'Sending to Gemini' },
  speaking:  { icon:'🗣️', label:'Speaking…',   sub:'Tap ⏸ to stop' },
  paused:    { icon:'🔇', label:'Muted',       sub:'Tap 🎙️ or wave to unmute' },
  triggered: { icon:'⚡', label:'Working…',    sub:'Running your command' },
};

const LOG_COLORS = {
  stt:'#93c5fd', wake:'#fbbf24', route:'#a78bfa', chat:'#86efac',
  tts:'#f9a8d4', camera:'#67e8f9', gesture:'#c084fc', error:'#f87171',
  warn:'#fbbf24', info:'#64748b', pinku:'#c084fc', laser:'#4ade80', source:'#86efac',
  you:'#93c5fd', user:'#93c5fd', llm:'#64748b', cam:'#67e8f9',
};
const LOG_ICONS = {
  stt:'🎙️', wake:'⚡', route:'🔀', chat:'💬', tts:'🔊', camera:'📷',
  gesture:'✋', error:'🔴', warn:'🟡', info:'⚪', pinku:'🤖', laser:'🎯', source:'📖',
  you:'🧑', user:'🧑', llm:'🤖', cam:'📷',
};

const DART_COLORS = {
  100: '#ef4444', 75: '#f97316', 50: '#eab308', 25: '#84cc16', 10: '#22d3ee', 0: '#6b7280',
};

function dartColor(score) {
  const keys = [100,75,50,25,10,0];
  for (const k of keys) { if (score >= k) return DART_COLORS[k]; }
  return '#6b7280';
}

function getFaceClass(status, detectFlash) {
  if (detectFlash) return 'triggered';
  if (status.muted) return 'paused';
  if (status.speaking) return 'speaking';
  const map = {
    idle:'sleeping', sleeping:'sleeping', awake:'awake',
    processing:'thinking', speaking:'speaking', muted:'paused', detection:'triggered',
  };
  return map[status.state] || 'sleeping';
}

function apiAction(name) {
  fetch('/api/action', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({action: name}),
  }).catch(() => {});
}

function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Face component ─────────────────────────────────────────────────────────────
function Face({ faceClass }) {
  return React.createElement('div', { className: 'face ' + faceClass },
    React.createElement('div', { className: 'head' },
      React.createElement('div', { className: 'hair-side left' }),
      React.createElement('div', { className: 'hair-side right' }),
      React.createElement('div', { className: 'hair-fringe' }),
      React.createElement('div', { className: 'brows' },
        React.createElement('div', { className: 'brow left' }),
        React.createElement('div', { className: 'brow right' })
      ),
      React.createElement('div', { className: 'eyes' },
        React.createElement('div', { className: 'eye eye-left' },
          React.createElement('span', { className: 'eye-lashes' },
            React.createElement('span', { className: 'lash' }),
            React.createElement('span', { className: 'lash' }),
            React.createElement('span', { className: 'lash' }),
            React.createElement('span', { className: 'lash' }),
            React.createElement('span', { className: 'lash' })
          )
        ),
        React.createElement('div', { className: 'eye eye-right' },
          React.createElement('span', { className: 'eye-lashes' },
            React.createElement('span', { className: 'lash' }),
            React.createElement('span', { className: 'lash' }),
            React.createElement('span', { className: 'lash' }),
            React.createElement('span', { className: 'lash' }),
            React.createElement('span', { className: 'lash' })
          )
        )
      ),
      React.createElement('div', { className: 'nose-bridge' }),
      React.createElement('div', { className: 'mouth' }),
      React.createElement('div', { className: 'cheek left' }),
      React.createElement('div', { className: 'cheek right' })
    ),
    React.createElement('div', { className: 'voice-wave' },
      React.createElement('span'), React.createElement('span'),
      React.createElement('span'), React.createElement('span'),
      React.createElement('span')
    )
  );
}

// ── Header ─────────────────────────────────────────────────────────────────────
function Header({ status, onWake, onPause, onMuteToggle, onSleep, onRestart }) {
  const faceClass = getFaceClass(status, false);
  const isMuted = status.muted;
  const isActive = status.state === 'awake' || status.state === 'processing' || status.speaking;

  function handleRestart() {
    if (!window.confirm('git pull + restart Pinku?')) return;
    onRestart();
  }

  return React.createElement('header', { className: 'header' },
    React.createElement('div', { className: 'header-left' },
      React.createElement('div', { className: 'dot-wrap' },
        React.createElement('div', { className: 'avatar-ring' }, 'P'),
        React.createElement('div', { className: 'status-dot ' + faceClass })
      ),
      React.createElement('div', null,
        React.createElement('div', { className: 'app-name' }, 'Pinku'),
        React.createElement('div', { className: 'status-pills' },
          React.createElement('span', {
            className: 'spill' + (status.persons > 0 ? ' on' : ''),
            title: 'Person detected'
          }, '👤'),
          React.createElement('span', {
            className: 'spill' + (isActive ? ' on' : ''),
            title: 'Session'
          }, '💬')
        )
      )
    ),
    React.createElement('div', { className: 'header-btns' },
      !isMuted && React.createElement('button', {
        className: 'hdr-btn stop-btn',
        title: isActive ? 'Sleep — end conversation, stay listening' : 'Sleep',
        onClick: onSleep
      }, '💤'),
      React.createElement('button', {
        className: 'hdr-btn mute-btn' + (isMuted ? ' active' : ''),
        title: isMuted ? 'Unmute — back to standby' : 'Mute — hard off, no listening',
        onClick: onMuteToggle
      }, isMuted ? '🎙️' : '🔇'),
      React.createElement('button', {
        className: 'hdr-btn restart-btn',
        title: 'git pull + restart Pinku',
        onClick: handleRestart
      }, '↺')
    )
  );
}

// ── HomeTab ────────────────────────────────────────────────────────────────────
function HomeTab({ status, detectFlash, connected, onWake, lastConvTime }) {
  const faceClass = getFaceClass(status, detectFlash);
  const ui = STATE_UI[faceClass] || STATE_UI.sleeping;
  const showBubble = faceClass === 'thinking' || faceClass === 'triggered';
  const showQuestion = !!status.last_transcript;
  const showReply = !!status.last_reply;
  const showPersonBadge = status.persons > 0;

  // Format HH:MM for the conversation timestamp
  function fmtTime(ts) {
    if (!ts) return '';
    var d = new Date(ts);
    var h = d.getHours(), m = d.getMinutes();
    return (h < 10 ? '0' : '') + h + ':' + (m < 10 ? '0' : '') + m;
  }

  // Dynamic sub text — show question when thinking, reply snippet when speaking
  var dynamicSub = ui.sub;
  if (faceClass === 'thinking' && status.last_transcript) {
    var q = status.last_transcript;
    dynamicSub = '“' + (q.length > 32 ? q.slice(0, 32) + '…' : q) + '”';
  } else if (faceClass === 'speaking' && status.last_reply) {
    var r = status.last_reply;
    dynamicSub = r.length > 38 ? r.slice(0, 38) + '…' : r;
  }

  var convTimeLabel = lastConvTime ? fmtTime(lastConvTime) : '';

  return React.createElement('div', { className: 'main-area' },
    React.createElement('div', { className: 'pinky-stage' },
      showQuestion && React.createElement('div', { className: 'question-cloud has-text' },
        React.createElement('span', { className: 'question-cloud-label' },
          'You said' + (convTimeLabel ? '  ·  ' + convTimeLabel : '')
        ),
        React.createElement('span', null, status.last_transcript)
      ),

      React.createElement('div', { className: 'face-with-actions' },
        React.createElement('button', {
          className: 'wake-btn-top',
          title: 'Tap to speak',
          onClick: onWake
        }, '🎙️'),

        React.createElement('div', { className: 'face-wrap ' + faceClass },
          React.createElement('div', {
            className: 'face-bubble' + (showBubble ? ' visible' : '') + (faceClass === 'triggered' ? ' triggered' : '')
          },
            React.createElement('span', { className: 'bubble-dot d1' }),
            React.createElement('span', { className: 'bubble-dot d2' }),
            React.createElement('span', { className: 'bubble-dot d3' })
          ),
          React.createElement(Face, { faceClass: faceClass }),
          React.createElement('div', { className: 'face-sparkles' },
            React.createElement('span', { className: 'sp sp1' }, '✦'),
            React.createElement('span', { className: 'sp sp2' }, '✦'),
            React.createElement('span', { className: 'sp sp3' }, '⋆'),
            React.createElement('span', { className: 'sp sp4' }, '⋆')
          )
        )
      ),

      showReply && React.createElement('div', { className: 'reply-cloud has-text' },
        React.createElement('span', { className: 'reply-cloud-label' }, 'Pinku said'),
        React.createElement('span', { className: 'reply-cloud-body' }, status.last_reply)
      ),

      React.createElement('div', { className: 'person-badge' + (showPersonBadge ? ' visible' : '') },
        React.createElement('span', { className: 'person-dot' }),
        React.createElement('span', null, status.persons > 1 ? status.persons + ' people here' : 'Someone here')
      ),

      React.createElement('div', { className: 'status-bar ' + faceClass },
        React.createElement('span', { className: 'status-bar-icon ' + faceClass }, ui.icon),
        React.createElement('div', { className: 'status-bar-text' },
          React.createElement('span', { className: 'status-bar-label' }, ui.label),
          React.createElement('span', { className: 'status-bar-sub' }, dynamicSub)
        ),
        React.createElement('div', { className: 'status-bar-right' },
          React.createElement('span', { className: 'ws-dot ' + (connected ? 'on' : 'off') })
        )
      )
    )
  );
}

// ── CameraTab ──────────────────────────────────────────────────────────────────
function CameraTab({ active, camStatus, detStatus, detections }) {
  const imgRef = React.useRef(null);
  const timerRef = React.useRef(null);
  const [camOnline, setCamOnline] = React.useState(false);
  const [permBtnText, setPermBtnText] = React.useState('🔓 Permission');
  const [permBtnDisabled, setPermBtnDisabled] = React.useState(false);

  function refreshCam() {
    if (!active) return;
    const tmp = new Image();
    tmp.onload = function() {
      if (imgRef.current) {
        imgRef.current.src = tmp.src;
        imgRef.current.className = 'visible';
      }
      setCamOnline(true);
      timerRef.current = setTimeout(refreshCam, 200);
    };
    tmp.onerror = function() {
      if (imgRef.current) imgRef.current.className = '';
      setCamOnline(false);
      timerRef.current = setTimeout(refreshCam, 2000);
    };
    tmp.src = '/api/camera.jpg?t=' + Date.now();
  }

  React.useEffect(function() {
    if (active) {
      refreshCam();
    }
    return function() {
      clearTimeout(timerRef.current);
    };
  }, [active]);

  function requestPermission() {
    setPermBtnText('⏳ Requesting…');
    setPermBtnDisabled(true);
    fetch('/api/camera/request-permission').then(function(r) { return r.json(); }).then(function(s) {
      if (s.ok) {
        setPermBtnText('✅ Granted!');
      } else {
        setPermBtnText('❌ Denied');
        setPermBtnDisabled(false);
      }
      setTimeout(function() { setPermBtnText('🔓 Permission'); setPermBtnDisabled(false); }, 4000);
    }).catch(function() { setPermBtnText('🔓 Permission'); setPermBtnDisabled(false); });
  }

  const statusMsg = camStatus && detStatus ? ('Camera: ' + camStatus + ' | Detection: ' + detStatus) : 'Checking…';
  const offlineMsg = camStatus ? ('📷 Camera: ' + camStatus + ' | Detection: ' + (detStatus || '?')) : '📷 Camera offline or not started';

  return React.createElement('div', { className: 'camera-area open' },
    React.createElement('div', { className: 'cam-feed-wrap' },
      React.createElement('img', { ref: imgRef, alt: 'Camera', draggable: false }),
      !camOnline && React.createElement('div', { className: 'cam-offline' }, offlineMsg)
    ),
    React.createElement('div', { className: 'cam-toolbar' },
      React.createElement('span', { style: { fontSize: '0.72rem', color: '#64748b' } }, statusMsg),
      React.createElement('button', {
        className: 'log-clear-btn',
        onClick: requestPermission,
        disabled: permBtnDisabled
      }, permBtnText)
    ),
    React.createElement('div', { className: 'cam-detect-feed' },
      detections.map(function(ev, i) {
        const ts = (ev.timestamp || '').split(' ')[1] || '';
        const chips = [];
        if (ev.persons > 0) chips.push(React.createElement('span', { key: 'p', className: 'det-chip person' }, '👤 ' + (ev.persons === 1 ? 'person' : ev.persons + ' people')));
        (ev.gestures || []).forEach(function(g, gi) { chips.push(React.createElement('span', { key: 'g'+gi, className: 'det-chip gesture' }, '✋ ' + g.gesture)); });
        (ev.objects  || []).forEach(function(o, oi) { chips.push(React.createElement('span', { key: 'o'+oi, className: 'det-chip object' }, o.label)); });
        return React.createElement('div', { key: i, className: 'det-row' },
          React.createElement('span', { className: 'det-time' }, ts),
          React.createElement('span', { className: 'det-body' }, chips.length ? chips : 'no detection')
        );
      })
    )
  );
}

// ── DartsTab ───────────────────────────────────────────────────────────────────
function DartsTab({ active, hits, game }) {
  const imgRef = React.useRef(null);
  const timerRef = React.useRef(null);
  const [camOnline, setCamOnline] = React.useState(false);
  const [speakDisabled, setSpeakDisabled] = React.useState(false);
  const [calBtnText, setCalBtnText] = React.useState('🎯 Calibrate');
  const [calDisabled, setCalDisabled] = React.useState(false);

  function refreshDartsCam() {
    if (!active) return;
    const tmp = new Image();
    tmp.onload = function() {
      if (imgRef.current) {
        imgRef.current.src = tmp.src;
        imgRef.current.className = 'visible';
      }
      setCamOnline(true);
      timerRef.current = setTimeout(refreshDartsCam, 200);
    };
    tmp.onerror = function() {
      if (imgRef.current) imgRef.current.className = '';
      setCamOnline(false);
      timerRef.current = setTimeout(refreshDartsCam, 2000);
    };
    tmp.src = '/api/camera.jpg?annotated=1&t=' + Date.now();
  }

  React.useEffect(function() {
    if (active) {
      refreshDartsCam();
      fetch('/api/laser/active', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({active:true})}).catch(function(){});
    } else {
      clearTimeout(timerRef.current);
      fetch('/api/laser/active', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({active:false})}).catch(function(){});
    }
    return function() { clearTimeout(timerRef.current); };
  }, [active]);

  function handleSpeak() {
    setSpeakDisabled(true);
    apiAction('dart_speak');
    setTimeout(function() { setSpeakDisabled(false); }, 1500);
  }

  function handleNextPlayer() {
    apiAction('dart_next_player');
  }

  function handleCalibrate() {
    setCalBtnText('⏳ Setting…');
    setCalDisabled(true);
    fetch('/api/laser/calibrate', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'})
      .then(function(r) { return r.json(); }).then(function(d) {
        setCalBtnText(d.ok ? '✅ Set!' : '❌ ' + (d.error || 'Failed'));
        setTimeout(function() { setCalBtnText('🎯 Calibrate'); setCalDisabled(false); }, 3500);
      }).catch(function() { setCalBtnText('❌ Error'); setTimeout(function() { setCalBtnText('🎯 Calibrate'); setCalDisabled(false); }, 3500); });
  }

  function handleNewGame() {
    if (!window.confirm('Start a new game? This clears all scores.')) return;
    apiAction('dart_new_game');
  }

  // Build leaderboard from game.rounds
  const rounds = (game && game.rounds) || [];
  const players = {};
  rounds.forEach(function(r) {
    if (!players[r.player]) players[r.player] = { turns:0, total:0, lastShots:[] };
    players[r.player].turns++;
    players[r.player].total += r.total;
    players[r.player].lastShots = r.shots || [];
  });
  const sortedPlayers = Object.keys(players).sort(function(a,b){ return players[b].total - players[a].total; });

  // Shot dots
  const shotsDone = (game && game.shots_done) || 0;
  const shotsTotal = (game && game.shots_total) || 5;
  const dots = [];
  for (let i = 0; i < shotsTotal; i++) {
    dots.push(React.createElement('span', {
      key: i,
      style: { color: i < shotsDone ? '#4ade80' : '#334155' }
    }, i < shotsDone ? '●' : '○'));
  }

  const awaiting = !!(game && game.awaiting);

  return React.createElement('div', { className: 'darts-area open' },
    React.createElement('div', { className: 'darts-top-row' },
      React.createElement('div', { className: 'cam-feed-wrap', style: { flex: 1, position: 'relative' } },
        React.createElement('img', { ref: imgRef, alt: 'Darts camera', draggable: false }),
        !camOnline && React.createElement('div', { className: 'cam-offline' }, '📷 Camera offline or not started'),
        React.createElement('div', { className: 'darts-hit-layer', style: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, pointerEvents: 'none' } },
          hits.map(function(hit, i) {
            const col = dartColor(hit.score);
            return React.createElement('div', {
              key: i,
              className: 'dart-hit-dot',
              style: {
                left: (hit.x * 100) + '%',
                top: (hit.y * 100) + '%',
                background: col,
                boxShadow: '0 0 10px ' + col + ', 0 0 20px ' + col + '60'
              }
            },
              React.createElement('span', { className: 'dart-score-badge' },
                hit.score === 100 ? '🎯' : hit.score === 0 ? '💨' : String(hit.score)
              )
            );
          })
        )
      ),
      React.createElement('div', { className: 'darts-leaderboard' },
        React.createElement('div', { className: 'dart-lb-title' }, '🏆 Leaderboard'),
        sortedPlayers.length === 0
          ? React.createElement('div', { className: 'dart-lb-empty' }, 'No scores yet')
          : sortedPlayers.map(function(pid) {
              const p = players[pid];
              const col = dartColor(p.total / Math.max(1, p.turns));
              const isActive = game && Number(pid) === game.player;
              return React.createElement('div', { key: pid, className: 'dart-lb-row' + (isActive ? ' lb-active' : '') },
                React.createElement('span', { className: 'dart-lb-player' }, 'P' + pid),
                React.createElement('span', { className: 'dart-lb-turns' }, p.turns + ' turn' + (p.turns > 1 ? 's' : '')),
                React.createElement('span', { className: 'dart-lb-shots' }, p.lastShots.map(function(s){ return s.score; }).join(' · ')),
                React.createElement('span', { className: 'dart-lb-total', style: { color: col } }, p.total)
              );
            })
      )
    ),

    React.createElement('div', { className: 'darts-gamebar' },
      React.createElement('span', { className: 'dgb-player' }, 'P' + ((game && game.player) || 1)),
      React.createElement('span', { className: 'dgb-dots' }, dots),
      React.createElement('span', { className: 'dgb-score' }, ((game && game.turn_total) || 0) + ' pts')
    ),

    React.createElement('div', { className: 'darts-toolbar' },
      React.createElement('button', { className: 'darts-btn speak-btn', onClick: handleSpeak, disabled: speakDisabled, title: 'Speak last score' }, '🔊'),
      React.createElement('button', { className: 'darts-btn next-btn' + (awaiting ? ' awaiting' : ''), onClick: handleNextPlayer },
        awaiting ? '→ Next Player ⚡' : '→ Next Player'
      ),
      React.createElement('button', { className: 'darts-btn cal-btn', onClick: handleCalibrate, disabled: calDisabled }, calBtnText),
      React.createElement('button', { className: 'darts-btn new-btn', onClick: handleNewGame }, '✕ New Game')
    ),

    React.createElement('div', { className: 'darts-score-list' },
      hits.slice().reverse().map(function(hit, i) {
        const col = dartColor(hit.score);
        const label = hit.score === 100 ? 'Bullseye!' : hit.score === 0 ? 'Miss' : hit.score + ' pts';
        return React.createElement('div', { key: i, className: 'dart-score-row' },
          React.createElement('span', { className: 'dart-score-num', style: { color: col } }, hit.score),
          React.createElement('span', { className: 'dart-score-label' }, label),
          React.createElement('span', { className: 'dart-score-pos' }, '(' + ((hit.x||0).toFixed(2)) + ', ' + ((hit.y||0).toFixed(2)) + ')')
        );
      })
    )
  );
}

// ── LogTab ─────────────────────────────────────────────────────────────────────
function LogTab({ logs, onClear }) {
  return React.createElement('div', { className: 'log-area open' },
    React.createElement('div', { className: 'log-toolbar' },
      React.createElement('span', null, 'Live log — newest first'),
      React.createElement('button', { className: 'log-clear-btn', onClick: onClear }, 'Clear')
    ),
    React.createElement('div', { className: 'log-entries' },
      logs.map(function(entry, i) {
        const icon = LOG_ICONS[entry.level] || '⚪';
        const color = LOG_COLORS[entry.level] || '#94a3b8';
        return React.createElement('div', {
          key: i,
          className: 'log-entry log-' + entry.level,
          style: { borderLeftColor: color }
        },
          React.createElement('span', { className: 'log-icon' }, icon),
          React.createElement('div', { className: 'log-body' },
            React.createElement('span', { className: 'log-msg' }, entry.msg),
            React.createElement('span', { className: 'log-ts' }, entry.ts + ' · ' + entry.level)
          )
        );
      })
    )
  );
}

// ── NavBar ─────────────────────────────────────────────────────────────────────
function NavBar({ tab, onTab }) {
  const tabs = [
    { id: 'home',  icon: '🏠', label: 'Home' },
    { id: 'cam',   icon: '📷', label: 'Camera' },
    { id: 'darts', icon: '🎯', label: 'Darts' },
    { id: 'log',   icon: '📋', label: 'Log' },
  ];
  return React.createElement('nav', { className: 'pinky-nav' },
    tabs.map(function(t) {
      return React.createElement('button', {
        key: t.id,
        className: 'pinky-nav-item' + (tab === t.id ? ' active' : ''),
        onClick: function() { onTab(t.id); }
      },
        React.createElement('span', { className: 'pinky-nav-ic' }, t.icon),
        t.label
      );
    })
  );
}

// ── App ────────────────────────────────────────────────────────────────────────
function App() {
  const [tab, setTab] = React.useState('home');
  const [status, setStatus] = React.useState({
    state: 'idle', muted: false, speaking: false,
    last_transcript: '', last_reply: '',
    cam_status: '', det_status: '',
    persons: 0,
  });
  const [detectFlash, setDetectFlash] = React.useState(false);
  const [detections, setDetections] = React.useState([]);
  const [logs, setLogs] = React.useState([]);
  const [dartHits, setDartHits] = React.useState([]);
  const [dartGame, setDartGame] = React.useState({ player:1, shots_done:0, shots_total:5, turn_total:0, awaiting:false, rounds:[] });
  const [connected, setConnected] = React.useState(false);
  const [lastConvTime, setLastConvTime] = React.useState(0);
  const [appHeight, setAppHeight] = React.useState(window.innerHeight);

  const detectTimerRef = React.useRef(null);
  const clearTimerRef  = React.useRef(null);
  const esRef = React.useRef(null);
  const statusRef = React.useRef(status);
  statusRef.current = status;

  // Window height (iOS 9 dvh fix)
  React.useEffect(function() {
    function onResize() { setAppHeight(window.innerHeight); }
    window.addEventListener('resize', onResize);
    return function() { window.removeEventListener('resize', onResize); };
  }, []);

  // Auto-clear conversation clouds after 3 minutes of inactivity
  React.useEffect(function() {
    if (!(status.last_transcript || status.last_reply)) return;
    clearTimeout(clearTimerRef.current);
    clearTimerRef.current = setTimeout(function() {
      setStatus(function(prev) {
        return Object.assign({}, prev, { last_transcript: '', last_reply: '' });
      });
    }, 3 * 60 * 1000);
  }, [status.last_transcript, status.last_reply]);

  // Load dart data when switching to darts tab
  React.useEffect(function() {
    if (tab === 'darts') {
      fetch('/api/dart/hits').then(function(r){ return r.json(); }).then(function(hits){ setDartHits(hits); }).catch(function(){});
      fetch('/api/dart/game').then(function(r){ return r.json(); }).then(function(g){ setDartGame(g); }).catch(function(){});
    }
  }, [tab]);

  // SSE connection
  React.useEffect(function() {
    function connect() {
      if (esRef.current) { try { esRef.current.close(); } catch(e) {} esRef.current = null; }
      var es = new EventSource('/api/stream');
      esRef.current = es;
      setConnected(false);

      es.onopen = function() { setConnected(true); };
      es.onerror = function() {
        setConnected(false);
        try { es.close(); } catch(e) {}
        esRef.current = null;
        setTimeout(connect, 3000);
      };
      es.onmessage = function(e) {
        try {
          var data = JSON.parse(e.data);
          if (data.type === 'status') {
            if (data.last_transcript || data.last_reply) setLastConvTime(Date.now());
            setStatus(function(prev) {
              return Object.assign({}, prev, {
                state: data.state || prev.state,
                muted: !!data.muted,
                speaking: !!data.speaking,
                last_transcript: data.last_transcript || prev.last_transcript,
                last_reply: data.last_reply || prev.last_reply,
                cam_status: data.cam_status !== undefined ? data.cam_status : prev.cam_status,
                det_status: data.det_status !== undefined ? data.det_status : prev.det_status,
              });
            });
          } else if (data.type === 'detection') {
            var persons = data.persons || 0;
            setStatus(function(prev) { return Object.assign({}, prev, { persons: persons }); });
            setDetections(function(prev) { return [data].concat(prev).slice(0, 60); });
            setDetectFlash(true);
            clearTimeout(detectTimerRef.current);
            detectTimerRef.current = setTimeout(function() { setDetectFlash(false); }, 1500);
          } else if (data.type === 'log') {
            setLogs(function(prev) {
              return [{ level: data.level || 'info', msg: data.msg || '', ts: data.ts || '' }].concat(prev).slice(0, 200);
            });
            // Show what Whisper heard immediately in the question cloud
            if (data.level === 'stt' && data.msg) {
              setLastConvTime(Date.now());
              setStatus(function(prev) { return Object.assign({}, prev, { last_transcript: data.msg }); });
            }
          } else if (data.type === 'dart_hit') {
            setDartHits(function(prev) { return prev.concat([data]).slice(-20); });
          } else if (data.type === 'dart_reset') {
            setDartHits([]);
            setDartGame(function(prev) { return Object.assign({}, prev, { rounds: [], turn_total: 0, shots_done: 0 }); });
          } else if (data.type === 'dart_hits_clear') {
            setDartHits([]);
          } else if (data.type === 'dart_game') {
            setDartGame(data);
          }
        } catch(err) {}
      };
    }
    connect();
    // Ensure laser off on load
    fetch('/api/laser/active', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({active:false})}).catch(function(){});
    return function() {
      if (esRef.current) { try { esRef.current.close(); } catch(e) {} }
    };
  }, []);

  function handleTab(newTab) {
    if (newTab !== 'darts' && tab === 'darts') {
      fetch('/api/laser/active', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({active:false})}).catch(function(){});
    }
    setTab(newTab);
  }

  function handleWake() {
    apiAction('wake');
    setStatus(function(prev) { return Object.assign({}, prev, { state: 'awake' }); });
  }
  function handleSleep() {
    apiAction('sleep');
    setStatus(function(prev) { return Object.assign({}, prev, { state: 'idle', speaking: false }); });
  }
  function handlePause() { apiAction('pause'); }
  function handleMuteToggle() { apiAction('mute_toggle'); }
  function handleRestart() {
    fetch('/api/restart', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({token:'pinku'}),
    }).then(function(r){ return r.json(); }).then(function(d){
      if (d.ok) setTimeout(function(){ location.reload(); }, 5000);
    }).catch(function(){});
  }

  return React.createElement('div', { id: 'app', style: { height: appHeight + 'px', display: 'flex', flexDirection: 'column', overflow: 'hidden' } },
    React.createElement(Header, { status: status, onWake: handleWake, onSleep: handleSleep, onPause: handlePause, onMuteToggle: handleMuteToggle, onRestart: handleRestart }),
    tab === 'home'  && React.createElement(HomeTab,   { status: status, detectFlash: detectFlash, connected: connected, onWake: handleWake, lastConvTime: lastConvTime }),
    tab === 'cam'   && React.createElement(CameraTab,  { active: tab === 'cam',   camStatus: status.cam_status, detStatus: status.det_status, detections: detections }),
    tab === 'darts' && React.createElement(DartsTab,   { active: tab === 'darts', hits: dartHits, game: dartGame }),
    tab === 'log'   && React.createElement(LogTab,     { logs: logs, onClear: function(){ setLogs([]); } }),
    React.createElement(NavBar, { tab: tab, onTab: handleTab })
  );
}

ReactDOM.render(React.createElement(App), document.getElementById('root'));
</script>
</body>
</html>
"""