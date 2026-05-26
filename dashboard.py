"""
Web dashboard — Pinku, using exact Pinky CSS face system (div-based, not SVG).
Face expressions, animations, sleeping state all identical to Pinky's index.html.
http://<mac-mini-ip>:5100
"""

import json
import time
import threading
from flask import Flask, Response, render_template_string
from config import DASHBOARD_PORT, DASHBOARD_HOST

_app    = Flask(__name__)
_logger = None
_status = {
    "state": "idle", "muted": False, "model": "",
    "last_transcript": "", "last_reply": "", "speaking": False,
    "detections": [],
}

_clients_lock = threading.Lock()
_clients: list = []

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

def log_message(level: str, msg: str):
    """Stream a log line to all dashboard clients (shown in Log tab)."""
    _broadcast({
        "type":  "log",
        "level": level,
        "msg":   msg,
        "ts":    time.strftime("%H:%M:%S"),
    })

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
        while True:
            time.sleep(0.05)
            while q:
                yield q.pop(0)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@_app.route("/api/status")
def api_status():
    from flask import jsonify
    return jsonify(_status)

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
    """Current camera frame as JPEG. Returns 503 if camera not active."""
    try:
        from camera import get_frame
        import cv2
        frame = get_frame()
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

@_app.route("/api/camera/status")
def camera_status():
    from flask import jsonify
    try:
        from camera import get_status
        return jsonify(get_status())
    except Exception as e:
        return jsonify({"camera": f"error: {e}", "detection": "unknown"})

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
<style>
/* ─────────────────────────────────────────────────────────────────────────────
   BASE  (copied verbatim from Pinky's index.html, adapted for Pinku)
───────────────────────────────────────────────────────────────────────────── */
:root {
  --bg:       #0a0a14;
  --surface:  #12121e;
  --border:   #24243a;
  --text:     #e8e8f4;
  --muted:    #666683;
  --accent:   #c084fc;
  --face-max: 440px;
}
@media (min-width: 500px) {
  :root { --face-max: min(62vmin, 460px); }
}
@media (min-width: 900px) {
  :root { --face-max: 520px; }
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
  display: flex; flex-direction: column;
  height: 100dvh; overflow: hidden;
  -webkit-font-smoothing: antialiased;
  user-select: none; -webkit-user-select: none;
}

/* ── Header ── */
.header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 16px 10px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  background: linear-gradient(180deg, rgba(124,92,191,0.07) 0%, transparent 100%);
}
.header-left { display: flex; align-items: center; gap: 10px; }
.dot-wrap { position: relative; }
.avatar-ring {
  width: 36px; height: 36px; border-radius: 50%;
  background: linear-gradient(135deg, #7c3aed, #4c1d95);
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.95rem; color: #fff;
  box-shadow: 0 0 12px rgba(124,58,237,0.45);
}
.status-dot {
  position: absolute; bottom: -1px; right: -1px;
  width: 11px; height: 11px; border-radius: 50%;
  border: 2px solid var(--bg);
  background: #4ade80;
  transition: background 0.5s ease;
}
.status-dot.sleeping  { background: #2a2a42; }
.status-dot.paused    { background: #f87171; }
.status-dot.thinking,
.status-dot.triggered { background: #c084fc; }
.status-dot.speaking  { background: #f472b6; }
.app-name {
  font-size: 1.05rem; font-weight: 700;
  background: linear-gradient(90deg, #c9b1ff, #f472b6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.status-pills { display: flex; gap: 5px; margin-top: 2px; }
.spill {
  font-size: 0.72rem; padding: 1px 6px; border-radius: 8px;
  background: rgba(255,255,255,0.05); border: 1px solid var(--border);
  color: var(--muted); transition: background 0.3s, color 0.3s;
}
.spill.on { background: rgba(74,222,128,0.12); color: #4ade80; border-color: rgba(74,222,128,0.3); }
.header-btns { display: flex; gap: 8px; }
.hdr-btn {
  padding: 6px 12px; border-radius: 10px; font-size: 0.85rem;
  border: 1px solid var(--border); background: rgba(255,255,255,0.05);
  color: var(--text); cursor: pointer; transition: background 0.18s;
  display: flex; align-items: center; justify-content: center;
  min-width: 40px; min-height: 36px;
}
.hdr-btn:hover { background: rgba(255,255,255,0.1); }
.hdr-btn.stop-btn  { border-color: rgba(248,113,113,0.4); color: #f87171; }
.hdr-btn.mute-btn  { border-color: rgba(248,113,113,0.35); }
.hdr-btn.resume-btn{ border-color: rgba(74,222,128,0.4); color: #4ade80; }
.hdr-btn.active    { background: rgba(192,132,252,0.18); border-color: rgba(192,132,252,0.5); }

/* ── Main area ── */
.main-area {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  overflow: hidden; gap: 0;
  padding: 8px 0 4px;
}

/* ── Thought bubble ── */
.face-bubble {
  position: absolute;
  top: 0%; right: -4%;
  background: rgba(14, 12, 28, 0.92);
  border: 1.5px solid var(--bubble-color, #c084fc);
  border-radius: 999px;
  padding: 5px 11px;
  display: flex; align-items: center; gap: 4px;
  z-index: 8; pointer-events: none;
  box-shadow: 0 4px 16px rgba(0,0,0,0.5), 0 0 12px rgba(192,132,252,0.18);
  backdrop-filter: blur(10px);
  opacity: 0;
  transform: translateY(8px) scale(0.86);
  transition: opacity 0.32s cubic-bezier(0.2,0.85,0.35,1), transform 0.34s cubic-bezier(0.34,1.15,0.52,1);
}
.face-bubble.visible { opacity: 1; transform: translateY(0) scale(1); }
.bubble-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--bubble-color, #c084fc);
  display: inline-block;
  animation: dot-bounce 1.22s cubic-bezier(0.45,0.05,0.55,0.95) infinite;
}
.bubble-dot.d2 { animation-delay: 0.16s; }
.bubble-dot.d3 { animation-delay: 0.32s; }
@keyframes dot-bounce {
  0%,100% { transform: translateY(0) scale(1); opacity: 0.82; }
  22%     { transform: translateY(0) scale(1.05); opacity: 1; }
  48%     { transform: translateY(-6px) scale(1.1); opacity: 1; }
  72%     { transform: translateY(-1px) scale(1); opacity: 0.9; }
}

/* ── Stage & layout ── */
.pinky-stage {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  width: 100%; max-width: 720px; margin: 0 auto;
}
.face-with-actions {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  width: 100%; max-width: min(94vw, 720px);
  padding: 2px 4px 10px;
}
.face-with-actions .wake-btn-top { margin-bottom: 12px; }
.face-with-actions .face-wrap {
  flex: 0 1 auto; width: 100%;
  max-width: min(92vw, var(--face-max));
  margin-left: auto; margin-right: auto;
}
.wake-btn-top {
  flex: 0 0 auto; width: 56px; height: 56px;
  border-radius: 50%;
  border: 2px solid rgba(216,180,254,0.9);
  background: linear-gradient(180deg, #7c3aed 0%, #5b21b6 52%, #4c1d95 100%);
  color: #fff; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 4px 0 rgba(49,10,90,0.75), 0 8px 18px rgba(91,33,182,0.38),
              0 0 0 1px rgba(255,255,255,0.2) inset;
  transition: transform 0.08s ease, filter 0.12s ease;
  font-size: 1.55rem; line-height: 1;
}
.wake-btn-top:active { transform: translateY(2px); }

/* ── Question & reply clouds ── */
.question-cloud {
  margin-bottom: 10px; width: 100%; max-width: 450px;
  padding: 10px 14px 12px; font-size: 0.86rem; line-height: 1.4;
  color: #0c1e2e; text-align: left;
  background: linear-gradient(165deg,#f8fcff 0%,#e8f4fc 45%,#dff0fb 100%);
  border: 2px solid rgba(56,189,248,0.55);
  border-radius: 26px; border-bottom-right-radius: 8px;
  box-shadow: 0 8px 22px rgba(0,0,0,0.18), 0 2px 0 rgba(255,255,255,0.7) inset;
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
  padding: 10px 14px 12px; font-size: 0.86rem; line-height: 1.4;
  color: #241830; text-align: left;
  background: linear-gradient(165deg,#fdfcff 0%,#f3e8ff 45%,#ede9fe 100%);
  border: 2px solid rgba(167,139,250,0.55);
  border-radius: 26px; border-bottom-left-radius: 8px;
  box-shadow: 0 10px 28px rgba(0,0,0,0.22), 0 2px 0 rgba(255,255,255,0.65) inset;
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
  display: flex; align-items: center; gap: 6px;
  background: rgba(74,222,128,0.10);
  border: 1px solid rgba(74,222,128,0.28);
  border-radius: 20px; padding: 4px 12px;
  font-size: 0.72rem; color: #4ade80;
  opacity: 0; transition: opacity 0.4s; pointer-events: none;
}
.person-badge.visible { opacity: 1; }
.person-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: #4ade80; animation: live-blink 1.6s ease-in-out infinite;
}
@keyframes live-blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Status bar ── */
.status-bar {
  flex-shrink: 0; margin-top: 10px;
  padding: 9px 14px 9px 12px; border-radius: 999px;
  background: linear-gradient(180deg, rgba(22,24,40,0.94), rgba(14,16,28,0.96));
  border: 1px solid rgba(100,116,139,0.4);
  box-shadow: 0 8px 24px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05) inset;
  backdrop-filter: blur(12px);
  max-width: min(96vw, 460px); width: 100%;
  display: flex; align-items: center; gap: 10px;
  transition: border-color 0.6s ease, box-shadow 0.6s ease;
}
.status-bar.listening { border-color: rgba(74,222,128,0.4); }
.status-bar.awake     { border-color: rgba(192,132,252,0.5); }
.status-bar.thinking,
.status-bar.triggered { border-color: rgba(232,121,249,0.48); }
.status-bar.speaking  { border-color: rgba(192,132,252,0.42); }
.status-bar.paused    { border-color: rgba(248,113,113,0.42); }
.status-bar-icon { font-size: 1.65rem; line-height: 1; flex-shrink: 0; }
.status-bar-icon.thinking,
.status-bar-icon.triggered { animation: notify-pulse 1.12s ease-in-out infinite; }
@keyframes notify-pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.7;transform:scale(0.9)} }
.status-bar-text { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.status-bar-label { font-size: 1.05rem; font-weight: 700; color: #e8e8f4; white-space: nowrap; }
.status-bar-sub   { font-size: 0.88rem; color: #8a8aaa; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 1px; }
.status-bar-right { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
.ws-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.ws-dot.on  { background: #4ade80; box-shadow: 0 0 5px rgba(74,222,128,0.6); }
.ws-dot.off { background: #64748b; }

/* ─────────────────────────────────────────────────────────────────────────────
   FACE WRAP  (exact copy from Pinky)
───────────────────────────────────────────────────────────────────────────── */
.face-wrap {
  position: relative; width: 100%; max-width: var(--face-max);
  margin: 0 auto; flex-shrink: 0;
  border-radius: 48% 48% 46% 46% / 44% 44% 56% 56%;
  border: 1px solid rgba(255,182,220,0.22);
  background:
    radial-gradient(ellipse 72% 58% at 50% 36%, rgba(255,210,232,0.22) 0%, transparent 58%),
    radial-gradient(circle at 32% 20%, #4a4478 0%, #322d58 38%, #1c1a34 100%);
  box-shadow: 0 24px 48px rgba(0,0,0,0.4),
              0 0 0 1px rgba(255,255,255,0.08) inset,
              inset 0 -16px 32px rgba(0,0,0,0.18);
  overflow: visible;
}
.face-wrap::before { content:''; display:block; padding-top:100%; }
@supports (aspect-ratio:1) {
  .face-wrap::before { display:none; padding-top:0; }
  .face-wrap {
    width: min(100%, var(--face-max));
    max-height: min(56vh, var(--face-max));
    aspect-ratio: 1; height: auto;
    display: flex; align-items: center; justify-content: center;
  }
  .face-wrap .face {
    position: relative; left:auto; right:auto; top:auto; bottom:auto;
    margin: 0 auto; transform: none;
    width: 71%; height: 86%; max-width:none; max-height:none;
  }
}
.face-wrap::after {
  content:''; position:absolute; left:-8%; right:-8%; top:-8%; bottom:-8%;
  border-radius:50%;
  border: 1px solid rgba(96,165,250,0.36);
  box-shadow: 0 0 24px rgba(59,130,246,0.28), inset 0 0 12px rgba(147,51,234,0.25);
  pointer-events:none; opacity:0.52;
  transition: opacity 0.5s ease;
}
@keyframes face-ambient-ring { 0%,100%{opacity:0.38} 50%{opacity:0.72} }
.face.listening .face-wrap::after, .face.awake .face-wrap::after { animation: face-ambient-ring 3.5s ease-in-out infinite; }
.face.thinking .face-wrap::after, .face.triggered .face-wrap::after { animation: face-ambient-ring 2.35s ease-in-out infinite; }
.face.paused .face-wrap::after, .face.sleeping .face-wrap::after { animation: none; opacity: 0.1; }

.face-wrap .face {
  position: absolute; left:0; right:0; top:0; bottom:0;
  width:71%; height:86%; margin:auto; transform:none;
  display:flex; flex-direction:column;
  align-items:center; justify-content:flex-start;
}

/* ─────────────────────────────────────────────────────────────────────────────
   HEAD & FEATURES  (exact copy from Pinky)
───────────────────────────────────────────────────────────────────────────── */
.head {
  position: relative; width: 100%; height: 88%;
  border-radius: 49% 49% 47% 47% / 47% 47% 55% 55%;
  background:
    radial-gradient(ellipse 120% 88% at 50% 22%,
      #fffefb 0%, #fff5f0 18%, #ffe8dc 38%, #ffd0c0 62%, #f5c4b0 92%);
  border: 1px solid rgba(240,180,165,0.35);
  box-shadow: inset 0 -10px 22px rgba(200,120,110,0.1),
              inset 12px 18px 32px rgba(255,255,255,0.55);
  transition: filter 0.45s ease, box-shadow 0.45s ease;
}
.hair-side {
  position: absolute; top: 16%; width: 26%; height: 86%;
  z-index: 1; pointer-events: none;
  background: linear-gradient(90deg,
    rgba(255,255,255,0.12) 0%, #ffc8dd 22%,
    #f9a8d4 50%, #f472b6 78%, #ec4899 100%);
  border-radius: 58% 42% 55% 45% / 22% 30% 88% 72%;
  box-shadow: inset -3px 0 8px rgba(255,255,255,0.25);
}
.hair-side.left  { left: -14%; transform: rotate(-5deg); }
.hair-side.right { right: -14%; transform: rotate(5deg) scaleX(-1); }
.hair-fringe {
  position: absolute; top: -4%; left: 10%; width: 80%; height: 18%;
  z-index: 2; pointer-events: none;
  background: linear-gradient(180deg, #ffe4f0 0%, #fda4d0 45%, #f472b6 100%);
  border-radius: 50% 50% 45% 45% / 85% 85% 28% 28%;
  clip-path: ellipse(92% 100% at 50% 0%);
  box-shadow: 0 2px 6px rgba(236,72,153,0.2);
}
.brows {
  position: absolute; top: 30.5%; left: 50%; transform: translateX(-50%);
  width: 58%; display: flex; justify-content: space-between; z-index: 4;
}
.brow {
  width: 32%; height: 3px; border-radius: 999px;
  background: linear-gradient(90deg, #b0887a, #8a5f52);
  opacity: 0.9; transform-origin: 50% 50%;
}
.brow.left  { transform: rotate(-5deg) translateY(1px); border-radius: 55% 45% 50% 50%; }
.brow.right { transform: rotate(5deg) translateY(1px); border-radius: 45% 55% 50% 50%; }
.eyes {
  position: absolute; top: 39.5%; left: 18%; width: 64%;
  display: flex; justify-content: space-between; align-items: center;
  z-index: 4; transform-origin: 50% 45%; will-change: transform;
}
.eye {
  width: clamp(34px, 9.2vmin, 60px); height: clamp(34px, 9.2vmin, 60px);
  border-radius: 50%;
  border: 2px solid rgba(55,65,81,0.22);
  background: radial-gradient(circle at 40% 28%, #ffffff 0%, #f5f9ff 62%, #dbeafe 100%);
  box-shadow: 0 0 16px rgba(255,255,255,0.5), 0 5px 16px rgba(59,130,246,0.2),
              inset 0 -4px 7px rgba(15,23,42,0.12);
  position: relative; transform-origin: center center;
}
.eye-lashes {
  position: absolute; left:-6%; right:-6%; top:-16%; height:42%;
  display: flex; justify-content: center; align-items: flex-end;
  padding-bottom: 2px; z-index: 6; pointer-events: none;
}
.eye-lashes .lash {
  flex: 0 0 auto; width: 2px;
  height: clamp(6px, 1.9vmin, 11px);
  margin: 0 1px; border-radius: 2px;
  background: linear-gradient(180deg, rgba(28,24,36,0.95) 0%, rgba(28,24,36,0.2) 100%);
  transform-origin: 50% 100%; opacity: 0.9;
  box-shadow: 0 0 1px rgba(0,0,0,0.25);
}
.eye .lash:nth-child(1) { transform: rotate(-22deg); height: 78%; }
.eye .lash:nth-child(2) { transform: rotate(-10deg); }
.eye .lash:nth-child(3) { transform: rotate(0deg); height: 110%; }
.eye .lash:nth-child(4) { transform: rotate(10deg); }
.eye .lash:nth-child(5) { transform: rotate(22deg); height: 78%; }
/* Iris via ::after */
.eye::after {
  content: ""; position: absolute;
  left: 50%; top: 56%; transform: translate(-50%,-50%);
  width: 42%; height: 48%; border-radius: 50%;
  background: radial-gradient(circle at 40% 34%, #7c3aed 0%, #4338ca 45%, #0f172a 84%, #020617 100%);
  box-shadow: 0 2px 10px rgba(76,29,149,0.45), inset 0 -3px 7px rgba(0,0,0,0.45);
}
/* Catchlight via ::before */
.eye::before {
  content: ""; position: absolute;
  left: 58%; top: 30%; width: 17%; height: 19%;
  border-radius: 50%; background: rgba(255,255,255,0.98);
  z-index: 1; box-shadow: 0 0 6px rgba(255,255,255,0.9);
}
.nose-bridge {
  position: absolute; top: 54%; left: 50%; transform: translateX(-50%);
  width: 12%; height: 14%;
  background: linear-gradient(90deg, transparent 0%,
    rgba(240,160,150,0.14) 35%, rgba(240,160,150,0.2) 50%,
    rgba(240,160,150,0.14) 65%, transparent 100%);
  border-radius: 40%; pointer-events: none; z-index: 2;
}
.mouth {
  position: absolute; left: 50%; transform: translateX(-50%);
  transform-origin: 50% 0%;
  top: 70.5%; width: 38%; max-width: 200px; height: 22px;
  border-radius: 0 0 52% 52%; border: none;
  background:
    radial-gradient(ellipse 95% 90% at 50% 0%, rgba(255,255,255,0.62) 0%, transparent 58%),
    linear-gradient(180deg, rgba(255,245,250,0.65) 0%, rgba(251,146,170,0.45) 38%, rgba(236,72,120,0.62) 100%);
  box-shadow: 0 3px 0 0 #db2777, 0 8px 14px rgba(219,39,119,0.32),
              inset 0 2px 8px rgba(255,255,255,0.45);
  z-index: 3;
}
.cheek {
  position: absolute; top: 56%; width: 21%; height: 11%;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(255,170,190,0.55) 0%, rgba(255,200,210,0.22) 42%, transparent 70%);
  z-index: 2;
}
.cheek.left { left: 6%; }
.cheek.right { right: 6%; }

/* ─────────────────────────────────────────────────────────────────────────────
   VOICE WAVE
───────────────────────────────────────────────────────────────────────────── */
.voice-wave {
  display: flex; align-items: flex-end; justify-content: center;
  width: 56%; height: 12%; margin-top: auto; padding-bottom: 2%;
  opacity: 0.18; position: relative; flex-shrink: 0;
}
.voice-wave span {
  width: 10%; margin: 0 1.2%; min-height: 16%; border-radius: 999px;
  background: #8fd5ff; box-shadow: 0 0 8px rgba(125,211,252,0.45);
  transform-origin: center bottom;
  animation: wave 1.2s cubic-bezier(0.4,0,0.2,1) infinite;
  animation-play-state: paused;
}
.voice-wave span:nth-child(1) { animation-delay: 0s; }
.voice-wave span:nth-child(2) { animation-delay: 0.1s; }
.voice-wave span:nth-child(3) { animation-delay: 0.2s; }
.voice-wave span:nth-child(4) { animation-delay: 0.32s; }
.voice-wave span:nth-child(5) { animation-delay: 0.44s; }
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
.face.triggered .voice-wave span { animation-play-state: running; }
.face.thinking .voice-wave span { animation-duration: 1.45s; }
.face.speaking .voice-wave span { animation-play-state: running; animation-duration: 0.55s; }
.face.speaking .voice-wave { opacity: 1; }
.face.paused .voice-wave span { background: #fca5a5; box-shadow: none; }

/* ─────────────────────────────────────────────────────────────────────────────
   SPARKLES
───────────────────────────────────────────────────────────────────────────── */
.face-sparkles { position:absolute; inset:0; pointer-events:none; z-index:10; overflow:visible; }
.sp {
  position: absolute; font-size: 0.6rem; line-height: 1;
  color: #f5d0fe; opacity: 0;
  text-shadow: 0 0 8px #c084fc, 0 0 18px rgba(192,132,252,0.45);
  animation: sparkle-float 4.2s ease-in-out infinite;
}
.sp1 { top:3%;  left:9%;  animation-delay:0s; }
.sp2 { top:7%;  right:11%; font-size:0.45rem; animation-delay:1.05s; }
.sp3 { top:1%;  left:52%; animation-delay:2.1s; }
.sp4 { top:11%; right:5%; font-size:0.42rem; animation-delay:0.62s; }
@keyframes sparkle-float {
  0%,100% { opacity:0;    transform: translateY(0)    rotate(0deg)  scale(0.5); }
  25%,75% { opacity:0.92; transform: translateY(-9px)  rotate(22deg) scale(1); }
  50%     { opacity:0.6;  transform: translateY(-16px) rotate(44deg) scale(0.82); }
}
.face.awake    ~ .face-sparkles .sp { color:#fde68a; text-shadow:0 0 8px #fb923c,0 0 18px rgba(251,146,60,0.4); animation-duration:2.2s; }
.face.thinking ~ .face-sparkles .sp { color:#e879f9; text-shadow:0 0 8px #c084fc,0 0 18px rgba(232,121,249,0.45); animation-duration:3s; }
.face.triggered ~ .face-sparkles .sp { color:#fb923c; text-shadow:0 0 8px #f97316,0 0 18px rgba(249,115,22,0.4); animation-duration:1.6s; }
.face.paused ~ .face-sparkles .sp,
.face.sleeping ~ .face-sparkles .sp { animation:none!important; opacity:0!important; }

/* ─────────────────────────────────────────────────────────────────────────────
   STATE EXPRESSIONS  (exact copy from Pinky)
───────────────────────────────────────────────────────────────────────────── */
@keyframes lip-soft {
  0%,100% { transform: translateX(-50%) scale(1,1); }
  28%     { transform: translateX(-50%) scale(1.03,1.05); }
  48%     { transform: translateX(-50%) scale(1.08,1.14); }
  72%     { transform: translateX(-50%) scale(1.02,1.04); }
}
@keyframes lip-think {
  0%,100% { transform: translateX(-50%) scale(1,1); opacity:0.9; }
  45%     { transform: translateX(-50%) scale(0.9,0.93); opacity:1; }
  70%     { transform: translateX(-50%) scale(0.95,0.98); opacity:0.94; }
}
@keyframes eye-glow {
  0%,100% { opacity:1;    transform:translate(-50%,-50%) scale(1); }
  32%     { opacity:0.78; transform:translate(-50%,-50%) scale(0.91); }
  55%     { opacity:0.68; transform:translate(-50%,-50%) scale(0.85); }
  82%     { opacity:0.92; transform:translate(-50%,-50%) scale(0.96); }
}
@keyframes eye-glow-think {
  0%,100% { opacity:1;    transform:translate(-46%,-50%) scale(1); }
  35%     { opacity:0.8;  transform:translate(-46%,-50%) scale(0.9); }
  60%     { opacity:0.72; transform:translate(-46%,-50%) scale(0.84); }
  85%     { opacity:0.88; transform:translate(-46%,-50%) scale(0.94); }
}
@keyframes eye-sparkle {
  0%,100% { transform:scale(1); opacity:1; }
  38%     { transform:scale(1.12); opacity:0.88; }
  62%     { transform:scale(1.04); opacity:0.95; }
}
@keyframes eye-scan {
  0%,100% { transform:translate(-50%,-50%); }
  22%     { transform:translate(-55%,-51%); }
  48%     { transform:translate(-42%,-49%); }
  72%     { transform:translate(-56%,-50%); }
}
@keyframes cheek-pulse { 0%,100%{opacity:0.65} 50%{opacity:1} }

/* Listening / idle-awake */
.face.listening .mouth, .face.awake .mouth {
  width: 36%; top: 70.5%; height: 21px;
  animation: lip-soft 2s ease-in-out infinite;
}
.face.listening .eye::after { animation: eye-glow 1.4s ease-in-out infinite; }
.face.listening .eye::before { animation: eye-sparkle 2.4s ease-in-out infinite; }
.face.listening .cheek { animation: cheek-pulse 3.8s ease-in-out infinite; }
/* Awake (session open) */
.face.awake .brow.left  { transform: rotate(-8deg) translateY(0); }
.face.awake .brow.right { transform: rotate(8deg) translateY(0); }
.face.awake .eye::after { animation: eye-glow 1.05s ease-in-out infinite; }
.face.awake .eye::before { animation: eye-sparkle 1.6s ease-in-out infinite; }
.face.awake .cheek { animation: cheek-pulse 1.5s ease-in-out infinite; }
.face.awake .mouth { animation: lip-soft 1.8s ease-in-out infinite; }
/* Thinking */
.face.thinking .eye::after { top:46%; left:52%; animation: eye-glow-think 2.2s ease-in-out infinite; }
.face.thinking .brow.left  { transform: rotate(-5deg) translateY(0); }
.face.thinking .brow.right { transform: rotate(5deg) translateY(0); }
.face.thinking .mouth {
  width: 32%; height: 13px; top: 71.5%; border-radius: 999px;
  background: linear-gradient(180deg, rgba(253,242,248,0.5), rgba(236,72,153,0.48));
  box-shadow: 0 3px 0 0 #9d174d, inset 0 1px 4px rgba(255,255,255,0.35);
  animation: lip-think 1.8s ease-in-out infinite;
}
.face.thinking .eye::before { animation: eye-sparkle 3.4s ease-in-out infinite; }
.face.thinking .cheek { animation: cheek-pulse 3s ease-in-out infinite; }
/* Triggered (action fired / detection) */
.face.triggered .mouth {
  width: 20%; height: 18%; border-radius: 50%; top: 69%;
  border: 3px solid #f472b6;
  background: radial-gradient(circle at 40% 35%, rgba(255,255,255,0.35), rgba(251,113,133,0.45));
  box-shadow: 0 4px 0 0 #be185d, inset 0 2px 8px rgba(255,255,255,0.25);
  animation: lip-soft 0.45s ease-in-out infinite;
}
.face.triggered .eye::after { animation: eye-glow 0.6s ease-in-out infinite; }
.face.triggered .eye::before { animation: eye-sparkle 1.1s ease-in-out infinite; }
/* Speaking */
.face.speaking .mouth { animation: lip-soft 0.32s ease-in-out infinite; }
.face.speaking .eye { animation: none; transform: scaleY(1); }
.face.speaking .eye::after { animation: eye-glow 1.8s ease-in-out infinite; }
/* Paused / muted */
.face.paused .eye { transform: none; animation: none; }
.face.paused .eye::before, .face.paused .eye::after { animation: none !important; }
.face.paused .eye-lashes { opacity: 0.38; filter: saturate(0.85); }
.face.paused .head { filter: saturate(0.82) brightness(0.96); }
.face.paused .eye::after { background: #5a3a40; }
.face.paused .mouth {
  width: 30%; height: 12px; border-radius: 999px; top: 74%;
  background: linear-gradient(180deg, rgba(80,50,58,0.5), rgba(120,60,72,0.65));
  box-shadow: 0 -3px 0 0 rgba(120,60,72,0.95), inset 0 2px 4px rgba(0,0,0,0.15);
  animation: none !important;
}
.face.paused .cheek { opacity: 0.3; }
/* Sleeping (idle, no session) */
@keyframes sleep-breathe { 0%,100%{opacity:1} 50%{opacity:0.7} }
.face.sleeping {
  filter: saturate(0.2) brightness(0.55);
  transition: filter 0.8s ease;
}
.face.sleeping .eye {
  transform: scaleY(0.18);
  transition: transform 0.6s ease;
  animation: none !important;
}
.face.sleeping .eye::before, .face.sleeping .eye::after { animation: none !important; opacity: 0 !important; }
.face.sleeping .mouth { width:16%; height:7px; border-radius:2px; top:69.5%; opacity:0.35; animation:none!important; }
.face.sleeping .cheek { opacity:0.15; animation:none!important; }
.face.sleeping .voice-wave { opacity:0!important; }

/* Smooth transitions when waking */
.face.awake, .face.thinking, .face.triggered, .face.listening { transition: filter 0.6s ease; }
.face.awake .eye, .face.thinking .eye, .face.triggered .eye { transition: transform 0.5s ease; }

/* ─────────────────────────────────────────────────────────────────────────────
   BOTTOM NAV
───────────────────────────────────────────────────────────────────────────── */
.pinky-nav {
  display: flex; border-top: 1px solid var(--border);
  flex-shrink: 0; background: rgba(0,0,0,0.35);
}
.pinky-nav-item {
  flex: 1; padding: 10px 0 14px;
  display: flex; flex-direction: column; align-items: center; gap: 3px;
  cursor: pointer; opacity: 0.45; transition: opacity 0.2s;
  border: none; background: none; color: var(--text);
  font-size: 0.62rem; letter-spacing: 0.3px;
}
.pinky-nav-item:hover { opacity: 0.75; }
.pinky-nav-item.active { opacity: 1; color: #c084fc; }
.pinky-nav-ic { font-size: 1.2rem; line-height: 1; }

/* Chat panel */
.chat-area { flex:1; overflow-y:auto; padding:14px; display:none; flex-direction:column; gap:10px; }
.chat-area.open { display:flex; }
.chat-area::-webkit-scrollbar { width:3px; }
.chat-area::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
.msg-row { display:flex; gap:8px; }
.msg-row.user { justify-content:flex-end; }
.msg-row.pinku { justify-content:flex-start; }
.bubble {
  max-width: 78%; padding: 9px 13px; border-radius: 18px;
  font-size: 0.88rem; line-height: 1.45;
}
.bubble.user { background: linear-gradient(135deg,#7c3aed,#4c1d95); border-bottom-right-radius:4px; }
.bubble.pinku { background:rgba(255,255,255,0.07); border:1px solid var(--border); border-bottom-left-radius:4px; }
.msg-time { font-size:0.65rem; color:var(--muted); margin-top:3px; }
.msg-row.user .msg-time { text-align:right; }
.day-sep { text-align:center; font-size:0.68rem; color:var(--muted); padding:8px 0; }

/* Camera panel */
.camera-area {
  flex:1; display:none; flex-direction:column;
  align-items:center; overflow:hidden; gap:0;
}
.camera-area.open { display:flex; }
.cam-feed-wrap {
  position:relative; width:100%; max-width:640px; flex-shrink:0;
  background:#000; aspect-ratio:4/3; overflow:hidden;
}
#cam-img { width:100%; height:100%; object-fit:contain; display:block; }
.cam-offline {
  position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  font-size:0.88rem; color:var(--muted); background:rgba(0,0,0,0.6);
}
.cam-toolbar {
  width:100%; max-width:640px; padding:5px 10px;
  display:flex; justify-content:space-between; align-items:center;
  border-bottom:1px solid var(--border); background:rgba(0,0,0,0.2);
}
.cam-detect-feed {
  flex:1; overflow-y:auto; width:100%; max-width:640px; padding:8px 12px;
}
.cam-detect-feed::-webkit-scrollbar { width:3px; }
.cam-detect-feed::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
.det-row {
  display:flex; align-items:baseline; gap:8px;
  padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.04);
  font-size:0.78rem; color:var(--muted);
}
.det-row .det-time { flex-shrink:0; font-size:0.68rem; color:#444466; }
.det-row .det-body { flex:1; color:var(--text); }
.det-chip {
  display:inline-block; padding:1px 6px; border-radius:6px; font-size:0.7rem;
  margin:0 2px;
}
.det-chip.person { background:rgba(74,222,128,0.15); color:#4ade80; border:1px solid rgba(74,222,128,0.3); }
.det-chip.gesture{ background:rgba(192,132,252,0.15); color:#c084fc; border:1px solid rgba(192,132,252,0.3); }
.det-chip.object { background:rgba(251,191,36,0.13); color:#fbbf24; border:1px solid rgba(251,191,36,0.28); }

/* Log panel */
.log-area {
  flex:1; overflow-y:auto; padding:0; display:none; flex-direction:column;
}
.log-area.open { display:flex; }
.log-area::-webkit-scrollbar { width:3px; }
.log-area::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
.log-toolbar {
  flex-shrink:0; padding:6px 14px; border-bottom:1px solid var(--border);
  display:flex; justify-content:space-between; align-items:center;
  background:rgba(0,0,0,0.2); font-size:0.72rem; color:var(--muted);
}
.log-clear-btn {
  font-size:0.7rem; padding:2px 8px; border-radius:6px; cursor:pointer;
  background:rgba(255,255,255,0.05); border:1px solid var(--border); color:var(--muted);
}
.log-clear-btn:hover { background:rgba(255,255,255,0.1); }
.log-entries { flex:1; overflow-y:auto; padding:6px 14px; display:flex; flex-direction:column; gap:1px; }
.log-entries::-webkit-scrollbar { width:3px; }
.log-entries::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
.log-entry {
  font-size:0.82rem; padding:4px 0;
  border-bottom:1px solid rgba(255,255,255,0.06);
  display:flex; gap:8px; align-items:baseline;
  font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
}
.log-entry .log-ts  { flex-shrink:0; font-size:0.70rem; color:#7c85a0; min-width:62px; }
.log-entry .log-lvl { flex-shrink:0; font-size:0.65rem; font-weight:600; padding:1px 5px;
                       border-radius:4px; background:rgba(255,255,255,0.07);
                       min-width:52px; text-align:center; letter-spacing:0.03em; }
.log-entry .log-msg { flex:1; word-break:break-word; color:#d1d5db; }
</style>
</head>
<body>

<!-- ── Header ── -->
<header class="header">
  <div class="header-left">
    <div class="dot-wrap">
      <div class="avatar-ring">P</div>
      <div class="status-dot" id="hdr-dot"></div>
    </div>
    <div>
      <div class="app-name">Pinku</div>
      <div class="status-pills">
        <span class="spill" id="pill-human" title="Person detected">👤</span>
        <span class="spill" id="pill-session" title="Session">💬</span>
      </div>
    </div>
  </div>
  <div class="header-btns">
    <button class="hdr-btn stop-btn" id="stop-btn" title="Stop speaking">⏹</button>
    <button class="hdr-btn mute-btn" id="mute-btn" title="Mute / Unmute">🔇</button>
  </div>
</header>

<!-- ── Main ── -->
<div class="main-area" id="main-area">
  <div class="pinky-stage">

    <!-- Question cloud -->
    <div class="question-cloud" id="q-cloud">
      <span class="question-cloud-label">You said</span>
      <span id="q-body"></span>
    </div>

    <div class="face-with-actions">
      <!-- Mic / wake button -->
      <button class="wake-btn-top" id="wake-btn" title="Tap to speak">🎙️</button>

      <!-- Face ring -->
      <div class="face-wrap" id="face-wrap-el">

        <!-- Thought bubble -->
        <div class="face-bubble" id="thought-bubble" style="--bubble-color:#c084fc">
          <span class="bubble-dot d1"></span>
          <span class="bubble-dot d2"></span>
          <span class="bubble-dot d3"></span>
        </div>

        <!-- THE FACE -->
        <div class="face sleeping" id="pinku-face">
          <div class="head">
            <div class="hair-side left"></div>
            <div class="hair-side right"></div>
            <div class="hair-fringe"></div>
            <div class="brows">
              <div class="brow left"></div>
              <div class="brow right"></div>
            </div>
            <div class="eyes">
              <div class="eye eye-left">
                <span class="eye-lashes">
                  <span class="lash"></span><span class="lash"></span>
                  <span class="lash"></span><span class="lash"></span>
                  <span class="lash"></span>
                </span>
              </div>
              <div class="eye eye-right">
                <span class="eye-lashes">
                  <span class="lash"></span><span class="lash"></span>
                  <span class="lash"></span><span class="lash"></span>
                  <span class="lash"></span>
                </span>
              </div>
            </div>
            <div class="nose-bridge"></div>
            <div class="mouth"></div>
            <div class="cheek left"></div>
            <div class="cheek right"></div>
          </div>
          <div class="voice-wave">
            <span></span><span></span><span></span><span></span><span></span>
          </div>
        </div><!-- face -->

        <!-- Sparkles -->
        <div class="face-sparkles">
          <span class="sp sp1">✦</span>
          <span class="sp sp2">✦</span>
          <span class="sp sp3">⋆</span>
          <span class="sp sp4">⋆</span>
        </div>

      </div><!-- face-wrap -->
    </div><!-- face-with-actions -->

    <!-- Reply cloud -->
    <div class="reply-cloud" id="r-cloud">
      <span class="reply-cloud-label">Pinku said</span>
      <span class="reply-cloud-body" id="r-body"></span>
    </div>

    <!-- Person badge -->
    <div class="person-badge" id="person-badge">
      <span class="person-dot"></span>
      <span id="person-name">Someone here</span>
    </div>

    <!-- Status bar -->
    <div class="status-bar sleeping" id="status-bar">
      <span class="status-bar-icon" id="sb-icon">😴</span>
      <div class="status-bar-text">
        <span class="status-bar-label" id="sb-label">Sleeping</span>
        <span class="status-bar-sub" id="sb-sub">Wake word: "Hey Pinku"</span>
      </div>
      <div class="status-bar-right">
        <span class="ws-dot off" id="ws-dot"></span>
      </div>
    </div>

  </div><!-- pinky-stage -->
</div><!-- main-area -->

<!-- Chat area (hidden by default) -->
<div class="chat-area" id="chat-area"></div>

<!-- Camera area (hidden by default) -->
<div class="camera-area" id="camera-area">
  <div class="cam-feed-wrap">
    <img id="cam-img" alt="Camera" style="display:none">
    <div class="cam-offline" id="cam-offline">📷 Camera offline or not started</div>
  </div>
  <div class="cam-toolbar">
    <span id="cam-status-text" style="font-size:0.72rem;color:#64748b">Checking…</span>
    <button class="log-clear-btn" id="cam-perm-btn" onclick="requestCameraPermission()">🔓 Request Permission</button>
  </div>
  <div class="cam-detect-feed" id="cam-detect-feed"></div>
</div>

<!-- Log area (hidden by default) -->
<div class="log-area" id="log-area">
  <div class="log-toolbar">
    <span>Live log — newest first</span>
    <button class="log-clear-btn" onclick="document.getElementById('log-entries').innerHTML=''">Clear</button>
  </div>
  <div class="log-entries" id="log-entries"></div>
</div>

<!-- Bottom nav -->
<nav class="pinky-nav">
  <button class="pinky-nav-item active" id="nav-home" onclick="showTab('home')">
    <span class="pinky-nav-ic">🏠</span>Home
  </button>
  <button class="pinky-nav-item" id="nav-chat" onclick="showTab('chat')">
    <span class="pinky-nav-ic">💬</span>Chat
  </button>
  <button class="pinky-nav-item" id="nav-cam" onclick="showTab('cam')">
    <span class="pinky-nav-ic">📷</span>Camera
  </button>
  <button class="pinky-nav-item" id="nav-log" onclick="showTab('log')">
    <span class="pinky-nav-ic">📋</span>Log
  </button>
</nav>

<script>
// ── State mapping ─────────────────────────────────────────────────────────────
const STATE_CSS = {
  idle:'sleeping', sleeping:'sleeping', awake:'awake',
  processing:'thinking', speaking:'speaking', muted:'paused', detection:'triggered',
};
const STATE_UI = {
  sleeping: { icon:'😴', label:'Sleeping',   sub:'Say "Hey Pinku" to wake me' },
  awake:    { icon:'👂', label:'Listening…', sub:'Speak freely — or tap 🎙️' },
  thinking: { icon:'⚡', label:'On it…',     sub:'Processing your request' },
  speaking: { icon:'🗣️', label:'Speaking…',  sub:'Tap ⏹ to stop' },
  paused:   { icon:'🔇', label:'Muted',      sub:'Tap mic or show ✊ to unmute' },
  triggered:{ icon:'⚡', label:'On it…',     sub:'Running your command' },
  listening:{ icon:'👁️', label:'Watching',   sub:'Step into view' },
};
const BUBBLE_STATES = new Set(['thinking','triggered']);
const BUBBLE_COLOR  = { thinking:'#c084fc', triggered:'#fb923c' };
const ALL_FACE_CSS  = ['sleeping','awake','listening','active','thinking','triggered','speaking','paused','connecting'];

let _lastStatus = {}, _lastTranscript = '', _lastReply = '';
let _detectRevertTimer = null;

// ── Face ─────────────────────────────────────────────────────────────────────
function setFaceClass(cls) {
  const face = document.getElementById('pinku-face');
  ALL_FACE_CSS.forEach(c => face.classList.remove(c));
  face.classList.add(cls);

  const sb = document.getElementById('status-bar');
  ALL_FACE_CSS.forEach(c => sb.classList.remove(c));
  sb.classList.add(cls);

  document.getElementById('hdr-dot').className = 'status-dot ' + cls;

  const bubble = document.getElementById('thought-bubble');
  if (BUBBLE_STATES.has(cls)) {
    bubble.style.setProperty('--bubble-color', BUBBLE_COLOR[cls] || '#c084fc');
    bubble.classList.add('visible');
  } else {
    bubble.classList.remove('visible');
  }

  const ui = STATE_UI[cls] || STATE_UI.sleeping;
  document.getElementById('sb-icon').textContent  = ui.icon;
  document.getElementById('sb-icon').className    = 'status-bar-icon ' + cls;
  document.getElementById('sb-label').textContent = ui.label;
  document.getElementById('sb-sub').textContent   = ui.sub;
}

// ── Status update from SSE ────────────────────────────────────────────────────
function applyStatus(data) {
  const isMuted    = data.muted;
  const isSpeaking = data.speaking;
  const rawState   = data.state || 'idle';

  let faceCls;
  if (isMuted)         faceCls = 'paused';
  else if (isSpeaking) faceCls = 'speaking';
  else                 faceCls = STATE_CSS[rawState] || 'sleeping';

  if (!_detectRevertTimer) setFaceClass(faceCls);

  // Question cloud
  if (data.last_transcript && data.last_transcript !== _lastTranscript) {
    _lastTranscript = data.last_transcript;
    document.getElementById('q-body').textContent = data.last_transcript;
    document.getElementById('q-cloud').classList.add('has-text');
  }
  // Reply cloud
  if (data.last_reply && data.last_reply !== _lastReply) {
    _lastReply = data.last_reply;
    document.getElementById('r-body').textContent = data.last_reply;
    document.getElementById('r-cloud').classList.add('has-text');
    addChat(data.last_transcript, data.last_reply);
  }

  // Mute button icon
  document.getElementById('mute-btn').textContent = isMuted ? '🎙️' : '🔇';
  document.getElementById('mute-btn').title = isMuted ? 'Unmute' : 'Mute';
  document.getElementById('mute-btn').classList.toggle('active', isMuted);

  // Camera status (included in initial SSE push and whenever camera.py changes state)
  if (data.cam_status !== undefined || data.det_status !== undefined) {
    const camTxt = data.cam_status || '?';
    const detTxt = data.det_status || '?';
    const msg = `Camera: ${camTxt} | Detection: ${detTxt}`;
    const offlineEl = document.getElementById('cam-offline');
    const statusEl  = document.getElementById('cam-status-text');
    if (offlineEl) offlineEl.textContent = '📷 ' + msg;
    if (statusEl)  statusEl.textContent  = msg;
  }
}

// ── Detection flash ───────────────────────────────────────────────────────────
function flashDetection(event) {
  setFaceClass('triggered');
  clearTimeout(_detectRevertTimer);
  _detectRevertTimer = setTimeout(() => {
    _detectRevertTimer = null;
    applyStatus(_lastStatus);
  }, 1500);

  const persons = event.persons || 0;
  if (persons > 0) {
    const badge = document.getElementById('person-badge');
    document.getElementById('person-name').textContent =
      persons > 1 ? `${persons} people here` : 'Someone here';
    badge.classList.add('visible');
    clearTimeout(badge._timer);
    badge._timer = setTimeout(() => badge.classList.remove('visible'), 5000);
    document.getElementById('pill-human').classList.add('on');
  }

  addDetectionRow(event);
  // Only write to the Log tab when a gesture is recognised — routine
  // person/object detections are already shown in the Camera tab feed.
  const gestures = event.gestures || [];
  if (gestures.length > 0) {
    addLog('gesture', '✋ ' + gestures.map(g => g.gesture).join(', '));
  }
}

// ── API action helper ─────────────────────────────────────────────────────────
function apiAction(name) {
  fetch('/api/action', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({action: name}),
  }).catch(() => {});
}

// ── Button wiring ─────────────────────────────────────────────────────────────
document.getElementById('wake-btn').addEventListener('click', () => {
  apiAction('wake');
  setFaceClass('awake');
});
document.getElementById('stop-btn').addEventListener('click', () => {
  apiAction('stop');
});
document.getElementById('mute-btn').addEventListener('click', () => {
  apiAction('mute_toggle');
});

// ── Chat panel ────────────────────────────────────────────────────────────────
let _chatLastTr = '', _chatLastRe = '';
function addChat(tr, re) {
  if (!tr && !re) return;
  if (tr === _chatLastTr && re === _chatLastRe) return;
  _chatLastTr = tr; _chatLastRe = re;
  const box = document.getElementById('chat-area');
  const now = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  if (tr) {
    const d = document.createElement('div');
    d.className = 'msg-row user';
    d.innerHTML = `<div><div class="bubble user">${esc(tr)}</div><div class="msg-time">${now}</div></div>`;
    box.appendChild(d);
  }
  if (re) {
    const d = document.createElement('div');
    d.className = 'msg-row pinku';
    d.innerHTML = `<div><div class="bubble pinku">${esc(re)}</div><div class="msg-time">${now}</div></div>`;
    box.appendChild(d);
  }
  box.scrollTop = box.scrollHeight;
}

// ── Camera panel ──────────────────────────────────────────────────────────────
let _camActive = false;
let _camTimer  = null;

function startCamera() {
  _camActive = true;
  refreshCam();
  fetch('/api/camera/status').then(r => r.json()).then(s => {
    const msg = `Camera: ${s.camera || '?'} | Detection: ${s.detection || '?'}`;
    document.getElementById('cam-offline').textContent = '📷 ' + msg;
    document.getElementById('cam-status-text').textContent = msg;
  }).catch(() => {});
}

function requestCameraPermission() {
  const btn = document.getElementById('cam-perm-btn');
  btn.textContent = '⏳ Requesting…';
  btn.disabled = true;
  addLog('camera', 'Requesting macOS camera permission…', '');
  fetch('/api/camera/request-permission').then(r => r.json()).then(s => {
    if (s.ok) {
      btn.textContent = '✅ Granted!';
      addLog('camera', 'Camera permission granted — restarting feed', '');
      document.getElementById('cam-status-text').textContent = 'Permission granted — reloading…';
      setTimeout(() => { stopCamera(); startCamera(); }, 1000);
    } else {
      btn.textContent = '❌ Denied';
      btn.disabled = false;
      addLog('error', 'Camera permission denied: ' + (s.output || ''), '');
      document.getElementById('cam-status-text').textContent =
        'Permission denied — go to System Settings → Privacy → Camera';
    }
    setTimeout(() => { btn.textContent = '🔓 Request Permission'; btn.disabled = false; }, 4000);
  }).catch(() => { btn.textContent = '🔓 Request Permission'; btn.disabled = false; });
}
function stopCamera() {
  _camActive = false;
  clearTimeout(_camTimer);
}
function refreshCam() {
  if (!_camActive) return;
  const img = document.getElementById('cam-img');
  const tmp = new Image();
  tmp.onload = () => {
    img.src = tmp.src;
    img.style.display = 'block';
    document.getElementById('cam-offline').style.display = 'none';
    _camTimer = setTimeout(refreshCam, 200);   // ~5 fps
  };
  tmp.onerror = () => {
    img.style.display = 'none';
    document.getElementById('cam-offline').style.display = 'flex';
    _camTimer = setTimeout(refreshCam, 2000);  // retry slower when offline
  };
  tmp.src = '/api/camera.jpg?t=' + Date.now();
}

function addDetectionRow(event) {
  const box = document.getElementById('cam-detect-feed');
  const row = document.createElement('div');
  row.className = 'det-row';
  const ts   = (event.timestamp || '').split(' ')[1] || '';
  let body = '';
  if (event.persons > 0)
    body += `<span class="det-chip person">👤 ${event.persons === 1 ? 'person' : event.persons + ' people'}</span>`;
  (event.gestures || []).forEach(g =>
    body += `<span class="det-chip gesture">✋ ${g.gesture}</span>`);
  (event.objects || []).forEach(o =>
    body += `<span class="det-chip object">${o.label}</span>`);
  row.innerHTML = `<span class="det-time">${esc(ts)}</span><span class="det-body">${body || 'no detection'}</span>`;
  box.insertBefore(row, box.firstChild);
  while (box.children.length > 60) box.removeChild(box.lastChild);
}

// ── Log panel ─────────────────────────────────────────────────────────────────
const LOG_ICONS = {
  stt:'🎙️', wake:'⚡', route:'🔀', chat:'💬', tts:'🔊',
  camera:'📷', gesture:'✋', error:'🔴', warn:'🟡',
  info:'⚪', you:'🧑', pinku:'🤖', cam:'📷',
  user:'🧑', source:'📖', llm:'🤖',
};
const LOG_COLORS = {
  stt:'#93c5fd', wake:'#fbbf24', route:'#a78bfa', chat:'#86efac',
  tts:'#f9a8d4', camera:'#67e8f9', gesture:'#c084fc', error:'#f87171',
  warn:'#fbbf24', info:'#64748b', you:'#93c5fd', pinku:'#c084fc', cam:'#67e8f9',
  user:'#93c5fd', source:'#86efac', llm:'#64748b',
};

function addLog(level, msg, ts) {
  const box = document.getElementById('log-entries');
  if (!box) return;
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  const t = ts || new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  const icon  = LOG_ICONS[level]  || '⚪';
  const color = LOG_COLORS[level] || '#94a3b8';
  entry.innerHTML =
    `<span class="log-ts">${esc(t)}</span>` +
    `<span class="log-lvl" style="color:${color};border:1px solid ${color}40">${icon} ${esc(level)}</span>` +
    `<span class="log-msg">${esc(msg)}</span>`;
  box.insertBefore(entry, box.firstChild);
  while (box.children.length > 200) box.removeChild(box.lastChild);
}

function formatDetEvent(e) {
  const parts = [];
  if (e.persons > 0) parts.push(`${e.persons} person${e.persons > 1 ? 's' : ''}`);
  (e.gestures || []).forEach(g => parts.push(`${g.gesture}`));
  (e.objects  || []).forEach(o => parts.push(`${o.label}(${o.conf})`));
  return parts.join(', ') || 'motion';
}

// ── Tab switching ─────────────────────────────────────────────────────────────
const _TABS = ['home', 'chat', 'cam', 'log'];
function showTab(name) {
  document.getElementById('main-area').style.display   = name === 'home' ? '' : 'none';
  document.getElementById('chat-area').className       = 'chat-area'   + (name === 'chat' ? ' open' : '');
  document.getElementById('camera-area').className     = 'camera-area' + (name === 'cam'  ? ' open' : '');
  document.getElementById('log-area').className        = 'log-area'    + (name === 'log'  ? ' open' : '');
  _TABS.forEach(t => {
    const el = document.getElementById('nav-' + t);
    if (el) el.classList.toggle('active', t === name);
  });
  if (name === 'cam') startCamera();
  else                stopCamera();
  if (name === 'chat') {
    const box = document.getElementById('chat-area');
    setTimeout(() => box.scrollTop = box.scrollHeight, 80);
  }
}

// ── SSE ───────────────────────────────────────────────────────────────────────
function connect() {
  const es = new EventSource('/api/stream');
  document.getElementById('ws-dot').className = 'ws-dot off';
  es.onopen  = () => document.getElementById('ws-dot').className = 'ws-dot on';
  es.onerror = () => {
    document.getElementById('ws-dot').className = 'ws-dot off';
    setTimeout(connect, 3000);
  };
  es.onmessage = e => {
    try {
      const data = JSON.parse(e.data);
      if (data.type === 'status') {
        _lastStatus = data;
        applyStatus(data);
      } else if (data.type === 'detection') {
        flashDetection(data);
      } else if (data.type === 'log') {
        addLog(data.level || 'info', data.msg || '', data.ts || '');
      }
    } catch(_) {}
  };
}
connect();

// ── Helpers ───────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>
"""
