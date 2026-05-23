"""
Web dashboard — Pinku avatar with full expressive face.
Matches Pinky character style: pink ears, eyelashes, round mouth,
context-sensitive expressions per state.
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

def push_detection(event: dict):
    _status["detections"].append(event)
    if len(_status["detections"]) > 50:
        _status["detections"] = _status["detections"][-50:]
    _broadcast({"type": "detection", **event})

# ── Routes ────────────────────────────────────────────────────────────────────

@_app.route("/")
def index():
    return render_template_string(HTML)

@_app.route("/api/stream")
def stream():
    import queue
    q: list = []
    with _clients_lock:
        _clients.append(q)
    def gen():
        yield f"data: {json.dumps({'type':'status',**_status})}\n\n"
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

# ── Start ─────────────────────────────────────────────────────────────────────

def start(logger=None, port=DASHBOARD_PORT):
    global _logger
    _logger = logger
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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pinku</title>
<style>
/* ── Reset & base ─────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:        #07090f;
  --surface:   #0f1118;
  --border:    rgba(255,255,255,0.07);
  --text:      #e0e0e0;
  --muted:     #888;
  --purple:    #7c5cbf;
  --purple-l:  #a07de0;
  --pink:      #e879a0;
  --green:     #4ade80;
  --amber:     #fbbf24;
  --red:       #f87171;
  --blue:      #60a5fa;
}
body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
  height: 100dvh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  user-select: none;
  -webkit-font-smoothing: antialiased;
}

/* ── Top bar ──────────────────────────────────────── */
#topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 20px 10px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  background: linear-gradient(180deg, rgba(124,92,191,0.08) 0%, transparent 100%);
}
#branding { display: flex; align-items: center; gap: 10px; }
#logo-chip {
  width: 34px; height: 34px; border-radius: 50%;
  background: linear-gradient(135deg, var(--purple), #3b2877);
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.9rem; color: #fff;
  box-shadow: 0 0 12px rgba(124,92,191,0.4);
}
#brand-name {
  font-size: 1.1rem; font-weight: 700;
  background: linear-gradient(90deg, #c9b1ff, var(--pink));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  letter-spacing: 0.5px;
}
#topbar-right { display: flex; gap: 12px; align-items: center; }
.top-btn {
  width: 32px; height: 32px; border-radius: 8px;
  background: rgba(255,255,255,0.06); border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem; cursor: pointer; transition: background 0.2s;
}
.top-btn:hover { background: rgba(255,255,255,0.12); }

/* ── Main area ────────────────────────────────────── */
#main {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 28px; overflow: hidden; padding: 10px;
  position: relative;
}

/* ── Avatar stage ─────────────────────────────────── */
#avatar-stage {
  position: relative;
  display: flex; align-items: center; justify-content: center;
}

/* Outer glow ring */
#orbit-ring {
  width: 320px; height: 320px; border-radius: 50%;
  background: radial-gradient(ellipse at center,
    rgba(30,20,55,0.95) 58%, rgba(15,10,30,0.98) 100%);
  position: relative;
  display: flex; align-items: center; justify-content: center;
  transition: box-shadow 0.6s ease;
  box-shadow:
    0 0 0 2px rgba(124,92,191,0.25),
    0 0 40px rgba(124,92,191,0.15),
    inset 0 0 60px rgba(0,0,0,0.5);
}

/* State-based ring glow */
body.state-awake    #orbit-ring { box-shadow: 0 0 0 2px rgba(74,222,128,0.4), 0 0 50px rgba(74,222,128,0.2), inset 0 0 60px rgba(0,0,0,0.5); }
body.state-processing #orbit-ring { box-shadow: 0 0 0 2px rgba(124,92,191,0.6), 0 0 60px rgba(124,92,191,0.35), inset 0 0 60px rgba(0,0,0,0.5); }
body.state-speaking #orbit-ring { box-shadow: 0 0 0 2px rgba(232,121,160,0.5), 0 0 55px rgba(232,121,160,0.3), inset 0 0 60px rgba(0,0,0,0.5); }
body.state-muted    #orbit-ring { box-shadow: 0 0 0 2px rgba(248,113,113,0.35), 0 0 40px rgba(248,113,113,0.15), inset 0 0 60px rgba(0,0,0,0.5); }

/* Listening pulse ring */
#pulse-ring {
  position: absolute;
  width: 320px; height: 320px; border-radius: 50%;
  border: 2px solid var(--green);
  opacity: 0; pointer-events: none;
  transform: scale(1);
}
body.state-awake #pulse-ring {
  animation: pulse-expand 2s ease-out infinite;
}
@keyframes pulse-expand {
  0%   { opacity: 0.6; transform: scale(1); }
  100% { opacity: 0;   transform: scale(1.18); }
}

/* Mic button (top of ring) */
#mic-btn {
  position: absolute;
  top: -14px;
  width: 34px; height: 34px; border-radius: 50%;
  background: linear-gradient(135deg, var(--purple), #3b2877);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.9rem;
  box-shadow: 0 2px 12px rgba(124,92,191,0.5);
  cursor: pointer; z-index: 10;
  transition: transform 0.2s;
}
#mic-btn:hover { transform: scale(1.08); }
body.state-muted #mic-btn { background: linear-gradient(135deg, var(--red), #a00); }

/* ── Character face SVG ───────────────────────────── */
#face-svg {
  width: 220px; height: 240px;
  overflow: visible;
  filter: drop-shadow(0 8px 24px rgba(0,0,0,0.5));
}

/* Eye eyelash lines */
.lash { stroke: #2a1a1a; stroke-width: 2.2; stroke-linecap: round; }

/* Pupils animate on state */
.pupil-l, .pupil-r { transition: cy 0.3s ease, cx 0.3s ease; }

/* Speaking bars below face */
#speak-bars {
  display: flex; gap: 4px; align-items: flex-end;
  height: 20px;
  position: absolute;
  bottom: 52px;
  opacity: 0; transition: opacity 0.3s;
}
body.state-speaking #speak-bars { opacity: 1; }
.bar {
  width: 6px; border-radius: 3px;
  background: linear-gradient(180deg, var(--pink), var(--purple));
  height: 4px;
}
body.state-speaking .bar { animation: wave 0.9s ease-in-out infinite; }
.bar:nth-child(1) { animation-delay: 0s; }
.bar:nth-child(2) { animation-delay: 0.12s; }
.bar:nth-child(3) { animation-delay: 0.24s; }
.bar:nth-child(4) { animation-delay: 0.36s; }
.bar:nth-child(5) { animation-delay: 0.18s; }
.bar:nth-child(6) { animation-delay: 0.06s; }
@keyframes wave {
  0%,100% { height: 4px; }
  50%      { height: 18px; }
}

/* Thinking dots */
#think-dots {
  position: absolute;
  top: 56px; right: 56px;
  display: none;
  gap: 4px;
}
body.state-processing #think-dots { display: flex; }
.dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--amber);
  animation: blink-dot 1.2s ease-in-out infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink-dot {
  0%,100% { opacity: 0.3; transform: scale(0.8); }
  50%      { opacity: 1;   transform: scale(1); }
}

/* Detection badge */
#detect-badge {
  position: absolute;
  bottom: 24px; left: 50%; transform: translateX(-50%);
  background: rgba(74,222,128,0.12);
  border: 1px solid rgba(74,222,128,0.3);
  border-radius: 20px;
  padding: 3px 12px;
  font-size: 0.68rem;
  color: var(--green);
  white-space: nowrap;
  opacity: 0; transition: opacity 0.4s;
}
#detect-badge.visible { opacity: 1; }

/* Idle breathe animation on face */
body.state-idle #face-svg, body.state-sleeping #face-svg {
  animation: breathe 4s ease-in-out infinite;
}
@keyframes breathe {
  0%,100% { transform: translateY(0); }
  50%      { transform: translateY(-4px); }
}

/* ── Status pill ──────────────────────────────────── */
#status-pill {
  display: flex; align-items: center; gap: 8px;
  background: rgba(255,255,255,0.05);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 8px 18px;
  font-size: 0.9rem;
  min-width: 160px;
  justify-content: center;
  transition: background 0.4s, border-color 0.4s;
}
body.state-awake     #status-pill { background: rgba(74,222,128,0.08); border-color: rgba(74,222,128,0.3); }
body.state-processing #status-pill { background: rgba(251,191,36,0.08); border-color: rgba(251,191,36,0.3); }
body.state-speaking  #status-pill { background: rgba(232,121,160,0.08); border-color: rgba(232,121,160,0.3); }
body.state-muted     #status-pill { background: rgba(248,113,113,0.08); border-color: rgba(248,113,113,0.3); }
#status-emoji { font-size: 1.1rem; }
#status-text  { font-weight: 600; letter-spacing: 0.3px; }
#status-live  {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--green); flex-shrink: 0;
  animation: live-blink 1.8s ease-in-out infinite;
}
body.state-idle #status-live,
body.state-sleeping #status-live { background: var(--muted); animation: none; }
body.state-muted #status-live { background: var(--red); }
body.state-processing #status-live { background: var(--amber); }
body.state-speaking #status-live { background: var(--pink); }
@keyframes live-blink {
  0%,100% { opacity: 1; }
  50%      { opacity: 0.3; }
}

/* ── Bottom nav ───────────────────────────────────── */
#nav {
  display: flex; border-top: 1px solid var(--border);
  flex-shrink: 0;
  background: rgba(0,0,0,0.3);
}
.nav-tab {
  flex: 1; padding: 10px 0 14px;
  display: flex; flex-direction: column;
  align-items: center; gap: 3px;
  cursor: pointer; opacity: 0.45;
  transition: opacity 0.2s;
  border: none; background: none;
  color: var(--text); font-size: 0.62rem;
  letter-spacing: 0.3px;
}
.nav-tab:hover { opacity: 0.75; }
.nav-tab.active { opacity: 1; color: var(--purple-l); }
.nav-tab.active .nav-icon { color: var(--purple-l); }
.nav-icon { font-size: 1.2rem; line-height: 1; }

/* ── Panels ───────────────────────────────────────── */
.panel { display: none; flex: 1; flex-direction: column; overflow: hidden; }
.panel.active { display: flex; }

/* Chat panel */
#chat-messages {
  flex: 1; overflow-y: auto; padding: 16px;
  display: flex; flex-direction: column; gap: 10px;
}
#chat-messages::-webkit-scrollbar { width: 3px; }
#chat-messages::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
.bubble {
  max-width: 78%; padding: 10px 14px;
  border-radius: 18px; font-size: 0.88rem; line-height: 1.45;
}
.bubble.you {
  align-self: flex-end;
  background: linear-gradient(135deg, var(--purple), #3b2877);
  border-bottom-right-radius: 4px;
}
.bubble.pinku {
  align-self: flex-start;
  background: rgba(255,255,255,0.07);
  border: 1px solid var(--border);
  border-bottom-left-radius: 4px;
}
.bubble-label {
  font-size: 0.65rem; color: var(--muted);
  margin-bottom: 2px; letter-spacing: 0.5px;
}

/* Log panel */
#log-list {
  flex: 1; overflow-y: auto; padding: 12px;
  display: flex; flex-direction: column; gap: 6px;
}
.log-item {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 12px;
  font-size: 0.78rem;
  display: flex; gap: 10px; align-items: baseline;
}
.log-time { color: var(--muted); flex-shrink: 0; font-size: 0.7rem; }
.log-label {
  font-size: 0.65rem; font-weight: 600;
  padding: 1px 6px; border-radius: 4px;
  background: rgba(124,92,191,0.2); color: var(--purple-l);
  flex-shrink: 0;
}
.log-body { color: var(--text); word-break: break-word; }

/* Music panel placeholder */
#music-panel {
  flex: 1; display: none; align-items: center; justify-content: center;
  flex-direction: column; gap: 12px; color: var(--muted); font-size: 0.9rem;
}
#music-panel.active { display: flex; }
</style>
</head>
<body class="state-idle">

<!-- Top bar -->
<div id="topbar">
  <div id="branding">
    <div id="logo-chip">P</div>
    <span id="brand-name">Pinku</span>
  </div>
  <div id="topbar-right">
    <div class="top-btn" id="model-chip" title="Model">🧠</div>
    <div class="top-btn" id="mute-btn" title="Mute">🎙️</div>
  </div>
</div>

<!-- ─── Panels wrapper ─── -->
<div style="flex:1;display:flex;flex-direction:column;overflow:hidden;" id="panels-wrapper">

  <!-- HOME panel -->
  <div class="panel active" id="home-panel">
    <div id="main">

      <!-- Avatar stage -->
      <div id="avatar-stage">
        <div id="pulse-ring"></div>
        <div id="orbit-ring">

          <!-- Mic button -->
          <div id="mic-btn">🎙️</div>

          <!-- Thinking dots -->
          <div id="think-dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
          </div>

          <!-- ── Character Face ── -->
          <svg id="face-svg" viewBox="0 0 200 230" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <radialGradient id="face-grad" cx="50%" cy="40%" r="55%">
                <stop offset="0%"   stop-color="#fef0e2"/>
                <stop offset="100%" stop-color="#f5d5b8"/>
              </radialGradient>
              <radialGradient id="ear-grad" cx="50%" cy="35%" r="60%">
                <stop offset="0%"   stop-color="#f09dc0"/>
                <stop offset="100%" stop-color="#d4679a"/>
              </radialGradient>
              <radialGradient id="eye-grad" cx="38%" cy="35%" r="60%">
                <stop offset="0%"   stop-color="#ffffff"/>
                <stop offset="100%" stop-color="#e8e8f0"/>
              </radialGradient>
              <radialGradient id="pupil-grad" cx="35%" cy="30%" r="55%">
                <stop offset="0%"   stop-color="#8a6bd0"/>
                <stop offset="100%" stop-color="#4a2d9c"/>
              </radialGradient>
              <filter id="soft-shadow">
                <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000" flood-opacity="0.35"/>
              </filter>
            </defs>

            <!-- Left ear -->
            <ellipse cx="34" cy="122" rx="24" ry="34" fill="url(#ear-grad)" filter="url(#soft-shadow)"/>
            <ellipse cx="34" cy="122" rx="14" ry="22" fill="#f5b8d2" opacity="0.6"/>

            <!-- Right ear -->
            <ellipse cx="166" cy="122" rx="24" ry="34" fill="url(#ear-grad)" filter="url(#soft-shadow)"/>
            <ellipse cx="166" cy="122" rx="14" ry="22" fill="#f5b8d2" opacity="0.6"/>

            <!-- Pink top / cap -->
            <rect x="62" y="16" width="76" height="20" rx="10" fill="#e879a0" filter="url(#soft-shadow)"/>
            <rect x="70" y="18" width="60" height="10" rx="5" fill="#f09dc0" opacity="0.5"/>

            <!-- Face body -->
            <rect x="46" y="30" width="108" height="150" rx="38" fill="url(#face-grad)" filter="url(#soft-shadow)"/>

            <!-- ── EYES — idle/sleeping state ── -->
            <g class="eye-sleep">
              <!-- Left eye closed -->
              <path id="eye-l-sleep" d="M66 105 Q78 98 90 105" stroke="#c48 " stroke-width="3" fill="none" stroke-linecap="round"/>
              <!-- Right eye closed -->
              <path id="eye-r-sleep" d="M110 105 Q122 98 134 105" stroke="#c48" stroke-width="3" fill="none" stroke-linecap="round"/>
              <!-- ZZZ -->
              <text x="150" y="72" font-size="9" fill="#c9b1ff" opacity="0.7" font-weight="bold">z</text>
              <text x="157" y="64" font-size="11" fill="#c9b1ff" opacity="0.5" font-weight="bold">z</text>
            </g>

            <!-- ── EYES — open state ── -->
            <g class="eye-open" style="display:none">
              <!-- Left eye white -->
              <ellipse id="eye-l-white" cx="78" cy="103" rx="19" ry="21" fill="url(#eye-grad)"/>
              <!-- Left pupil -->
              <circle class="pupil-l" cx="80" cy="105" r="10" fill="url(#pupil-grad)"/>
              <circle class="pupil-l-dot" cx="83" cy="102" r="3.5" fill="white"/>
              <!-- Left eyelashes -->
              <line x1="63" y1="90" x2="67" y2="84" class="lash"/>
              <line x1="74" y1="85" x2="76" y2="79" class="lash"/>
              <line x1="85" y1="86" x2="89" y2="81" class="lash"/>

              <!-- Right eye white -->
              <ellipse id="eye-r-white" cx="122" cy="103" rx="19" ry="21" fill="url(#eye-grad)"/>
              <!-- Right pupil -->
              <circle class="pupil-r" cx="124" cy="105" r="10" fill="url(#pupil-grad)"/>
              <circle class="pupil-r-dot" cx="127" cy="102" r="3.5" fill="white"/>
              <!-- Right eyelashes -->
              <line x1="111" y1="90" x2="115" y2="84" class="lash"/>
              <line x1="120" y1="85" x2="122" y2="79" class="lash"/>
              <line x1="131" y1="86" x2="135" y2="81" class="lash"/>
            </g>

            <!-- ── EYES — surprised state ── -->
            <g class="eye-surprise" style="display:none">
              <!-- Left eye wide -->
              <ellipse cx="78" cy="101" rx="21" ry="24" fill="url(#eye-grad)"/>
              <circle cx="78" cy="103" r="11" fill="url(#pupil-grad)"/>
              <circle cx="82" cy="99" r="4" fill="white"/>
              <!-- Left highlight arc -->
              <path d="M58 88 Q78 80 98 88" stroke="#f09dc0" stroke-width="2.5" fill="none" stroke-linecap="round"/>

              <!-- Right eye wide -->
              <ellipse cx="122" cy="101" rx="21" ry="24" fill="url(#eye-grad)"/>
              <circle cx="122" cy="103" r="11" fill="url(#pupil-grad)"/>
              <circle cx="126" cy="99" r="4" fill="white"/>
              <path d="M102 88 Q122 80 142 88" stroke="#f09dc0" stroke-width="2.5" fill="none" stroke-linecap="round"/>
            </g>

            <!-- ── EYES — squint/muted state ── -->
            <g class="eye-squint" style="display:none">
              <!-- Left squint -->
              <path d="M60 104 Q78 110 96 104" stroke="#c48" stroke-width="3.5" fill="none" stroke-linecap="round"/>
              <path d="M61 100 Q78 106 95 100" stroke="#c48" stroke-width="1.5" fill="none" stroke-linecap="round" opacity="0.4"/>
              <!-- X marks -->
              <line x1="70" y1="98" x2="76" y2="92" stroke="#f87171" stroke-width="2" stroke-linecap="round"/>
              <line x1="76" y1="98" x2="70" y2="92" stroke="#f87171" stroke-width="2" stroke-linecap="round"/>
              <!-- Right squint -->
              <path d="M104 104 Q122 110 140 104" stroke="#c48" stroke-width="3.5" fill="none" stroke-linecap="round"/>
              <path d="M105 100 Q122 106 139 100" stroke="#c48" stroke-width="1.5" fill="none" stroke-linecap="round" opacity="0.4"/>
              <line x1="114" y1="98" x2="120" y2="92" stroke="#f87171" stroke-width="2" stroke-linecap="round"/>
              <line x1="120" y1="98" x2="114" y2="92" stroke="#f87171" stroke-width="2" stroke-linecap="round"/>
            </g>

            <!-- ── MOUTH states ── -->

            <!-- mouth-idle: tiny soft smile -->
            <g class="mouth-idle">
              <path d="M88 148 Q100 156 112 148" stroke="#d4679a" stroke-width="2.5" fill="none" stroke-linecap="round"/>
            </g>

            <!-- mouth-talk: open oval, animated -->
            <g class="mouth-talk" style="display:none">
              <ellipse id="mouth-oval" cx="100" cy="152" rx="14" ry="16" fill="#e879a0"/>
              <ellipse cx="100" cy="152" rx="9" ry="10" fill="#7a2040" opacity="0.5"/>
              <!-- tongue hint -->
              <ellipse cx="100" cy="160" rx="6" ry="3.5" fill="#f09dc0" opacity="0.7"/>
            </g>

            <!-- mouth-hmm: thinking pressed lips -->
            <g class="mouth-hmm" style="display:none">
              <path d="M86 150 Q100 145 114 150" stroke="#d4679a" stroke-width="3" fill="none" stroke-linecap="round"/>
              <path d="M90 153 Q100 157 110 153" stroke="#d4679a" stroke-width="2" fill="none" stroke-linecap="round" opacity="0.5"/>
            </g>

            <!-- mouth-ooh: small excited O -->
            <g class="mouth-ooh" style="display:none">
              <ellipse cx="100" cy="150" rx="10" ry="12" fill="#e879a0"/>
              <ellipse cx="100" cy="150" rx="6" ry="7" fill="#7a2040" opacity="0.5"/>
            </g>

            <!-- mouth-muted: flat line -->
            <g class="mouth-muted" style="display:none">
              <path d="M86 150 L114 150" stroke="#d4679a" stroke-width="3" stroke-linecap="round"/>
            </g>

            <!-- Cheek blush -->
            <ellipse cx="58" cy="125" rx="11" ry="7" fill="#f09dc0" opacity="0.35"/>
            <ellipse cx="142" cy="125" rx="11" ry="7" fill="#f09dc0" opacity="0.35"/>
          </svg>
          <!-- end face SVG -->

          <!-- Speaking bars (inside orbit ring, below face) -->
          <div id="speak-bars">
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
          </div>

        </div><!-- orbit-ring -->

        <!-- Detection badge -->
        <div id="detect-badge">👤 Someone here</div>
      </div><!-- avatar-stage -->

      <!-- Status pill -->
      <div id="status-pill">
        <span id="status-emoji">😴</span>
        <span id="status-text">Sleeping</span>
        <div id="status-live"></div>
      </div>

    </div><!-- main -->
  </div><!-- home-panel -->

  <!-- CHAT panel -->
  <div class="panel" id="chat-panel">
    <div id="chat-messages"></div>
  </div>

  <!-- LOG panel -->
  <div class="panel" id="log-panel">
    <div id="log-list"></div>
  </div>

  <!-- MUSIC panel -->
  <div id="music-panel">
    <div style="font-size:2.5rem;">🎵</div>
    <div>Music control coming soon</div>
  </div>

</div><!-- panels-wrapper -->

<!-- Bottom nav -->
<nav id="nav">
  <button class="nav-tab active" onclick="switchTab('home')" id="tab-home">
    <span class="nav-icon">🏠</span>Home
  </button>
  <button class="nav-tab" onclick="switchTab('chat')" id="tab-chat">
    <span class="nav-icon">💬</span>Chat
  </button>
  <button class="nav-tab" onclick="switchTab('log')" id="tab-log">
    <span class="nav-icon">📋</span>Log
  </button>
  <button class="nav-tab" onclick="switchTab('music')" id="tab-music">
    <span class="nav-icon">🎵</span>Music
  </button>
</nav>

<script>
// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.panel, #music-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  if (name === 'music') {
    document.getElementById('music-panel').classList.add('active');
  } else {
    document.getElementById(name + '-panel').classList.add('active');
  }
  document.getElementById('tab-' + name).classList.add('active');
}

// ── Face expression engine ───────────────────────────────────────────────────
const EXPRESSIONS = {
  //          eyes           mouth
  sleeping:  ['sleep',      'idle'],
  idle:      ['sleep',      'idle'],
  awake:     ['open',       'ooh'],       // alert, listening — wide eyes + excited mouth
  processing:['open',       'hmm'],       // thinking — eyes open, pressed lips
  speaking:  ['open',       'talk'],      // talking — eyes open, open mouth
  muted:     ['squint',     'muted'],     // muted — squinting X eyes, flat mouth
  detection: ['surprise',   'ooh'],       // person detected — surprised
};

// Pupil positions per state (cx offset from center, cy offset)
const PUPIL_POS = {
  sleeping:   { lx: 80,  ly: 108, rx: 124, ry: 108 },
  idle:       { lx: 80,  ly: 108, rx: 124, ry: 108 },
  awake:      { lx: 78,  ly: 104, rx: 122, ry: 104 },
  processing: { lx: 76,  ly: 100, rx: 120, ry: 100 }, // pupils up-left (thinking)
  speaking:   { lx: 80,  ly: 105, rx: 124, ry: 105 },
  muted:      { lx: 80,  ly: 106, rx: 124, ry: 106 },
  detection:  { lx: 78,  ly: 103, rx: 122, ry: 103 },
};

function setExpression(state) {
  const [eyeState, mouthState] = EXPRESSIONS[state] || EXPRESSIONS.idle;

  // Eyes
  document.querySelector('.eye-sleep').style.display    = eyeState === 'sleep'    ? '' : 'none';
  document.querySelector('.eye-open').style.display     = eyeState === 'open'     ? '' : 'none';
  document.querySelector('.eye-surprise').style.display = eyeState === 'surprise' ? '' : 'none';
  document.querySelector('.eye-squint').style.display   = eyeState === 'squint'   ? '' : 'none';

  // Pupils
  const pos = PUPIL_POS[state] || PUPIL_POS.idle;
  document.querySelectorAll('.pupil-l').forEach(el => {
    el.setAttribute('cx', pos.lx); el.setAttribute('cy', pos.ly);
  });
  document.querySelectorAll('.pupil-l-dot').forEach(el => {
    el.setAttribute('cx', pos.lx + 3); el.setAttribute('cy', pos.ly - 3);
  });
  document.querySelectorAll('.pupil-r').forEach(el => {
    el.setAttribute('cx', pos.rx); el.setAttribute('cy', pos.ry);
  });
  document.querySelectorAll('.pupil-r-dot').forEach(el => {
    el.setAttribute('cx', pos.rx + 3); el.setAttribute('cy', pos.ry - 3);
  });

  // Mouth
  document.querySelector('.mouth-idle').style.display  = mouthState === 'idle'  ? '' : 'none';
  document.querySelector('.mouth-talk').style.display  = mouthState === 'talk'  ? '' : 'none';
  document.querySelector('.mouth-hmm').style.display   = mouthState === 'hmm'   ? '' : 'none';
  document.querySelector('.mouth-ooh').style.display   = mouthState === 'ooh'   ? '' : 'none';
  document.querySelector('.mouth-muted').style.display = mouthState === 'muted' ? '' : 'none';
}

// ── State display map ─────────────────────────────────────────────────────────
const STATE_UI = {
  sleeping:   { emoji: '😴', text: 'Sleeping' },
  idle:       { emoji: '😴', text: 'Dozing' },
  awake:      { emoji: '👂', text: 'Listening…' },
  processing: { emoji: '⚡', text: 'On it…' },
  speaking:   { emoji: '🗣️', text: 'Speaking' },
  muted:      { emoji: '🔇', text: 'Muted' },
};

let _lastDetect = 0;

function applyStatus(data) {
  const state = data.muted ? 'muted' : (data.speaking ? 'speaking' : data.state);
  const bodyClass = 'state-' + (state || 'idle');

  // Body class
  document.body.className = bodyClass;

  // Expression
  setExpression(state);

  // Status pill
  const ui = STATE_UI[state] || STATE_UI.idle;
  document.getElementById('status-emoji').textContent = ui.emoji;
  document.getElementById('status-text').textContent  = ui.text;

  // Mic btn icon
  document.getElementById('mic-btn').textContent = data.muted ? '🔇' : '🎙️';

  // Model chip
  if (data.model) document.getElementById('model-chip').title = data.model;

  // Chat messages
  if (data.last_transcript || data.last_reply) {
    addChat(data.last_transcript, data.last_reply);
  }
}

// ── Chat ──────────────────────────────────────────────────────────────────────
let _lastTranscript = '', _lastReply = '';
function addChat(tr, re) {
  if (!tr && !re) return;
  if (tr === _lastTranscript && re === _lastReply) return;
  _lastTranscript = tr; _lastReply = re;

  const box = document.getElementById('chat-messages');
  if (tr) {
    const d = document.createElement('div');
    d.innerHTML = `<div class="bubble-label">YOU</div><div class="bubble you">${esc(tr)}</div>`;
    box.appendChild(d);
  }
  if (re) {
    const d = document.createElement('div');
    d.innerHTML = `<div class="bubble-label">PINKU</div><div class="bubble pinku">${esc(re)}</div>`;
    box.appendChild(d);
  }
  box.scrollTop = box.scrollHeight;
}

// ── Log ───────────────────────────────────────────────────────────────────────
function addLog(event) {
  const list = document.getElementById('log-list');
  const now  = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  const label = event.type === 'detection' ? (event.label || 'detect') : event.type;
  const body  = event.label
    ? `${event.label} ${event.confidence ? Math.round(event.confidence * 100) + '%' : ''}`
    : JSON.stringify(event).slice(0, 80);
  const el = document.createElement('div');
  el.className = 'log-item';
  el.innerHTML = `<span class="log-time">${now}</span>
                  <span class="log-label">${esc(label)}</span>
                  <span class="log-body">${esc(body)}</span>`;
  list.appendChild(el);
  if (list.children.length > 80) list.removeChild(list.firstChild);
  list.scrollTop = list.scrollHeight;
}

// ── Detection badge ───────────────────────────────────────────────────────────
function flashDetection(event) {
  const badge = document.getElementById('detect-badge');
  const label = event.label || 'Object';
  badge.textContent = (label === 'person' ? '👤 ' : '📦 ') + label + ' detected';
  badge.classList.add('visible');
  clearTimeout(window._detectTimer);
  window._detectTimer = setTimeout(() => badge.classList.remove('visible'), 4000);
  addLog(event);
}

// ── SSE ───────────────────────────────────────────────────────────────────────
function connect() {
  const es = new EventSource('/api/stream');
  es.onmessage = e => {
    try {
      const data = JSON.parse(e.data);
      if (data.type === 'status') applyStatus(data);
      else if (data.type === 'detection') flashDetection(data);
    } catch(_) {}
  };
  es.onerror = () => setTimeout(connect, 3000);
}
connect();

// ── Helpers ───────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Speaking mouth animation ──────────────────────────────────────────────────
// Animate the oval mouth size when speaking
setInterval(() => {
  const oval = document.getElementById('mouth-oval');
  if (!oval) return;
  if (document.body.classList.contains('state-speaking')) {
    const r = 10 + Math.random() * 8;
    oval.setAttribute('ry', r.toFixed(1));
  }
}, 150);
</script>
</body>
</html>
"""
