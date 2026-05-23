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
<!-- Pinku Dashboard v3 — feminine character face -->
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
  bottom: 32px; left: 50%; transform: translateX(-50%);
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

/* Gesture badge */
#gesture-badge {
  position: absolute;
  bottom: 8px; left: 50%; transform: translateX(-50%);
  background: rgba(124,92,191,0.15);
  border: 1px solid rgba(124,92,191,0.35);
  border-radius: 20px;
  padding: 3px 12px;
  font-size: 0.68rem;
  color: var(--purple-l);
  white-space: nowrap;
  opacity: 0; transition: opacity 0.4s;
}
#gesture-badge.visible { opacity: 1; }

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

          <!-- ── Character Face (Pinky-style exact match) ── -->
          <svg id="face-svg" viewBox="0 0 200 230" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <!-- Face: warm cream -->
              <radialGradient id="face-grad" cx="42%" cy="32%" r="58%">
                <stop offset="0%"   stop-color="#fff5ea"/>
                <stop offset="65%"  stop-color="#fde0c4"/>
                <stop offset="100%" stop-color="#f5cba8"/>
              </radialGradient>
              <!-- Ears: pink -->
              <radialGradient id="ear-grad" cx="38%" cy="28%" r="62%">
                <stop offset="0%"   stop-color="#f8b8d8"/>
                <stop offset="100%" stop-color="#d96aa0"/>
              </radialGradient>
              <!-- Cap/hair: bright pink -->
              <linearGradient id="cap-grad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%"   stop-color="#f5a0c8"/>
                <stop offset="100%" stop-color="#d4508a"/>
              </linearGradient>
              <!-- Eye white -->
              <radialGradient id="eye-grad" cx="32%" cy="28%" r="58%">
                <stop offset="0%"   stop-color="#ffffff"/>
                <stop offset="100%" stop-color="#e8e8f2"/>
              </radialGradient>
              <!-- Iris: purple -->
              <radialGradient id="iris-grad" cx="30%" cy="25%" r="56%">
                <stop offset="0%"   stop-color="#9878e8"/>
                <stop offset="55%"  stop-color="#5c35b8"/>
                <stop offset="100%" stop-color="#321878"/>
              </radialGradient>
              <!-- Mouth disc: pink -->
              <radialGradient id="mouth-grad" cx="40%" cy="35%" r="60%">
                <stop offset="0%"   stop-color="#f5a0c0"/>
                <stop offset="100%" stop-color="#d46090"/>
              </radialGradient>
              <filter id="fsh" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000" flood-opacity="0.28"/>
              </filter>
              <filter id="fsh2" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000" flood-opacity="0.2"/>
              </filter>
            </defs>

            <!-- ── EARS (chunky rounded rectangles, behind face) ── -->
            <rect x="14" y="88" width="38" height="72" rx="19" fill="url(#ear-grad)" filter="url(#fsh)"/>
            <rect x="20" y="96" width="24" height="54" rx="12" fill="#f9d0e8" opacity="0.5"/>

            <rect x="148" y="88" width="38" height="72" rx="19" fill="url(#ear-grad)" filter="url(#fsh)"/>
            <rect x="156" y="96" width="24" height="54" rx="12" fill="#f9d0e8" opacity="0.5"/>

            <!-- ── HAIR CAP (flat rounded rect on top) ── -->
            <rect x="56" y="20" width="88" height="30" rx="15" fill="url(#cap-grad)" filter="url(#fsh)"/>
            <!-- Cap highlight -->
            <rect x="64" y="23" width="72" height="12" rx="7" fill="#fce0f0" opacity="0.4"/>

            <!-- ── FACE BODY (wide, squarish, very rounded) ── -->
            <rect x="44" y="38" width="112" height="152" rx="44" fill="url(#face-grad)" filter="url(#fsh)"/>

            <!-- ── EYES — SLEEPING ── -->
            <g class="eye-sleep">
              <!-- Left eye closed — curved line + ticks -->
              <path d="M66 103 Q80 95 94 103" stroke="#3a1858" stroke-width="3.5" fill="none" stroke-linecap="round"/>
              <line x1="68"  y1="102" x2="66"  y2="95"  stroke="#3a1858" stroke-width="2"   stroke-linecap="round"/>
              <line x1="75"  y1="97"  x2="74"  y2="90"  stroke="#3a1858" stroke-width="2.2" stroke-linecap="round"/>
              <line x1="83"  y1="95"  x2="83"  y2="88"  stroke="#3a1858" stroke-width="2.2" stroke-linecap="round"/>
              <line x1="91"  y1="97"  x2="93"  y2="91"  stroke="#3a1858" stroke-width="2"   stroke-linecap="round"/>
              <!-- Right eye closed -->
              <path d="M106 103 Q120 95 134 103" stroke="#3a1858" stroke-width="3.5" fill="none" stroke-linecap="round"/>
              <line x1="108" y1="102" x2="106" y2="95"  stroke="#3a1858" stroke-width="2"   stroke-linecap="round"/>
              <line x1="115" y1="97"  x2="114" y2="90"  stroke="#3a1858" stroke-width="2.2" stroke-linecap="round"/>
              <line x1="123" y1="95"  x2="123" y2="88"  stroke="#3a1858" stroke-width="2.2" stroke-linecap="round"/>
              <line x1="131" y1="97"  x2="133" y2="91"  stroke="#3a1858" stroke-width="2"   stroke-linecap="round"/>
              <!-- ZZZ -->
              <text x="146" y="74" font-size="9"  fill="#c9b1ff" opacity="0.85" font-weight="700">z</text>
              <text x="154" y="65" font-size="12" fill="#c9b1ff" opacity="0.5"  font-weight="700">z</text>
            </g>

            <!-- ── EYES — OPEN (awake / speaking) ── -->
            <g class="eye-open" style="display:none">
              <!-- LEFT EYE — round white -->
              <circle cx="80" cy="103" r="20" fill="url(#eye-grad)" filter="url(#fsh2)"/>
              <!-- Iris + pupil -->
              <circle class="pupil-l" cx="80" cy="103" r="11" fill="url(#iris-grad)"/>
              <circle class="pupil-l" cx="80" cy="103" r="5"  fill="#180a38"/>
              <!-- Catchlight -->
              <circle class="pupil-l-dot" cx="85" cy="98"  r="4"  fill="white"/>
              <circle cx="76"  cy="109" r="1.8" fill="white" opacity="0.35"/>
              <!-- Upper lash ticks (4 short lines fanning upward from top of eye) -->
              <line x1="64"  y1="95"  x2="61"  y2="87"  stroke="#3a1858" stroke-width="2.2" stroke-linecap="round"/>
              <line x1="73"  y1="87"  x2="72"  y2="79"  stroke="#3a1858" stroke-width="2.4" stroke-linecap="round"/>
              <line x1="82"  y1="84"  x2="82"  y2="76"  stroke="#3a1858" stroke-width="2.4" stroke-linecap="round"/>
              <line x1="91"  y1="86"  x2="94"  y2="79"  stroke="#3a1858" stroke-width="2.2" stroke-linecap="round"/>
              <!-- Eyebrow — thin, slightly arched -->
              <path d="M62 80 Q80 73 98 79" stroke="#7a4060" stroke-width="2.8" fill="none" stroke-linecap="round"/>

              <!-- RIGHT EYE -->
              <circle cx="120" cy="103" r="20" fill="url(#eye-grad)" filter="url(#fsh2)"/>
              <circle class="pupil-r" cx="120" cy="103" r="11" fill="url(#iris-grad)"/>
              <circle class="pupil-r" cx="120" cy="103" r="5"  fill="#180a38"/>
              <circle class="pupil-r-dot" cx="125" cy="98"  r="4"  fill="white"/>
              <circle cx="116"  cy="109" r="1.8" fill="white" opacity="0.35"/>
              <line x1="104" y1="95"  x2="101" y2="87"  stroke="#3a1858" stroke-width="2.2" stroke-linecap="round"/>
              <line x1="113" y1="87"  x2="112" y2="79"  stroke="#3a1858" stroke-width="2.4" stroke-linecap="round"/>
              <line x1="122" y1="84"  x2="122" y2="76"  stroke="#3a1858" stroke-width="2.4" stroke-linecap="round"/>
              <line x1="131" y1="86"  x2="134" y2="79"  stroke="#3a1858" stroke-width="2.2" stroke-linecap="round"/>
              <path d="M102 79 Q120 73 138 80" stroke="#7a4060" stroke-width="2.8" fill="none" stroke-linecap="round"/>
            </g>

            <!-- ── EYES — SURPRISED (wider circles, bigger irises) ── -->
            <g class="eye-surprise" style="display:none">
              <!-- Left wide -->
              <circle cx="80" cy="103" r="23" fill="url(#eye-grad)" filter="url(#fsh2)"/>
              <circle class="pupil-l" cx="80" cy="103" r="13" fill="url(#iris-grad)"/>
              <circle class="pupil-l" cx="80" cy="103" r="6"  fill="#180a38"/>
              <circle class="pupil-l-dot" cx="86" cy="97"  r="5"  fill="white"/>
              <!-- More dramatic lashes -->
              <line x1="61"  y1="93"  x2="57"  y2="83"  stroke="#3a1858" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="71"  y1="84"  x2="70"  y2="74"  stroke="#3a1858" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="81"  y1="81"  x2="81"  y2="71"  stroke="#3a1858" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="91"  y1="83"  x2="95"  y2="74"  stroke="#3a1858" stroke-width="2.5" stroke-linecap="round"/>
              <!-- Raised brow -->
              <path d="M58 76 Q80 67 102 75" stroke="#7a4060" stroke-width="2.8" fill="none" stroke-linecap="round"/>
              <!-- Right wide -->
              <circle cx="120" cy="103" r="23" fill="url(#eye-grad)" filter="url(#fsh2)"/>
              <circle class="pupil-r" cx="120" cy="103" r="13" fill="url(#iris-grad)"/>
              <circle class="pupil-r" cx="120" cy="103" r="6"  fill="#180a38"/>
              <circle class="pupil-r-dot" cx="126" cy="97"  r="5"  fill="white"/>
              <line x1="101" y1="93"  x2="97"  y2="83"  stroke="#3a1858" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="111" y1="84"  x2="110" y2="74"  stroke="#3a1858" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="121" y1="81"  x2="121" y2="71"  stroke="#3a1858" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="131" y1="83"  x2="135" y2="74"  stroke="#3a1858" stroke-width="2.5" stroke-linecap="round"/>
              <path d="M98 75 Q120 67 142 76" stroke="#7a4060" stroke-width="2.8" fill="none" stroke-linecap="round"/>
            </g>

            <!-- ── EYES — SQUINT / MUTED ── -->
            <g class="eye-squint" style="display:none">
              <path d="M63 105 Q80 114 97 105" stroke="#3a1858" stroke-width="4"   fill="none" stroke-linecap="round"/>
              <path d="M65 101 Q80 108 95 101" stroke="#3a1858" stroke-width="2"   fill="none" stroke-linecap="round" opacity="0.35"/>
              <line x1="70" y1="99" x2="79" y2="90" stroke="#f87171" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="79" y1="99" x2="70" y2="90" stroke="#f87171" stroke-width="2.5" stroke-linecap="round"/>
              <!-- furrowed brow -->
              <path d="M62 83 Q80 79 97 84" stroke="#7a4060" stroke-width="2.8" fill="none" stroke-linecap="round"/>
              <!-- right -->
              <path d="M103 105 Q120 114 137 105" stroke="#3a1858" stroke-width="4"   fill="none" stroke-linecap="round"/>
              <path d="M105 101 Q120 108 135 101" stroke="#3a1858" stroke-width="2"   fill="none" stroke-linecap="round" opacity="0.35"/>
              <line x1="110" y1="99" x2="119" y2="90" stroke="#f87171" stroke-width="2.5" stroke-linecap="round"/>
              <line x1="119" y1="99" x2="110" y2="90" stroke="#f87171" stroke-width="2.5" stroke-linecap="round"/>
              <path d="M103 84 Q120 79 138 83" stroke="#7a4060" stroke-width="2.8" fill="none" stroke-linecap="round"/>
            </g>

            <!-- ── MOUTH — IDLE: soft small smile ── -->
            <g class="mouth-idle">
              <path d="M88 152 Q100 162 112 152" stroke="#d46090" stroke-width="3" fill="none" stroke-linecap="round"/>
            </g>

            <!-- ── MOUTH — TALK: big round pink disc (signature Pinky look) ── -->
            <g class="mouth-talk" style="display:none">
              <circle id="mouth-oval" cx="100" cy="155" r="17" fill="url(#mouth-grad)" filter="url(#fsh2)"/>
              <!-- inner dark -->
              <circle cx="100" cy="156" r="10" fill="#8a2848" opacity="0.6"/>
              <!-- highlight -->
              <ellipse cx="94" cy="150" rx="5" ry="3" fill="white" opacity="0.25"/>
            </g>

            <!-- ── MOUTH — HMM: pressed flat line ── -->
            <g class="mouth-hmm" style="display:none">
              <path d="M88 155 Q100 151 112 155" stroke="#d46090" stroke-width="3.5" fill="none" stroke-linecap="round"/>
            </g>

            <!-- ── MOUTH — OOH: medium disc ── -->
            <g class="mouth-ooh" style="display:none">
              <circle cx="100" cy="154" r="12" fill="url(#mouth-grad)" filter="url(#fsh2)"/>
              <circle cx="100" cy="155" r="7"  fill="#8a2848" opacity="0.55"/>
              <ellipse cx="95" cy="150" rx="3.5" ry="2" fill="white" opacity="0.25"/>
            </g>

            <!-- ── MOUTH — MUTED: flat line ── -->
            <g class="mouth-muted" style="display:none">
              <line x1="86" y1="155" x2="114" y2="155" stroke="#d46090" stroke-width="3.5" stroke-linecap="round"/>
            </g>

            <!-- ── CHEEKS ── -->
            <ellipse cx="60"  cy="122" rx="14" ry="9" fill="#f09dc0" opacity="0.32"/>
            <ellipse cx="140" cy="122" rx="14" ry="9" fill="#f09dc0" opacity="0.32"/>
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
        <!-- Gesture badge -->
        <div id="gesture-badge"></div>
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

// Pupil/iris positions — round eyes at cx=80/120, cy=103
const PUPIL_POS = {
  sleeping:   { lx: 80,  ly: 106, rx: 120, ry: 106 },
  idle:       { lx: 80,  ly: 106, rx: 120, ry: 106 },
  awake:      { lx: 80,  ly: 103, rx: 120, ry: 103 },  // centred, alert
  processing: { lx: 77,  ly: 99,  rx: 117, ry: 99  },  // up-left = thinking
  speaking:   { lx: 80,  ly: 103, rx: 120, ry: 103 },
  muted:      { lx: 80,  ly: 105, rx: 120, ry: 105 },
  detection:  { lx: 80,  ly: 103, rx: 120, ry: 103 },
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
const GESTURE_EMOJI = {
  'Thumbs Up': '👍', 'Thumbs Down': '👎', 'Open Hand': '🖐',
  'Fist': '✊', 'Peace': '✌️', 'Pointing': '☝️',
  'Rock On': '🤘', 'Call Me': '🤙', 'Custom': '🤚',
};
const GESTURE_ACTION = {
  'Thumbs Up': 'Wake / listen', 'Thumbs Down': 'Stop speaking',
  'Open Hand': 'Pause', 'Fist': 'Mute toggle', 'Peace': 'What time is it?',
  'Pointing': 'Describe scene', 'Call Me': 'Unmute + wake',
};

function flashDetection(event) {
  // Person / object badge
  if (event.persons || (event.objects && event.objects.length)) {
    const badge = document.getElementById('detect-badge');
    const label = event.persons ? 'person' : (event.objects[0]?.label || 'object');
    badge.textContent = (label === 'person' ? '👤 ' : '📦 ') + label + ' detected';
    badge.classList.add('visible');
    clearTimeout(window._detectTimer);
    window._detectTimer = setTimeout(() => badge.classList.remove('visible'), 4000);
  }
  // Gesture badge
  if (event.gestures && event.gestures.length) {
    const g = event.gestures[0];
    const gBadge = document.getElementById('gesture-badge');
    const em  = GESTURE_EMOJI[g.gesture] || '✋';
    const act = GESTURE_ACTION[g.gesture] || g.gesture;
    gBadge.textContent = em + ' ' + g.gesture + ' → ' + act;
    gBadge.classList.add('visible');
    clearTimeout(window._gestureTimer);
    window._gestureTimer = setTimeout(() => gBadge.classList.remove('visible'), 3500);
  }
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

// ── Speaking mouth animation — pulse the big disc radius ─────────────────────
setInterval(() => {
  const disc = document.getElementById('mouth-oval');
  if (!disc) return;
  if (document.body.classList.contains('state-speaking')) {
    const r = 14 + Math.random() * 6;   // 14–20px radius
    disc.setAttribute('r', r.toFixed(1));
  }
}, 140);
</script>
</body>
</html>
"""
