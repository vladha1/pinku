"""
Web dashboard — Pinky-style avatar UI with animated face, state-reactive.
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
}

HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pinku</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: #07090f;
  color: #e0e0e0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  user-select: none;
}

/* ── Top bar ─────────────────────────────────────────────── */
#topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 20px 8px;
  flex-shrink: 0;
}
#branding { display: flex; align-items: center; gap: 10px; }
#avatar-chip {
  width: 32px; height: 32px; border-radius: 50%;
  background: linear-gradient(135deg, #7c5cbf, #4a3a8a);
  display: flex; align-items: center; justify-content: center;
  font-weight: bold; font-size: 0.9rem; color: #fff;
}
#brand-name { font-size: 1rem; font-weight: 600; color: #fff; }
#top-controls { display: flex; gap: 8px; }
.ctrl-btn {
  width: 36px; height: 36px; border-radius: 50%;
  background: #1a1d2e; border: 1px solid #2a2d3e;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; font-size: 1rem; transition: background 0.2s;
}
.ctrl-btn:hover { background: #252840; }
.ctrl-btn.muted { background: #3a1a1a; border-color: #662222; }
#temp-badge {
  background: #1a1d2e; border: 1px solid #2a2d3e;
  border-radius: 20px; padding: 4px 10px; font-size: 0.75rem; color: #888;
}

/* ── Main area ───────────────────────────────────────────── */
#main {
  flex: 1;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  padding: 10px 20px;
  gap: 20px;
}

/* ── Avatar circle ───────────────────────────────────────── */
#avatar-ring {
  position: relative;
  width: 260px; height: 260px;
}

#mic-btn {
  position: absolute; top: -16px; left: 50%; transform: translateX(-50%);
  width: 36px; height: 36px; border-radius: 50%; z-index: 10;
  background: linear-gradient(135deg, #7c5cbf, #5a3fa0);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.9rem; cursor: pointer; border: none;
  box-shadow: 0 2px 12px rgba(124,92,191,0.5);
  transition: all 0.2s;
}
#mic-btn:hover { transform: translateX(-50%) scale(1.1); }

#orbit {
  width: 260px; height: 260px; border-radius: 50%;
  background: radial-gradient(circle at 40% 35%, #3a3060 0%, #1e1838 50%, #0f0d20 100%);
  border: 2px solid #2d2650;
  display: flex; align-items: center; justify-content: center;
  position: relative;
  box-shadow: 0 0 40px rgba(100,80,180,0.15), inset 0 0 60px rgba(0,0,0,0.4);
  transition: box-shadow 0.4s;
}
#orbit.awake  { box-shadow: 0 0 50px rgba(0,255,136,0.2), inset 0 0 60px rgba(0,0,0,0.3); }
#orbit.speak  { box-shadow: 0 0 50px rgba(160,100,255,0.3), inset 0 0 60px rgba(0,0,0,0.3); }
#orbit.muted  { box-shadow: 0 0 30px rgba(255,60,60,0.15), inset 0 0 60px rgba(0,0,0,0.4); }

/* Robot face SVG */
#face-svg { width: 160px; height: 180px; }

/* Eyes */
.eye { transition: all 0.3s; }
.eye-open { opacity: 1; }
.eye-shut  { opacity: 1; }

/* Speaking wave bars */
#speak-bars {
  display: flex; gap: 3px; align-items: flex-end;
  height: 16px; margin-top: 4px;
}
.bar {
  width: 4px; border-radius: 2px;
  background: linear-gradient(to top, #7c5cbf, #00ff88);
  animation: none;
}
.bar.active { animation: wave 0.6s ease-in-out infinite; }
.bar:nth-child(1) { animation-delay: 0s;    height: 6px; }
.bar:nth-child(2) { animation-delay: 0.1s;  height: 10px; }
.bar:nth-child(3) { animation-delay: 0.2s;  height: 14px; }
.bar:nth-child(4) { animation-delay: 0.1s;  height: 10px; }
.bar:nth-child(5) { animation-delay: 0s;    height: 6px; }
@keyframes wave {
  0%,100% { transform: scaleY(0.4); }
  50%      { transform: scaleY(1.0); }
}

/* Listening pulse ring */
#listen-ring {
  position: absolute; inset: -12px; border-radius: 50%;
  border: 2px solid transparent; transition: all 0.4s;
  pointer-events: none;
}
#listen-ring.active {
  border-color: rgba(0,255,136,0.4);
  animation: pulse-ring 1.5s ease-in-out infinite;
}
@keyframes pulse-ring {
  0%,100% { transform: scale(1); opacity: 0.6; }
  50%      { transform: scale(1.04); opacity: 1; }
}

/* ── Status pills ────────────────────────────────────────── */
#status-pill {
  background: #13152a; border: 1px solid #2a2d45;
  border-radius: 24px; padding: 8px 20px;
  font-size: 0.9rem; font-weight: 500;
  display: flex; align-items: center; gap: 8px;
  transition: all 0.3s;
}

#status-bar {
  background: #0e1020; border: 1px solid #1e2035;
  border-radius: 16px; padding: 10px 16px;
  display: flex; align-items: center; gap: 10px;
  width: 100%; max-width: 420px;
}
#status-bar-icon { font-size: 1.1rem; }
#status-bar-text { flex: 1; }
#status-bar-title { font-size: 0.82rem; font-weight: 600; color: #ccc; }
#status-bar-sub   { font-size: 0.72rem; color: #555; margin-top: 1px; }
#live-dot { width: 8px; height: 8px; border-radius: 50%; background: #333; flex-shrink: 0; }
#live-dot.live { background: #00ff88; box-shadow: 0 0 6px #00ff88; }

/* ── Chat tab ────────────────────────────────────────────── */
#chat-view {
  display: none; flex: 1; overflow-y: auto;
  padding: 12px 20px; flex-direction: column; gap: 8px;
}
.chat-msg { max-width: 80%; padding: 10px 14px; border-radius: 14px;
            font-size: 0.85rem; line-height: 1.5; }
.chat-you  { background: #1e2240; color: #ccc; align-self: flex-end; border-radius: 14px 14px 4px 14px; }
.chat-pinku { background: #131520; color: #e0e0e0; align-self: flex-start; border-radius: 14px 14px 14px 4px;
              border: 1px solid #1e2035; }

/* ── Bottom nav ──────────────────────────────────────────── */
#bottom-nav {
  display: flex; justify-content: space-around;
  padding: 10px 0 16px;
  border-top: 1px solid #13152a;
  flex-shrink: 0;
}
.nav-btn {
  display: flex; flex-direction: column; align-items: center;
  gap: 4px; cursor: pointer; padding: 4px 20px;
  border-radius: 12px; transition: all 0.2s;
  border: none; background: transparent; color: #555;
}
.nav-btn:hover { color: #888; }
.nav-btn.active { color: #7c5cbf; }
.nav-btn .nav-icon { font-size: 1.3rem; }
.nav-btn .nav-lbl  { font-size: 0.6rem; letter-spacing: 0.5px; }
</style>
</head>
<body>

<!-- Top bar -->
<div id="topbar">
  <div id="branding">
    <div id="avatar-chip">P</div>
    <span id="brand-name">Pinku</span>
  </div>
  <div id="top-controls">
    <div class="ctrl-btn" id="mute-btn" title="Mute">🔇</div>
    <div class="ctrl-btn" id="cam-btn" title="Camera">📷</div>
  </div>
</div>

<!-- Home view -->
<div id="main">

  <!-- Avatar -->
  <div id="avatar-ring">
    <button id="mic-btn">🎙️</button>
    <div id="orbit">
      <div id="listen-ring"></div>
      <div style="display:flex;flex-direction:column;align-items:center;gap:6px">

        <!-- SVG robot face -->
        <svg id="face-svg" viewBox="0 0 160 180" xmlns="http://www.w3.org/2000/svg">
          <!-- Head -->
          <rect x="30" y="30" width="100" height="110" rx="22" ry="22"
                fill="#b8a898" />
          <!-- Forehead band -->
          <rect x="30" y="30" width="100" height="28" rx="22" ry="22"
                fill="#8a7a6a" />
          <rect x="30" y="48" width="100" height="10" fill="#8a7a6a" />
          <!-- Left ear speaker -->
          <rect x="14" y="62" width="16" height="40" rx="8" fill="#9a8878" />
          <rect x="17" y="70" width="3" height="6" rx="1" fill="#6a5a4a" opacity="0.6"/>
          <rect x="17" y="80" width="3" height="6" rx="1" fill="#6a5a4a" opacity="0.6"/>
          <rect x="17" y="90" width="3" height="6" rx="1" fill="#6a5a4a" opacity="0.6"/>
          <!-- Right ear speaker -->
          <rect x="130" y="62" width="16" height="40" rx="8" fill="#9a8878" />
          <rect x="140" y="70" width="3" height="6" rx="1" fill="#6a5a4a" opacity="0.6"/>
          <rect x="140" y="80" width="3" height="6" rx="1" fill="#6a5a4a" opacity="0.6"/>
          <rect x="140" y="90" width="3" height="6" rx="1" fill="#6a5a4a" opacity="0.6"/>
          <!-- Eyes - sleeping (closed lines) -->
          <g id="eyes-sleep">
            <path d="M52 80 Q65 76 78 80" stroke="#6a5a4a" stroke-width="3"
                  fill="none" stroke-linecap="round"/>
            <path d="M82 80 Q95 76 108 80" stroke="#6a5a4a" stroke-width="3"
                  fill="none" stroke-linecap="round"/>
          </g>
          <!-- Eyes - awake (circles) -->
          <g id="eyes-awake" style="display:none">
            <circle cx="65" cy="79" r="10" fill="#2a2040"/>
            <circle cx="65" cy="79" r="6"  fill="#4a3a70"/>
            <circle cx="67" cy="77" r="2"  fill="#fff" opacity="0.8"/>
            <circle cx="95" cy="79" r="10" fill="#2a2040"/>
            <circle cx="95" cy="79" r="6"  fill="#4a3a70"/>
            <circle cx="97" cy="77" r="2"  fill="#fff" opacity="0.8"/>
          </g>
          <!-- Eyes - thinking (half open) -->
          <g id="eyes-think" style="display:none">
            <rect x="52" y="76" width="26" height="10" rx="5" fill="#2a2040"/>
            <rect x="82" y="76" width="26" height="10" rx="5" fill="#2a2040"/>
          </g>
          <!-- Mouth (neutral line) -->
          <path d="M62 118 Q80 122 98 118" stroke="#8a7a6a" stroke-width="2.5"
                fill="none" stroke-linecap="round" id="mouth"/>
          <!-- Chin -->
          <rect x="65" y="130" width="30" height="10" rx="5" fill="#9a8878"/>
        </svg>

        <!-- Speaking bars -->
        <div id="speak-bars">
          <div class="bar"></div>
          <div class="bar"></div>
          <div class="bar"></div>
          <div class="bar"></div>
          <div class="bar"></div>
        </div>

      </div>
    </div>
  </div>

  <!-- Status pill -->
  <div id="status-pill">
    <span id="pill-emoji">😴</span>
    <span id="pill-text">Sleeping</span>
  </div>

  <!-- Status bar -->
  <div id="status-bar">
    <span id="status-bar-icon">😴</span>
    <div id="status-bar-text">
      <div id="status-bar-title">Sleeping</div>
      <div id="status-bar-sub">Say "Pinku" to wake</div>
    </div>
    <div id="live-dot"></div>
  </div>

</div>

<!-- Chat view (hidden by default) -->
<div id="chat-view"></div>

<!-- Bottom nav -->
<div id="bottom-nav">
  <button class="nav-btn active" data-tab="home">
    <span class="nav-icon">🏠</span><span class="nav-lbl">HOME</span>
  </button>
  <button class="nav-btn" data-tab="chat">
    <span class="nav-icon">💬</span><span class="nav-lbl">CHAT</span>
  </button>
  <button class="nav-btn" data-tab="log">
    <span class="nav-icon">📋</span><span class="nav-lbl">LOG</span>
  </button>
  <button class="nav-btn" data-tab="music">
    <span class="nav-icon">🎵</span><span class="nav-lbl">MUSIC</span>
  </button>
</div>

<script>
// ── Tab switching ──────────────────────────────────────────
const tabs = { home: document.getElementById('main'), chat: document.getElementById('chat-view') };
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    document.getElementById('main').style.display = (tab==='home') ? 'flex' : 'none';
    const cv = document.getElementById('chat-view');
    cv.style.display = (tab==='chat') ? 'flex' : 'none';
  });
});

// ── State machine ──────────────────────────────────────────
const STATE = {
  sleeping: {
    emoji:'😴', pill:'Sleeping', sub:'Say "Pinku" to wake',
    eyes:'sleep', orbit:'', ring:false, bars:false
  },
  idle: {
    emoji:'😴', pill:'Sleeping', sub:'Say "Pinku" to wake',
    eyes:'sleep', orbit:'', ring:false, bars:false
  },
  awake: {
    emoji:'👂', pill:'Listening…', sub:'I\'m listening — go ahead',
    eyes:'awake', orbit:'awake', ring:true, bars:false
  },
  processing: {
    emoji:'🤔', pill:'Thinking…', sub:'Give me a moment…',
    eyes:'think', orbit:'awake', ring:false, bars:false
  },
  speaking: {
    emoji:'🗣️', pill:'Speaking', sub:'',
    eyes:'awake', orbit:'speak', ring:false, bars:true
  },
  muted: {
    emoji:'🔇', pill:'Muted', sub:'Star laser to unmute',
    eyes:'sleep', orbit:'muted', ring:false, bars:false
  },
};

function applyState(key) {
  const s = STATE[key] || STATE.idle;
  document.getElementById('pill-emoji').textContent      = s.emoji;
  document.getElementById('pill-text').textContent       = s.pill;
  document.getElementById('status-bar-icon').textContent = s.emoji;
  document.getElementById('status-bar-title').textContent = s.pill;
  if (s.sub) document.getElementById('status-bar-sub').textContent = s.sub;

  // Eyes
  document.getElementById('eyes-sleep').style.display = s.eyes==='sleep' ? '' : 'none';
  document.getElementById('eyes-awake').style.display = s.eyes==='awake' ? '' : 'none';
  document.getElementById('eyes-think').style.display = s.eyes==='think' ? '' : 'none';

  // Orbit ring
  const orbit = document.getElementById('orbit');
  orbit.className = s.orbit;

  // Listen pulse ring
  document.getElementById('listen-ring').className = s.ring ? 'active' : '';

  // Speaking bars
  document.querySelectorAll('.bar').forEach(b => {
    b.classList.toggle('active', s.bars);
  });
}

// ── Chat history ───────────────────────────────────────────
const chatView = document.getElementById('chat-view');
function addChat(you, pinku) {
  if (you) {
    const d = document.createElement('div');
    d.className = 'chat-msg chat-you';
    d.textContent = you;
    chatView.appendChild(d);
  }
  if (pinku) {
    const d = document.createElement('div');
    d.className = 'chat-msg chat-pinku';
    d.textContent = pinku;
    chatView.appendChild(d);
  }
  chatView.scrollTop = chatView.scrollHeight;
}

// ── SSE ────────────────────────────────────────────────────
applyState('idle');

const es = new EventSource('/api/stream');
es.onopen = () => {
  document.getElementById('live-dot').className = 'live';
};
es.onmessage = e => {
  const d = JSON.parse(e.data);
  if (d.type === 'status') {
    let key = d.muted ? 'muted' : d.speaking ? 'speaking' : d.state;
    applyState(key);
    if (d.last_transcript && d.last_reply) {
      // Update status bar sub with last transcript
      document.getElementById('status-bar-sub').textContent = d.last_transcript;
      if (!d.speaking && !d.muted && d.state !== 'idle') {
        addChat(d.last_transcript, d.last_reply);
      }
    }
    if (d.model) document.getElementById('brand-name').textContent = 'Pinku';
  }
};
es.onerror = () => {
  document.getElementById('live-dot').className = '';
};

// Blink animation when sleeping
let blinkTimer;
function scheduleBlink() {
  const delay = 4000 + Math.random() * 4000;
  blinkTimer = setTimeout(() => {
    const eyes = document.getElementById('eyes-sleep');
    if (eyes.style.display !== 'none') {
      // brief squint on sleep eyes (already closed, so just a subtle twitch)
    }
    scheduleBlink();
  }, delay);
}
scheduleBlink();
</script>
</body>
</html>"""


_subscribers: list = []
_sub_lock = threading.Lock()


def _push(data: dict):
    msg = f"data: {json.dumps(data)}\n\n"
    with _sub_lock:
        for q in list(_subscribers):
            q.append(msg)


def update_status(**kwargs):
    _status.update(kwargs)
    _push({"type": "status", **_status})


def push_detection(event: dict):
    _push({"type": "detection", "event": event})


@_app.route("/")
def index():
    return render_template_string(HTML)


@_app.route("/api/history")
def history():
    return Response(json.dumps(_logger.recent(200) if _logger else []),
                    mimetype="application/json")


@_app.route("/api/stream")
def stream():
    from collections import deque
    q: deque = deque(maxlen=100)
    with _sub_lock:
        _subscribers.append(q)
    q.append(f"data: {json.dumps({'type':'status',**_status})}\n\n")

    def generate():
        try:
            last_ping = time.time()
            while True:
                if q:
                    yield q.popleft()
                else:
                    if time.time() - last_ping > 15:
                        yield ": ping\n\n"
                        last_ping = time.time()
                    time.sleep(0.05)
        finally:
            with _sub_lock:
                try:
                    _subscribers.remove(q)
                except ValueError:
                    pass

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def start(logger=None, port: int = DASHBOARD_PORT):
    global _logger
    _logger = logger
    _status["model"] = __import__("config").OLLAMA_MODEL
    threading.Thread(
        target=lambda: _app.run(host=DASHBOARD_HOST, port=port,
                                threaded=True, use_reloader=False),
        daemon=True,
    ).start()
    print(f"[Dashboard] http://0.0.0.0:{port}")
