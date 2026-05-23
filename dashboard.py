"""
Web dashboard — SSE-powered, Pinky-style interface.
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

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pinku</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#0a0a0a; color:#e0e0e0; font-family:monospace; padding:20px; }

/* ── Header ── */
.header { display:flex; align-items:center; gap:16px; margin-bottom:20px; }
.header h1 { color:#00ff88; font-size:1.4rem; letter-spacing:2px; }
#status-pill {
  padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:bold;
  letter-spacing:1px; transition: all 0.3s;
}
.pill-idle    { background:#1a1a1a; color:#555; border:1px solid #333; }
.pill-awake   { background:#003320; color:#00ff88; border:1px solid #00aa55; }
.pill-processing { background:#001a33; color:#00aaff; border:1px solid #0055aa; animation: pulse 1s infinite; }
.pill-muted   { background:#330000; color:#ff4444; border:1px solid #aa0000; }
.pill-speaking { background:#1a0033; color:#cc88ff; border:1px solid #6600cc; animation: pulse 0.8s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

#live-dot { position:fixed; top:20px; right:20px; font-size:0.72rem; color:#333; }
#live-dot.live { color:#00ff88; }

/* ── Stats ── */
#stats { display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap; }
.stat {
  background:#111; border:1px solid #222; border-radius:8px;
  padding:12px 18px; min-width:110px; text-align:center;
}
.stat .val { font-size:1.8rem; font-weight:bold; color:#00ccff; }
.stat .lbl { font-size:0.65rem; color:#555; margin-top:4px; letter-spacing:1px; }

/* ── Transcript / reply ── */
.bubble-wrap { display:flex; flex-direction:column; gap:8px; margin-bottom:20px; }
.bubble {
  padding:10px 14px; border-radius:8px; font-size:0.88rem;
  min-height:36px; line-height:1.5; transition:all 0.3s;
}
.bubble-user  { background:#111; border:1px solid #222; color:#aaa; }
.bubble-user span  { color:#555; font-size:0.7rem; margin-right:8px; }
.bubble-pinku { background:#0a1a0a; border:1px solid #1a3a1a; color:#ccc; }
.bubble-pinku span { color:#00aa55; font-size:0.7rem; margin-right:8px; }

/* ── Detection log ── */
.section-title { color:#444; font-size:0.7rem; letter-spacing:2px; margin-bottom:8px; }
.wrap { max-height:calc(100vh - 420px); overflow-y:auto; }
table { width:100%; border-collapse:collapse; font-size:0.8rem; }
th { background:#111; color:#555; text-align:left; padding:6px 10px;
     border-bottom:1px solid #222; position:sticky; top:0; letter-spacing:1px; }
td { padding:5px 10px; border-bottom:1px solid #161616; vertical-align:top; }
tr:hover td { background:#111; }

.tag { display:inline-block; border-radius:3px; padding:1px 6px;
       font-size:0.72rem; margin:1px; }
.tag-obj    { background:#1a1400; color:#ccaa00; border:1px solid #443300; }
.tag-person { background:#001522; color:#0099cc; border:1px solid #003355; }
.tag-gest   { background:#001400; color:#00cc66; border:1px solid #003311; }
</style>
</head>
<body>

<div class="header">
  <h1>PINKU</h1>
  <div id="status-pill" class="pill-idle">IDLE</div>
  <span style="color:#333;font-size:0.75rem" id="model-lbl"></span>
</div>

<div id="stats">
  <div class="stat"><div class="val" id="s-events">0</div><div class="lbl">EVENTS</div></div>
  <div class="stat"><div class="val" id="s-persons">0</div><div class="lbl">PERSONS</div></div>
  <div class="stat"><div class="val" id="s-objects">0</div><div class="lbl">OBJECTS</div></div>
  <div class="stat"><div class="val" id="s-gestures">0</div><div class="lbl">GESTURES</div></div>
</div>

<div class="bubble-wrap">
  <div class="bubble bubble-user"><span>YOU</span><span id="transcript">—</span></div>
  <div class="bubble bubble-pinku"><span>PINKU</span><span id="reply">—</span></div>
</div>

<div class="section-title">DETECTION LOG</div>
<div class="wrap">
<table>
  <thead><tr>
    <th style="width:150px">TIME</th>
    <th style="width:55px">PERSONS</th>
    <th>OBJECTS</th>
    <th>GESTURES</th>
  </tr></thead>
  <tbody id="tbody"></tbody>
</table>
</div>

<div id="live-dot">○ connecting</div>

<script>
let totEvents=0, totPersons=0, totObjects=0, totGestures=0;
const MAX_ROWS = 200;

function setPill(state, muted, speaking) {
  const el = document.getElementById('status-pill');
  if (muted) {
    el.className = 'pill-muted'; el.textContent = 'MUTED'; return;
  }
  if (speaking) {
    el.className = 'pill-speaking'; el.textContent = 'SPEAKING'; return;
  }
  const map = {
    idle:       ['pill-idle',       'IDLE'],
    awake:      ['pill-awake',      'LISTENING'],
    processing: ['pill-processing', 'THINKING'],
    triggered:  ['pill-processing', 'TRIGGERED'],
  };
  const [cls, lbl] = map[state] || ['pill-idle', state.toUpperCase()];
  el.className = cls; el.textContent = lbl;
}

function addRow(e) {
  totEvents++;
  totPersons  += e.persons;
  totObjects  += e.objects.length;
  totGestures += e.gestures.length;
  document.getElementById('s-events').textContent   = totEvents;
  document.getElementById('s-persons').textContent  = totPersons;
  document.getElementById('s-objects').textContent  = totObjects;
  document.getElementById('s-gestures').textContent = totGestures;

  const tbody = document.getElementById('tbody');
  const tr = document.createElement('tr');
  const objTags = e.objects.map(o =>
    `<span class="tag tag-obj">${o.label} ${Math.round(o.conf*100)}%</span>`).join('');
  const gTags = e.gestures.map(g =>
    `<span class="tag tag-gest">${g.hand}: ${g.gesture}</span>`).join('');
  const pTag = e.persons > 0
    ? `<span class="tag tag-person">${e.persons} person${e.persons>1?'s':''}</span>` : '—';
  tr.innerHTML = `<td>${e.timestamp}</td><td>${pTag}</td><td>${objTags||'—'}</td><td>${gTags||'—'}</td>`;
  tbody.insertBefore(tr, tbody.firstChild);
  if (tbody.children.length > MAX_ROWS) tbody.removeChild(tbody.lastChild);
}

fetch('/api/history').then(r=>r.json()).then(ev => ev.slice().reverse().forEach(addRow));

const es = new EventSource('/api/stream');
es.onopen = () => {
  document.getElementById('live-dot').textContent = '● live';
  document.getElementById('live-dot').className = 'live';
};
es.onmessage = e => {
  const d = JSON.parse(e.data);
  if (d.type === 'status') {
    setPill(d.state, d.muted, d.speaking);
    if (d.model) document.getElementById('model-lbl').textContent = d.model;
    if (d.last_transcript) document.getElementById('transcript').textContent = d.last_transcript;
    if (d.last_reply)      document.getElementById('reply').textContent      = d.last_reply;
  } else if (d.type === 'detection') {
    addRow(d.event);
  }
};
es.onerror = () => {
  document.getElementById('live-dot').textContent = '○ reconnecting';
  document.getElementById('live-dot').className = '';
};
</script>
</body>
</html>"""


# ── SSE broadcast ─────────────────────────────────────────────────────────────

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


# ── Routes ────────────────────────────────────────────────────────────────────

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
