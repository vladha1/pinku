"""
Web dashboard — matches Pinky's visual style.
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
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d0d; color: #e0e0e0; font-family: monospace; padding: 16px; }
  h1 { color: #00ff88; margin-bottom: 12px; font-size: 1.2rem; }

  #stats { display: flex; gap: 24px; margin-bottom: 16px; flex-wrap: wrap; }
  .stat { background: #1a1a1a; border: 1px solid #333; border-radius: 6px;
          padding: 10px 18px; text-align: center; }
  .stat .val { font-size: 2rem; font-weight: bold; color: #00ccff; }
  .stat .lbl { font-size: 0.7rem; color: #888; margin-top: 2px; }

  #transcript-box { background: #111; border: 1px solid #222; border-radius: 6px;
                    padding: 8px 12px; margin-bottom: 16px; min-height: 60px; }
  .tx-row { font-size: 0.82rem; margin: 3px 0; line-height: 1.5; }
  .tx-you   { color: #888; }
  .tx-you   span { color: #555; margin-right: 6px; }
  .tx-pinku { color: #ccc; }
  .tx-pinku span { color: #00ff88; margin-right: 6px; }

  #log { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  #log th { background: #1a1a1a; color: #888; text-align: left;
             padding: 6px 10px; border-bottom: 1px solid #333;
             position: sticky; top: 0; }
  #log td { padding: 5px 10px; border-bottom: 1px solid #1e1e1e; vertical-align: top; }
  tr:hover td { background: #151515; }

  .tag { display: inline-block; border-radius: 3px; padding: 1px 6px;
         font-size: 0.75rem; margin: 1px; }
  .tag-obj    { background: #2a2200; color: #ffcc00; border: 1px solid #554400; }
  .tag-person { background: #002233; color: #00ccff; border: 1px solid #005566; }
  .tag-gest   { background: #002200; color: #00ff88; border: 1px solid #005500; }
  .tag-state  { background: #1a1a1a; color: #888;    border: 1px solid #333; }
  .tag-muted  { background: #330000; color: #ff4444; border: 1px solid #660000; }
  .tag-awake  { background: #002200; color: #00ff88; border: 1px solid #004400; }
  .tag-speak  { background: #1a0033; color: #cc88ff; border: 1px solid #440066; }

  #status { position: fixed; top: 12px; right: 16px; font-size: 0.75rem; color: #555; }
  #status.live { color: #00ff88; }
  .state-tag { position: fixed; top: 10px; right: 80px; }
  .wrap { max-height: calc(100vh - 220px); overflow-y: auto; }
</style>
</head>
<body>
<h1>Pinku &nbsp;
  <span style="font-size:0.75rem;color:#555" id="model-lbl"></span>
</h1>

<div id="stats">
  <div class="stat"><div class="val" id="s-events">0</div><div class="lbl">EVENTS</div></div>
  <div class="stat"><div class="val" id="s-persons">0</div><div class="lbl">PERSONS SEEN</div></div>
  <div class="stat"><div class="val" id="s-objects">0</div><div class="lbl">OBJECTS SEEN</div></div>
  <div class="stat"><div class="val" id="s-gestures">0</div><div class="lbl">GESTURES</div></div>
</div>

<div id="transcript-box">
  <div class="tx-row tx-you"><span>YOU</span><span id="tx-text">—</span></div>
  <div class="tx-row tx-pinku"><span>PINKU</span><span id="reply-text">—</span></div>
</div>

<div class="wrap">
<table id="log">
  <thead><tr>
    <th style="width:160px">Time</th>
    <th style="width:60px">Persons</th>
    <th>Objects</th>
    <th>Gestures</th>
  </tr></thead>
  <tbody id="tbody"></tbody>
</table>
</div>

<div class="state-tag"><span id="state-tag" class="tag tag-state">IDLE</span></div>
<div id="status">connecting…</div>

<script>
let totEvents=0, totPersons=0, totObjects=0, totGestures=0;
const MAX_ROWS = 200;

function updateState(d) {
  const el = document.getElementById('state-tag');
  if (d.muted) {
    el.className = 'tag tag-muted'; el.textContent = 'MUTED'; return;
  }
  if (d.speaking) {
    el.className = 'tag tag-speak'; el.textContent = '● SPEAKING'; return;
  }
  const map = {
    idle:       ['tag-state', "SAY 'PINKU'"],
    awake:      ['tag-awake', '● LISTENING'],
    processing: ['tag-awake', '● THINKING'],
    triggered:  ['tag-awake', '● TRIGGERED'],
  };
  const [cls, lbl] = map[d.state] || ['tag-state', d.state.toUpperCase()];
  el.className = 'tag ' + cls;
  el.textContent = lbl;
}

function addRow(e) {
  totEvents++; totPersons += e.persons;
  totObjects += e.objects.length; totGestures += e.gestures.length;
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
  document.getElementById('status').textContent = '● live';
  document.getElementById('status').className = 'live';
};
es.onmessage = e => {
  const d = JSON.parse(e.data);
  if (d.type === 'status') {
    updateState(d);
    if (d.model) document.getElementById('model-lbl').textContent = d.model;
    if (d.last_transcript) document.getElementById('tx-text').textContent = d.last_transcript;
    if (d.last_reply)      document.getElementById('reply-text').textContent = d.last_reply;
  } else if (d.type === 'detection') {
    addRow(d.event);
  }
};
es.onerror = () => {
  document.getElementById('status').textContent = '○ reconnecting…';
  document.getElementById('status').className = '';
};
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
