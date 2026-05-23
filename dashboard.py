"""
Web dashboard — SSE-powered live detection log + system status.
Accessible at http://<mac-mini-ip>:5100
"""

import json
import time
import threading
from flask import Flask, Response, render_template_string
from config import DASHBOARD_PORT, DASHBOARD_HOST

_app    = Flask(__name__)
_logger = None
_status = {"state": "idle", "muted": False, "model": "", "last_transcript": ""}

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pinku Dashboard</title>
<style>
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:#0d0d0d; color:#e0e0e0; font-family:monospace; padding:16px; }
  h1   { color:#00ff88; margin-bottom:12px; font-size:1.2rem; }

  #info { display:flex; gap:16px; margin-bottom:16px; flex-wrap:wrap; }
  .card { background:#1a1a1a; border:1px solid #333; border-radius:6px;
          padding:10px 16px; }
  .card .val { font-size:1.6rem; font-weight:bold; color:#00ccff; }
  .card .lbl { font-size:0.7rem; color:#888; margin-top:2px; }

  #transcript { background:#111; border:1px solid #222; border-radius:4px;
                padding:8px 12px; margin-bottom:16px; font-size:0.9rem;
                color:#ccc; min-height:32px; }

  #log { width:100%; border-collapse:collapse; font-size:0.82rem; }
  #log th { background:#1a1a1a; color:#888; text-align:left;
             padding:6px 10px; border-bottom:1px solid #333;
             position:sticky; top:0; }
  #log td { padding:5px 10px; border-bottom:1px solid #1e1e1e; vertical-align:top; }
  tr:hover td { background:#151515; }

  .tag { display:inline-block; border-radius:3px; padding:1px 6px;
         font-size:0.75rem; margin:1px; }
  .tag-obj  { background:#2a2200; color:#ffcc00; border:1px solid #554400; }
  .tag-person { background:#002233; color:#00ccff; border:1px solid #005566; }
  .tag-gesture { background:#002200; color:#00ff88; border:1px solid #005500; }

  #status-dot { position:fixed; top:14px; right:16px; font-size:0.75rem; color:#555; }
  #status-dot.live { color:#00ff88; }
  .wrap { max-height:calc(100vh - 200px); overflow-y:auto; }

  .state-awake  { color:#00ff88; }
  .state-muted  { color:#ff4444; }
  .state-idle   { color:#555; }
</style>
</head>
<body>
<h1>Pinku</h1>

<div id="info">
  <div class="card">
    <div class="val" id="s-state">idle</div>
    <div class="lbl">STATE</div>
  </div>
  <div class="card">
    <div class="val" id="s-events">0</div>
    <div class="lbl">DETECTIONS</div>
  </div>
  <div class="card">
    <div class="val" id="s-persons">0</div>
    <div class="lbl">PERSONS</div>
  </div>
  <div class="card">
    <div class="val" id="s-gestures">0</div>
    <div class="lbl">GESTURES</div>
  </div>
  <div class="card">
    <div class="val" id="s-model" style="font-size:0.9rem;padding-top:6px">—</div>
    <div class="lbl">LLM MODEL</div>
  </div>
</div>

<div id="transcript">—</div>

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
<div id="status-dot">connecting…</div>

<script>
let totEvents=0,totPersons=0,totGestures=0;
const MAX_ROWS=200;

function addRow(e){
  totEvents++;
  totPersons+=e.persons;
  totGestures+=e.gestures.length;
  document.getElementById('s-events').textContent=totEvents;
  document.getElementById('s-persons').textContent=totPersons;
  document.getElementById('s-gestures').textContent=totGestures;
  const tbody=document.getElementById('tbody');
  const tr=document.createElement('tr');
  const objTags=e.objects.map(o=>`<span class="tag tag-obj">${o.label} ${Math.round(o.conf*100)}%</span>`).join('');
  const gTags=e.gestures.map(g=>`<span class="tag tag-gesture">${g.hand}: ${g.gesture}</span>`).join('');
  const pTag=e.persons>0?`<span class="tag tag-person">${e.persons} person${e.persons>1?'s':''}</span>`:'—';
  tr.innerHTML=`<td>${e.timestamp}</td><td>${pTag}</td><td>${objTags||'—'}</td><td>${gTags||'—'}</td>`;
  tbody.insertBefore(tr,tbody.firstChild);
  if(tbody.children.length>MAX_ROWS) tbody.removeChild(tbody.lastChild);
}

// History
fetch('/api/history').then(r=>r.json()).then(ev=>ev.slice().reverse().forEach(addRow));

// Status SSE
const ss=new EventSource('/api/status-stream');
ss.onmessage=e=>{
  const d=JSON.parse(e.data);
  if(d.type==='status'){
    const el=document.getElementById('s-state');
    el.textContent=d.muted?'muted':d.state;
    el.className='val state-'+(d.muted?'muted':d.state==='awake'?'awake':'idle');
    if(d.model) document.getElementById('s-model').textContent=d.model;
    if(d.last_transcript) document.getElementById('transcript').textContent=d.last_transcript;
  } else if(d.type==='detection'){
    addRow(d.event);
  }
};
ss.onopen=()=>{
  document.getElementById('status-dot').textContent='● live';
  document.getElementById('status-dot').className='live';
};
ss.onerror=()=>{
  document.getElementById('status-dot').textContent='○ reconnecting…';
  document.getElementById('status-dot').className='';
};
</script>
</body>
</html>"""


def update_status(**kwargs):
    """Call from main to push state to dashboard subscribers."""
    _status.update(kwargs)
    _push_event({"type": "status", **_status})


def push_detection(event: dict):
    _push_event({"type": "detection", "event": event})


_subscribers: list = []
_sub_lock = threading.Lock()


def _push_event(data: dict):
    msg = f"data: {json.dumps(data)}\n\n"
    with _sub_lock:
        for q in list(_subscribers):
            q.append(msg)


@_app.route("/")
def index():
    return render_template_string(HTML)


@_app.route("/api/history")
def history():
    return Response(json.dumps(_logger.recent(200) if _logger else []),
                    mimetype="application/json")


@_app.route("/api/status-stream")
def status_stream():
    from collections import deque
    q: deque[str] = deque(maxlen=50)
    with _sub_lock:
        _subscribers.append(q)
    # Send current state immediately
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
    t = threading.Thread(
        target=lambda: _app.run(host=DASHBOARD_HOST, port=port,
                                threaded=True, use_reloader=False),
        daemon=True,
    )
    t.start()
    print(f"[Dashboard] http://0.0.0.0:{port}")
