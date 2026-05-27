#!/usr/bin/env python3
"""
Pinku Music — standalone AI music generation app.

Separate from the Pinku voice assistant so music generation (MusicGen ~300 MB
model) doesn't interfere with voice recognition or muting.

Usage:
  .venv/bin/python3 music_app.py          # default port 5101
  .venv/bin/python3 music_app.py --port 5200

Open in browser:  http://localhost:5101
Pinku dashboard:  http://localhost:5100
"""

from __future__ import annotations
import argparse
import json
import queue
import threading
import time

from flask import Flask, Response, jsonify, request

import music

_app  = Flask(__name__)
_PORT = 5101

# ── SSE broadcast ─────────────────────────────────────────────────────────────

_clients: list[queue.SimpleQueue] = []
_clients_lock = threading.Lock()


def _broadcast(data: dict):
    msg = "data: " + json.dumps(data) + "\n\n"
    with _clients_lock:
        dead = []
        for q in _clients:
            try:
                q.put_nowait(msg)
            except Exception:
                dead.append(q)
        for q in dead:
            _clients.remove(q)


def _push_state(state: dict | None = None):
    _broadcast({"type": "music_state", **(state or music.get_state())})


def _push_library():
    _broadcast({"type": "music_library_updated"})


# ── Routes ────────────────────────────────────────────────────────────────────

@_app.route("/events")
def sse():
    q: queue.SimpleQueue = queue.SimpleQueue()
    with _clients_lock:
        _clients.append(q)
    init = json.dumps({"type": "music_state", **music.get_state()})

    def stream():
        yield f"data: {init}\n\n"
        while True:
            try:
                msg = q.get(timeout=25)
                yield msg
            except queue.Empty:
                yield ": ping\n\n"   # keep-alive
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                              "X-Accel-Buffering": "no"})


@_app.route("/api/state")
def api_state():
    return jsonify(music.get_state())


@_app.route("/api/start", methods=["POST"])
def api_start():
    data     = request.get_json(silent=True) or {}
    theme    = (data.get("theme") or "chill").strip()
    duration = int(data.get("duration_sec") or 300)
    music.start(theme, duration_sec=duration, on_state_change=_push_state)
    return jsonify({"ok": True, "theme": theme, "duration_sec": duration})


@_app.route("/api/stop", methods=["POST"])
def api_stop():
    music.stop()
    return jsonify({"ok": True})


@_app.route("/api/library")
def api_library():
    return jsonify(music.list_library())


@_app.route("/api/play_library", methods=["POST"])
def api_play_library():
    data    = request.get_json(silent=True) or {}
    item_id = data.get("id", "")
    try:
        music.play_library_item(item_id, on_state_change=_push_state)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@_app.route("/api/library/<item_id>", methods=["DELETE"])
def api_delete_library(item_id):
    ok = music.delete_library_item(item_id)
    if ok:
        _push_library()
    return jsonify({"ok": ok})


# ── UI ────────────────────────────────────────────────────────────────────────

@_app.route("/")
def index():
    return _HTML


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Pinku Music</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:      #0a0a14;
  --surface: #12121e;
  --border:  #24243a;
  --text:    #e8e8f4;
  --muted:   #666683;
  --accent:  #c084fc;
}
body {
  background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', 'Segoe UI', sans-serif;
  min-height: 100dvh; display: flex; flex-direction: column;
}

/* ── Header ── */
header {
  padding: 14px 20px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 12px; flex-shrink: 0;
}
.logo { font-size: 1.5rem; }
.app-title  { font-size: 1.1rem; font-weight: 700; }
.app-sub    { font-size: 0.72rem; color: var(--muted); margin-top: 1px; }
header a { margin-left: auto; font-size: 0.75rem; color: var(--muted);
  text-decoration: none; padding: 4px 10px; border-radius: 8px;
  border: 1px solid var(--border); }
header a:hover { background: rgba(255,255,255,0.06); color: var(--text); }

/* ── Main scroll area ── */
main { flex: 1; overflow-y: auto; padding: 0 0 32px;
  display: flex; flex-direction: column; align-items: center; gap: 0; }

/* ── Status bar ── */
.status-bar {
  width: 100%; max-width: 600px; display: flex; align-items: center; gap: 12px;
  padding: 16px 20px 0; flex-shrink: 0;
}
.status-icon  { font-size: 1.8rem; line-height: 1; flex-shrink: 0; }
.status-text  { flex: 1; min-width: 0; }
.status-label {
  display: block; font-size: 1rem; font-weight: 700; color: var(--accent);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.status-sub {
  display: block; font-size: 0.75rem; color: #64748b; margin-top: 2px;
}
.stop-btn {
  padding: 7px 16px; border-radius: 9px; font-size: 0.85rem; font-weight: 700;
  border: 1px solid rgba(248,113,113,0.4); background: rgba(248,113,113,0.1);
  color: #f87171; cursor: pointer; flex-shrink: 0; display: none;
}
.stop-btn:hover { background: rgba(248,113,113,0.22); }

/* ── Progress bar ── */
.progress-wrap { width: 100%; max-width: 600px; padding: 10px 20px 0; display: none; }
.progress-track {
  height: 5px; border-radius: 3px; background: rgba(255,255,255,0.06);
  overflow: hidden;
}
.progress-fill {
  height: 100%; border-radius: 3px;
  background: linear-gradient(90deg, #c084fc, #818cf8);
  transition: width 0.5s linear; width: 0%;
}

/* ── Preset grid ── */
.presets {
  width: 100%; max-width: 600px; display: grid;
  grid-template-columns: repeat(4, 1fr); gap: 8px; padding: 16px 20px 0;
}
.preset-btn {
  display: flex; flex-direction: column; align-items: center; gap: 4px;
  padding: 11px 4px; border-radius: 13px; font-size: 0.78rem; font-weight: 600;
  border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.04);
  color: #94a3b8; cursor: pointer; transition: all 0.15s; user-select: none;
}
.preset-btn:hover  { background: rgba(192,132,252,0.12); border-color: rgba(192,132,252,0.35); }
.preset-btn.active { background: rgba(192,132,252,0.18); border-color: rgba(192,132,252,0.55); color: var(--accent); }
.preset-emoji { font-size: 1.6rem; line-height: 1; }

/* ── Custom input ── */
.custom-row { width: 100%; max-width: 600px; padding: 10px 20px 0; }
.theme-input {
  width: 100%; padding: 10px 13px; border-radius: 11px;
  background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12);
  color: var(--text); font-size: 0.9rem; outline: none;
}
.theme-input:focus  { border-color: rgba(192,132,252,0.45); }
.theme-input::placeholder { color: #475569; }

/* ── Duration row ── */
.dur-row {
  width: 100%; max-width: 600px; padding: 10px 20px 0;
  display: flex; align-items: center; gap: 10px;
}
.dur-label { font-size: 0.75rem; color: #64748b; flex-shrink: 0; }
.dur-btns  { display: flex; gap: 6px; flex-wrap: wrap; }
.dur-btn {
  padding: 5px 12px; border-radius: 8px; font-size: 0.78rem; cursor: pointer;
  border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.04);
  color: #94a3b8; transition: all 0.15s;
}
.dur-btn:hover  { background: rgba(255,255,255,0.1); color: var(--text); }
.dur-btn.active { background: rgba(251,191,36,0.15); border-color: rgba(251,191,36,0.45); color: #fbbf24; }

/* ── Generate button ── */
.play-row { width: 100%; max-width: 600px; padding: 14px 20px 0; }
.play-btn {
  width: 100%; padding: 14px; border-radius: 13px; font-size: 1.05rem; font-weight: 700;
  border: 1px solid rgba(192,132,252,0.45); background: rgba(192,132,252,0.14);
  color: var(--accent); cursor: pointer; transition: all 0.15s; letter-spacing: 0.03em;
}
.play-btn:hover    { background: rgba(192,132,252,0.26); }
.play-btn:disabled { opacity: 0.42; cursor: default; }
.play-btn.playing  { background: rgba(74,222,128,0.14); border-color: rgba(74,222,128,0.45); color: #4ade80; }

/* ── Library ── */
.library-section { width: 100%; max-width: 600px; padding: 20px 20px 0; display: none; }
.library-title {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: #64748b; padding-bottom: 8px;
}
.lib-item {
  display: flex; align-items: center; gap: 8px;
  padding: 9px 11px; border-radius: 11px; margin-bottom: 7px;
  background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
  transition: border-color 0.15s;
}
.lib-item:hover   { border-color: rgba(192,132,252,0.3); }
.lib-item.playing { border-color: rgba(74,222,128,0.45); background: rgba(74,222,128,0.06); }
.lib-info { flex: 1; min-width: 0; }
.lib-theme { font-size: 0.9rem; font-weight: 600; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.lib-meta  { font-size: 0.68rem; color: #64748b; margin-top: 1px; }
.lib-play  {
  padding: 5px 11px; border-radius: 8px; font-size: 0.8rem; cursor: pointer; flex-shrink: 0;
  border: 1px solid rgba(192,132,252,0.4); background: rgba(192,132,252,0.1); color: var(--accent);
}
.lib-play:hover { background: rgba(192,132,252,0.22); }
.lib-del {
  padding: 5px 9px; border-radius: 8px; font-size: 0.8rem; cursor: pointer; flex-shrink: 0;
  border: 1px solid rgba(248,113,113,0.3); background: transparent; color: #f87171;
}
.lib-del:hover { background: rgba(248,113,113,0.12); }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
.status-label.generating { animation: pulse 1.4s ease-in-out infinite; }
</style>
</head>
<body>

<header>
  <span class="logo">🎵</span>
  <div>
    <div class="app-title">Pinku Music</div>
    <div class="app-sub">AI music generator</div>
  </div>
  <a href="http://localhost:5100" target="_blank">← Pinku</a>
</header>

<main>

  <!-- Status -->
  <div class="status-bar">
    <span class="status-icon" id="s-icon">🎵</span>
    <div class="status-text">
      <span class="status-label" id="s-label">Ready to play</span>
      <span class="status-sub"   id="s-sub">Choose a theme below</span>
    </div>
    <button class="stop-btn" id="stop-btn" onclick="doStop()">■ Stop</button>
  </div>

  <!-- Progress -->
  <div class="progress-wrap" id="progress-wrap">
    <div class="progress-track"><div class="progress-fill" id="progress-fill"></div></div>
  </div>

  <!-- Preset grid -->
  <div class="presets" id="presets"></div>

  <!-- Custom theme -->
  <div class="custom-row">
    <input class="theme-input" id="theme-input"
           placeholder="Custom theme… e.g. rainy Bollywood evening" maxlength="80">
  </div>

  <!-- Duration -->
  <div class="dur-row">
    <span class="dur-label">Duration</span>
    <div class="dur-btns">
      <button class="dur-btn"        data-sec="60">1 min</button>
      <button class="dur-btn active" data-sec="300">5 min</button>
      <button class="dur-btn"        data-sec="600">10 min</button>
      <button class="dur-btn"        data-sec="1800">30 min</button>
      <button class="dur-btn"        data-sec="0">∞</button>
    </div>
  </div>

  <!-- Play button -->
  <div class="play-row">
    <button class="play-btn" id="play-btn" onclick="doPlay()">▶ Generate &amp; Play</button>
  </div>

  <!-- Library -->
  <div class="library-section" id="library-section">
    <div class="library-title">📂 Saved sessions</div>
    <div id="library-list"></div>
  </div>

</main>

<script>
// ── Preset definitions ────────────────────────────────────────────────────────
const PRESETS = [
  {key:'chill',      emoji:'☁️'},  {key:'focus',      emoji:'🎯'},
  {key:'bollywood',  emoji:'🪘'},  {key:'jazz',       emoji:'🎷'},
  {key:'classical',  emoji:'🎻'},  {key:'ambient',    emoji:'🌊'},
  {key:'party',      emoji:'🎉'},  {key:'meditation', emoji:'🧘'},
  {key:'sleep',      emoji:'🌙'},  {key:'bhajan',     emoji:'🪔'},
  {key:'folk',       emoji:'🪈'},  {key:'rock',       emoji:'🎸'},
];

let _theme   = 'chill';
let _durSec  = 300;
let _library = [];
let _lastState = null;

// ── Build preset buttons ──────────────────────────────────────────────────────
const grid = document.getElementById('presets');
grid.innerHTML = PRESETS.map(p =>
  `<button class="preset-btn${p.key===_theme?' active':''}" id="pb-${p.key}"
           onclick="selectPreset('${p.key}')">
     <span class="preset-emoji">${p.emoji}</span>${p.key}
   </button>`
).join('');

// ── Duration buttons ──────────────────────────────────────────────────────────
document.querySelectorAll('.dur-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.dur-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    _durSec = parseInt(btn.dataset.sec) || 0;
  });
});

// ── Custom theme input ────────────────────────────────────────────────────────
document.getElementById('theme-input').addEventListener('input', function() {
  if (this.value.trim()) {
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    _theme = this.value.trim();
  }
});

function selectPreset(key) {
  _theme = key;
  document.querySelectorAll('.preset-btn').forEach(b =>
    b.classList.toggle('active', b.id === 'pb-' + key)
  );
  document.getElementById('theme-input').value = '';
}

// ── API calls ─────────────────────────────────────────────────────────────────
function doPlay() {
  const inp = document.getElementById('theme-input');
  const theme = (inp && inp.value.trim()) ? inp.value.trim() : _theme;
  fetch('/api/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({theme, duration_sec: _durSec}),
  }).catch(console.error);
}

function doStop() {
  fetch('/api/stop', {method: 'POST'}).catch(console.error);
}

function playLibrary(id) {
  fetch('/api/play_library', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id}),
  }).then(r => r.json()).then(d => {
    if (!d.ok) alert('Error: ' + (d.error || '?'));
  }).catch(console.error);
}

function deleteLibrary(id) {
  if (!confirm('Delete this saved track?')) return;
  fetch('/api/library/' + id, {method: 'DELETE'})
    .then(() => loadLibrary())
    .catch(console.error);
}

function loadLibrary() {
  fetch('/api/library').then(r => r.json()).then(items => {
    _library = items || [];
    renderLibrary();
  }).catch(() => {});
}

// ── State rendering ───────────────────────────────────────────────────────────
function applyState(s) {
  if (!s) return;
  _lastState = s;
  const state    = s.state || 'idle';
  const icon     = document.getElementById('s-icon');
  const label    = document.getElementById('s-label');
  const sub      = document.getElementById('s-sub');
  const stopBtn  = document.getElementById('stop-btn');
  const playBtn  = document.getElementById('play-btn');
  const progWrap = document.getElementById('progress-wrap');
  const progFill = document.getElementById('progress-fill');

  const active = ['loading','generating','playing'].includes(state);
  stopBtn.style.display  = active ? '' : 'none';
  progWrap.style.display = active ? '' : 'none';
  playBtn.disabled       = active;
  playBtn.textContent    = active ? '♪ Playing…' : '▶ Generate & Play';
  playBtn.classList.toggle('playing', state === 'playing');
  label.classList.toggle('generating', state === 'generating' || state === 'loading');

  if (state === 'loading') {
    icon.textContent  = '⏳';
    label.textContent = 'Loading model…';
    sub.textContent   = 'First run downloads MusicGen (~300 MB)';
  } else if (state === 'generating') {
    icon.textContent  = '🎼';
    label.textContent = `Generating "${s.theme || ''}"…`;
    sub.textContent   = `Chunk ${(s.chunk||0)+1} of ${s.chunks_total||'?'} · please wait`;
  } else if (state === 'playing') {
    icon.textContent  = '♪';
    label.textContent = `♪ ${s.theme || ''}`;
    label.classList.remove('generating');
    const el = s.elapsed || 0, tot = s.total || 0;
    sub.textContent = tot > 0
      ? `${fmt(el)} / ${fmt(tot)} · chunk ${s.chunk||1}`
      : `${fmt(el)} elapsed`;
    progFill.style.width = (tot > 0 ? Math.min(100,(el/tot*100)).toFixed(1) : 50) + '%';
  } else if (state === 'error') {
    icon.textContent  = '⚠️';
    label.textContent = 'Error';
    sub.textContent   = s.error || 'Check terminal for details';
  } else {
    icon.textContent  = '🎵';
    label.textContent = 'Ready to play';
    sub.textContent   = 'Choose a theme below';
    progFill.style.width = '0%';
  }
  renderLibrary();
}

function fmt(sec) {
  const m = Math.floor(sec/60), s = sec%60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderLibrary() {
  const section = document.getElementById('library-section');
  const list    = document.getElementById('library-list');
  if (!_library.length) { section.style.display = 'none'; return; }
  section.style.display = '';
  const curTheme = _lastState && _lastState.state === 'playing' ? _lastState.theme : null;
  list.innerHTML = _library.map(item => {
    const mins = item.duration_sec > 0 ? Math.round(item.duration_sec/60)+' min' : '';
    const size = item.size_kb > 1024 ? (item.size_kb/1024).toFixed(1)+'MB' : item.size_kb+'KB';
    const active = curTheme === item.theme;
    return `<div class="lib-item${active?' playing':''}" id="li-${esc(item.id)}">
      <div class="lib-info">
        <div class="lib-theme">${esc(item.theme)}</div>
        <div class="lib-meta">${esc(item.created_at)}${mins?' · '+mins:''} · ${size}</div>
      </div>
      <button class="lib-play" onclick="playLibrary('${esc(item.id)}')">▶ Play</button>
      <button class="lib-del"  onclick="deleteLibrary('${esc(item.id)}')">🗑</button>
    </div>`;
  }).join('');
}

// ── SSE ───────────────────────────────────────────────────────────────────────
function connectSSE() {
  const es = new EventSource('/events');
  es.onmessage = e => {
    try {
      const data = JSON.parse(e.data);
      if (data.type === 'music_state')       applyState(data);
      if (data.type === 'music_library_updated') loadLibrary();
    } catch(_) {}
  };
  es.onerror = () => setTimeout(connectSSE, 3000);
}

// ── Init ──────────────────────────────────────────────────────────────────────
connectSSE();
loadLibrary();
</script>
</body>
</html>
"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Pinku Music — standalone AI music app")
    ap.add_argument("--port", type=int, default=_PORT)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    print("=" * 50)
    print("  Pinku Music")
    print(f"  Open: http://localhost:{args.port}")
    print(f"  Library: {music.LIBRARY_DIR}")
    print("=" * 50)

    # Warm up MusicGen model in background so first generate is faster
    music.preload()

    _app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
