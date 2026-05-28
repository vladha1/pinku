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
import time

import logging
logging.getLogger("werkzeug").setLevel(logging.ERROR)   # suppress per-request access logs

from flask import Flask, jsonify, request

import music

_app  = Flask(__name__)
_PORT = 5101


def _push_state(state: dict | None = None):
    pass   # polling — no push needed; kept so music.py callback wiring still works


def _push_library():
    pass   # polling — clients refresh on their own schedule


# ── Routes ────────────────────────────────────────────────────────────────────

@_app.route("/api/state")
def api_state():
    return jsonify(music.get_state())


@_app.route("/api/start", methods=["POST"])
def api_start():
    data     = request.get_json(silent=True) or {}
    theme    = (data.get("theme") or "chill").strip()
    duration = int(data.get("duration_sec") or 300)
    music.start(theme, duration_sec=duration,
                on_state_change=_push_state,
                on_library_change=_push_library)
    return jsonify({"ok": True, "theme": theme, "duration_sec": duration})


@_app.route("/api/stop", methods=["POST"])
def api_stop():
    music.stop()
    return jsonify({"ok": True})


@_app.route("/api/pause", methods=["POST"])
def api_pause():
    music.pause_toggle()
    return jsonify({"ok": True, "paused": music.get_state().get("paused", False)})


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


@_app.route("/api/library/<item_id>", methods=["PATCH"])
def api_rename_library(item_id):
    data      = request.get_json(silent=True) or {}
    new_theme = (data.get("theme") or "").strip()
    if not new_theme:
        return jsonify({"ok": False, "error": "theme required"}), 400
    ok = music.rename_library_item(item_id, new_theme)
    if ok:
        _push_library()
    return jsonify({"ok": ok})


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

/* ── Player controls (shown when playing/paused) ── */
.player-controls {
  width: 100%; max-width: 600px; padding: 10px 20px 0;
  display: none; gap: 8px; flex-wrap: wrap;
}
.player-controls.visible { display: flex; }
.ctrl-btn {
  padding: 8px 18px; border-radius: 9px; font-size: 0.9rem; font-weight: 700;
  cursor: pointer; flex-shrink: 0; transition: all 0.15s;
}
.ctrl-pause {
  border: 1px solid rgba(251,191,36,0.45); background: rgba(251,191,36,0.12);
  color: #fbbf24;
}
.ctrl-pause:hover  { background: rgba(251,191,36,0.24); }
.ctrl-pause.paused {
  border-color: rgba(74,222,128,0.5); background: rgba(74,222,128,0.14);
  color: #4ade80;
}
.ctrl-stop {
  border: 1px solid rgba(248,113,113,0.4); background: rgba(248,113,113,0.1);
  color: #f87171;
}
.ctrl-stop:hover { background: rgba(248,113,113,0.22); }

/* ── Progress bar ── */
.progress-wrap { width: 100%; max-width: 600px; padding: 10px 20px 0; display: none; }
.progress-track {
  height: 5px; border-radius: 3px; background: rgba(255,255,255,0.06); overflow: hidden;
}
.progress-fill {
  height: 100%; border-radius: 3px;
  background: linear-gradient(90deg, #c084fc, #818cf8);
  transition: width 0.8s linear; width: 0%;
}
/* Chunk generation sub-bar (amber, shown only while generating) */
.chunk-track {
  height: 4px; border-radius: 3px; background: rgba(255,255,255,0.04);
  overflow: hidden; margin-top: 5px;
}
.chunk-fill {
  height: 100%; border-radius: 3px;
  background: linear-gradient(90deg, #fbbf24, #f59e0b);
  transition: width 1s linear; width: 0%;
}
.progress-meta {
  display: flex; justify-content: space-between;
  font-size: 0.7rem; color: #64748b; margin-top: 5px;
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
.library-section { width: 100%; max-width: 600px; padding: 20px 20px 0; }
.library-header {
  display: flex; align-items: center; justify-content: space-between;
  padding-bottom: 10px;
}
.library-title {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b;
}
.library-count { font-size: 0.68rem; color: #475569; }
.library-empty {
  padding: 20px; text-align: center; color: #475569; font-size: 0.82rem;
  border: 1px dashed rgba(255,255,255,0.08); border-radius: 11px;
}

/* Card layout: info row on top, buttons row below */
.lib-item {
  display: flex; flex-direction: column; gap: 8px;
  padding: 11px 13px; border-radius: 11px; margin-bottom: 8px;
  background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
  transition: border-color 0.15s;
}
.lib-item:hover   { border-color: rgba(192,132,252,0.3); }
.lib-item.playing { border-color: rgba(74,222,128,0.45); background: rgba(74,222,128,0.06); }
.lib-info  { min-width: 0; }
.lib-theme { font-size: 0.92rem; font-weight: 600;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lib-meta  { font-size: 0.7rem; color: #64748b; margin-top: 2px; }

/* Button row — always full width, never overflows */
.lib-actions { display: flex; gap: 7px; }
.lib-actions button {
  flex: 1; padding: 7px 4px; border-radius: 8px; font-size: 0.8rem; font-weight: 600;
  cursor: pointer; border-width: 1px; border-style: solid; transition: background 0.15s;
  white-space: nowrap;
}
.lib-play   { border-color: rgba(192,132,252,0.4); background: rgba(192,132,252,0.1); color: var(--accent); }
.lib-play:hover   { background: rgba(192,132,252,0.22); }
.lib-play.stop    { border-color: rgba(74,222,128,0.45); background: rgba(74,222,128,0.1); color: #4ade80; }
.lib-play.stop:hover { background: rgba(74,222,128,0.22); }
.lib-rename { border-color: rgba(148,163,184,0.25); background: transparent; color: #94a3b8; }
.lib-rename:hover { background: rgba(148,163,184,0.1); color: var(--text); }
.lib-del    { border-color: rgba(248,113,113,0.3); background: transparent; color: #f87171; }
.lib-del:hover    { background: rgba(248,113,113,0.12); }

/* inline rename edit row */
.lib-edit-wrap  { display: flex; gap: 7px; align-items: center; }
.lib-edit-input {
  flex: 1; padding: 6px 10px; border-radius: 8px; font-size: 0.9rem; font-weight: 600;
  background: rgba(255,255,255,0.07); border: 1px solid rgba(192,132,252,0.5);
  color: var(--text); outline: none;
}
.lib-edit-input:focus { border-color: var(--accent); }
.lib-edit-ok {
  padding: 6px 13px; border-radius: 8px; font-size: 0.82rem; font-weight: 600;
  cursor: pointer; flex-shrink: 0;
  border: 1px solid rgba(74,222,128,0.45); background: rgba(74,222,128,0.1); color: #4ade80;
}
.lib-edit-ok:hover { background: rgba(74,222,128,0.22); }
.lib-edit-cancel {
  padding: 6px 11px; border-radius: 8px; font-size: 0.82rem; font-weight: 600;
  cursor: pointer; flex-shrink: 0;
  border: 1px solid rgba(248,113,113,0.3); background: transparent; color: #f87171;
}
.lib-edit-cancel:hover { background: rgba(248,113,113,0.12); }

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
  </div>

  <!-- Player controls — shown only while generating / playing / paused -->
  <div class="player-controls" id="player-controls">
    <button class="ctrl-btn ctrl-pause" id="pause-btn" onclick="doPause()">⏸ Pause</button>
    <button class="ctrl-btn ctrl-stop"  id="stop-btn"  onclick="doStop()">■ Stop</button>
  </div>

  <!-- Progress -->
  <div class="progress-wrap" id="progress-wrap">
    <div class="progress-track"><div class="progress-fill" id="progress-fill"></div></div>
    <div class="chunk-track"  id="chunk-track" style="display:none">
      <div class="chunk-fill" id="chunk-fill"></div>
    </div>
    <div class="progress-meta" id="progress-meta" style="display:none">
      <span id="meta-left"></span>
      <span id="meta-right"></span>
    </div>
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
    <div class="library-header">
      <span class="library-title">📂 Saved tracks</span>
      <span class="library-count" id="library-count"></span>
    </div>
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

let _theme      = 'chill';
let _durSec     = 300;
let _library    = [];
let _lastState  = null;
let _renamingId = null;   // set while an inline rename input is visible

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

function doPause() {
  fetch('/api/pause', {method: 'POST'}).catch(console.error);
}

function playLibrary(id) {
  // If this item is already playing, stop it
  if (_lastState && _lastState.playing_id === id) {
    doStop();
    return;
  }
  fetch('/api/play_library', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id}),
  }).then(r => r.json()).then(d => {
    if (!d.ok) alert('Error: ' + (d.error || '?'));
  }).catch(console.error);
}

function deleteLibrary(id) {
  // Optimistic remove — update local state immediately so the 1s poller
  // doesn't re-show the item while the DELETE request is in-flight.
  _library = _library.filter(x => x.id !== id);
  renderLibrary();
  fetch('/api/library/' + id, {method: 'DELETE'})
    .then(r => r.json())
    .then(d => {
      if (!d.ok) console.warn('Delete returned ok=false for', id);
      loadLibrary();   // confirm server state
    })
    .catch(err => { console.error(err); loadLibrary(); });
}

function loadLibrary() {
  _lastLibraryTs = Date.now();
  fetch('/api/library').then(r => r.json()).then(items => {
    _library = items || [];
    renderLibrary();
  }).catch(() => {});
}

// ── State rendering ───────────────────────────────────────────────────────────
function applyState(s) {
  if (!s) return;
  _lastState = s;
  const state    = s.state  || 'idle';
  const paused   = !!s.paused;
  const icon     = document.getElementById('s-icon');
  const label    = document.getElementById('s-label');
  const sub      = document.getElementById('s-sub');
  const controls = document.getElementById('player-controls');
  const pauseBtn = document.getElementById('pause-btn');
  const playBtn  = document.getElementById('play-btn');
  const progWrap = document.getElementById('progress-wrap');
  const progFill = document.getElementById('progress-fill');

  const active     = ['loading','generating','playing'].includes(state);
  const canPause   = state === 'playing';
  const chunkTrack = document.getElementById('chunk-track');
  const chunkFill  = document.getElementById('chunk-fill');
  const metaRow    = document.getElementById('progress-meta');
  const metaLeft   = document.getElementById('meta-left');
  const metaRight  = document.getElementById('meta-right');

  // Player controls visibility
  controls.classList.toggle('visible', active);
  pauseBtn.style.display = canPause ? '' : 'none';
  pauseBtn.classList.toggle('paused', paused);
  pauseBtn.textContent = paused ? '▶ Resume' : '⏸ Pause';

  progWrap.style.display = active ? '' : 'none';
  playBtn.disabled       = active;
  playBtn.textContent    = active ? '♪ Playing…' : '▶ Generate & Play';
  playBtn.classList.toggle('playing', state === 'playing' && !paused);
  label.classList.toggle('generating', state === 'generating' || state === 'loading');

  if (state === 'loading') {
    icon.textContent  = '⏳';
    label.textContent = s.cached ? 'Loading model from cache…' : 'Downloading model (~300 MB)…';
    sub.textContent   = s.cached ? 'Usually takes ~10s on M4' : 'First run only — saved to ~/pinku/models/';
    chunkTrack.style.display = 'none';
    metaRow.style.display    = 'none';
    progFill.style.width = '0%';
  } else if (state === 'generating') {
    const chunkNum   = (s.chunk || 0) + 1;
    const chunkTot   = s.chunks_total || '?';
    const chunkSec   = s.chunk_sec   || 30;
    const genStart   = s.gen_start   || 0;
    const elapsed    = genStart > 0 ? Math.round(Date.now()/1000 - genStart) : 0;
    const pct        = chunkSec > 0 ? Math.min(98, Math.round(elapsed / chunkSec * 100)) : 0;
    const remaining  = chunkSec > 0 ? Math.max(0, chunkSec - elapsed) : 0;

    icon.textContent  = '🎼';
    label.textContent = `Generating "${s.theme || ''}"…`;
    sub.textContent   = `Chunk ${chunkNum}${chunkTot !== '?' ? ' of ' + chunkTot : ''} · ~${remaining}s left`;

    // Amber chunk progress bar
    chunkTrack.style.display = '';
    chunkFill.style.width    = pct + '%';

    // Overall playlist progress (chunks done / total)
    const overallPct = chunkTot !== '?' ? Math.round(((chunkNum - 1) / chunkTot) * 100) : 0;
    progFill.style.width = overallPct + '%';

    metaRow.style.display = '';
    metaLeft.textContent  = `Chunk ${chunkNum}${chunkTot !== '?' ? '/' + chunkTot : ''} — ${elapsed}s elapsed`;
    metaRight.textContent = `~${Math.round(remaining)}s remaining`;
  } else if (state === 'playing') {
    icon.textContent  = paused ? '⏸' : '♪';
    label.textContent = paused ? `⏸ ${s.theme || ''}` : `♪ ${s.theme || ''}`;
    label.classList.remove('generating');
    const el = s.elapsed || 0, tot = s.total || 0;
    sub.textContent = paused
      ? 'Paused · tap Resume to continue'
      : (tot > 0
          ? `${fmt(el)} / ${fmt(tot)} · chunk ${s.chunk||1}`
          : `${fmt(el)} elapsed`);
    progFill.style.width = (tot > 0 ? Math.min(100,(el/tot*100)).toFixed(1) : 50) + '%';
    chunkTrack.style.display = 'none';
    metaRow.style.display    = 'none';
  } else if (state === 'error') {
    icon.textContent  = '⚠️';
    label.textContent = 'Error — tap Generate to retry';
    label.classList.remove('generating');
    sub.textContent   = s.error || 'Check terminal for details';
    playBtn.disabled  = false;
    playBtn.textContent = '↺ Retry';
    playBtn.classList.remove('playing');
  } else {
    icon.textContent  = '🎵';
    label.textContent = 'Ready to play';
    sub.textContent   = 'Choose a theme below';
    progFill.style.width = '0%';
    chunkTrack.style.display = 'none';
    metaRow.style.display    = 'none';
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
  // Don't rebuild the list while an inline rename input is open —
  // the 1s poller would destroy the text field the user is typing in.
  if (_renamingId) return;

  const list      = document.getElementById('library-list');
  const countEl   = document.getElementById('library-count');
  const playingId = _lastState ? (_lastState.playing_id || '') : '';

  if (!_library.length) {
    countEl.textContent = '';
    list.innerHTML = `<div class="library-empty">
      No saved tracks yet — generate some music and it will appear here automatically.
    </div>`;
    return;
  }

  countEl.textContent = _library.length + ' track' + (_library.length !== 1 ? 's' : '');

  list.innerHTML = _library.map(item => {
    const mins   = item.duration_sec > 0 ? Math.round(item.duration_sec / 60) + ' min' : '';
    const size   = item.size_kb > 1024
                   ? (item.size_kb / 1024).toFixed(1) + ' MB'
                   : item.size_kb + ' KB';
    const active = playingId && playingId === item.id;
    const id     = esc(item.id);
    return `<div class="lib-item${active ? ' playing' : ''}" id="li-${id}">
      <div class="lib-info" id="li-info-${id}">
        <div class="lib-theme">${esc(item.theme)}</div>
        <div class="lib-meta">${esc(item.created_at)}${mins ? ' · ' + mins : ''} · ${size}</div>
      </div>
      <div class="lib-actions">
        <button class="lib-play ${active ? 'stop' : ''}"
                onclick="playLibrary('${id}')">${active ? '■ Stop' : '▶ Play'}</button>
        <button class="lib-rename"
                data-id="${id}" data-theme="${esc(item.theme)}"
                onclick="startRename(this.dataset.id, this.dataset.theme)">✏ Rename</button>
        <button class="lib-del"
                onclick="deleteLibrary('${id}')">🗑 Delete</button>
      </div>
    </div>`;
  }).join('');
}

// ── Inline rename ─────────────────────────────────────────────────────────────
function cancelRename() {
  _renamingId = null;
  loadLibrary();
}

function startRename(id, currentTheme) {
  const infoEl = document.getElementById('li-info-' + id);
  if (!infoEl) return;
  _renamingId = id;   // block renderLibrary() while input is open
  // Replace the info block content with an inline edit row
  infoEl.innerHTML = `
    <div class="lib-edit-wrap">
      <input class="lib-edit-input" id="re-inp-${id}"
             value="${currentTheme.replace(/"/g,'&quot;')}" maxlength="80">
      <button class="lib-edit-ok"     onclick="confirmRename('${id}')">✓ Save</button>
      <button class="lib-edit-cancel" onclick="cancelRename()">✗</button>
    </div>`;
  const inp = document.getElementById('re-inp-' + id);
  if (!inp) return;
  inp.focus(); inp.select();
  inp.addEventListener('keydown', e => {
    if (e.key === 'Enter')  { e.preventDefault(); confirmRename(id); }
    if (e.key === 'Escape') cancelRename();
  });
}

function confirmRename(id) {
  const inp = document.getElementById('re-inp-' + id);
  if (!inp) return;
  const newTheme = inp.value.trim();
  if (!newTheme) { inp.focus(); return; }
  inp.disabled = true;
  fetch('/api/library/' + id, {
    method:  'PATCH',
    headers: {'Content-Type': 'application/json'},
    body:    JSON.stringify({theme: newTheme}),
  }).then(r => r.json()).then(d => {
    _renamingId = null;   // allow re-render before loadLibrary rebuilds
    if (d.ok) loadLibrary();
    else { inp.disabled = false; inp.focus(); alert('Rename failed'); }
  }).catch(() => { _renamingId = null; inp.disabled = false; loadLibrary(); });
}

// ── Polling (replaces SSE — no spinning browser tab) ─────────────────────────
let _lastLibraryTs = 0;

function pollState() {
  fetch('/api/state')
    .then(r => r.json())
    .then(s => {
      applyState(s);
      // Refresh library when a new track finishes (state goes playing → idle)
      const wasPlaying = _lastState && ['playing','generating'].includes(_lastState.state);
      const nowIdle    = s.state === 'idle';
      const libAge     = Date.now() - _lastLibraryTs;
      if ((wasPlaying && nowIdle) || libAge > 15000) loadLibrary();
    })
    .catch(() => {});
}

// ── Init ──────────────────────────────────────────────────────────────────────
pollState();
loadLibrary();
// While generating/playing poll every 1s for live progress; otherwise every 2s
setInterval(() => {
  const active = _lastState && ['loading','generating','playing'].includes(_lastState.state);
  pollState();
}, 1000);
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
