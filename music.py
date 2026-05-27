"""
Music generation + playback for Pinku.

Primary backend : Meta MusicGen (AudioCraft) — local, MPS-accelerated on Apple Silicon.
Fallback backend: Ollama → ABC notation → abc2midi → FluidSynth (fully local, lightweight).

Install deps:
  pip install audiocraft          # MusicGen
  brew install abcmidi fluidsynth # ABC fallback

Usage:
  import music
  music.start("rainy day jazz", duration_sec=300, on_state_change=my_cb)
  music.stop()
"""

from __future__ import annotations
import json
import math
import os
import queue
import subprocess
import threading
import time
import uuid
from typing import Callable

from config import (
    MUSIC_BACKEND, MUSICGEN_MODEL, MUSIC_CHUNK_SEC, MUSIC_MODEL_CACHE,
)

# ── Library (persistent saved sessions) ──────────────────────────────────────

LIBRARY_DIR  = os.path.expanduser("~/pinku/music_library")
_INDEX_FILE  = os.path.join(LIBRARY_DIR, "_index.json")
_lib_lock    = threading.Lock()

def _ensure_library():
    os.makedirs(LIBRARY_DIR, exist_ok=True)

def _load_index() -> list[dict]:
    try:
        with open(_INDEX_FILE) as f:
            return json.load(f)
    except Exception:
        return []

def _save_index(items: list[dict]):
    _ensure_library()
    with open(_INDEX_FILE, "w") as f:
        json.dump(items, f, indent=2)

def list_library() -> list[dict]:
    """Return all saved sessions, newest first."""
    with _lib_lock:
        items = _load_index()
    # Attach live file sizes and filter out missing files
    valid = []
    for item in items:
        chunks = [c for c in item.get("chunks", []) if os.path.exists(c)]
        if chunks:
            item["chunks"]   = chunks
            item["size_kb"]  = sum(os.path.getsize(c) for c in chunks) // 1024
            valid.append(item)
    return sorted(valid, key=lambda x: x.get("created_at", ""), reverse=True)

def _add_to_library(session_id: str, theme: str, prompt: str,
                    duration_sec: int, chunk_paths: list[str]) -> dict:
    entry = {
        "id":          session_id,
        "theme":       theme,
        "prompt":      prompt,
        "duration_sec": duration_sec,
        "chunks":      chunk_paths,
        "size_kb":     sum(os.path.getsize(p) for p in chunk_paths) // 1024,
        "created_at":  time.strftime("%Y-%m-%d %H:%M"),
    }
    with _lib_lock:
        items = _load_index()
        items.append(entry)
        _save_index(items)
    return entry

def delete_library_item(item_id: str) -> bool:
    """Delete a saved session (files + index entry). Returns True if found."""
    with _lib_lock:
        items = _load_index()
        found = next((x for x in items if x["id"] == item_id), None)
        if not found:
            return False
        items = [x for x in items if x["id"] != item_id]
        _save_index(items)
    for path in found.get("chunks", []):
        try:
            os.unlink(path)
        except Exception:
            pass
    return True


def rename_library_item(item_id: str, new_theme: str) -> bool:
    """Rename a saved session. Returns True if found."""
    new_theme = new_theme.strip()
    if not new_theme:
        return False
    with _lib_lock:
        items = _load_index()
        found = False
        for item in items:
            if item["id"] == item_id:
                item["theme"] = new_theme
                found = True
                break
        if found:
            _save_index(items)
    return found


# ── Preset themes ─────────────────────────────────────────────────────────────
# Keys are shown as dashboard buttons; values are MusicGen prompts.

PRESETS: dict[str, str] = {
    "chill":       "lo-fi chill beats, soft piano, gentle bass, warm vinyl crackle, relaxing, 75 bpm",
    "focus":       "minimal piano, quiet ambient pads, no percussion, concentration music, 85 bpm",
    "bollywood":   "upbeat Bollywood dance, dhol, tabla, sitar, brass, energetic, 120 bpm",
    "jazz":        "smooth jazz quartet, piano, upright bass, brushed drums, trumpet, late night, 95 bpm",
    "classical":   "Baroque harpsichord and strings, Bach-style, gentle and ornate, 80 bpm",
    "ambient":     "floating ambient electronic, soft reverb pads, no rhythm, meditative, ethereal",
    "party":       "high energy EDM, synthesizer bass drop, four-on-the-floor kick, euphoric, 128 bpm",
    "meditation":  "Tibetan singing bowls, gentle drone, soft overtone chanting, peaceful, very slow",
    "sleep":       "very quiet sleep music, slow breathing pads, no melody, barely audible, 50 bpm",
    "bhajan":      "Indian devotional, harmonium, tabla, gentle humming, bhajan style, peaceful",
    "folk":        "Indian folk, bansuri flute, gentle acoustic strings, morning raga feeling",
    "rock":        "classic rock, electric guitar riff, drums, bass, energetic, driving rhythm",
}

PRESET_EMOJIS: dict[str, str] = {
    "chill": "☁️", "focus": "🎯", "bollywood": "🪘", "jazz": "🎷",
    "classical": "🎻", "ambient": "🌊", "party": "🎉", "meditation": "🧘",
    "sleep": "🌙", "bhajan": "🪔", "folk": "🪈", "rock": "🎸",
}


# ── State ─────────────────────────────────────────────────────────────────────

_state_lock = threading.Lock()
_state: dict = {
    "state":        "idle",   # idle | loading | generating | playing | error
    "theme":        "",
    "prompt":       "",
    "elapsed":      0,
    "total":        0,
    "chunk":        0,
    "chunks_total": 0,
    "error":        "",
    "paused":       False,
    "cached":       False,
    "playing_id":   "",      # library item ID currently playing (empty = live generation)
}

_on_state_change:   Callable[[dict], None] | None = None
_on_library_change: Callable[[], None] | None = None
_stop_event  = threading.Event()
_play_thread: threading.Thread | None = None
_afplay_lock = threading.Lock()
_afplay_proc: subprocess.Popen | None = None


def _dlog(msg: str, level: str = "music"):
    """Log to stdout and dashboard Log tab."""
    print(f"[Music] {msg}")
    try:
        import dashboard as _db
        _db.log_message(level, f"♪ {msg}")
    except Exception:
        pass


def _set_state(**kwargs):
    with _state_lock:
        _state.update(kwargs)
    cb = _on_state_change
    if cb:
        try:
            cb(dict(_state))
        except Exception:
            pass


def get_state() -> dict:
    with _state_lock:
        return dict(_state)


# ── MusicGen backend (via HuggingFace transformers — no C build deps) ─────────
# Install: .venv/bin/pip install transformers scipy accelerate
#
# Token rate: MusicGen encodes audio at 50 tokens/sec.
# chunk_sec=30  →  max_new_tokens=1500

_mg_model     = None
_mg_processor = None
_mg_device    = "cpu"
_mg_lock      = threading.Lock()


def _best_device() -> str:
    """Return 'mps' on Apple Silicon, 'cuda' if available, else 'cpu'."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _load_musicgen():
    """Load MusicGen model onto the best available device.
    Returns (model, processor) or raises RuntimeError with a user-friendly message.
    On failure the state is set to 'error' so the UI doesn't stay on 'Loading model…'.
    """
    global _mg_model, _mg_processor, _mg_device
    if _mg_model is not None:
        return _mg_model, _mg_processor
    with _mg_lock:
        if _mg_model is not None:
            return _mg_model, _mg_processor
        device = _best_device()
        os.makedirs(MUSIC_MODEL_CACHE, exist_ok=True)
        _from_cache = os.path.isdir(os.path.join(
            MUSIC_MODEL_CACHE,
            "models--" + MUSICGEN_MODEL.replace("/", "--"),
            "snapshots",
        ))
        _set_state(state="loading", cached=_from_cache)
        print(f"[Music] {'Loading' if _from_cache else 'Downloading'} MusicGen "
              f"{MUSICGEN_MODEL!r} on {device} "
              f"({'cache: ' + MUSIC_MODEL_CACHE if _from_cache else '~300 MB first run'}) …")
        try:
            from transformers import (          # type: ignore
                MusicgenForConditionalGeneration,
                AutoProcessor,
            )
            import torch
            _mg_processor = AutoProcessor.from_pretrained(
                MUSICGEN_MODEL, cache_dir=MUSIC_MODEL_CACHE)
            _mg_model     = MusicgenForConditionalGeneration.from_pretrained(
                MUSICGEN_MODEL,
                cache_dir=MUSIC_MODEL_CACHE,
                low_cpu_mem_usage=True,
            )
            _mg_model = _mg_model.to(device)
            _mg_device = device
            print(f"[Music] MusicGen ready ✓  device={device}")
            # If preload() finished and nobody started playback yet, go back to idle
            if get_state()["state"] == "loading":
                _set_state(state="idle")
        except ImportError:
            err = "transformers not installed — run: pip install transformers scipy accelerate"
            _set_state(state="error", error=err)
            raise RuntimeError(err)
        except Exception as exc:
            import traceback
            err = f"Model load failed: {exc}"
            print(f"[Music] ERROR loading model:\n{traceback.format_exc()}")
            _set_state(state="error", error=str(exc)[:200])
            # Reset so the user can retry by pressing Play
            _mg_model     = None
            _mg_processor = None
            raise RuntimeError(err) from exc
    return _mg_model, _mg_processor


def _generate_chunk_musicgen(prompt: str, chunk_sec: int, idx: int) -> str:
    """Generate one WAV chunk via MusicGen (transformers). Returns WAV path."""
    import torch
    import scipy.io.wavfile   # type: ignore
    import numpy as np

    model, processor = _load_musicgen()

    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    _dlog(f"generating chunk {idx} — {chunk_sec}s ({chunk_sec*50} tokens)…")

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=chunk_sec * 50,   # 50 tokens ≈ 1 second
        )

    _dlog(f"raw output shape: {tuple(audio_values.shape)}")

    # Shape is [batch, channels, samples] or [batch, samples] depending on version
    if audio_values.ndim == 3:
        data = audio_values[0, 0].cpu().numpy()
    elif audio_values.ndim == 2:
        data = audio_values[0].cpu().numpy()
    else:
        data = audio_values.cpu().numpy().flatten()

    # Sample rate: try config first, fall back to 32000 Hz (MusicGen default)
    try:
        sr = model.config.audio_encoder.sampling_rate
    except Exception:
        sr = 32000

    _dlog(f"audio: {len(data)} samples @ {sr}Hz = {len(data)/sr:.1f}s")

    # Normalise to int16
    peak     = float(np.abs(data).max()) or 1.0
    data_i16 = (data / peak * 32767 * 0.82).astype(np.int16)

    path = f"/tmp/pinku_music_{idx}.wav"
    scipy.io.wavfile.write(path, rate=sr, data=data_i16)

    size_kb = os.path.getsize(path) // 1024
    _dlog(f"wrote {path} ({size_kb} KB)")
    if size_kb < 10:
        raise RuntimeError(f"WAV too small ({size_kb} KB) — generation likely failed")
    return path


# ── ABC/MIDI fallback backend ─────────────────────────────────────────────────

_SF2_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/opt/homebrew/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/local/share/sounds/sf2/FluidR3_GM.sf2",
    # macOS fluid-soundfont-gm via brew
    "/opt/homebrew/Cellar/fluid-soundfont-gm/FluidR3_GM.sf2",
]

_ABC_SYS = (
    "You are a music composer. Output ONLY valid ABC notation — no markdown, no explanation.\n"
    "Format: X:1, T:title, M:meter, L:1/8, Q:bpm, K:key, then bars.\n"
    "Write exactly 16 bars."
)


def _generate_chunk_abc(theme: str, idx: int) -> str | None:
    """Ollama → ABC → midi → WAV. Returns WAV path or None on failure."""
    abc_path = f"/tmp/pinku_music_{idx}.abc"
    mid_path = f"/tmp/pinku_music_{idx}.mid"
    wav_path = f"/tmp/pinku_music_{idx}.wav"
    try:
        # Ask Ollama for ABC notation
        import urllib.request, json as _json
        body = _json.dumps({
            "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            "system": _ABC_SYS,
            "prompt": f"Write a short {theme} melody in ABC notation, 16 bars.",
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate",
            data=body, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            abc_text = _json.loads(r.read())["response"].strip()

        # Extract ABC block if wrapped in markdown
        if "X:1" not in abc_text:
            import re
            m = re.search(r'(X:1.*?)(?:```|\Z)', abc_text, re.DOTALL)
            abc_text = m.group(1).strip() if m else abc_text

        with open(abc_path, "w") as f:
            f.write(abc_text)

        # abc2midi
        r2 = subprocess.run(["abc2midi", abc_path, "-o", mid_path],
                            capture_output=True, timeout=15)
        if r2.returncode != 0 or not os.path.exists(mid_path):
            print(f"[Music] abc2midi failed: {r2.stderr.decode()[:200]}")
            return None

        # FluidSynth
        sf = next((s for s in _SF2_CANDIDATES if os.path.exists(s)), None)
        if not sf:
            print("[Music] No SF2 soundfont found — install fluid-soundfont-gm")
            return None
        subprocess.run(
            ["fluidsynth", "-ni", "-F", wav_path, sf, mid_path],
            capture_output=True, timeout=60,
        )
        return wav_path if os.path.exists(wav_path) else None

    except Exception as e:
        print(f"[Music] ABC chunk {idx} error: {e}")
        return None
    finally:
        for p in (abc_path, mid_path):
            try:
                os.path.exists(p) and os.unlink(p)
            except Exception:
                pass


# ── Theme → prompt ────────────────────────────────────────────────────────────

_NO_VOCALS = ", no vocals, instrumental only"

def resolve_prompt(theme: str) -> str:
    """Map a user theme to a MusicGen prompt. Uses presets, then passes through.
    Always appends a no-vocals tag so MusicGen never generates singing."""
    t = theme.strip().lower()
    if t in PRESETS:
        base = PRESETS[t]
    else:
        base = next((p for k, p in PRESETS.items() if k in t), theme)
    # Strip any accidental duplicate before appending
    base = base.replace(_NO_VOCALS, "").rstrip(", ")
    return base + _NO_VOCALS


# ── afplay playback ───────────────────────────────────────────────────────────

def _play_wav_async(path: str) -> subprocess.Popen:
    global _afplay_proc
    with _afplay_lock:
        proc = subprocess.Popen(
            ["afplay", "-v", "0.82", path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _afplay_proc = proc
    return proc


def _kill_afplay():
    global _afplay_proc
    with _afplay_lock:
        p = _afplay_proc
        _afplay_proc = None
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            p.kill()


def _cleanup(*paths: str):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass


# ── Main entry points ─────────────────────────────────────────────────────────

def start(
    theme: str,
    duration_sec: int = 0,
    on_state_change: Callable[[dict], None] | None = None,
    on_library_change: Callable[[], None] | None = None,
):
    """
    Begin generating and playing music.
    duration_sec=0 → play until stop() is called.
    on_library_change: called (no args) when a new session is saved to the library.
    """
    global _on_state_change, _on_library_change, _play_thread, _stop_event

    stop()   # cancel any in-progress playback first

    # Only overwrite the callbacks if the caller explicitly provides one.
    # This preserves the permanent default registered by music_app at startup.
    if on_state_change is not None:
        _on_state_change   = on_state_change
    if on_library_change is not None:
        _on_library_change = on_library_change
    _stop_event      = threading.Event()
    dur              = max(0, int(duration_sec))

    def _run():
        prompt        = resolve_prompt(theme)
        chunk_sec     = MUSIC_CHUNK_SEC
        chunks_needed = math.ceil(dur / chunk_sec) if dur > 0 else 10_000
        session_id    = uuid.uuid4().hex[:10]
        saved_chunks: list[str] = []

        _ensure_library()

        _set_state(
            state="generating", theme=theme, prompt=prompt,
            elapsed=0, total=dur, chunk=0, chunks_total=chunks_needed, error="",
        )
        print(f"[Music] ▶ theme={theme!r}  dur={dur}s  session={session_id}")

        chunk_q: queue.Queue[str | None] = queue.Queue(maxsize=2)

        # ── Generator thread: fills chunk_q ahead of playback ─────────────────
        def _generator():
            for i in range(chunks_needed):
                if _stop_event.is_set():
                    chunk_q.put(None)
                    return
                try:
                    _set_state(state="generating")
                    # Save directly to library dir with stable filename
                    lib_path = os.path.join(LIBRARY_DIR, f"{session_id}_{i}.wav")
                    if MUSIC_BACKEND == "abc":
                        tmp = _generate_chunk_abc(theme, i)
                        if tmp:
                            import shutil
                            shutil.move(tmp, lib_path)
                            path = lib_path
                        else:
                            path = None
                    else:
                        # Override the /tmp path to go straight to library
                        import scipy.io.wavfile, numpy as np, torch
                        model, processor = _load_musicgen()
                        # Move inputs to the same device as the model (MPS / CPU)
                        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
                        inputs = {k: v.to(_mg_device) for k, v in inputs.items()}
                        _dlog(f"generating chunk {i} ({chunk_sec*50} tokens) on {_mg_device}…")
                        with torch.no_grad():
                            audio_values = model.generate(**inputs, max_new_tokens=chunk_sec * 50)
                        _dlog(f"raw shape: {tuple(audio_values.shape)}")
                        if audio_values.ndim == 3:
                            data = audio_values[0, 0].cpu().numpy()
                        elif audio_values.ndim == 2:
                            data = audio_values[0].cpu().numpy()
                        else:
                            data = audio_values.cpu().numpy().flatten()
                        try:
                            sr = model.config.audio_encoder.sampling_rate
                        except Exception:
                            sr = 32000
                        peak     = float(np.abs(data).max()) or 1.0
                        data_i16 = (data / peak * 32767 * 0.82).astype(np.int16)
                        scipy.io.wavfile.write(lib_path, rate=sr, data=data_i16)
                        size_kb = os.path.getsize(lib_path) // 1024
                        _dlog(f"saved {lib_path} ({size_kb} KB)")
                        path = lib_path if size_kb >= 10 else None

                    if not path:
                        _set_state(error="Generation failed — see log")
                        chunk_q.put(None)
                        return
                    saved_chunks.append(path)
                    chunk_q.put(path)
                except Exception as e:
                    import traceback
                    _dlog(f"chunk {i} error: {e}", level="error")
                    _dlog(traceback.format_exc()[:300], level="error")
                    _set_state(error=str(e)[:160])
                    chunk_q.put(None)
                    return
            chunk_q.put(None)   # sentinel

        gen_t = threading.Thread(target=_generator, daemon=True, name="music-gen")
        gen_t.start()

        # ── Playback loop ──────────────────────────────────────────────────────
        # NOTE: start_ts is set on FIRST playback, not on generation start,
        # so duration counts actual listening time — not model loading time.
        start_ts: float | None = None
        chunk_idx = 0

        while not _stop_event.is_set():
            try:
                path = chunk_q.get(timeout=120)   # wait up to 2 min for first chunk
            except queue.Empty:
                _set_state(error="Generation timed out")
                break
            if path is None:
                break

            chunk_idx += 1
            if start_ts is None:
                start_ts = time.time()   # clock starts when music actually begins

            _set_state(state="playing", chunk=chunk_idx, elapsed=0)
            _dlog(f"▶ playing chunk {chunk_idx}: {path}")

            proc = _play_wav_async(path)

            while proc.poll() is None:
                if _stop_event.is_set():
                    proc.terminate()
                    break
                now_elapsed = time.time() - start_ts
                _set_state(elapsed=int(now_elapsed))
                if dur > 0 and now_elapsed >= dur:
                    _stop_event.set()
                    proc.terminate()
                    break
                time.sleep(0.4)

        _kill_afplay()
        gen_t.join(timeout=5)
        final_elapsed = int(time.time() - start_ts) if start_ts else 0

        # Save to library index (keep files — don't clean up)
        if saved_chunks:
            entry = _add_to_library(session_id, theme, prompt, final_elapsed, saved_chunks)
            _dlog(f"saved to library: {entry['id']} ({entry['size_kb']} KB)")
            # Notify music_app.py to refresh the library list via the library callback
            cb = _on_library_change
            if cb:
                try:
                    cb()
                except Exception:
                    pass

        _set_state(state="idle", elapsed=final_elapsed, chunk=0, chunks_total=0)
        print(f"[Music] ■ done  elapsed={final_elapsed}s")

    _play_thread = threading.Thread(target=_run, daemon=True, name="music-play")
    _play_thread.start()


def stop():
    """Stop playback and generation immediately."""
    global _stop_event
    _stop_event.set()
    _kill_afplay()
    t = _play_thread
    if t and t.is_alive():
        t.join(timeout=3)
    _set_state(state="idle", elapsed=0, chunk=0, chunks_total=0, error="", paused=False, playing_id="")


def pause():
    """Pause current afplay with SIGSTOP (instant, no data loss)."""
    import signal
    if get_state()["state"] != "playing":
        return
    with _afplay_lock:
        p = _afplay_proc
    if p and p.poll() is None:
        try:
            p.send_signal(signal.SIGSTOP)
            _set_state(paused=True)
        except Exception:
            pass


def resume():
    """Resume a SIGSTOP-paused afplay with SIGCONT."""
    import signal
    with _afplay_lock:
        p = _afplay_proc
    if p and p.poll() is None:
        try:
            p.send_signal(signal.SIGCONT)
            _set_state(paused=False)
        except Exception:
            pass


def pause_toggle():
    """Toggle pause/resume."""
    if get_state().get("paused"):
        resume()
    else:
        pause()


def play_library_item(
    item_id: str,
    on_state_change: Callable[[dict], None] | None = None,
):
    """Play a saved library session by ID (no generation — instant playback)."""
    global _on_state_change, _play_thread, _stop_event

    items = list_library()
    entry = next((x for x in items if x["id"] == item_id), None)
    if not entry:
        raise ValueError(f"Library item {item_id!r} not found")

    chunks = entry["chunks"]
    theme  = entry["theme"]
    total  = sum(os.path.getsize(c) for c in chunks) / (32000 * 2)  # approx seconds

    stop()

    _on_state_change = on_state_change
    _stop_event      = threading.Event()

    def _run_lib():
        _set_state(state="playing", theme=theme, prompt=entry.get("prompt",""),
                   elapsed=0, total=int(total), chunk=0,
                   chunks_total=len(chunks), error="", playing_id=item_id)
        _dlog(f"▶ library play: {theme!r}  {len(chunks)} chunk(s)")

        start_ts  = time.time()
        for idx, path in enumerate(chunks, 1):
            if _stop_event.is_set():
                break
            if not os.path.exists(path):
                _dlog(f"chunk missing: {path}", level="error")
                continue
            _set_state(state="playing", chunk=idx,
                       elapsed=int(time.time() - start_ts))
            proc = _play_wav_async(path)
            while proc.poll() is None:
                if _stop_event.is_set():
                    proc.terminate()
                    break
                _set_state(elapsed=int(time.time() - start_ts))
                time.sleep(0.4)

        _kill_afplay()
        _set_state(state="idle", elapsed=int(time.time() - start_ts),
                   chunk=0, chunks_total=0, playing_id="")
        _dlog(f"■ library playback done")

    _play_thread = threading.Thread(target=_run_lib, daemon=True, name="music-lib-play")
    _play_thread.start()


def is_playing() -> bool:
    return get_state()["state"] in ("loading", "generating", "playing")


def duck():
    """Temporarily pause music playback (SIGSTOP) so Pinku's reply is audible.
    Call unduck() to resume. Safe to call when nothing is playing."""
    import signal
    with _afplay_lock:
        p = _afplay_proc
    if p and p.poll() is None:
        try:
            p.send_signal(signal.SIGSTOP)
        except Exception:
            pass


def unduck():
    """Resume music playback (SIGCONT) after Pinku finishes speaking."""
    import signal
    with _afplay_lock:
        p = _afplay_proc
    if p and p.poll() is None:
        try:
            p.send_signal(signal.SIGCONT)
        except Exception:
            pass


def preload():
    """Warm up the MusicGen model in the background (call at startup)."""
    if MUSIC_BACKEND != "musicgen":
        return

    def _do_preload():
        try:
            _load_musicgen()
        except Exception as e:
            # Error is already reflected in state; just log here so terminal shows it
            print(f"[Music] preload failed: {e}")

    threading.Thread(target=_do_preload, daemon=True, name="music-preload").start()
