#!/usr/bin/env python3
"""
Enroll a speaker into Pinku's speaker identification system.

Usage:
    # Record directly from the mic (recommended — 10+ seconds of natural speech):
    python enroll_speakers.py --name vlad --record 12

    # From an existing WAV file (mono or stereo, any sample rate):
    python enroll_speakers.py --name vlad --wav /path/to/recording.wav

    # List enrolled profiles:
    python enroll_speakers.py --list

    # Remove a profile:
    python enroll_speakers.py --remove vlad

Tips:
  - Speak naturally for 10–15 seconds (a few sentences, normal conversational pace)
  - Use the same mic and room position you use when talking to Pinku
  - Avoid background noise, TV, or music during enrollment
  - Re-enroll if accuracy drops (different room, new mic, illness affecting voice)
  - Enroll separately for each family member who should be able to trigger Pinku
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import wave

PROFILES_DIR = os.path.join(os.path.expanduser("~"), "pinku", "profiles")


def record_pcm(seconds: int, sample_rate: int = 16000) -> bytes:
    """Record seconds of audio from the default mic. Returns raw 16-bit mono PCM."""
    try:
        import pyaudio
    except ImportError:
        print("pyaudio is required: pip install pyaudio")
        sys.exit(1)

    pa    = pyaudio.PyAudio()
    chunk = 1024
    stream = pa.open(
        format=pyaudio.paInt16, channels=1, rate=sample_rate,
        input=True, frames_per_buffer=chunk,
    )

    print(f"  Recording {seconds}s — speak now …")
    frames = []
    total  = int(sample_rate / chunk * seconds)
    for i in range(total):
        frames.append(stream.read(chunk, exception_on_overflow=False))
        pct = int((i + 1) / total * 20)
        print(f"\r  [{'#' * pct}{'.' * (20 - pct)}] {i + 1}/{total}", end="", flush=True)
    print()

    stream.stop_stream()
    stream.close()
    pa.terminate()
    return b"".join(frames)


def load_wav_as_pcm(path: str, target_rate: int = 16000) -> bytes:
    """Load a WAV (mono or stereo) and return mono 16-bit PCM at target_rate."""
    with wave.open(path, "rb") as wf:
        channels  = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        src_rate  = wf.getframerate()
        raw       = wf.readframes(wf.getnframes())

    audio = np.frombuffer(raw, dtype=np.int16 if sampwidth == 2 else np.int8)
    audio = audio.astype(np.float32)

    # Mix down to mono
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

    # Normalise to int16 range
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 32767.0

    # Resample to target_rate if needed (simple linear for small ratio differences)
    if src_rate != target_rate:
        from resemblyzer import preprocess_wav
        wav_f32 = audio.astype(np.float32) / 32768.0
        # preprocess_wav handles resampling internally via librosa
        wav_f32 = preprocess_wav(wav_f32)
        return (wav_f32 * 32768.0).astype(np.int16).tobytes()

    return audio.astype(np.int16).tobytes()


def embed_pcm(pcm: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Extract a 256-d Resemblyzer embedding from raw PCM bytes."""
    from resemblyzer import VoiceEncoder, preprocess_wav
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    wav   = preprocess_wav(audio)
    enc   = VoiceEncoder()
    print("  Extracting voice embedding …")
    emb   = enc.embed_utterance(wav)
    # Normalise to unit length (speaker_id.py also does this, but belt-and-suspenders)
    norm  = np.linalg.norm(emb)
    if norm > 1e-8:
        emb = emb / norm
    return emb


def cmd_list():
    if not os.path.isdir(PROFILES_DIR):
        print("No profiles directory found.")
        return
    files = [f for f in os.listdir(PROFILES_DIR) if f.endswith(".npy")]
    if not files:
        print("No profiles enrolled yet.")
        return
    print(f"Enrolled profiles in {PROFILES_DIR}:")
    for f in sorted(files):
        path = os.path.join(PROFILES_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f[:-4]:20s}  ({size} bytes)")


def cmd_remove(name: str):
    path = os.path.join(PROFILES_DIR, f"{name}.npy")
    if not os.path.exists(path):
        print(f"No profile found for '{name}'.")
        return
    os.remove(path)
    print(f"Removed profile: {name}")


def main():
    ap = argparse.ArgumentParser(
        description="Enroll speaker profiles for Pinku",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--name",   help="Speaker name to enroll (e.g. vlad)")
    ap.add_argument("--wav",    help="Path to a WAV file to use for enrollment")
    ap.add_argument("--record", type=int, metavar="SECONDS",
                    help="Record N seconds of speech from the mic")
    ap.add_argument("--list",   action="store_true", help="List enrolled profiles")
    ap.add_argument("--remove", metavar="NAME", help="Remove an enrolled profile")
    args = ap.parse_args()

    if args.list:
        cmd_list()
        return

    if args.remove:
        cmd_remove(args.remove)
        return

    if not args.name:
        ap.error("--name is required for enrollment")

    if not args.wav and not args.record:
        ap.error("Provide --wav or --record to supply audio")

    os.makedirs(PROFILES_DIR, exist_ok=True)
    out_path = os.path.join(PROFILES_DIR, f"{args.name}.npy")

    if os.path.exists(out_path):
        ans = input(f"Profile '{args.name}' already exists. Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return

    print(f"\nEnrolling: {args.name}")

    if args.wav:
        print(f"  Loading {args.wav} …")
        pcm = load_wav_as_pcm(args.wav)
    else:
        pcm = record_pcm(args.record)

    duration = len(pcm) / (16000 * 2)
    print(f"  Audio: {duration:.1f}s")
    if duration < 4.0:
        print("  Warning: less than 4 seconds — accuracy may be lower. "
              "Try enrolling with 10+ seconds.")

    emb = embed_pcm(pcm)
    np.save(out_path, emb)
    print(f"  Saved → {out_path}")
    print(f"\nDone. '{args.name}' is now enrolled.")
    print("Restart Pinku for the new profile to take effect.")


if __name__ == "__main__":
    main()
