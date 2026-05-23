"""
Camera — YOLO object detection + MediaPipe pose/hands + laser wake detection.

Runs in a background thread, posting detection events to a callback.
On M4 Mac Mini, YOLOv8 runs via PyTorch MPS backend (Metal) automatically.

Laser detection (ceiling-mounted camera):
  Single dot (1–LASER_STARS_MIN dots)  → call on_laser_wake()
  Star field  (≥ LASER_STARS_MIN dots) → call on_laser_stars()
"""

from __future__ import annotations
import time
import threading
import base64
import numpy as np
import cv2

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    YOLO_MODEL, YOLO_CONF, DETECT_EVERY,
    LASER_STARS_MIN, LASER_COOLDOWN,
)

# ── Shared frame store ────────────────────────────────────────────────────────
_frame_lock = threading.Lock()
_last_frame: np.ndarray | None = None


def get_frame() -> np.ndarray | None:
    with _frame_lock:
        return _last_frame.copy() if _last_frame is not None else None


def frame_to_b64(frame: np.ndarray) -> str:
    """Encode a BGR frame as base64 JPEG."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


# ── Laser dot counter ─────────────────────────────────────────────────────────

def _laser_dot_count(frame: np.ndarray) -> int:
    """
    Count distinct green laser dots visible in frame.

    Two mask strategies combined:
    1. Green halo  — saturated green (hue 40-85, sat ≥ 80, val ≥ 170)
    2. Hot core    — overexposed white-green (sat < 80, val ≥ 245)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    k   = np.ones((3, 3), dtype=np.uint8)

    mask_grn = cv2.inRange(hsv,
                           np.array([40,  80, 170], dtype=np.uint8),
                           np.array([85, 255, 255], dtype=np.uint8))
    mask_grn = cv2.morphologyEx(mask_grn, cv2.MORPH_OPEN, k)

    mask_hot = cv2.inRange(hsv,
                           np.array([0,   0, 245], dtype=np.uint8),
                           np.array([180, 80, 255], dtype=np.uint8))
    mask_hot = cv2.morphologyEx(mask_hot, cv2.MORPH_OPEN, k)

    combined = cv2.bitwise_or(mask_grn, mask_hot)
    cnts, _  = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len([c for c in cnts if 2 <= cv2.contourArea(c) <= 120])


# ── Detection thread ──────────────────────────────────────────────────────────

class CameraDetector:
    """
    Background thread that captures frames and runs:
    - YOLOv8 object / person detection
    - MediaPipe pose + hand gesture classification
    - Laser dot detection

    Callbacks (all called from the detection thread):
      on_detection(event)   — dict: {timestamp, objects, persons, gestures}
      on_laser_wake()       — single laser dot appeared (rising edge)
      on_laser_stars(muted) — star field; muted=True means currently muted
    """

    def __init__(self,
                 on_detection=None,
                 on_laser_wake=None,
                 on_laser_stars=None,
                 is_muted_fn=None):
        self.on_detection  = on_detection  or (lambda e: None)
        self.on_laser_wake = on_laser_wake or (lambda: None)
        self.on_laser_stars= on_laser_stars or (lambda muted: None)
        self.is_muted_fn   = is_muted_fn   or (lambda: False)

        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._stop    = threading.Event()

        # Laser edge-detection state
        self._laser_trigger_at:     float = 0.0
        self._laser_consec:         int   = 0
        self._laser_dot_was_present: bool = False

        # Detection dedup state
        self._last_snapshot = None

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ── Gesture classifier ────────────────────────────────────────────────────

    @staticmethod
    def _classify_gesture(hand_lm, handedness) -> str:
        lm   = hand_lm.landmark
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        is_right = handedness.classification[0].label == "Right"
        extended = [lm[4].x < lm[3].x if is_right else lm[4].x > lm[3].x]
        extended += [lm[t].y < lm[p].y for t, p in zip(tips[1:], pips[1:])]
        th, ix, mi, ri, pi = extended
        if all(extended):                          return "Open Hand"
        if not any(extended):                      return "Fist"
        if th and not ix and not mi and not ri and not pi:
            return "Thumbs Up" if lm[4].y < lm[3].y else "Thumbs Down"
        if ix and mi and not ri and not pi and not th: return "Peace"
        if ix and not mi and not ri and not pi:        return "Pointing"
        if ix and pi and not mi and not ri:            return "Rock On"
        if th and pi and not ix and not mi and not ri: return "Call Me"
        return "Custom"

    # ── Laser check ───────────────────────────────────────────────────────────

    def _check_laser(self, frame: np.ndarray):
        dot_count = _laser_dot_count(frame)

        if dot_count == 0:
            self._laser_consec          = 0
            self._laser_dot_was_present = False
            return

        self._laser_consec += 1
        is_star = dot_count >= LASER_STARS_MIN

        # Star field: require 2 stable frames
        if is_star and self._laser_consec < 2:
            return

        # Single dot: edge-triggered only
        if not is_star:
            if self._laser_dot_was_present:
                return
            self._laser_dot_was_present = True

        # Cooldown
        if time.time() - self._laser_trigger_at < LASER_COOLDOWN:
            return

        self._laser_trigger_at = time.time()
        self._laser_consec     = 0

        if is_star:
            print(f"[Laser] Star field ({dot_count} dots) → mute toggle")
            self.on_laser_stars(self.is_muted_fn())
        else:
            print(f"[Laser] Single dot → wake")
            self.on_laser_wake()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run(self):
        global _last_frame
        from ultralytics import YOLO
        import mediapipe as mp

        print("[Camera] Loading YOLO …")
        yolo  = YOLO(YOLO_MODEL)
        mp_pose  = mp.solutions.pose
        mp_hands = mp.solutions.hands
        pose  = mp_pose.Pose(min_detection_confidence=0.5,
                             min_tracking_confidence=0.5,
                             model_complexity=0)
        hands = mp_hands.Hands(max_num_hands=2,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5,
                               model_complexity=0)

        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

        if not cap.isOpened():
            print(f"[Camera] ERROR: cannot open camera {CAMERA_INDEX}")
            return

        print(f"[Camera] Running (detect every {DETECT_EVERY}s)")
        last_detect = 0.0

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            with _frame_lock:
                _last_frame = frame.copy()

            now = time.time()
            if now - last_detect < DETECT_EVERY:
                continue
            last_detect = now

            # ── Laser ────────────────────────────────────────────────────────
            self._check_laser(frame)

            # ── YOLO ─────────────────────────────────────────────────────────
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            event = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "objects": [], "persons": 0, "gestures": [],
            }

            results = yolo(frame, conf=YOLO_CONF, verbose=False)[0]
            for box in results.boxes:
                label = yolo.names[int(box.cls[0])]
                conf  = round(float(box.conf[0]), 2)
                if label == "person":
                    event["persons"] += 1
                else:
                    event["objects"].append({"label": label, "conf": conf})

            # ── Pose ─────────────────────────────────────────────────────────
            pose_res = pose.process(rgb)
            if pose_res.pose_landmarks and event["persons"] == 0:
                event["persons"] = 1

            # ── Hands ────────────────────────────────────────────────────────
            hand_res = hands.process(rgb)
            if hand_res.multi_hand_landmarks:
                for lm, hd in zip(hand_res.multi_hand_landmarks,
                                  hand_res.multi_handedness):
                    g    = self._classify_gesture(lm, hd)
                    side = hd.classification[0].label
                    event["gestures"].append({"hand": side, "gesture": g})

            # ── Dedup + fire callback ─────────────────────────────────────────
            snap = (
                event["persons"],
                tuple(sorted(o["label"] for o in event["objects"])),
                tuple(sorted(f"{g['hand']}:{g['gesture']}" for g in event["gestures"])),
            )
            has_det = event["objects"] or event["persons"] or event["gestures"]
            if has_det and snap != self._last_snapshot:
                self.on_detection(event)
            self._last_snapshot = snap if has_det else None

        cap.release()
        pose.close()
        hands.close()
        print("[Camera] Stopped")
