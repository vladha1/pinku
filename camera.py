"""
Camera — YOLO object detection + MediaPipe pose/hands + laser wake detection.

Runs two background threads:
  1. Capture thread  — opens camera, reads frames into _last_frame (fast, no heavy libs)
  2. Detect thread   — runs YOLO + MediaPipe on captured frames every DETECT_EVERY seconds

Separating capture from detection means the live feed works in the dashboard
even while YOLO is still loading or if MediaPipe fails.

macOS note: uses cv2.CAP_AVFOUNDATION backend; auto-scans indices 0-3 if 0 fails.
"""

from __future__ import annotations
import time
import threading
import base64
import traceback
import numpy as np
import cv2

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    YOLO_MODEL, YOLO_CONF, DETECT_EVERY,
    LASER_STARS_MIN, LASER_COOLDOWN,
)

# ── Shared state ──────────────────────────────────────────────────────────────
_frame_lock  = threading.Lock()
_last_frame: np.ndarray | None = None

_cam_status  = "starting"   # "starting" | "ok" | "error: <msg>"
_det_status  = "starting"   # "starting" | "ok" | "error: <msg>"


def get_frame() -> np.ndarray | None:
    with _frame_lock:
        return _last_frame.copy() if _last_frame is not None else None


def get_status() -> dict:
    return {"camera": _cam_status, "detection": _det_status}


def frame_to_b64(frame: np.ndarray) -> str:
    """Encode a BGR frame as base64 JPEG."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


# ── Laser dot counter ─────────────────────────────────────────────────────────

def _laser_dot_count(frame: np.ndarray) -> int:
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


# ── Camera open helper ────────────────────────────────────────────────────────

def _open_camera(preferred_index: int) -> cv2.VideoCapture | None:
    """
    Try to open camera, using AVFoundation on macOS.
    Auto-scans indices 0-3 if the preferred index fails.
    Returns an open VideoCapture, or None.
    """
    indices = list(dict.fromkeys([preferred_index, 0, 1, 2, 3]))

    for idx in indices:
        # Try AVFoundation first (macOS native, required for permissions)
        for backend in [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
                    # Verify we can actually read a frame
                    ret, _ = cap.read()
                    if ret:
                        backend_name = "AVFoundation" if backend == cv2.CAP_AVFOUNDATION else "default"
                        print(f"[Camera] Opened camera index={idx} backend={backend_name}")
                        return cap
                    cap.release()
            except Exception:
                pass

    return None


# ── Detection thread ──────────────────────────────────────────────────────────

class CameraDetector:
    """
    Manages two daemon threads:
      _capture_thread — reads frames from webcam continuously
      _detect_thread  — runs YOLO + MediaPipe every DETECT_EVERY seconds

    Callbacks (called from detect thread):
      on_detection(event)    — {timestamp, objects, persons, gestures}
      on_laser_wake()        — single laser dot (rising edge)
      on_laser_stars(muted)  — star field detected
    """

    def __init__(self,
                 on_detection=None,
                 on_laser_wake=None,
                 on_laser_stars=None,
                 is_muted_fn=None):
        self.on_detection   = on_detection   or (lambda e: None)
        self.on_laser_wake  = on_laser_wake  or (lambda: None)
        self.on_laser_stars = on_laser_stars or (lambda muted: None)
        self.is_muted_fn    = is_muted_fn    or (lambda: False)

        self._stop = threading.Event()

        # Laser state
        self._laser_trigger_at      = 0.0
        self._laser_consec          = 0
        self._laser_dot_was_present = False

        # Dedup
        self._last_snapshot = None

    def start(self):
        threading.Thread(target=self._capture_loop, daemon=True, name="cam-capture").start()
        threading.Thread(target=self._detect_loop,  daemon=True, name="cam-detect").start()

    def stop(self):
        self._stop.set()

    # ── Capture loop ──────────────────────────────────────────────────────────

    def _capture_loop(self):
        global _last_frame, _cam_status
        print("[Camera] Starting capture …")

        cap = _open_camera(CAMERA_INDEX)
        if cap is None:
            msg = (
                f"Cannot open any camera (tried indices 0-3 with AVFoundation + default). "
                f"Check System Settings → Privacy & Security → Camera and grant access to Terminal/Python."
            )
            print(f"[Camera] ERROR: {msg}")
            _cam_status = f"error: no camera found"
            return

        _cam_status = "ok"
        consec_fail = 0

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                consec_fail += 1
                if consec_fail > 30:
                    print("[Camera] Too many read failures — trying to reopen …")
                    cap.release()
                    cap = _open_camera(CAMERA_INDEX)
                    if cap is None:
                        _cam_status = "error: camera disconnected"
                        break
                    consec_fail = 0
                time.sleep(0.05)
                continue

            consec_fail = 0
            with _frame_lock:
                _last_frame = frame.copy()

        cap.release()
        print("[Camera] Capture stopped")

    # ── Detect loop ───────────────────────────────────────────────────────────

    def _detect_loop(self):
        global _det_status
        print("[Camera] Starting detection …")

        # Load heavy models — wrap each separately so partial failures are visible
        try:
            from ultralytics import YOLO
            print("[Camera] Loading YOLO …")
            yolo = YOLO(YOLO_MODEL)
            print("[Camera] YOLO ready")
        except Exception as e:
            print(f"[Camera] YOLO load failed: {e}")
            _det_status = f"error: YOLO {e}"
            return

        try:
            import mediapipe as mp
            mp_pose  = mp.solutions.pose
            mp_hands = mp.solutions.hands
            pose  = mp_pose.Pose(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5,
                                 model_complexity=0)
            hands = mp_hands.Hands(max_num_hands=2,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5,
                                   model_complexity=0)
            print("[Camera] MediaPipe ready")
        except Exception as e:
            print(f"[Camera] MediaPipe load failed: {e}")
            _det_status = f"error: MediaPipe {e}"
            return

        _det_status = "ok"
        last_detect = 0.0

        while not self._stop.is_set():
            # Wait until there's a frame
            frame = get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            now = time.time()
            if now - last_detect < DETECT_EVERY:
                time.sleep(0.05)
                continue
            last_detect = now

            try:
                self._check_laser(frame)
                self._run_detection(frame, yolo, pose, hands)
            except Exception:
                print(f"[Camera] Detection error:\n{traceback.format_exc()}")
                time.sleep(1.0)

        pose.close()
        hands.close()
        print("[Camera] Detection stopped")

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
        if all(extended):                                                return "Open Hand"
        if not any(extended):                                            return "Fist"
        if th and not ix and not mi and not ri and not pi:
            return "Thumbs Up" if lm[4].y < lm[3].y else "Thumbs Down"
        if ix and mi and not ri and not pi and not th:                   return "Peace"
        if ix and not mi and not ri and not pi:                          return "Pointing"
        if ix and pi and not mi and not ri:                              return "Rock On"
        if th and pi and not ix and not mi and not ri:                   return "Call Me"
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

        if is_star and self._laser_consec < 2:
            return
        if not is_star:
            if self._laser_dot_was_present:
                return
            self._laser_dot_was_present = True

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

    # ── YOLO + MediaPipe detection ────────────────────────────────────────────

    def _run_detection(self, frame, yolo, pose, hands):
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

        pose_res = pose.process(rgb)
        if pose_res.pose_landmarks and event["persons"] == 0:
            event["persons"] = 1

        hand_res = hands.process(rgb)
        if hand_res.multi_hand_landmarks:
            for lm, hd in zip(hand_res.multi_hand_landmarks,
                              hand_res.multi_handedness):
                g    = self._classify_gesture(lm, hd)
                side = hd.classification[0].label
                event["gestures"].append({"hand": side, "gesture": g})

        snap = (
            event["persons"],
            tuple(sorted(o["label"] for o in event["objects"])),
            tuple(sorted(f"{g['hand']}:{g['gesture']}" for g in event["gestures"])),
        )
        has_det = event["objects"] or event["persons"] or event["gestures"]
        if has_det and snap != self._last_snapshot:
            self.on_detection(event)
        self._last_snapshot = snap if has_det else None
