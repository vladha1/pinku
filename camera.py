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
    YOLO_MODEL, YOLO_CONF, YOLO_IGNORE, DETECT_EVERY,
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


def _push_status():
    """Push camera status to dashboard SSE stream (best-effort, no import error if dashboard not ready)."""
    try:
        import dashboard
        dashboard.push_camera_status(_cam_status, _det_status)
    except Exception:
        pass


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
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
                # AVFoundation needs a moment to warm up — try up to 10 frames
                got_frame = False
                for _ in range(10):
                    ret, _ = cap.read()
                    if ret:
                        got_frame = True
                        break
                    time.sleep(0.1)
                if got_frame:
                    backend_name = "AVFoundation" if backend == cv2.CAP_AVFOUNDATION else "default"
                    print(f"[Camera] Opened camera index={idx} backend={backend_name}")
                    return cap
                cap.release()
            except Exception as e:
                print(f"[Camera] index={idx} backend={backend}: {e}")
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
            _push_status()
            return

        _cam_status = "ok"
        _push_status()
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
                        _push_status()
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

        # ── YOLO (required) ───────────────────────────────────────────────────
        try:
            from ultralytics import YOLO
            print("[Camera] Loading YOLO …")
            yolo = YOLO(YOLO_MODEL)
            print("[Camera] YOLO ready")
        except Exception as e:
            print(f"[Camera] YOLO load failed: {e}")
            _det_status = f"error: YOLO {e}"
            _push_status()
            return

        # ── MediaPipe Tasks API (optional — Apple Silicon M1/M2/M3/M4) ────────
        # mediapipe ≥ 0.10 on Apple Silicon dropped the old `solutions` API.
        # We try the new Tasks API first, then the legacy API, then skip gestures.
        hand_detector = None
        pose_detector = None

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            # ── Download task models if needed ────────────────────────────────
            import urllib.request, os

            _HAND_MODEL = "hand_landmarker.task"
            _POSE_MODEL = "pose_landmarker_lite.task"

            if not os.path.exists(_HAND_MODEL):
                print("[Camera] Downloading hand_landmarker.task …")
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/"
                    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    _HAND_MODEL,
                )
                print("[Camera] hand_landmarker.task downloaded")

            if not os.path.exists(_POSE_MODEL):
                print("[Camera] Downloading pose_landmarker_lite.task …")
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/"
                    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                    _POSE_MODEL,
                )
                print("[Camera] pose_landmarker_lite.task downloaded")

            hand_opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_HAND_MODEL),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            hand_detector = mp_vision.HandLandmarker.create_from_options(hand_opts)

            pose_opts = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_POSE_MODEL),
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            pose_detector = mp_vision.PoseLandmarker.create_from_options(pose_opts)
            print("[Camera] MediaPipe Tasks API ready (hand + pose)")

        except Exception as e:
            # Try legacy solutions API (older mediapipe / non-Apple-Silicon)
            print(f"[Camera] Tasks API failed ({e}) — trying legacy solutions API …")
            try:
                import mediapipe as mp
                _legacy_pose  = mp.solutions.pose.Pose(
                    min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
                _legacy_hands = mp.solutions.hands.Hands(
                    max_num_hands=2, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5, model_complexity=0)
                hand_detector = ("legacy", _legacy_hands)
                pose_detector = ("legacy", _legacy_pose)
                print("[Camera] MediaPipe legacy solutions API ready")
            except Exception as e2:
                print(f"[Camera] MediaPipe not available ({e2}) — running YOLO-only (no gestures)")

        _det_status = "ok" if hand_detector else "ok (YOLO only — no gestures)"
        print(f"[Camera] Detection status: {_det_status}")
        _push_status()
        last_detect = 0.0

        while not self._stop.is_set():
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
                self._run_detection(frame, yolo, hand_detector, pose_detector)
            except Exception:
                print(f"[Camera] Detection error:\n{traceback.format_exc()}")
                time.sleep(1.0)

        # Cleanup
        try:
            if isinstance(hand_detector, tuple):
                hand_detector[1].close()
                pose_detector[1].close()
            elif hand_detector:
                hand_detector.close()
                if pose_detector:
                    pose_detector.close()
        except Exception:
            pass
        print("[Camera] Detection stopped")

    # ── Gesture classifier (works for both Tasks API and legacy landmarks) ────

    @staticmethod
    def _classify_gesture_lm(lm, is_right: bool) -> str:
        """
        Classify from a list of 21 landmark objects with .x .y .z

        Landmark indices (MediaPipe Hand):
          0=wrist  4=thumb tip  3=thumb IP  2=thumb MCP
          5=index MCP  8=index tip
          9=middle MCP 12=middle tip
          13=ring MCP  16=ring tip
          17=pinky MCP 20=pinky tip
        """
        # ── Finger extension (tip above its MCP knuckle in image coords) ──────
        ix = lm[8].y  < lm[5].y
        mi = lm[12].y < lm[9].y
        ri = lm[16].y < lm[13].y
        pi = lm[20].y < lm[17].y
        no_fingers = not ix and not mi and not ri and not pi

        # ── Thumb: sideways extension (used for Open Hand / Call Me / Fist) ───
        th_side = lm[4].x < lm[3].x if is_right else lm[4].x > lm[3].x

        # ── Thumbs Up / Down: thumb vertical, all fingers curled ─────────────
        # Normalise by wrist→middle-MCP distance so it works at any hand size
        hand_h = abs(lm[0].y - lm[9].y) + 1e-6
        if no_fingers:
            if lm[4].y < lm[0].y - hand_h * 0.25:   # tip well ABOVE wrist
                return "Thumbs Up"
            if lm[4].y > lm[0].y + hand_h * 0.25:   # tip well BELOW wrist
                return "Thumbs Down"

        # ── Other gestures ────────────────────────────────────────────────────
        if ix and mi and ri and pi:                            return "Open Hand"
        if not ix and not mi and not ri and not pi:            return "Fist"
        if ix and mi and not ri and not pi:                    return "Peace"
        if ix and not mi and not ri and not pi:                return "Pointing"
        if ix and pi and not mi and not ri:                    return "Rock On"
        if th_side and pi and not ix and not mi and not ri:    return "Call Me"
        return "Custom"

    @staticmethod
    def _classify_gesture(hand_lm, handedness) -> str:
        """Legacy solutions API wrapper."""
        lm       = hand_lm.landmark
        is_right = handedness.classification[0].label == "Right"
        return CameraDetector._classify_gesture_lm(lm, is_right)

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

    def _run_detection(self, frame, yolo, hand_detector, pose_detector):
        import mediapipe as mp
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "objects": [], "persons": 0, "gestures": [],
        }

        # ── YOLO ─────────────────────────────────────────────────────────────
        results = yolo(frame, conf=YOLO_CONF, verbose=False)[0]
        for box in results.boxes:
            label = yolo.names[int(box.cls[0])]
            conf  = round(float(box.conf[0]), 2)
            if label in YOLO_IGNORE:
                continue
            if label == "person":
                event["persons"] += 1
            else:
                event["objects"].append({"label": label, "conf": conf})

        # ── MediaPipe ─────────────────────────────────────────────────────────
        if hand_detector is not None:
            is_legacy = isinstance(hand_detector, tuple)

            if is_legacy:
                # Legacy solutions API
                _, legacy_hands = hand_detector
                _, legacy_pose  = pose_detector
                pose_res = legacy_pose.process(rgb)
                if pose_res.pose_landmarks and event["persons"] == 0:
                    event["persons"] = 1
                hand_res = legacy_hands.process(rgb)
                if hand_res.multi_hand_landmarks:
                    for lm, hd in zip(hand_res.multi_hand_landmarks,
                                      hand_res.multi_handedness):
                        g    = self._classify_gesture(lm, hd)
                        side = hd.classification[0].label
                        event["gestures"].append({"hand": side, "gesture": g})
            else:
                # New Tasks API (mediapipe ≥ 0.10, Apple Silicon)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                if pose_detector:
                    pose_res = pose_detector.detect(mp_image)
                    if pose_res.pose_landmarks and event["persons"] == 0:
                        event["persons"] = 1

                hand_res = hand_detector.detect(mp_image)
                for i, lm_list in enumerate(hand_res.hand_landmarks):
                    try:
                        hand_info = hand_res.handedness[i]
                        is_right  = hand_info[0].category_name == "Right"
                        g    = self._classify_gesture_lm(lm_list, is_right)
                        side = "Right" if is_right else "Left"
                        event["gestures"].append({"hand": side, "gesture": g})
                    except Exception:
                        pass

        snap = (
            event["persons"],
            tuple(sorted(o["label"] for o in event["objects"])),
            tuple(sorted(f"{g['hand']}:{g['gesture']}" for g in event["gestures"])),
        )
        has_det = event["objects"] or event["persons"] or event["gestures"]
        if has_det and snap != self._last_snapshot:
            self.on_detection(event)
        self._last_snapshot = snap if has_det else None
