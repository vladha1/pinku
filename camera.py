"""
Camera — two local YOLO models, no MediaPipe.

  YOLO_MODEL    (yolov8n.pt)  — person + object detection
  GESTURE_MODEL               — hand gesture classification (HaGRID dataset)
                                auto-downloaded from HuggingFace on first run

Runs two background threads:
  1. Capture thread  — reads frames into _last_frame (fast, no heavy libs)
  2. Detect thread   — runs both YOLO models every DETECT_EVERY seconds

macOS: uses cv2.CAP_AVFOUNDATION backend; auto-scans indices 0-3 if 0 fails.
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
)

# ── Pose keypoint indices (COCO 17-point) ─────────────────────────────────────
# 0:nose  5:l-shoulder 6:r-shoulder  7:l-elbow  8:r-elbow
# 9:l-wrist 10:r-wrist 11:l-hip 12:r-hip
_KP_NOSE    = 0
_KP_L_SHO   = 5;  _KP_R_SHO   = 6
_KP_L_ELB   = 7;  _KP_R_ELB   = 8
_KP_L_WRI   = 9;  _KP_R_WRI   = 10
_KP_L_HIP   = 11; _KP_R_HIP   = 12

# ── Shared state ──────────────────────────────────────────────────────────────
_frame_lock  = threading.Lock()
_last_frame: np.ndarray | None = None

_cam_status  = "starting"
_det_status  = "starting"


def get_frame() -> np.ndarray | None:
    with _frame_lock:
        return _last_frame.copy() if _last_frame is not None else None


def get_status() -> dict:
    return {"camera": _cam_status, "detection": _det_status}


def _push_status():
    try:
        import dashboard
        dashboard.push_camera_status(_cam_status, _det_status)
    except Exception:
        pass


def frame_to_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


# ── Camera open helper ────────────────────────────────────────────────────────

def _open_camera(preferred_index: int) -> cv2.VideoCapture | None:
    indices = list(dict.fromkeys([preferred_index, 0, 1, 2, 3]))
    for idx in indices:
        for backend in [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release(); continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
                got_frame = False
                for _ in range(10):
                    ret, _ = cap.read()
                    if ret: got_frame = True; break
                    time.sleep(0.1)
                if got_frame:
                    bname = "AVFoundation" if backend == cv2.CAP_AVFOUNDATION else "default"
                    print(f"[Camera] Opened camera index={idx} backend={bname}")
                    return cap
                cap.release()
            except Exception as e:
                print(f"[Camera] index={idx} backend={backend}: {e}")
    return None


# ── Detection thread ──────────────────────────────────────────────────────────

class CameraDetector:
    """
    Two daemon threads: capture + detect.
    Callback: on_detection(event) — {timestamp, objects, persons, gestures}
    """

    def __init__(self, on_detection=None):
        self.on_detection   = on_detection or (lambda e: None)
        self._stop          = threading.Event()
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
            print("[Camera] ERROR: no camera found — check System Settings → Privacy → Camera")
            _cam_status = "error: no camera found"
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
                    print("[Camera] Too many failures — reopening …")
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

        # ── Object / person YOLO ──────────────────────────────────────────────
        try:
            from ultralytics import YOLO
            print(f"[Camera] Loading object model {YOLO_MODEL} …")
            yolo = YOLO(YOLO_MODEL)
            print("[Camera] Object YOLO ready")
        except Exception as e:
            print(f"[Camera] Object YOLO failed: {e}")
            _det_status = f"error: {e}"
            _push_status()
            return

        # ── Gesture YOLO (HaGRID) ─────────────────────────────────────────────
        # ── Pose YOLO for gesture detection ───────────────────────────────────
        # yolov8n-pose.pt is an official ultralytics model — auto-downloads
        # from GitHub, no authentication required.
        pose_model = None
        try:
            print("[Camera] Loading pose model yolov8n-pose.pt …")
            pose_model = YOLO("yolov8n-pose.pt")
            print("[Camera] Pose YOLO ready — gesture detection from wrist/shoulder keypoints")
        except Exception as e:
            print(f"[Camera] Pose YOLO not available ({e}) — no gesture detection")

        _det_status = "ok" if pose_model else "ok (no gesture model)"
        print(f"[Camera] Detection status: {_det_status}")
        _push_status()

        last_detect = 0.0
        while not self._stop.is_set():
            frame = get_frame()
            if frame is None:
                time.sleep(0.1); continue

            now = time.time()
            if now - last_detect < DETECT_EVERY:
                time.sleep(0.05); continue
            last_detect = now

            try:
                self._run_detection(frame, yolo, pose_model)
            except Exception:
                print(f"[Camera] Detection error:\n{traceback.format_exc()}")
                time.sleep(1.0)

        print("[Camera] Detection stopped")

    # ── Gesture from pose keypoints ───────────────────────────────────────────

    @staticmethod
    def _gesture_from_pose(kps) -> str | None:
        """
        Classify a gesture from 17 COCO pose keypoints.
        kps: array of shape (17, 3) — [x, y, confidence] per keypoint.

        Pose only gives wrist position — not finger data — so we only
        detect what's actually reliable: arm raised = Open Hand (wake).
        """
        def vis(i): return kps[i][2] > 0.3
        def y(i):   return kps[i][1]   # increases downward

        if not (vis(_KP_L_SHO) or vis(_KP_R_SHO)):
            return None   # shoulders not visible — can't classify

        l_raised = vis(_KP_L_WRI) and vis(_KP_L_SHO) and y(_KP_L_WRI) < y(_KP_L_SHO)
        r_raised = vis(_KP_R_WRI) and vis(_KP_R_SHO) and y(_KP_R_WRI) < y(_KP_R_SHO)

        if l_raised or r_raised:
            return "Open Hand"   # arm raised = wave = wake Pinku

        return None

    # ── Combined detection ────────────────────────────────────────────────────

    def _run_detection(self, frame, yolo, pose_model):
        event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "objects": [], "persons": 0, "gestures": [],
        }

        # ── Object / person detection ─────────────────────────────────────────
        for box in yolo(frame, conf=YOLO_CONF, verbose=False)[0].boxes:
            label = yolo.names[int(box.cls[0])]
            conf  = round(float(box.conf[0]), 2)
            if label in YOLO_IGNORE:
                continue
            if label == "person":
                event["persons"] += 1
            else:
                event["objects"].append({"label": label, "conf": conf})

        # ── Pose / gesture detection ──────────────────────────────────────────
        if pose_model is not None:
            pose_res = pose_model(frame, conf=0.5, verbose=False)[0]
            if pose_res.keypoints is not None:
                for kps in pose_res.keypoints.data:   # one set of kps per person
                    kps_np = kps.cpu().numpy()
                    if event["persons"] == 0:
                        event["persons"] = 1          # pose confirms a person
                    gesture = self._gesture_from_pose(kps_np)
                    if gesture:
                        event["gestures"].append({"gesture": gesture})

        # ── Dedup — only fire callback when scene changes ─────────────────────
        snap = (
            event["persons"],
            tuple(sorted(o["label"] for o in event["objects"])),
            tuple(sorted(g["gesture"] for g in event["gestures"])),
        )
        has_det = event["objects"] or event["persons"] or event["gestures"]
        if has_det and snap != self._last_snapshot:
            self.on_detection(event)
        self._last_snapshot = snap if has_det else None
