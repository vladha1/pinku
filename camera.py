"""
Camera — YOLO pose + OpenCV hand gesture classification.

  YOLO_MODEL       (yolov8n.pt)       — object detection (persons counted via face only)
  yolov8n-pose.pt                     — body pose; nose keypoint = face present
  Hand gesture CV                     — skin mask + convexity defects on wrist crop

Gestures detected:
  Open Hand   — 4+ finger gaps (wave / open palm) → wake
  Fist        — closed hand, high solidity         → sleep / mute

macOS: cv2.CAP_AVFOUNDATION backend, auto-scans indices 0-3.
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
    YOLO_MODEL, YOLO_CONF, YOLO_IGNORE, DETECT_EVERY, MIN_SHOULDER_PX,
)

# ── COCO 17-point pose keypoint indices ───────────────────────────────────────
_KP_NOSE  = 0
_KP_L_SHO = 5;  _KP_R_SHO = 6
_KP_L_ELB = 7;  _KP_R_ELB = 8
_KP_L_WRI = 9;  _KP_R_WRI = 10
_KP_L_HIP = 11; _KP_R_HIP = 12

# ── Shared state ──────────────────────────────────────────────────────────────
_frame_lock = threading.Lock()
_last_frame: np.ndarray | None = None
_cam_status = "starting"
_det_status = "starting"


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


# ── Hand gesture via skin mask + convexity defects ────────────────────────────

def _classify_hand(frame: np.ndarray, wx: float, wy: float) -> str | None:
    """
    Crop a region around the wrist (wx, wy) and classify the hand gesture
    using skin segmentation + convex hull analysis.

    Returns gesture label or None if hand not clearly visible.
    """
    h, w = frame.shape[:2]
    sz  = max(90, int(h * 0.22))       # crop ~22% of frame height
    x1  = max(0, int(wx) - sz // 2)
    y1  = max(0, int(wy) - sz // 2)
    x2  = min(w, x1 + sz)
    y2  = min(h, y1 + sz)
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] < 30 or roi.shape[1] < 30:
        return None

    # ── Skin mask in YCrCb (robust across skin tones + indoor lighting) ───────
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    mask  = cv2.inRange(ycrcb,
                        np.array([0,  133,  77], np.uint8),
                        np.array([255, 173, 127], np.uint8))
    k    = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.dilate(mask, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt  = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    # Must cover at least 8% of crop — filters out noise
    if area < roi.shape[0] * roi.shape[1] * 0.08:
        return None

    # ── Convex hull + defects ─────────────────────────────────────────────────
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if len(hull_idx) < 3:
        return None
    hull_pts = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull_pts)
    if hull_area < 1:
        return None
    solidity = area / hull_area      # 1.0 = perfectly convex (fist)

    defects = cv2.convexityDefects(cnt, hull_idx)
    n_gaps  = 0
    if defects is not None:
        for d in defects:
            _, _, _, depth = d[0]
            # depth is in 8.8 fixed-point → divide by 256 for pixels
            if depth / 256.0 > sz * 0.08:   # gap > 8% of crop size
                n_gaps += 1

    # ── Classify ──────────────────────────────────────────────────────────────
    # Only report Open Hand — a clearly open palm with 4+ finger gaps.
    # Fist is too common a false positive (normal resting hand = zero gaps, high solidity).
    if n_gaps >= 4:
        return "Open Hand"

    return None


# ── Detection class ───────────────────────────────────────────────────────────

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
            print("[Camera] ERROR: no camera found")
            _cam_status = "error: no camera found"
            _push_status(); return

        _cam_status = "ok"
        _push_status()
        consec_fail = 0

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                consec_fail += 1
                if consec_fail > 30:
                    cap.release()
                    cap = _open_camera(CAMERA_INDEX)
                    if cap is None:
                        _cam_status = "error: camera disconnected"
                        _push_status(); break
                    consec_fail = 0
                time.sleep(0.05); continue
            consec_fail = 0
            with _frame_lock:
                _last_frame = frame.copy()

        cap.release()
        print("[Camera] Capture stopped")

    # ── Detect loop ───────────────────────────────────────────────────────────

    def _detect_loop(self):
        global _det_status
        print("[Camera] Starting detection …")

        try:
            from ultralytics import YOLO
            print(f"[Camera] Loading object model {YOLO_MODEL} …")
            yolo = YOLO(YOLO_MODEL)
            print("[Camera] Object YOLO ready")
        except Exception as e:
            _det_status = f"error: {e}"; _push_status(); return

        pose_model = None
        try:
            print("[Camera] Loading pose model yolov8n-pose.pt …")
            pose_model = YOLO("yolov8n-pose.pt")
            print("[Camera] Pose YOLO ready")
        except Exception as e:
            print(f"[Camera] Pose model unavailable ({e})")

        _det_status = "ok" if pose_model else "ok (no pose model)"
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

    # ── Combined detection ────────────────────────────────────────────────────

    def _run_detection(self, frame, yolo, pose_model):
        h = frame.shape[0]
        event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "objects": [], "persons": 0, "gestures": [],
        }

        # ── Object detection (YOLO) — persons counted separately via face ────────
        yolo_person_seen = False
        for box in yolo(frame, conf=YOLO_CONF, verbose=False)[0].boxes:
            label = yolo.names[int(box.cls[0])]
            conf  = round(float(box.conf[0]), 2)
            if label in YOLO_IGNORE:
                continue
            if label == "person":
                yolo_person_seen = True   # used as fallback only if pose model absent
            else:
                event["objects"].append({"label": label, "conf": conf})

        # ── Pose + face detection + hand gesture ──────────────────────────────
        # Person is only counted when the nose keypoint (face) is visible —
        # much more reliable than YOLO "person" which fires on partial bodies.
        # Falls back to YOLO person count if pose model not loaded.
        if pose_model is not None:
            pose_res = pose_model(frame, conf=0.5, verbose=False)[0]
            if pose_res.keypoints is not None:
                for kps in pose_res.keypoints.data:
                    kps_np = kps.cpu().numpy()   # (17, 3)

                    def vis(i): return kps_np[i][2] > 0.3
                    def kx(i):  return float(kps_np[i][0])
                    def ky(i):  return float(kps_np[i][1])

                    # Only count as present if face (nose) is visible AND
                    # the person is close enough (inter-shoulder width ≥ MIN_SHOULDER_PX).
                    # This prevents distant faces (across the room, on TV) from waking Pinku.
                    if vis(_KP_NOSE):
                        if vis(_KP_L_SHO) and vis(_KP_R_SHO):
                            shoulder_w = abs(kx(_KP_R_SHO) - kx(_KP_L_SHO))
                            close_enough = shoulder_w >= MIN_SHOULDER_PX
                        else:
                            # Shoulders not detected — use nose keypoint confidence as proxy
                            # (high conf usually means the person is close/clear)
                            close_enough = float(kps_np[_KP_NOSE][2]) >= 0.75
                        if close_enough:
                            event["persons"] += 1

                    # Check each visible wrist
                    for wri, sho in [(_KP_R_WRI, _KP_R_SHO), (_KP_L_WRI, _KP_L_SHO)]:
                        if not vis(wri):
                            continue
                        # Classify hand shape at this wrist
                        gesture = _classify_hand(frame, kx(wri), ky(wri))
                        if gesture:
                            # Upgrade Open Hand to Thumbs Up if arm is fully raised
                            if gesture == "Open Hand" and vis(sho) and ky(wri) < ky(sho):
                                gesture = "Open Hand"   # arm raised + open palm
                            event["gestures"].append({"gesture": gesture})
                            break   # one gesture per frame is enough

        else:
            # No pose model — fall back to YOLO person detection
            if yolo_person_seen:
                event["persons"] = 1

        # ── Dedup ─────────────────────────────────────────────────────────────
        snap = (
            event["persons"],
            tuple(sorted(o["label"] for o in event["objects"])),
            tuple(sorted(g["gesture"] for g in event["gestures"])),
        )
        has_det = event["objects"] or event["persons"] or event["gestures"]
        if has_det and snap != self._last_snapshot:
            self.on_detection(event)
        self._last_snapshot = snap if has_det else None
