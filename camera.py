"""
Camera — YOLO pose + OpenCV hand gesture classification.

  YOLO_MODEL       (yolov8n.pt)       — object detection (persons counted via face only)
  yolov8n-pose.pt                     — body pose; nose keypoint = face present
  Hand gesture CV                     — skin mask + convexity defects on wrist crop

Gestures detected:
  Hands Up    — open palm with wrist above shoulder → wake from muted / idle
  Open Hand   — 4+ finger gaps, arm down (wave)     → stop speech / extend session
  Fist        — closed hand, high solidity           → sleep / mute

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
    LASER_DETECT, LASER_HUE_LO, LASER_HUE_HI, LASER_SAT_MIN, LASER_VAL_MIN,
    LASER_POLL_SEC, LASER_MOVE_THRESH,
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

# Last known laser dots — drawn on camera.jpg overlay
_laser_overlay_lock = threading.Lock()
_laser_overlay_dots: list[dict] = []   # list of {px, py, r}

# Calibrated bullseye centre in normalised coords — updated by pinku.py
_laser_bull: tuple[float, float] = (0.5, 0.5)


def set_laser_bull(x: float, y: float):
    """Update the bullseye position drawn on the annotated camera feed."""
    global _laser_bull
    _laser_bull = (float(x), float(y))


def get_frame(annotated: bool = False) -> np.ndarray | None:
    """Return current frame. If annotated=True, draws laser dot + bullseye overlay."""
    with _frame_lock:
        frame = _last_frame.copy() if _last_frame is not None else None
    if frame is None or not annotated:
        return frame

    h, w = frame.shape[:2]

    # ── Bullseye calibration target ───────────────────────────────────────────
    # Rings match the scoring thresholds in pinku.py _dart_score().
    # dist formula: dist = sqrt((dx*2)²+(dy*2)²) / sqrt(2)
    # → pixel_rx = dist * w / sqrt(2),  pixel_ry = dist * h / sqrt(2)
    # Drawn as ellipses so they look circular on 4:3 frames.
    bx = int(_laser_bull[0] * w)
    by = int(_laser_bull[1] * h)
    _RINGS = [
        (0.035, 100, "BULL"),
        (0.09,   75, "75"),
        (0.16,   50, "50"),
        (0.24,   25, "25"),
        (0.35,   10, "10"),
    ]
    for i, (dist_t, score, lbl) in enumerate(_RINGS):
        rx = max(4, int(dist_t * w / 1.414))
        ry = max(4, int(dist_t * h / 1.414))
        thickness = -1 if i == 0 else 1
        cv2.ellipse(frame, (bx, by), (rx, ry), 0, 0, 360,
                    (0, 165, 255), thickness, cv2.LINE_AA)
        # Score label at right edge of each ring
        cv2.putText(frame, lbl, (bx + rx + 3, by + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1, cv2.LINE_AA)
    # Cross-hair (extends just past the outermost ring)
    arm = max(4, int(0.35 * w / 1.414)) + 12
    cv2.line(frame, (bx - arm, by), (bx + arm, by), (0, 165, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (bx, by - arm), (bx, by + arm), (0, 165, 255), 1, cv2.LINE_AA)

    # ── Detected laser dot(s) ─────────────────────────────────────────────────
    with _laser_overlay_lock:
        dots = list(_laser_overlay_dots)
    for d in dots:
        px, py = d.get("px", 0), d.get("py", 0)
        r  = max(10, int(d.get("r", 6) * 2.5))
        # Outer glow ring
        cv2.circle(frame, (px, py), r + 6, (0, 200, 0), 2, cv2.LINE_AA)
        # Bright filled dot
        cv2.circle(frame, (px, py), r, (0, 255, 80), -1, cv2.LINE_AA)
        # Cross-hair
        cv2.line(frame, (px - r - 8, py), (px + r + 8, py), (0, 255, 80), 1, cv2.LINE_AA)
        cv2.line(frame, (px, py - r - 8), (px, py + r + 8), (0, 255, 80), 1, cv2.LINE_AA)
        # Score label
        score_text = f"{d.get('score', '?')} pts" if 'score' in d else f"({d['x']:.2f},{d['y']:.2f})"
        cv2.putText(frame, score_text, (px + r + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 1, cv2.LINE_AA)
    return frame


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


# ── Green laser dot detection ─────────────────────────────────────────────────

def _detect_laser_dots(frame: np.ndarray, debug: bool = False) -> list[dict]:
    """
    Detect ALL green laser pointer dots via HSV masking.
    debug=True: log every candidate blob with its stats before filtering.
    """
    if not LASER_DETECT:
        return []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([LASER_HUE_LO, LASER_SAT_MIN, LASER_VAL_MIN], np.uint8),
        np.array([LASER_HUE_HI, 255,            255           ], np.uint8),
    )
    # Morphological close to merge adjacent pixels — avoids blurring tiny dots away
    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask_px = int(np.count_nonzero(mask))
    if debug:
        print(f"[Laser debug] H:{LASER_HUE_LO}-{LASER_HUE_HI}  S≥{LASER_SAT_MIN}  V≥{LASER_VAL_MIN}  →  {mask_px} px in mask")

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        if debug:
            print("[Laser debug] no contours found")
        return []

    h, w     = frame.shape[:2]
    frame_px = w * h
    dots: list[dict] = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        circ  = (4 * np.pi * area / (perim * perim)) if perim > 0 else 0
        M     = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        # Sample HSV at centroid for debug
        hx, hy = int(cx), int(cy)
        hx = max(0, min(w-1, hx)); hy = max(0, min(h-1, hy))
        h_val, s_val, v_val = hsv[hy, hx]

        if debug:
            print(f"[Laser debug]   blob area={area:.0f} circ={circ:.2f} "
                  f"HSV=({h_val},{s_val},{v_val}) at ({cx:.0f},{cy:.0f})  "
                  f"{'PASS' if area >= 8 and circ >= 0.2 else 'FAIL'}")

        if area < 8 or area > frame_px * 0.02:
            continue
        if circ < 0.20:
            continue

        dots.append({
            "x":  round(cx / w, 3),
            "y":  round(cy / h, 3),
            "r":  round(float(np.sqrt(area / np.pi)), 1),
            "px": int(cx),
            "py": int(cy),
        })

    dots.sort(key=lambda d: d["x"])
    return dots


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
        self._last_laser_key: tuple = ()   # dedup key: quantized dot positions
        self._laser_warmup  = 8            # ignore first N laser frames (lets LEDs settle)

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

        last_yolo  = 0.0
        last_laser = 0.0
        while not self._stop.is_set():
            frame = get_frame()
            if frame is None:
                time.sleep(0.05); continue
            now = time.time()

            # ── Laser: fast independent loop (~8 fps) ─────────────────────────
            if LASER_DETECT and now - last_laser >= LASER_POLL_SEC:
                last_laser = now
                try:
                    self._run_laser(frame)
                except Exception:
                    pass   # never let laser errors kill the loop

            # ── YOLO + pose: slow loop (every DETECT_EVERY seconds) ───────────
            if now - last_yolo >= DETECT_EVERY:
                last_yolo = now
                try:
                    self._run_detection(frame, yolo, pose_model)
                except Exception:
                    print(f"[Camera] Detection error:\n{traceback.format_exc()}")
                    time.sleep(1.0)
            else:
                time.sleep(0.05)

        print("[Camera] Detection stopped")

    # ── Laser-only fast path ──────────────────────────────────────────────────

    def _run_laser(self, frame):
        """Fast laser detection — runs ~8 fps independent of YOLO."""
        laser_dots = _detect_laser_dots(frame)

        # Warmup: consume first N frames silently so always-on LEDs settle
        # into the baseline key and are never reported as "appeared"
        if self._laser_warmup > 0:
            self._laser_warmup -= 1
            q = int(1 / max(LASER_MOVE_THRESH, 0.01))
            self._last_laser_key = tuple(
                (round(d["x"] * q), round(d["y"] * q)) for d in laser_dots
            )
            if self._laser_warmup == 0:
                print(f"[Laser] warmup done — baseline has {len(laser_dots)} dot(s) (LEDs/noise ignored)")
            return

        # Always update the camera overlay
        with _laser_overlay_lock:
            _laser_overlay_dots.clear()
            _laser_overlay_dots.extend(laser_dots)

        # Log every change (appear / move / disappear)
        q = int(1 / max(LASER_MOVE_THRESH, 0.01))
        laser_key = tuple(
            (round(d["x"] * q), round(d["y"] * q)) for d in laser_dots
        )
        if laser_key == self._last_laser_key:
            return
        self._last_laser_key = laser_key

        if laser_dots:
            coords = "  ".join(f"({d['x']:.2f},{d['y']:.2f}) r={d['r']:.0f}px"
                               for d in laser_dots)
            print(f"[Laser] {len(laser_dots)} dot(s): {coords}")
            try:
                import dashboard as _db
                _db.log_message("laser", f"🔴 {len(laser_dots)} dot(s) — {coords}")
            except Exception:
                pass
        else:
            print("[Laser] dot gone")
            try:
                import dashboard as _db
                _db.log_message("laser", "🔴 dot gone")
            except Exception:
                pass

        self.on_detection({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "objects": [], "persons": 0, "gestures": [],
            "laser": laser_dots,
        })

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
                            nose_conf = float(kps_np[_KP_NOSE][2])
                            close_enough = nose_conf >= 0.75
                        if close_enough:
                            event["persons"] += 1

                    # Check each visible wrist
                    for wri, sho in [(_KP_R_WRI, _KP_R_SHO), (_KP_L_WRI, _KP_L_SHO)]:
                        if not vis(wri):
                            continue
                        # Classify hand shape at this wrist
                        gesture = _classify_hand(frame, kx(wri), ky(wri))
                        if gesture:
                            # Distinguish Hands Up (wrist above shoulder) from Open Hand (arm down)
                            if gesture == "Open Hand" and vis(sho) and ky(wri) < ky(sho):
                                gesture = "Hands Up"   # arm raised above shoulder
                            event["gestures"].append({"gesture": gesture})
                            break   # one gesture per frame is enough

        else:
            # No pose model — fall back to YOLO person detection
            if yolo_person_seen:
                event["persons"] = 1

        # ── Dedup ─────────────────────────────────────────────────────────────
        # Dedup on persons + gestures only — objects are excluded so that
        # TV content changing (different objects on screen every 3 s) doesn't
        # spam the log or fire on_detection when nobody is actually present.
        snap = (
            event["persons"],
            tuple(sorted(g["gesture"] for g in event["gestures"])),
        )
        has_actionable = bool(event["persons"] or event["gestures"])
        if has_actionable and snap != self._last_snapshot:
            self.on_detection(event)
        self._last_snapshot = snap if has_actionable else None
