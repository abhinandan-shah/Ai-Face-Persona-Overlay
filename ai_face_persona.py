import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False

mp_face = mp.solutions.face_mesh

LEFT_MOUTH = 61
RIGHT_MOUTH = 291
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

SCREENSHOT_DIR = "screenshots"
MODEL_PATH = "models/emotion.onnx"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

EMO_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
INPUT_SIZE = (64, 64)

onnx_session = None
if ONNX_AVAILABLE and os.path.isfile(MODEL_PATH):
    try:
        onnx_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        print("Loaded ONNX model:", MODEL_PATH)
    except Exception as e:
        print("Failed loading ONNX model:", e)
        onnx_session = None
elif not ONNX_AVAILABLE and os.path.isfile(MODEL_PATH):
    print("ONNX model found but onnxruntime not installed. Install with: pip install onnxruntime")
else:
    print("No ONNX model found (or onnxruntime not installed). Using heuristics.")

def landmarks_to_xy(landmark, frame_w, frame_h):
    return int(landmark.x * frame_w), int(landmark.y * frame_h)

def mouth_aspect_ratio(landmarks, w, h):
    (x_l, y_l) = landmarks_to_xy(landmarks[LEFT_MOUTH], w, h)
    (x_r, y_r) = landmarks_to_xy(landmarks[RIGHT_MOUTH], w, h)
    horizontal = np.hypot(x_r - x_l, y_r - y_l) + 1e-6
    (x_u, y_u) = landmarks_to_xy(landmarks[UPPER_LIP], w, h)
    (x_lo, y_lo) = landmarks_to_xy(landmarks[LOWER_LIP], w, h)
    vertical = np.hypot(x_lo - x_u, y_lo - y_u) + 1e-6
    mar = vertical / horizontal
    return mar, (x_l, y_l, x_r, y_r), horizontal, vertical

def eye_opening_ratio(landmarks, top_idx, bottom_idx, w, h):
    (xt, yt) = landmarks_to_xy(landmarks[top_idx], w, h)
    (xb, yb) = landmarks_to_xy(landmarks[bottom_idx], w, h)
    return np.hypot(xt - xb, yt - yb)

def crop_face_square(frame, bbox, pad=0.25, size=INPUT_SIZE):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)
    pad_px = int(max(fw, fh) * pad)
    cx1 = max(0, x1 - pad_px)
    cy1 = max(0, y1 - pad_px)
    cx2 = min(w, x2 + pad_px)
    cy2 = min(h, y2 + pad_px)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    m = max(ch, cw)
    top = (m - ch) // 2
    bottom = m - ch - top
    left = (m - cw) // 2
    right = m - cw - left
    crop_square = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    resized = cv2.resize(crop_square, size)
    return resized

def predict_emotion_onnx(face_img):
    if onnx_session is None:
        return None, 0.0
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    arr = gray.astype(np.float32) / 255.0
    try:
        inp_name = onnx_session.get_inputs()[0].name
        input_arr = arr[np.newaxis, np.newaxis, :, :].astype(np.float32)
        out = onnx_session.run(None, {inp_name: input_arr})[0]
    except Exception:
        input_arr = arr[np.newaxis, :, :, np.newaxis].astype(np.float32)
        out = onnx_session.run(None, {inp_name: input_arr})[0]
    if out.ndim == 2:
        logits = out[0]
    elif out.ndim == 1:
        logits = out
    else:
        logits = out.flatten()
    exps = np.exp(logits - np.max(logits))
    probs = exps / (exps.sum() + 1e-9)
    idx = int(np.argmax(probs))
    label = EMO_CLASSES[idx] if idx < len(EMO_CLASSES) else "unknown"
    return label, float(probs[idx])

class Calibrator:
    def __init__(self):
        self.samples = []
        self.active = False
    def start(self):
        self.samples = []
        self.active = True
        self.start_time = time.time()
    def add(self, data):
        if not self.active:
            return
        self.samples.append(data)
    def stop(self):
        self.active = False
        if not self.samples:
            return None
        arr = np.array(self.samples)
        baselines = np.median(arr, axis=0)
        return {
            'mar_base': float(baselines[0]),
            'mouthw_base': float(baselines[1]),
            'eye_open_base': float(baselines[2])
        }

class LabelSmoother:
    def __init__(self, maxlen=8):
        self.maxlen = maxlen
        self.history = []
    def add(self, label, conf):
        self.history.append((label, conf))
        if len(self.history) > self.maxlen:
            self.history.pop(0)
    def get(self):
        if not self.history:
            return "No face", 0.0
        uniq = {}
        for l, c in self.history:
            uniq.setdefault(l, []).append(c)
        best = None
        best_score = -1.0
        for l, confs in uniq.items():
            score = len(confs) + 0.5 * (sum(confs)/len(confs))
            if score > best_score:
                best_score = score
                best = (l, float(sum(confs)/len(confs)))
        return best

calibrator = Calibrator()
smoother = LabelSmoother(maxlen=10)
neutral_baseline = None

def heuristic_emotion_with_baseline(landmarks, bbox, w, h, baseline=None):
    mar, mouth_coords, mouth_horiz, mouth_vert = mouth_aspect_ratio(landmarks, w, h)
    left_eye = eye_opening_ratio(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, w, h)
    right_eye = eye_opening_ratio(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, w, h)
    eye_avg_px = (left_eye + right_eye) / 2.0
    face_w = max(1, bbox[2] - bbox[0])
    mouth_w_px = np.hypot(mouth_coords[2]-mouth_coords[0], mouth_coords[3]-mouth_coords[1])
    rel_mouth_w = mouth_w_px / face_w
    rel_eye_open = eye_avg_px / face_w
    if baseline:
        mar_base = baseline.get('mar_base', 0.28)
        mouthw_base = baseline.get('mouthw_base', 0.35)
        eye_base = baseline.get('eye_open_base', 0.05)
    else:
        mar_base = 0.28
        mouthw_base = 0.36
        eye_base = 0.05
    mar_rel = mar / (mar_base + 1e-6)
    mouthw_rel = rel_mouth_w / (mouthw_base + 1e-6)
    eye_rel = rel_eye_open / (eye_base + 1e-6)
    if mar_rel > 1.7 or eye_rel > 1.6:
        return "surprise", min(0.98, 0.6 + 0.4*(mar_rel-1.7))
    if mouthw_rel > 1.35 and mar_rel < 1.15:
        return "happy", min(0.97, 0.6 + 0.5*(mouthw_rel-1.35))
    if mar_rel > 1.15 and mar_rel <= 1.7:
        return "fear/talking", 0.6
    if eye_rel < 0.5 and mouthw_rel < 0.85:
        return "angry/disgust", 0.75
    if mouthw_rel < 0.9 and eye_rel <= 1.0:
        return "sad", 0.6
    if eye_rel < 0.25:
        return "sleepy", 0.9
    return "neutral", 0.55

def draw_rounded_rect(img, tl, br, color, radius=12, thickness=-1, alpha=0.6):
    overlay = img.copy()
    x1, y1 = tl
    x2, y2 = br
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def neon_text(img, text, org, scale=0.8, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = org
    for i in range(5, 0, -1):
        cv2.putText(img, text, (x, y), font, scale, (30, 180, 255), thickness + i, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (230, 230, 255), thickness, cv2.LINE_AA)

def save_screenshot(img, prefix='screenshot'):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = os.path.join(SCREENSHOT_DIR, f"{prefix}_{ts}.png")
    cv2.imwrite(fname, img)
    return fname

def main():
    global neutral_baseline
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    prev_time = time.time()
    fps = 0
    smoothing_label = None
    label_fade = 1.0
    dl_enabled = bool(onnx_session)

    print("Controls: ESC quit | S save | D toggle DL | C calibrate neutral baseline (2s)")

    calibrating = False
    cal_start = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        label_text = "No face"
        label_conf = 0.0
        bbox = None

        if calibrating and time.time() - cal_start > 2.0:
            neutral_baseline = calibrator.stop()
            calibrating = False
            print("Calibration done. Baseline:", neutral_baseline)

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            landmarks = results.multi_face_landmarks[0].landmark
            xs = [p.x for p in landmarks]
            ys = [p.y for p in landmarks]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bbox = (int(min_x * w), int(min_y * h), int(max_x * w), int(max_y * h))

            mar, mouth_coords, _, _ = mouth_aspect_ratio(landmarks, w, h)
            mouth_w_px = np.hypot(mouth_coords[2]-mouth_coords[0], mouth_coords[3]-mouth_coords[1])
            left_eye_px = eye_opening_ratio(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, w, h)
            right_eye_px = eye_opening_ratio(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, w, h)
            eye_avg_px = (left_eye_px + right_eye_px) / 2.0
            face_w_px = max(1, bbox[2] - bbox[0])
            rel_mouth_w = mouth_w_px / face_w_px
            rel_eye_open = eye_avg_px / face_w_px

            if calibrating:
                calibrator.add([mar, rel_mouth_w, rel_eye_open])

            label_from_dl = None
            if dl_enabled and onnx_session is not None:
                face_crop = crop_face_square(frame, bbox, pad=0.25, size=INPUT_SIZE)
                if face_crop is not None:
                    try:
                        label_from_dl, conf_from_dl = predict_emotion_onnx(face_crop)
                        if label_from_dl is not None:
                            label_text = label_from_dl
                            label_conf = conf_from_dl
                        else:
                            label_text, label_conf = heuristic_emotion_with_baseline(landmarks, bbox, w, h, baseline=neutral_baseline)
                    except Exception:
                        label_text, label_conf = heuristic_emotion_with_baseline(landmarks, bbox, w, h, baseline=neutral_baseline)
            else:
                label_text, label_conf = heuristic_emotion_with_baseline(landmarks, bbox, w, h, baseline=neutral_baseline)

            smoother.add(label_text, label_conf)
            sm_label, sm_conf = smoother.get()
            label_text, label_conf = sm_label, sm_conf

            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 200, 255), 1)
            try:
                xl, yl, xr, yr = mouth_coords
                cv2.circle(frame, (xl, yl), 3, (30, 200, 255), -1)
                cv2.circle(frame, (xr, yr), 3, (30, 200, 255), -1)
            except Exception:
                pass
        else:
            smoother.add("No face", 0.0)
            label_text, label_conf = smoother.get()

        hud_w, hud_h = 480, 150
        hud_x, hud_y = 18, 18
        hud_br = (hud_x + hud_w, hud_y + hud_h)
        hud_color = (10, 120, 150)
        draw_rounded_rect(frame, (hud_x, hud_y), hud_br, hud_color, radius=18, thickness=-1, alpha=0.24)
        scanline_y = hud_y + int((time.time() * 120) % hud_h)
        cv2.line(frame, (hud_x + 8, scanline_y), (hud_x + hud_w - 8, scanline_y), (80, 200, 255), 1)

        display_label = f"{label_text}"
        if label_conf > 0:
            display_label = f"{display_label} ({label_conf:.2f})"
        neon_text(frame, display_label, (hud_x + 24, hud_y + 62), scale=0.95, thickness=1)

        status = f"DL: {'ON' if dl_enabled else 'OFF'}  |  Cal: {'YES' if neutral_baseline else 'NO'}"
        cv2.putText(frame, status, (hud_x + 24, hud_y + 116), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = (fps * 0.9) + (1.0 / dt) * 0.1
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        cv2.imshow("AI Face Persona HUD (press C to calibrate)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("s") or key == ord("S"):
            fname = save_screenshot(frame)
            print("Saved", fname)
        elif key == ord("d"):
            if onnx_session is None:
                print("ONNX model not available; place model at:", MODEL_PATH)
            else:
                dl_enabled = not dl_enabled
                print("DL mode:", dl_enabled)
        elif key == ord("c") or key == ord("C"):
            print("Starting 2s calibration: please look neutral (no smile, eyes open natural)")
            calibrator.start()
            calibrating = True
            cal_start = time.time()

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
