"""
Virtual makeup built on the existing 68-point landmark model (300W format).

Key features
- Reuses the trained Keras model at models/face_landmarks_300w.h5.
-.png overlays are NOT used; makeup is applied through polygon masks from landmarks.
- Eyeshadow is drawn on one eye, then mirrored to the other with landmark-based warping.
- Works for both images and webcam.
"""

import argparse
import copy
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
import tensorflow as tf

# =========================
# Paths / settings
# =========================
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "face_landmarks_300w.h5"
CASCADE_PATH = ROOT / "haarcascade_frontalface_default.xml"
IMG_SIZE = 224

# Orientation helpers
ROTATE_CODES = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# =========================
# 68-point landmark indices (300W/Multi-PIE convention)
# =========================
JAW = list(range(0, 17))
RIGHT_BROW = list(range(17, 22))   # viewer-left eyebrow
LEFT_BROW = list(range(22, 27))    # viewer-right eyebrow
NOSE = list(range(27, 36))
LEFT_EYE = list(range(36, 42))     # subject's right, viewer-left
RIGHT_EYE = list(range(42, 48))    # subject's left, viewer-right
LIPS_OUTER = list(range(48, 60))
LIPS_INNER = list(range(60, 68))


# =========================
# Utility helpers
# =========================
def detect_face_bbox(frame: np.ndarray, cascade: cv2.CascadeClassifier) -> Tuple[int, int, int, int]:
    """Return (x, y, w, h) for the largest detected face; fall back to full frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        h, w = frame.shape[:2]
        return 0, 0, w, h
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.12 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(frame.shape[1], x + w + pad)
    y1 = min(frame.shape[0], y + h + pad)
    return x0, y0, x1 - x0, y1 - y0


def predict_landmarks(frame: np.ndarray, model: tf.keras.Model, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Run the 68-point model on the face crop and map results back to the frame.
    Returns (68,2) float32 in absolute pixel coords.
    """
    x, y, w, h = bbox
    face = frame[y:y + h, x:x + w]
    if face.size == 0:
        h_full, w_full = frame.shape[:2]
        face = frame
        x, y, w, h = 0, 0, w_full, h_full

    # Letterbox the face to IMG_SIZE while preserving aspect ratio
    scale = min(IMG_SIZE / float(w), IMG_SIZE / float(h))
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(face, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    pad_x = (IMG_SIZE - new_w) // 2
    pad_y = (IMG_SIZE - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    face_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    raw = model.predict(face_rgb[None, ...], verbose=0)[0]

    # Handle interleaved vs split formats
    if raw.shape == (136,):
        xs, ys = raw[:68], raw[68:]
        cand1 = np.stack([xs, ys], axis=1)
        cand2 = raw.reshape(68, 2)
        var1 = np.var(cand1, axis=0).sum()
        var2 = np.var(cand2, axis=0).sum()
        pred = cand1 if var1 >= var2 else cand2
    elif raw.shape == (68, 2):
        pred = raw
    else:
        pred = raw.reshape(68, 2)

    pred = pred.astype(np.float32)

    # Decide normalization explicitly (no heuristics beyond thresholds)
    pmin, pmax = float(pred.min()), float(pred.max())
    print(f"[debug] raw preds min/max/mean/std: {pmin:.4f}/{pmax:.4f}/{pred.mean():.4f}/{pred.std():.4f} first5: {pred[:5]}")
    if 0.0 <= pmin and pmax <= 1.2:
        norm_case = "0..1"
        pred = pred * IMG_SIZE
    elif -1.2 <= pmin and pmax <= 1.2:
        norm_case = "-1..1"
        pred = (pred + 1.0) * 0.5 * IMG_SIZE
    else:
        norm_case = "pixels"
        # keep as-is
    print(f"[debug] normalization case: {norm_case}")

    # Detect x/y swap via variance
    var_orig = np.var(pred[:, 0]) + np.var(pred[:, 1])
    swapped = pred[:, ::-1].copy()
    var_swapped = np.var(swapped[:, 0]) + np.var(swapped[:, 1])
    if var_swapped > var_orig:
        pred = swapped

    # Undo letterbox to face-crop coordinates
    lms_face = np.zeros((68, 2), dtype=np.float32)
    lms_face[:, 0] = (pred[:, 0] - pad_x) / scale
    lms_face[:, 1] = (pred[:, 1] - pad_y) / scale

    # Map to full-frame coordinates
    landmarks = np.zeros((68, 2), dtype=np.float32)
    landmarks[:, 0] = x + lms_face[:, 0]
    landmarks[:, 1] = y + lms_face[:, 1]

    return landmarks

def polygon_mask(shape: Tuple[int, int, int], pts: np.ndarray, blur: int = 15) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = pts.astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 255)
    if blur > 0:
        k = max(3, int(blur) | 1)  # odd kernel
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def alpha_blend(frame: np.ndarray, color: Tuple[int, int, int], mask: np.ndarray, opacity: float) -> np.ndarray:
    alpha = (mask.astype(np.float32) / 255.0) * float(opacity)
    alpha = np.clip(alpha, 0.0, 1.0)
    overlay = np.full_like(frame, color, dtype=np.uint8)
    out = frame.astype(np.float32)
    out = out * (1.0 - alpha[..., None]) + overlay.astype(np.float32) * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def orient_frame(frame: np.ndarray, rotation: int = 0, mirror: bool = False) -> np.ndarray:
    """Rotate (0/90/180/270) then optional mirror to get a consistent processing orientation."""
    if rotation not in ROTATE_CODES:
        raise ValueError("rotation must be one of 0,90,180,270")
    out = frame
    if ROTATE_CODES[rotation] is not None:
        out = cv2.rotate(out, ROTATE_CODES[rotation])
    if mirror:
        out = cv2.flip(out, 1)
    return out


def deorient_frame(frame: np.ndarray, rotation: int = 0, mirror: bool = False) -> np.ndarray:
    """Invert orient_frame: undo mirror then rotate back."""
    out = frame
    if mirror:
        out = cv2.flip(out, 1)
    if rotation == 90:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif rotation == 270:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    return out


def letterbox_landmarks(lms: np.ndarray, src_shape: Tuple[int, int], dst_shape: Tuple[int, int]) -> np.ndarray:
    """
    Map landmarks from src image shape to a letterboxed canvas of dst_shape (w,h),
    useful when Flutter displays preview with aspect-fit and black bars.
    """
    src_h, src_w = src_shape
    dst_w, dst_h = dst_shape
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w, new_h = src_w * scale, src_h * scale
    pad_x = (dst_w - new_w) * 0.5
    pad_y = (dst_h - new_h) * 0.5
    out = lms.copy().astype(np.float32)
    out[:, 0] = out[:, 0] * scale + pad_x
    out[:, 1] = out[:, 1] * scale + pad_y
    return out


def region_light_adjust(frame: np.ndarray, pts: np.ndarray, base: float = 1.0) -> float:
    """Adjust opacity by local brightness so makeup is softer on bright skin."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    y_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    mean_y = cv2.mean(y_channel, mask=mask)[0] / 255.0
    return float(np.clip(base * (0.9 + (0.5 - mean_y) * 0.6), 0.35, 1.15))


# =========================
# Region builders
# =========================
def eyeshadow_band(lms: np.ndarray, eye_idx: list, brow_idx: list) -> np.ndarray:
    """
    Build a thin polygon strip between upper eyelid and brow.
    Uses upper lid arc, lifted toward brow but capped below the brow.
    """
    eye = lms[eye_idx]
    brow = lms[brow_idx]
    # upper lid: sort by x for stable arc
    upper = eye[[0, 1, 2, 3]]
    upper = upper[np.argsort(upper[:, 0])]
    eye_width = np.linalg.norm(eye[3] - eye[0])
    lift = max(4.0, eye_width * 0.12)
    top_limit = np.min(brow[:, 1]) - 2

    lifted = upper.copy()
    lifted[:, 1] = np.minimum(upper[:, 1] - lift, top_limit)

    band = np.vstack([upper, lifted[::-1]])
    return band


def eyebrow_polygon(lms: np.ndarray, idxs: list) -> np.ndarray:
    return cv2.convexHull(lms[idxs].astype(np.float32)).reshape(-1, 2)


# =========================
# Makeup layers
# =========================
def apply_lipstick(frame: np.ndarray, lms: np.ndarray, color=(90, 40, 180), opacity=0.55, feather=17) -> np.ndarray:
    outer = lms[LIPS_OUTER]
    inner = lms[LIPS_INNER]

    outer_mask = polygon_mask(frame.shape, outer, blur=feather)
    inner_mask = polygon_mask(frame.shape, inner, blur=feather // 2)
    lip_mask = cv2.subtract(outer_mask, inner_mask)

    alpha = opacity * region_light_adjust(frame, outer)
    painted = alpha_blend(frame, color, lip_mask, alpha)

    # subtle gloss on upper lip center
    gloss = np.zeros_like(frame)
    center_top = np.mean(outer[[2, 3, 4]], axis=0).astype(int)
    cv2.ellipse(gloss, tuple(center_top), (20, 8), 0, 0, 360, (235, 235, 255), -1)
    gloss_mask = polygon_mask(frame.shape, outer, blur=25)
    painted = alpha_blend(painted, (235, 235, 255), gloss_mask, opacity * 0.08)
    return painted


def apply_blush(frame: np.ndarray, lms: np.ndarray, color=(80, 65, 180), opacity=0.28, feather=49) -> np.ndarray:
    """
    Approximate cheek ellipses using jaw and nose anchors to stay on cheek mass.
    """
    # viewer-left cheek anchors (subject right)
    left_anchors = lms[[2, 3, 4, 31, 48]]
    right_anchors = lms[[13, 14, 15, 35, 54]]

    def ellipse_from_anchors(pts: np.ndarray):
        cx, cy = np.mean(pts, axis=0)
        span = np.ptp(pts, axis=0)
        ax = max(18.0, span[0] * 0.55)
        ay = max(14.0, span[1] * 0.60)
        return (int(cx), int(cy + 10)), (int(ax), int(ay))

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    center_l, axes_l = ellipse_from_anchors(left_anchors)
    center_r, axes_r = ellipse_from_anchors(right_anchors)

    cv2.ellipse(mask, center_l, axes_l, 0, 0, 360, 255, -1)
    cv2.ellipse(mask, center_r, axes_r, 0, 0, 360, 255, -1)

    face_mask = cv2.convexHull(lms.astype(np.float32)).astype(np.int32)
    face_mask_img = np.zeros_like(mask)
    cv2.fillConvexPoly(face_mask_img, face_mask, 255)
    mask = cv2.bitwise_and(mask, face_mask_img)

    k = max(3, int(feather) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    cheek_pts = np.vstack([left_anchors, right_anchors])
    alpha = opacity * region_light_adjust(frame, cheek_pts, base=1.0)
    return alpha_blend(frame, color, mask, alpha)


def apply_eyeshadow(frame: np.ndarray, lms: np.ndarray, color=(120, 85, 190), opacity=0.38, feather=35) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame)

    # Build band on LEFT_EYE (viewer-left), then mirror to RIGHT_EYE with affine warp.
    left_band = eyeshadow_band(lms, LEFT_EYE, LEFT_BROW)
    left_mask = polygon_mask(frame.shape, left_band, blur=feather)

    # Paint left eye
    overlay[:] = color
    out = alpha_blend(frame, color, left_mask, opacity * region_light_adjust(frame, left_band))

    # Mirror to right eye using landmark alignment (outer corner, inner corner, center)
    src_tri = np.float32([lms[36], lms[39], (lms[36] + lms[39]) / 2.0])
    dst_tri = np.float32([lms[42], lms[45], (lms[42] + lms[45]) / 2.0])
    M, _ = cv2.estimateAffinePartial2D(src_tri, dst_tri, method=cv2.LMEDS)

    warped_mask = cv2.warpAffine(left_mask, M, (w, h))
    warped_mask = np.clip(warped_mask, 0, 255).astype(np.uint8)
    out = alpha_blend(out, color, warped_mask, opacity * region_light_adjust(frame, lms[RIGHT_EYE]))
    return out


def apply_eyebrow_shading(frame: np.ndarray, lms: np.ndarray, color=(45, 35, 30), opacity=0.22, feather=15) -> np.ndarray:
    left = eyebrow_polygon(lms, LEFT_BROW)
    right = eyebrow_polygon(lms, RIGHT_BROW)

    left_mask = polygon_mask(frame.shape, left, blur=feather)
    right_mask = polygon_mask(frame.shape, right, blur=feather)

    out = alpha_blend(frame, color, left_mask, opacity)
    out = alpha_blend(out, color, right_mask, opacity)
    return out


def apply_highlighter(
    frame: np.ndarray,
    lms: np.ndarray,
    color=(230, 230, 255),
    opacity=0.26,
    feather=25,
) -> np.ndarray:
    """
    Adds soft highlights on brow bone and inner eye corners.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Brow bone centers
    left_brow_mid = np.mean(lms[LEFT_BROW], axis=0)
    right_brow_mid = np.mean(lms[RIGHT_BROW], axis=0)

    # Inner eye corners (tear duct)
    left_inner = lms[39]
    right_inner = lms[42]

    for center, axes in [
        (left_brow_mid, (18, 8)),
        (right_brow_mid, (18, 8)),
        (left_inner, (10, 6)),
        (right_inner, (10, 6)),
    ]:
        cx, cy = int(center[0]), int(center[1])
        ax, ay = axes
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    k = max(3, int(feather) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return alpha_blend(frame, color, mask, opacity)


# =========================
# Pipeline
# =========================
def render_makeup_layers(frame: np.ndarray, lms: np.ndarray, style: Dict) -> np.ndarray:
    result = frame.copy()
    result = apply_eyeshadow(result, lms, **style["eyeshadow"])
    result = apply_eyebrow_shading(result, lms, **style["eyebrow"])
    result = apply_highlighter(result, lms, **style["highlighter"])
    result = apply_blush(result, lms, **style["blush"])
    result = apply_lipstick(result, lms, **style["lipstick"])
    return result


def draw_debug_overlay(frame: np.ndarray, lms: np.ndarray, bbox, tight_bbox=None) -> np.ndarray:
    out = frame.copy()

    # Green = Haar bbox
    x, y, w, h = bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Blue = tight bbox computed from landmarks
    if tight_bbox is not None:
        tx, ty, tw, th = tight_bbox
        cv2.rectangle(out, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 2)

    # Red points
    for (px, py) in lms.astype(int):
        cv2.circle(out, (px, py), 3, (0, 0, 255), -1)

    # Yellow hull to visualize overall scale
    if len(lms) >= 3:
        hull = cv2.convexHull(lms.astype(np.int32))
        cv2.polylines(out, [hull], isClosed=True, color=(0, 255, 255), thickness=2)

    return out

def tighten_bbox_from_landmarks(lms: np.ndarray, frame_shape, margin=0.12):
    h, w = frame_shape[:2]
    x0 = max(0, int(np.min(lms[:, 0])))
    y0 = max(0, int(np.min(lms[:, 1])))
    x1 = min(w - 1, int(np.max(lms[:, 0])))
    y1 = min(h - 1, int(np.max(lms[:, 1])))

    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    pad = int(margin * max(bw, bh))

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    return (x0, y0, x1 - x0, y1 - y0)

def apply_makeup(
    frame: np.ndarray,
    model: tf.keras.Model,
    cascade: cv2.CascadeClassifier,
    style: Dict,
    debug: bool = False,
    return_landmarks: bool = False,
):
    bbox = detect_face_bbox(frame, cascade)
    lms1 = predict_landmarks(frame, model, bbox)

    tight_bbox = tighten_bbox_from_landmarks(lms1, frame.shape, margin=0.12)
    lms2 = predict_landmarks(frame, model, tight_bbox)

    debug_img=draw_debug_overlay(frame,lms2, bbox=bbox , tight_bbox=tight_bbox)
    return debug_img

    if debug:
        debug_img = draw_debug_overlay(frame, lms2, bbox, tight_bbox)
        cv2.putText(
            debug_img,
            "DEBUG LANDMARKS MODE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        print("DEBUG: returning landmarks + bbox only")
        if return_landmarks:
            return debug_img, lms2, tight_bbox
        return debug_img

    result = render_makeup_layers(frame, lms2, style)
    if return_landmarks:
        return result, lms2, tight_bbox
    return result


def apply_makeup_smoothed(
    frame: np.ndarray,
    model: tf.keras.Model,
    cascade: cv2.CascadeClassifier,
    style: Dict,
    prev_landmarks: Optional[np.ndarray],
    ema_momentum: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    bbox = detect_face_bbox(frame, cascade)
    lms = predict_landmarks(frame, model, bbox)

    if prev_landmarks is not None:
        m = float(np.clip(ema_momentum, 0.0, 0.95))
        lms = (prev_landmarks * m + lms * (1.0 - m)).astype(np.float32)

    return render_makeup_layers(frame, lms, style), lms


# =========================
# Public entrypoints
# =========================
def parse_color(text: str):
    """Accept hex like #aabbcc or comma RGB like 80,35,170 -> BGR tuple."""
    text = text.strip()
    if text.startswith("#") and len(text) == 7:
        r = int(text[1:3], 16); g = int(text[3:5], 16); b = int(text[5:7], 16)
        return (b, g, r)
    if "," in text:
        parts = [int(p) for p in text.split(",")]
        if len(parts) == 3:
            return (parts[2], parts[1], parts[0])  # convert RGB to BGR
    raise ValueError("Color must be #RRGGBB or R,G,B")


STYLES = {
    "simple": {
        "eyeshadow": {"color": (120, 80, 175), "opacity": 0.42, "feather": 33},
        "eyebrow": {"color": (30, 25, 20), "opacity": 0.18, "feather": 13},
        "blush": {"color": (80, 65, 180), "opacity": 0.24, "feather": 55},
        "lipstick": {"color": (80, 35, 170), "opacity": 0.58, "feather": 19},
        "highlighter": {"color": (235, 235, 255), "opacity": 0.24, "feather": 27},
    },
    "glam": {
        "eyeshadow": {"color": (140, 100, 210), "opacity": 0.55, "feather": 31},
        "eyebrow": {"color": (25, 20, 18), "opacity": 0.26, "feather": 15},
        "blush": {"color": (90, 70, 200), "opacity": 0.32, "feather": 57},
        "lipstick": {"color": (60, 20, 170), "opacity": 0.72, "feather": 17},
        "highlighter": {"color": (245, 245, 255), "opacity": 0.30, "feather": 27},
    },
}


def build_style(args) -> Dict:
    base = {k: v.copy() for k, v in STYLES[args.style].items()}
    # override color fields if provided
    if args.lip_color:
        base["lipstick"]["color"] = parse_color(args.lip_color)
    if args.shadow_color:
        base["eyeshadow"]["color"] = parse_color(args.shadow_color)
    if args.blush_color:
        base["blush"]["color"] = parse_color(args.blush_color)
    if args.highlight_color:
        base["highlighter"]["color"] = parse_color(args.highlight_color)
    return base


def run_on_image(image_path: str, save_path: str, style: Dict, rotation: int = 0, mirror: bool = False):
    model = tf.keras.models.load_model(MODEL_PATH)
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    oriented = orient_frame(bgr, rotation=rotation, mirror=mirror)
    out_oriented = apply_makeup(oriented, model, cascade, style)
    out = deorient_frame(out_oriented, rotation=rotation, mirror=mirror)
    cv2.imwrite(save_path, out)
    print(f"Saved: {save_path}")


def run_webcam(camera_id: int, style: Dict, rotation: int = 0, mirror: bool = False, ema: float = 0.0):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Camera not available.")
        return

    print("Press Q to quit.")
    prev_lms = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        oriented = orient_frame(frame, rotation=rotation, mirror=mirror)
        if ema > 0:
            output_oriented, prev_lms = apply_makeup_smoothed(
                oriented, model, cascade, style, prev_lms, ema_momentum=ema
            )
        else:
            output_oriented = apply_makeup(oriented, model, cascade, style)
        output = deorient_frame(output_oriented, rotation=rotation, mirror=mirror)
        cv2.imshow("Virtual Makeup (68 landmarks)", output)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Virtual makeup using 68-point landmarks.")
    parser.add_argument("--image", type=str, help="Path to input image (if set, runs image mode).")
    parser.add_argument("--out", type=str, default="makeup_result.png", help="Output path for image mode.")
    parser.add_argument("--camera", type=int, default=0, help="Camera id for webcam mode.")
    parser.add_argument("--style", choices=list(STYLES.keys()), default="simple", help="Preset style.")
    parser.add_argument("--lip-color", type=str, help="Override lipstick color (#RRGGBB or R,G,B).")
    parser.add_argument("--shadow-color", type=str, help="Override eyeshadow color (#RRGGBB or R,G,B).")
    parser.add_argument("--blush-color", type=str, help="Override blush color (#RRGGBB or R,G,B).")
    parser.add_argument("--highlight-color", type=str, help="Override highlighter color (#RRGGBB or R,G,B).")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="Rotate input before processing to fix portrait/landscape.")
    parser.add_argument("--mirror", action="store_true", help="Mirror input before processing (e.g., front camera).")
    parser.add_argument("--ema", type=float, default=0.0, help="EMA momentum for landmark smoothing (0 disables).")
    args = parser.parse_args()

    style = build_style(args)

    if args.image:
        run_on_image(args.image, args.out, style, rotation=args.rotate, mirror=args.mirror)
    else:
        run_webcam(args.camera, style, rotation=args.rotate, mirror=args.mirror, ema=args.ema)


if __name__ == "__main__":
    main()
