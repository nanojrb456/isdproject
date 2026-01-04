import cv2
import numpy as np
import mediapipe as mp

# =======================
# FaceMesh indices
# =======================
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

LEFT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

LEFT_UPPER_LID  = [33, 160, 158, 133]    # outer -> inner-ish
RIGHT_UPPER_LID = [263, 387, 385, 362]   # outer -> inner-ish

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

LEFT_CHEEK_ANCHORS  = [234, 93, 132, 58, 172, 136]
RIGHT_CHEEK_ANCHORS = [454, 323, 361, 288, 397, 365]

# Brow hint (only to compute safe eyeshadow top)
LEFT_BROW_HINT  = [70, 63, 105, 66]
RIGHT_BROW_HINT = [300, 293, 334, 296]

# Nose tip-ish (for blush direction)
NOSE_TIP = 4


# =======================
# Global tuning knobs (easy to edit)
# =======================
SIMPLE = {
    "skin_strength": 0.35,
    "eyeshadow_intensity": 0.30,   # كان ضعيف -> زاد
    "eyeliner_alpha": 0.40,
    "eyeliner_thickness": 2,
    "blush_intensity": 0.25,       # قوة منطقية
    "lip_intensity": 0.42          # الحمرة كانت خفيفة
}

GLAM = {
    "skin_strength": 0.70,
    "eyeshadow_intensity": 0.85,
    "glitter_strength": 0.30,
    "eyeliner_alpha": 0.65,
    "eyeliner_thickness": 2,
    "wing_len": 10,                # كان طويل -> قصّرناه
    "wing_up": 7,
    "blush_intensity": 0.30,       # كان قوي كتير -> خففناه
    "lip_intensity": 0.80
}


# =======================
# Helpers (mirror-safe)
# =======================
def lm_xy(face_lms, idx, w, h, mirror=True):
    lm = face_lms.landmark[idx]
    x = int(lm.x * w)
    y = int(lm.y * h)
    if mirror:
        x = (w - 1) - x
    return np.array([x, y], dtype=np.int32)

def poly(face_lms, idxs, w, h, mirror=True):
    pts = np.array([lm_xy(face_lms, i, w, h, mirror) for i in idxs], dtype=np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts

def smooth_mask(mask, k=31):
    k = max(3, int(k) | 1)
    return cv2.GaussianBlur(mask, (k, k), 0)

def alpha_blend(base_bgr, overlay_bgr, alpha_mask_01):
    a = np.clip(alpha_mask_01, 0, 1).astype(np.float32)
    base = base_bgr.astype(np.float32)
    over = overlay_bgr.astype(np.float32)
    out = base * (1 - a[..., None]) + over * a[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)

def fill(mask, pts, v=255):
    if pts is not None and len(pts) >= 3:
        cv2.fillPoly(mask, [pts], v)
    return mask

def adaptive_intensity(frame, mask, base, clamp=(0.25, 1.15)):
    """Scale intensity by local brightness (Y channel) so makeup softens on bright skin."""
    if mask is None or mask.size == 0:
        return base
    if np.count_nonzero(mask) == 0:
        return base
    y = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    mean_y = cv2.mean(y, mask=mask)[0] / 255.0
    scaled = base * (0.9 + (0.5 - mean_y) * 0.6)
    return float(np.clip(scaled, clamp[0], clamp[1]))


# =======================
# Face / exclude masks (for smoothing + keep makeup inside face)
# =======================
def build_face_mask(face_lms, w, h, mirror=True):
    m = np.zeros((h, w), dtype=np.uint8)
    fill(m, poly(face_lms, FACE_OVAL, w, h, mirror), 255)
    return smooth_mask(m, 41)

def build_exclude_mask(face_lms, w, h, mirror=True):
    m = np.zeros((h, w), dtype=np.uint8)
    fill(m, poly(face_lms, LEFT_EYE,  w, h, mirror), 255)
    fill(m, poly(face_lms, RIGHT_EYE, w, h, mirror), 255)
    fill(m, poly(face_lms, LIPS_OUTER, w, h, mirror), 255)
    return smooth_mask(m, 31)

def skin_smooth_only_face(frame, face_mask, exclude_mask, strength=0.4):
    area = cv2.subtract(face_mask, exclude_mask)
    a = (area.astype(np.float32) / 255.0) * strength
    smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=60, sigmaSpace=60)
    return alpha_blend(frame, smooth, a)


# =======================
# Eyeliner
# SIMPLE: full liner NO wing
# GLAM: wing OUTER only + short + capped
# =======================
def apply_eyeliner(frame, face_lms, w, h, style="simple", mirror=True):
    out = frame.copy()

    # corners
    L_OUT = lm_xy(face_lms, 33,  w, h, mirror)
    L_IN  = lm_xy(face_lms, 133, w, h, mirror)
    R_OUT = lm_xy(face_lms, 263, w, h, mirror)
    R_IN  = lm_xy(face_lms, 362, w, h, mirror)

    if style == "glam":
        thickness = GLAM["eyeliner_thickness"]
        alpha = GLAM["eyeliner_alpha"]
        wing_len = GLAM["wing_len"]
        wing_up  = GLAM["wing_up"]
    else:
        thickness = SIMPLE["eyeliner_thickness"]
        alpha = SIMPLE["eyeliner_alpha"]
        wing_len = 0
        wing_up  = 0

    color = (15, 15, 15)

    # Lid line (FULL from inner to outer)
    L_lid = poly(face_lms, LEFT_UPPER_LID, w, h, mirror)
    R_lid = poly(face_lms, RIGHT_UPPER_LID, w, h, mirror)
    cv2.polylines(out, [L_lid[::-1]], False, color, thickness, cv2.LINE_AA)
    cv2.polylines(out, [R_lid[::-1]], False, color, thickness, cv2.LINE_AA)

    # Wing only in GLAM
    if style == "glam":
        # direction inner->outer, extend beyond outer
        vL = (L_OUT - L_IN).astype(np.float32); vL /= (np.linalg.norm(vL) + 1e-6)
        vR = (R_OUT - R_IN).astype(np.float32); vR /= (np.linalg.norm(vR) + 1e-6)

        wingL = (L_OUT + vL * wing_len + np.array([0, -wing_up], np.float32)).astype(np.int32)
        wingR = (R_OUT + vR * wing_len + np.array([0, -wing_up], np.float32)).astype(np.int32)

        # cap wing so it doesn't go crazy
        wingL[0] = np.clip(wingL[0], 0, w - 1); wingL[1] = np.clip(wingL[1], 0, h - 1)
        wingR[0] = np.clip(wingR[0], 0, w - 1); wingR[1] = np.clip(wingR[1], 0, h - 1)

        cv2.line(out, tuple(L_OUT), tuple(wingL), color, thickness, cv2.LINE_AA)
        cv2.line(out, tuple(R_OUT), tuple(wingR), color, thickness, cv2.LINE_AA)

    return cv2.addWeighted(out, alpha, frame, 1 - alpha, 0)


# =======================
# Eyeshadow + Glitter
# FIX: eyeshadow band ABOVE eye only (not inside eyeball)
# Glitter also clipped to shadow band and excludes eye region
# =======================
def upper_eye_arc(eye_pts):
    # take points with smallest y (upper lid zone)
    ys = eye_pts[:, 1]
    thr = np.percentile(ys, 55)
    upper = eye_pts[ys <= thr]
    if len(upper) < 6:
        upper = eye_pts
    # sort by x for stable band shape
    upper = upper[np.argsort(upper[:, 0])]
    return upper

def build_shadow_band(face_lms, w, h, which="left", mirror=True):
    if which == "left":
        eye = poly(face_lms, LEFT_EYE, w, h, mirror)
        brow = poly(face_lms, LEFT_BROW_HINT, w, h, mirror)
    else:
        eye = poly(face_lms, RIGHT_EYE, w, h, mirror)
        brow = poly(face_lms, RIGHT_BROW_HINT, w, h, mirror)

    upper = upper_eye_arc(eye)

    brow_y = int(np.min(brow[:, 1])) if len(brow) else 0
    top_limit = brow_y + 12  # keep margin below brow

    lifted = upper.copy()
    lifted[:, 1] = lifted[:, 1] - 24
    lifted[:, 1] = np.maximum(lifted[:, 1], top_limit)

    # band polygon: upper arc + reversed lifted arc (so it becomes a strip)
    band = np.vstack([upper, lifted[::-1]])
    band[:, 0] = np.clip(band[:, 0], 0, w - 1)
    band[:, 1] = np.clip(band[:, 1], 0, h - 1)
    return band

def apply_eyeshadow(frame, face_lms, w, h, style="simple", mirror=True, face_mask=None):
    overlay = frame.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    if style == "glam":
        intensity = GLAM["eyeshadow_intensity"]
        color = (70, 40, 90)   # warm glam tone
        blur_k = 41
    else:
        intensity = SIMPLE["eyeshadow_intensity"]
        color = (100, 65, 85)
        blur_k = 50

    lp = build_shadow_band(face_lms, w, h, "left", mirror)
    rp = build_shadow_band(face_lms, w, h, "right", mirror)
    fill(mask, lp, 255)
    fill(mask, rp, 255)

    # IMPORTANT: keep it inside face to avoid glitter on brow/hair
    if face_mask is not None:
        mask = cv2.bitwise_and(mask, face_mask)

    mask = smooth_mask(mask, blur_k)
    adj = adaptive_intensity(frame, mask, intensity)
    alpha = (mask.astype(np.float32) / 255.0) * adj
    overlay[:] = color
    out = alpha_blend(frame, overlay, alpha)

    # GLITTER: only on shadow band, EXCLUDE eyeball region
    if style == "glam":
        eye_mask = np.zeros((h, w), dtype=np.uint8)
        fill(eye_mask, poly(face_lms, LEFT_EYE,  w, h, mirror), 255)
        fill(eye_mask, poly(face_lms, RIGHT_EYE, w, h, mirror), 255)
        eye_mask = smooth_mask(eye_mask, 21)

        glitter_area = cv2.subtract(mask, eye_mask)  # remove inside eye
        glitter_area = smooth_mask(glitter_area, 31)

        pts = np.column_stack(np.where(glitter_area > 120))
        glitter = np.zeros_like(frame)
        if len(pts) > 0:
            rng = np.random.default_rng()
            sample_n = min(350, len(pts))
            pick = pts[rng.choice(len(pts), size=sample_n, replace=False)]
            for (y, x) in pick:
                glitter[y, x] = (230, 230, 255)

        g_alpha = (glitter_area.astype(np.float32) / 255.0) * GLAM["glitter_strength"]
        out = alpha_blend(out, glitter, g_alpha)

    return out


# =======================
# Blush (FIX: position outward + keep inside face + reduce glam strength)
# =======================
def cheek_center_scale(face_lms, w, h, anchors, mirror=True):
    pts = np.array([lm_xy(face_lms, i, w, h, mirror) for i in anchors], dtype=np.int32)
    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))

    xspan = np.max(pts[:, 0]) - np.min(pts[:, 0])
    yspan = np.max(pts[:, 1]) - np.min(pts[:, 1])
    ax = max(22, int(xspan * 0.55))
    ay = max(18, int(yspan * 0.50))
    return (cx, cy), (ax, ay)

def apply_blush(frame, face_lms, w, h, style="simple", mirror=True, face_mask=None):
    overlay = frame.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    if style == "glam":
        intensity = GLAM["blush_intensity"]
        blur_k = 61
        color = (120, 60, 220)
    else:
        intensity = SIMPLE["blush_intensity"]
        blur_k = 65
        color = (90, 55, 150)

    nose = lm_xy(face_lms, NOSE_TIP, w, h, mirror).astype(np.float32)

    (lc, la) = cheek_center_scale(face_lms, w, h, LEFT_CHEEK_ANCHORS, mirror)
    (rc, ra) = cheek_center_scale(face_lms, w, h, RIGHT_CHEEK_ANCHORS, mirror)

    lc = np.array(lc, np.float32); rc = np.array(rc, np.float32)

    # move blush OUTWARD (away from nose) + slightly DOWN to sit on cheek
    for c in [lc, rc]:
        v = c - nose
        v /= (np.linalg.norm(v) + 1e-6)
        c += v * 18          # outward
        c[1] += 8            # down

    lc = (int(lc[0]), int(lc[1]))
    rc = (int(rc[0]), int(rc[1]))

    cv2.ellipse(mask, lc, la, 0, 0, 360, 255, -1)
    cv2.ellipse(mask, rc, ra, 0, 0, 360, 255, -1)

    # clip to face to stop blush going on hair
    if face_mask is not None:
        mask = cv2.bitwise_and(mask, face_mask)

    mask = smooth_mask(mask, blur_k)
    adj = adaptive_intensity(frame, mask, intensity)
    alpha = (mask.astype(np.float32) / 255.0) * adj

    overlay[:] = color
    return alpha_blend(frame, overlay, alpha)


# =======================
# Lipstick (outer - inner) avoid teeth
# =======================
def apply_lipstick(frame, face_lms, w, h, style="simple", mirror=True, face_mask=None):
    overlay = frame.copy()
    outer = np.zeros((h, w), dtype=np.uint8)
    inner = np.zeros((h, w), dtype=np.uint8)

    if style == "glam":
        intensity = GLAM["lip_intensity"]
        color = (60, 30, 170)   # deep rose/burgundy
        blur_k = 35
        gloss = 0.18
    else:
        intensity = SIMPLE["lip_intensity"]
        color = (60, 30, 170)   # natural pink
        blur_k = 41
        gloss = 0.2

    fill(outer, poly(face_lms, LIPS_OUTER, w, h, mirror), 255)
    fill(inner, poly(face_lms, LIPS_INNER, w, h, mirror), 255)

    lip = cv2.subtract(outer, smooth_mask(inner, 21))
    if face_mask is not None:
        lip = cv2.bitwise_and(lip, face_mask)

    lip = smooth_mask(lip, blur_k)

    adj = adaptive_intensity(frame, lip, intensity)
    alpha = (lip.astype(np.float32) / 255.0) * adj
    overlay[:] = color
    out = alpha_blend(frame, overlay, alpha)

    # subtle gloss
    if gloss > 0:
        gloss_img = np.zeros_like(frame)
        top = lm_xy(face_lms, 0, w, h, mirror)
        gx, gy = int(top[0]), int(top[1])
        cv2.ellipse(gloss_img, (gx, gy + 8), (26, 10), 0, 0, 360, (230, 230, 255), -1)
        gmask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(gmask, (gx, gy + 8), (26, 10), 0, 0, 360, 255, -1)
        gmask = cv2.bitwise_and(gmask, lip)
        gmask = smooth_mask(gmask, 31)
        out = alpha_blend(out, gloss_img, (gmask.astype(np.float32) / 255.0) * gloss)

    return out


# =======================
# MAIN
# =======================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    mode = "OFF"
    print("Controls: [S]=Simple  [G]=Glam  [O]=OFF  [Q]=Quit")

    while True:
        ok, raw = cap.read()
        if not ok:
            break

        frame = cv2.flip(raw, 1)  # mirror display
        h, w = frame.shape[:2]

        # FaceMesh on raw (unflipped)
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)

        if res.multi_face_landmarks:
            face_lms = res.multi_face_landmarks[0]

            face_mask = build_face_mask(face_lms, w, h, mirror=True)
            ex_mask = build_exclude_mask(face_lms, w, h, mirror=True)

            if mode == "SIMPLE":
                frame = skin_smooth_only_face(frame, face_mask, ex_mask, SIMPLE["skin_strength"])
                frame = apply_eyeshadow(frame, face_lms, w, h, "simple", True, face_mask)
                frame = apply_eyeliner(frame, face_lms, w, h, "simple", True)  # NO wing
                frame = apply_blush(frame, face_lms, w, h, "simple", True, face_mask)
                frame = apply_lipstick(frame, face_lms, w, h, "simple", True, face_mask)

            elif mode == "GLAM":
                frame = skin_smooth_only_face(frame, face_mask, ex_mask, GLAM["skin_strength"])
                frame = apply_eyeshadow(frame, face_lms, w, h, "glam", True, face_mask)  # glitter fixed
                frame = apply_eyeliner(frame, face_lms, w, h, "glam", True)             # wing shorter
                frame = apply_blush(frame, face_lms, w, h, "glam", True, face_mask)     # clip to face
                frame = apply_lipstick(frame, face_lms, w, h, "glam", True, face_mask)

        cv2.putText(frame, f"MODE: {mode}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "S:Simple  G:Glam  O:Off  Q:Quit", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Makeup Filter (Realistic)", frame)

        k = cv2.waitKey(1) & 0xFF
        if k in [ord('q'), ord('Q')]:
            break
        elif k in [ord('s'), ord('S')]:
            mode = "SIMPLE"
        elif k in [ord('g'), ord('G')]:
            mode = "GLAM"
        elif k in [ord('o'), ord('O')]:
            mode = "OFF"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
