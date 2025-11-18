import os, glob, cv2, numpy as np
from config import TARGET_W, TARGET_H, OUTPUT_DIR, EXTS

def list_images(folder):
    paths = []
    for e in EXTS:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)

def read_bgr(path, ensure_size=True):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Không đọc được ảnh: {path}")
    if ensure_size:
        if img.shape[1] != TARGET_W or img.shape[0] != TARGET_H:
            img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    return img

def save_jpg_png(base_name_no_ext, img_bgr):
    jpg = os.path.join(OUTPUT_DIR, f"{base_name_no_ext}.jpg")
    png = os.path.join(OUTPUT_DIR, f"{base_name_no_ext}.png")
    cv2.imwrite(jpg, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(png, img_bgr)

def center_crop_quarter(img):
    h, w = img.shape[:2]
    nw, nh = w // 2, h // 2
    cx, cy = w // 2, h // 2
    x1, y1 = cx - nw // 2, cy - nh // 2
    x2, y2 = x1 + nw, y1 + nh
    return img[y1:y2, x1:x2]

def rotate_animation(img, steps=100, step_deg=5, delay_ms=100, title="Xoay ảnh"):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    angle = 0
    for _ in range(steps):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        cv2.imshow(title, rotated)
        k = cv2.waitKey(delay_ms) & 0xFF
        if k == 27:
            break
        angle += step_deg
    cv2.destroyWindow(title)