"""
process_recording.py
====================
Offline post-processor for stereo recordings made with `record_stereo.py`.

Takes a recording directory containing `left.mp4` and `right.mp4` and runs
the full pipeline on it (rectify -> SGBM -> depth -> YOLO -> distance labels),
writing an annotated output MP4 that you can play at the symposium.

This reuses the SAME calibration, SAME stereo_depth_omp library, and SAME
YOLO model as threecam.py — it's designed to be dropped into
`ComputerVision-main/multicam/` next to threecam.py.

No realtime pressure — CPU is fine. Processes as fast as it can and saves.

Usage:
    # from ComputerVision-main/multicam/
    python process_recording.py recordings/20260421_140015

    # Optional flags:
    #   --output custom_name.mp4     (default: annotated.mp4 in same folder)
    #   --every-n 1                  (process every Nth frame; default 1)
    #   --max-frames N               (stop after N frames; default: all)
    #   --no-yolo                    (skip YOLO; depth-only output)
"""

import argparse
import ctypes
import json
import os
import sys
import time

import cv2
import numpy as np


# =============================================================================
# ARG PARSING
# =============================================================================
parser = argparse.ArgumentParser(description="Process a stereo recording into an annotated video.")
parser.add_argument("session_dir", help="path to a recording dir containing left.mp4 and right.mp4")
parser.add_argument("--output", default=None, help="output MP4 filename (default: <session>/annotated.mp4)")
parser.add_argument("--every-n", type=int, default=1, help="process every Nth frame")
parser.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
parser.add_argument("--no-yolo", action="store_true", help="skip YOLO (depth only)")
args = parser.parse_args()

session_dir = args.session_dir
left_path = os.path.join(session_dir, "left.mp4")
right_path = os.path.join(session_dir, "right.mp4")
output_path = args.output or os.path.join(session_dir, "annotated.mp4")

if not os.path.isfile(left_path) or not os.path.isfile(right_path):
    print(f"ERROR: missing left.mp4 or right.mp4 in {session_dir}")
    sys.exit(1)


# =============================================================================
# CALIBRATION + STEREO LIB — copied verbatim from threecam.py structure
# These values come from your check_calib.py output. Keep in sync.
# =============================================================================
IMG_SIZE = (640, 480)  # (W, H)

# ---- PASTE your calibration block here (same as threecam.py) -----------------
# This matches the structure your threecam.py uses. If your threecam.py has
# updated calibration values, copy them in.
M_left  = np.array([[458.0,   0.0,   320.0],
                    [  0.0, 458.0,   240.0],
                    [  0.0,   0.0,     1.0]], dtype=np.float64)
M_right = np.array([[458.0,   0.0,   320.0],
                    [  0.0, 458.0,   240.0],
                    [  0.0,   0.0,     1.0]], dtype=np.float64)
D_left  = np.zeros((1, 5), dtype=np.float64)
D_right = np.zeros((1, 5), dtype=np.float64)
R       = np.eye(3, dtype=np.float64)
T       = np.array([[-0.0749], [0.0], [0.0]], dtype=np.float64)  # 74.9 mm baseline

# IMPORTANT: these toy defaults are just to make the script import cleanly.
# Before you run this, replace M_left/M_right/D_left/D_right/R/T with the
# actual block from the top of your threecam.py. Same applies to anything
# else below marked "match threecam.py".
# -----------------------------------------------------------------------------

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    M_left, D_left, M_right, D_right, IMG_SIZE, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

focal_length_px = float(P1[0, 0])
baseline_m      = float(-P2[0, 3] / P2[0, 0])

print(f"[calib] focal_length_px = {focal_length_px:.2f}")
print(f"[calib] baseline        = {baseline_m*100:.2f} cm")

map1_left,  map2_left  = cv2.initUndistortRectifyMap(M_left,  D_left,  R1, P1, IMG_SIZE, cv2.CV_32FC1)
map1_right, map2_right = cv2.initUndistortRectifyMap(M_right, D_right, R2, P2, IMG_SIZE, cv2.CV_32FC1)

NUM_DISP   = 128
BLOCK_SIZE = 7


# ----------------------- stereo_depth_omp loader ------------------------------
OMP_AVAILABLE = False
stereo_lib = None

LIB_CANDIDATES = ["./stereo_depth_omp.dll", "./stereo_depth_omp.so",
                  "stereo_depth_omp.dll", "stereo_depth_omp.so"]
for candidate in LIB_CANDIDATES:
    if os.path.isfile(candidate):
        try:
            stereo_lib = ctypes.CDLL(candidate)
            # Minimum required signature bindings — same as threecam.py
            stereo_lib.rectify_remap.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.uint8,   flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.uint8,   flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int
            ]
            stereo_lib.stereo_disparity_sgbm.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.uint8,   flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.uint8,   flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            stereo_lib.disparity_to_depth.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float
            ]
            stereo_lib.get_median_depth_roi.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_float
            ]
            stereo_lib.get_median_depth_roi.restype = ctypes.c_float
            OMP_AVAILABLE = True
            print(f"[OMP] loaded {candidate}")
            break
        except Exception as exc:
            print(f"[OMP] couldn't load {candidate}: {exc}")

if not OMP_AVAILABLE:
    print("[OMP] falling back to OpenCV SGBM")
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISP,
        blockSize=BLOCK_SIZE,
        P1=8 * 3 * BLOCK_SIZE * BLOCK_SIZE,
        P2=32 * 3 * BLOCK_SIZE * BLOCK_SIZE,
        disp12MaxDiff=1,
        uniquenessRatio=12,
        speckleWindowSize=120,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


# =============================================================================
# PIPELINE FUNCTIONS — mirror threecam.py so the output matches the live demo
# =============================================================================
def compute_depth_map(left_gray, right_gray):
    h, w = left_gray.shape
    if OMP_AVAILABLE:
        left_rect  = np.zeros_like(left_gray)
        right_rect = np.zeros_like(right_gray)
        stereo_lib.rectify_remap(left_gray,  left_rect,  map1_left,  map2_left,  h, w)
        stereo_lib.rectify_remap(right_gray, right_rect, map1_right, map2_right, h, w)

        disparity = np.zeros((h, w), dtype=np.float32)
        stereo_lib.stereo_disparity_sgbm(left_rect, right_rect, disparity, h, w, NUM_DISP, BLOCK_SIZE)

        depth = np.zeros((h, w), dtype=np.float32)
        stereo_lib.disparity_to_depth(disparity, depth, h, w,
                                       ctypes.c_float(focal_length_px),
                                       ctypes.c_float(baseline_m))
    else:
        left_rect  = cv2.remap(left_gray,  map1_left,  map2_left,  cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_gray, map1_right, map2_right, cv2.INTER_LINEAR)
        disparity  = sgbm.compute(left_rect, right_rect).astype(np.float32) / 16.0
        depth = np.zeros_like(disparity)
        valid = disparity > 1.0
        depth[valid] = (focal_length_px * baseline_m) / disparity[valid]
    return depth, disparity, left_rect, right_rect


def get_object_depth_feet(depth_m, x1, y1, x2, y2):
    h, w = depth_m.shape
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if OMP_AVAILABLE:
        result = stereo_lib.get_median_depth_roi(
            depth_m, w, x1, y1, x2, y2, ctypes.c_float(0.1))
        return result if result > 0 else None
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * 0.1), int(bh * 0.1)
    roi = depth_m[y1+my:y2-my, x1+mx:x2-mx]
    valid = roi[(roi > 0.1) & (roi < 30.0)]
    if len(valid) < 10:
        return None
    return float(np.median(valid)) * 3.28084


# =============================================================================
# YOLO — lazy load, optional
# =============================================================================
yolo_model = None
if not args.no_yolo:
    try:
        from ultralytics import YOLO
        # Adjust path if yolo11x.pt lives elsewhere in your repo
        for candidate in ["../yolo11x.pt", "./yolo11x.pt", "yolo11x.pt"]:
            if os.path.isfile(candidate):
                yolo_model = YOLO(candidate)
                print(f"[YOLO] loaded {candidate}")
                break
        if yolo_model is None:
            print("[YOLO] couldn't find yolo11x.pt — continuing without YOLO")
    except Exception as exc:
        print(f"[YOLO] import/load failed: {exc} — continuing without YOLO")


def draw_overlay(rectified_bgr, depth_m, yolo_results):
    """Draw YOLO boxes + distance in feet on the rectified left frame."""
    out = rectified_bgr.copy()
    if yolo_results is None:
        return out

    for r in yolo_results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id] if yolo_model else str(cls_id)
            conf  = float(box.conf[0])

            dist_ft = get_object_depth_feet(depth_m, x1, y1, x2, y2)
            tag = f"{label} {conf:.2f}"
            if dist_ft is not None:
                tag += f"  {dist_ft:.1f} ft"

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(out, tag, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return out


# =============================================================================
# MAIN — read both MP4s, process frame by frame, write annotated MP4
# =============================================================================
cap_left  = cv2.VideoCapture(left_path)
cap_right = cv2.VideoCapture(right_path)

fps   = cap_left.get(cv2.CAP_PROP_FPS) or 30.0
total = int(min(cap_left.get(cv2.CAP_PROP_FRAME_COUNT),
                cap_right.get(cv2.CAP_PROP_FRAME_COUNT)))
if args.max_frames is not None:
    total = min(total, args.max_frames)

print(f"[read] {total} frames @ {fps:.1f} fps from {session_dir}")

# First frame to size the writer
ok_l, f_l = cap_left.read()
ok_r, f_r = cap_right.read()
if not (ok_l and ok_r):
    print("ERROR: couldn't read first frame from recording")
    sys.exit(1)

if len(f_l.shape) == 3: f_l = cv2.cvtColor(f_l, cv2.COLOR_BGR2GRAY)
if len(f_r.shape) == 3: f_r = cv2.cvtColor(f_r, cv2.COLOR_BGR2GRAY)

out_h, out_w = f_l.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h), isColor=True)

# Rewind and process from the top
cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Temporal smoothing on depth — match threecam.py's 3-frame weighted avg
depth_history = []
SMOOTH_N = 3
SMOOTH_WEIGHTS = np.array([0.2, 0.3, 0.5], dtype=np.float32)  # oldest -> newest

frame_idx = 0
processed = 0
t_start = time.time()

while True:
    ok_l, f_l = cap_left.read()
    ok_r, f_r = cap_right.read()
    if not (ok_l and ok_r):
        break
    if args.max_frames is not None and processed >= args.max_frames:
        break
    if frame_idx % args.every_n != 0:
        frame_idx += 1
        continue
    frame_idx += 1

    if len(f_l.shape) == 3: f_l = cv2.cvtColor(f_l, cv2.COLOR_BGR2GRAY)
    if len(f_r.shape) == 3: f_r = cv2.cvtColor(f_r, cv2.COLOR_BGR2GRAY)

    depth_m, disparity, left_rect, right_rect = compute_depth_map(f_l, f_r)

    # Temporal smoothing: weighted avg of last N depth maps
    depth_history.append(depth_m)
    if len(depth_history) > SMOOTH_N:
        depth_history.pop(0)
    if len(depth_history) == SMOOTH_N:
        stacked = np.stack(depth_history, axis=0)
        w = SMOOTH_WEIGHTS.reshape(SMOOTH_N, 1, 1)
        depth_smooth = (stacked * w).sum(axis=0)
    else:
        depth_smooth = depth_m

    rect_bgr = cv2.cvtColor(left_rect, cv2.COLOR_GRAY2BGR)

    yolo_results = None
    if yolo_model is not None:
        yolo_results = yolo_model(rect_bgr, verbose=False)

    annotated = draw_overlay(rect_bgr, depth_smooth, yolo_results)

    # HUD
    cv2.putText(annotated, f"frame {processed+1}/{total}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    writer.write(annotated)
    processed += 1

    if processed % 30 == 0:
        elapsed = time.time() - t_start
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"  ... {processed}/{total}  ({rate:.1f} fps processing)")

writer.release()
cap_left.release()
cap_right.release()

elapsed = time.time() - t_start
print(f"[done] wrote {output_path}")
print(f"       {processed} frames processed in {elapsed:.1f}s "
      f"({processed/elapsed:.1f} fps processing)")