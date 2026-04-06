"""
collect_depth_data.py — Stereo depth accuracy data collection tool
  Simple SGBM + Temporal Smoothing (no census, no LR, no WLS)

Setup:
  1. Place camera on tripod, mark floor at 0 point
  2. Mark distances: 2ft, 4ft, 6ft, 8ft, 10ft, 12ft, 15ft, 20ft
  3. Have someone stand at each mark facing the camera

Controls:
  s = save reading (will prompt for actual distance in terminal)
  c = change lighting condition label
  e = toggle Sobel edges
  d = toggle depth on/off
  m = toggle disparity heatmap
  q = quit and save CSV

Output:
  depth_accuracy_data.csv with columns:
    timestamp, actual_ft, reported_ft, error_ft, error_pct,
    object, confidence, condition, depth_compute_ms, x1, y1, x2, y2
"""

import cv2
import numpy as np
import depthai as dai
import sys
import os
import time
import ctypes
import csv
import numpy.ctypeslib as npct
from ultralytics import YOLO
from datetime import datetime

# ============================================================
# YOLO MODEL
# ============================================================
model = YOLO("yolo11x.pt")

# ============================================================
# GPU SOBEL (fallback to CPU)
# ============================================================
sys.path.append("../GPU")

GPU_AVAILABLE = True
try:
    from sobel_gpu_api import sobel_gpu_edges
except Exception:
    GPU_AVAILABLE = False
    def sobel_gpu_edges(frame_bgr, threshold=80):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 else frame_bgr
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.convertScaleAbs(mag)
        _, edges = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
        return edges

def overlay_edges_on_rgb(frame_bgr, edges_gray, alpha=0.45):
    overlay = frame_bgr.copy()
    edges_gray = cv2.erode(edges_gray, np.ones((2, 2), np.uint8), iterations=1)
    overlay[edges_gray > 0] = (0, 255, 0)
    return cv2.addWeighted(frame_bgr, 1 - alpha, overlay, alpha, 0)

# ============================================================
# LOAD OpenMP C++ STEREO DEPTH LIBRARY
# ============================================================
if sys.platform == "win32":
    LIB_NAME = "stereo_depth_omp.dll"
else:
    LIB_NAME = "stereo_depth_omp.so"

try:
    stereo_lib = ctypes.CDLL(os.path.join(".", LIB_NAME))
    OMP_AVAILABLE = True
    print(f"[OpenMP stereo library loaded: {LIB_NAME}]")
except OSError as e:
    print(f"[WARNING] Could not load {LIB_NAME}: {e}")
    OMP_AVAILABLE = False

if OMP_AVAILABLE:
    stereo_lib.stereo_disparity_sgbm.argtypes = [
        npct.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
    ]
    stereo_lib.stereo_disparity_sgbm.restype = None

    stereo_lib.disparity_to_depth.argtypes = [
        npct.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
    ]
    stereo_lib.disparity_to_depth.restype = None

    stereo_lib.rectify_remap.argtypes = [
        npct.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int, ctypes.c_int,
    ]
    stereo_lib.rectify_remap.restype = None

    stereo_lib.get_median_depth_roi.argtypes = [
        npct.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
    ]
    stereo_lib.get_median_depth_roi.restype = ctypes.c_float

# ============================================================
# STEREO CALIBRATION
# ============================================================
M_left = np.array([
    [458.79290771484375, 0.0, 330.3569641113281],
    [0.0, 458.92999267578125, 248.63180541992188],
    [0.0, 0.0, 1.0]
])
M_right = np.array([
    [449.5007629394531, 0.0, 317.2297058105469],
    [0.0, 449.496826171875, 251.83892822265625],
    [0.0, 0.0, 1.0]
])

D_left = np.array([234.60887145996094, -623.8788452148438, -0.0005015762872062624,
                    0.00030968087958171964, 426.3470153808594, 235.93760681152344,
                    -627.5354614257812, 429.06951904296875, 0.0, 0.0, 0.0, 0.0,
                    0.005349033512175083, 0.0020748842507600784])
D_right = np.array([-2.113563060760498, 0.003960954025387764, -0.0019331664079800248,
                     -4.7757501306477934e-05, 1.580187201499939, -2.129826784133911,
                     0.04710356146097183, 1.5510843992233276, 0.0, 0.0, 0.0, 0.0,
                     0.016708601266145706, -0.005866835359483957])

R = np.array([
    [0.9999504685401917, -0.004337015096098185, -0.00894914660602808],
    [0.004344655200839043, 0.9999902248382568, 0.0008343947120010853],
    [0.008945440873503685, -0.0008732345886528492, 0.9999595880508423]
])
T = np.array([-7.492964744567871, 0.002622523345053196, -0.01669374108314514])
T_meters = T / 100.0

IMG_SIZE = (640, 480)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    M_left, D_left, M_right, D_right, IMG_SIZE, R, T_meters,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

focal_length_px = P1[0, 0]
baseline_m = abs(T_meters[0])

print(f"[Stereo] Focal length: {focal_length_px:.2f} px")
print(f"[Stereo] Baseline: {baseline_m * 100:.2f} cm ({baseline_m:.4f} m)")

map1_left, map2_left = cv2.initUndistortRectifyMap(M_left, D_left, R1, P1, IMG_SIZE, cv2.CV_32FC1)
map1_right, map2_right = cv2.initUndistortRectifyMap(M_right, D_right, R2, P2, IMG_SIZE, cv2.CV_32FC1)

NUM_DISP = 128
BLOCK_SIZE = 7

# ============================================================
# TEMPORAL SMOOTHING
# ============================================================
class TemporalSmoother:
    def __init__(self, num_frames=3, weight_decay=0.7):
        self.num_frames = num_frames
        self.weight_decay = weight_decay
        self.depth_history = []

    def smooth(self, depth_m):
        self.depth_history.append(depth_m.copy())
        if len(self.depth_history) > self.num_frames:
            self.depth_history.pop(0)

        if len(self.depth_history) == 1:
            return depth_m

        result = np.zeros_like(depth_m)
        total_weight = np.zeros_like(depth_m)

        for i, frame in enumerate(reversed(self.depth_history)):
            weight = self.weight_decay ** i
            valid = frame > 0.1
            result[valid] += frame[valid] * weight
            total_weight[valid] += weight

        nonzero = total_weight > 0
        result[nonzero] /= total_weight[nonzero]
        return result

depth_smoother = TemporalSmoother(num_frames=3, weight_decay=0.7)

# ============================================================
# DEPTH COMPUTATION — Simple SGBM + Temporal Smoothing
# ============================================================

def compute_depth_map(left_gray, right_gray):
    h, w = left_gray.shape

    if OMP_AVAILABLE:
        left_rect = np.zeros_like(left_gray)
        right_rect = np.zeros_like(right_gray)
        stereo_lib.rectify_remap(left_gray, left_rect, map1_left, map2_left, h, w)
        stereo_lib.rectify_remap(right_gray, right_rect, map1_right, map2_right, h, w)

        disparity_raw = np.zeros((h, w), dtype=np.float32)
        stereo_lib.stereo_disparity_sgbm(left_rect, right_rect, disparity_raw,
                                          h, w, NUM_DISP, BLOCK_SIZE)

        depth = np.zeros((h, w), dtype=np.float32)
        stereo_lib.disparity_to_depth(disparity_raw, depth, h, w,
                                       ctypes.c_float(focal_length_px),
                                       ctypes.c_float(baseline_m))
        disparity_filtered = disparity_raw
    else:
        left_rect = cv2.remap(left_gray, map1_left, map2_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_gray, map1_right, map2_right, cv2.INTER_LINEAR)
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=NUM_DISP, blockSize=BLOCK_SIZE,
            P1=8*3*BLOCK_SIZE*BLOCK_SIZE, P2=32*3*BLOCK_SIZE*BLOCK_SIZE,
            disp12MaxDiff=1, uniquenessRatio=12, speckleWindowSize=120,
            speckleRange=2, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        disp_raw = sgbm.compute(left_rect, right_rect).astype(np.float32) / 16.0
        disparity_filtered = disp_raw
        depth = np.zeros_like(disparity_filtered)
        valid = disparity_filtered > 1.0
        depth[valid] = (focal_length_px * baseline_m) / disparity_filtered[valid]

    # Temporal smoothing
    depth_smoothed = depth_smoother.smooth(depth)

    return depth_smoothed, disparity_filtered


def get_object_depth_feet(depth_m, x1, y1, x2, y2):
    h, w = depth_m.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if OMP_AVAILABLE:
        result = stereo_lib.get_median_depth_roi(depth_m, w, x1, y1, x2, y2, ctypes.c_float(0.1))
        return result if result > 0 else None
    else:
        margin = 0.1
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * margin), int(bh * margin)
        cx1, cy1 = x1 + mx, y1 + my
        cx2, cy2 = x2 - mx, y2 - my
        roi = depth_m[cy1:cy2, cx1:cx2]
        valid = roi[(roi > 0.1) & (roi < 30.0)]
        if len(valid) < 10:
            return None
        return float(np.median(valid)) * 3.28084


def colorize_disparity(disp):
    d = disp.copy()
    d[d < 0] = 0
    mx = float(np.max(d)) if d.size else 0.0
    if mx > 1e-6:
        norm = (d / mx * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(d, dtype=np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


# ============================================================
# DATA COLLECTION
# ============================================================

CSV_FILE = "depth_accuracy_data.csv"
CSV_HEADER = [
    "timestamp", "actual_ft", "reported_ft", "error_ft", "error_pct",
    "object", "confidence", "condition", "depth_compute_ms",
    "x1", "y1", "x2", "y2"
]

data_rows = []
current_condition = "indoor_normal"

def save_csv():
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(data_rows)
    print(f"\n[SAVED] {len(data_rows)} readings to {CSV_FILE}")


# ============================================================
# MAIN LOOP
# ============================================================

device = dai.Device(dai.UsbSpeed.HIGH)
with dai.Pipeline(device) as pipeline:
    cam_color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    q_color = cam_color.requestOutput((640, 480)).createOutputQueue()

    cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    q_left = cam_left.requestOutput((640, 480)).createOutputQueue()

    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    q_right = cam_right.requestOutput((640, 480)).createOutputQueue()

    sobel_on = False
    depth_on = True
    show_disparity = False

    frame_count = 0
    start_time = time.time()
    fps_display = 0.0

    current_depth_m = None
    current_disparity = None
    current_results = None
    current_depth_ms = 0.0

    pipeline.start()
    print("\n" + "=" * 50)
    print("  DEPTH ACCURACY DATA COLLECTION")
    print("  SGBM + Temporal Smoothing (3 frames)")
    print("=" * 50)
    print(f"  Condition: {current_condition}")
    print(f"  Output: {CSV_FILE}")
    print(f"  Engine: {'OpenMP SGBM + Temporal' if OMP_AVAILABLE else 'OpenCV + Temporal'}")
    print(f"  Expected disparity at 2ft: ~{focal_length_px * baseline_m / 0.6096:.0f} px")
    print("=" * 50)
    print("Controls:")
    print("  s = SAVE reading (then type actual distance)")
    print("  c = change lighting condition")
    print("  e = toggle Sobel edges")
    print("  d = toggle depth")
    print("  m = toggle disparity heatmap")
    print("  q = quit and save CSV")
    print("=" * 50 + "\n")

    while pipeline.isRunning():
        color_frame = q_color.get().getCvFrame()
        left_frame = q_left.get().getCvFrame()
        right_frame = q_right.get().getCvFrame()

        left_gray = left_frame if len(left_frame.shape) == 2 else cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = right_frame if len(right_frame.shape) == 2 else cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        if len(left_frame.shape) == 2:
            left_frame = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR)
        if len(right_frame.shape) == 2:
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_GRAY2BGR)

        # Depth computation
        current_depth_m = None
        current_disparity = None
        if depth_on:
            t_depth = time.time()
            current_depth_m, current_disparity = compute_depth_map(left_gray, right_gray)
            current_depth_ms = (time.time() - t_depth) * 1000

            # DEBUG: print disparity stats every 30 frames
            if frame_count % 30 == 0:
                valid_disp = current_disparity[current_disparity > 0]
                if len(valid_disp) > 0:
                    print(f"[DEBUG] Disp — min:{valid_disp.min():.1f} max:{valid_disp.max():.1f} "
                          f"median:{np.median(valid_disp):.1f} mean:{np.mean(valid_disp):.1f} "
                          f"| Expected ~{focal_length_px * baseline_m / 0.6096:.0f} at 2ft")
                else:
                    print("[DEBUG] No valid disparity pixels!")

        # Sobel
        if sobel_on:
            color_frame = overlay_edges_on_rgb(color_frame, sobel_gpu_edges(color_frame))

        # YOLO + depth annotations
        current_results = model(color_frame, verbose=False)
        r = current_results[0]
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            conf = float(box.conf)
            name = r.names[cls]

            cv2.rectangle(color_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(color_frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 0, 0), -1)
            cv2.putText(color_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if depth_on and current_depth_m is not None:
                dist_ft = get_object_depth_feet(current_depth_m, x1, y1, x2, y2)
                if dist_ft is not None:
                    depth_label = f"{dist_ft:.1f} ft"
                    (dw, dh), _ = cv2.getTextSize(depth_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    dy = y2 + dh + 8
                    cv2.rectangle(color_frame, (x1, y2), (x1 + dw, dy + 4), (0, 0, 0), -1)
                    cv2.putText(color_frame, depth_label, (x1, dy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps_display = frame_count / elapsed

        # Display
        small_size = (640, 480)
        color_disp = cv2.resize(color_frame, small_size)

        panels = [color_disp]

        if show_disparity and current_disparity is not None:
            disp_color = colorize_disparity(current_disparity)
            disp_color = cv2.resize(disp_color, small_size)
            cv2.putText(disp_color, "DISPARITY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            panels.append(disp_color)

        combined = np.hstack(panels)

        # Status bar
        status = (f"FPS: {fps_display:.1f} | Readings: {len(data_rows)} | "
                  f"Condition: {current_condition} | "
                  f"Depth: {current_depth_ms:.0f}ms | "
                  f"[S]ave [C]ondition [Q]uit")
        cv2.putText(combined, status, (10, combined.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        cv2.putText(combined, "Press S to save a reading", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.imshow("Depth Data Collection", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            save_csv()
            break

        if key == ord('e'):
            sobel_on = not sobel_on

        if key == ord('d'):
            depth_on = not depth_on

        if key == ord('m'):
            show_disparity = not show_disparity

        if key == ord('c'):
            print("\n--- CHANGE CONDITION ---")
            print("Options: indoor_normal, indoor_dim, outdoor_sun, outdoor_cloudy,")
            print("         outdoor_fog, outdoor_rain, window_glare, backlit")
            new_cond = input("Enter condition: ").strip()
            if new_cond:
                current_condition = new_cond
                print(f"[Condition set to: {current_condition}]")

        if key == ord('s'):
            if current_depth_m is None:
                print("[WARNING] Depth not computed — press 'd' to enable depth first")
                continue

            if current_results is None:
                print("[WARNING] No YOLO results available")
                continue

            r = current_results[0]
            if len(r.boxes) == 0:
                print("[WARNING] No objects detected — make sure you're in frame")
                continue

            best_box = None
            best_area = 0
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box = box

            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cls = int(best_box.cls)
            conf = float(best_box.conf)
            name = r.names[cls]
            reported_ft = get_object_depth_feet(current_depth_m, x1, y1, x2, y2)

            if reported_ft is None:
                print("[WARNING] Could not compute depth for detected object")
                continue

            # DEBUG: print disparity in the bounding box region
            roi_disp = current_disparity[y1:y2, x1:x2]
            valid_roi = roi_disp[roi_disp > 0]
            if len(valid_roi) > 0:
                print(f"  [DEBUG BOX] Disp in box — min:{valid_roi.min():.1f} max:{valid_roi.max():.1f} "
                      f"median:{np.median(valid_roi):.1f}")

            print(f"\n--- SAVE READING ---")
            print(f"  Detected: {name} ({conf:.2f})")
            print(f"  Reported depth: {reported_ft:.2f} ft")
            actual_str = input("  Enter ACTUAL distance in feet: ").strip()

            try:
                actual_ft = float(actual_str)
            except ValueError:
                print("[SKIPPED] Invalid number")
                continue

            error_ft = reported_ft - actual_ft
            error_pct = (abs(error_ft) / actual_ft * 100) if actual_ft > 0 else 0

            row = [
                datetime.now().isoformat(),
                f"{actual_ft:.2f}",
                f"{reported_ft:.2f}",
                f"{error_ft:.2f}",
                f"{error_pct:.1f}",
                name,
                f"{conf:.3f}",
                current_condition,
                f"{current_depth_ms:.1f}",
                x1, y1, x2, y2
            ]
            data_rows.append(row)
            print(f"  [LOGGED] actual={actual_ft:.1f}ft reported={reported_ft:.2f}ft "
                  f"error={error_ft:+.2f}ft ({error_pct:.1f}%)")
            print(f"  Total readings: {len(data_rows)}\n")

    cv2.destroyAllWindows()
    print(f"\nDone! {len(data_rows)} readings saved to {CSV_FILE}")