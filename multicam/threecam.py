"""
threecam.py — OAK-D Lite triple camera pipeline
  - 3 live feeds (LEFT | COLOR | RIGHT)
  - Sobel edge overlay (GPU or CPU fallback)
  - YOLO object detection
  - OpenMP-parallelized stereo depth (C++ library, 8 threads)
  - Distance in feet per detected object

Compile the C++ library first:
  Windows (MinGW):  g++ -O3 -fopenmp -shared -o stereo_depth_omp.dll stereo_depth_omp.cpp
  Windows (MSVC):   cl /O2 /openmp /LD /Fe:stereo_depth_omp.dll stereo_depth_omp.cpp
  Linux:            g++ -O3 -fopenmp -shared -fPIC -o stereo_depth_omp.so stereo_depth_omp.cpp

Controls:
  e = toggle Sobel edges
  y = toggle YOLO detection
  d = toggle depth calculation (auto-enables YOLO)
  m = toggle disparity heatmap panel
  q = quit
"""

import cv2
import numpy as np
import depthai as dai
import sys
import os
import time
import ctypes
import numpy.ctypeslib as npct
from ultralytics import YOLO

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
except Exception as e:
    print("[GPU Sobel unavailable — using CPU Sobel fallback]")
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
    print("[WARNING] Falling back to OpenCV SGBM (no OpenMP acceleration)")
    OMP_AVAILABLE = False

if OMP_AVAILABLE:
    # stereo_disparity_sgbm
    stereo_lib.stereo_disparity_sgbm.argtypes = [
        npct.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),   # left
        npct.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),   # right
        npct.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), # disparity out
        ctypes.c_int, ctypes.c_int,  # height, width
        ctypes.c_int, ctypes.c_int,  # num_disparities, block_size
    ]
    stereo_lib.stereo_disparity_sgbm.restype = None

    # disparity_to_depth
    stereo_lib.disparity_to_depth.argtypes = [
        npct.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),  # disparity
        npct.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),  # depth out
        ctypes.c_int, ctypes.c_int,  # height, width
        ctypes.c_float, ctypes.c_float,  # focal_px, baseline_m
    ]
    stereo_lib.disparity_to_depth.restype = None

    # rectify_remap
    stereo_lib.rectify_remap.argtypes = [
        npct.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),    # src
        npct.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),    # dst
        npct.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),  # map_x
        npct.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),  # map_y
        ctypes.c_int, ctypes.c_int,  # height, width
    ]
    stereo_lib.rectify_remap.restype = None

    # get_median_depth_roi
    stereo_lib.get_median_depth_roi.argtypes = [
        npct.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),  # depth
        ctypes.c_int,  # img_width
        ctypes.c_int, ctypes.c_int,  # x1, y1
        ctypes.c_int, ctypes.c_int,  # x2, y2
        ctypes.c_float,  # margin
    ]
    stereo_lib.get_median_depth_roi.restype = ctypes.c_float


# ============================================================
# STEREO CALIBRATION — YOUR OAK-D LITE'S REAL NUMBERS
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

# Stereo rectification (computed once)
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    M_left, D_left, M_right, D_right, IMG_SIZE, R, T_meters,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

focal_length_px = P1[0, 0]
baseline_m = abs(T_meters[0])

print(f"[Stereo] Focal length: {focal_length_px:.2f} px")
print(f"[Stereo] Baseline: {baseline_m * 100:.2f} cm ({baseline_m:.4f} m)")

# Precompute rectification maps
map1_left, map2_left = cv2.initUndistortRectifyMap(M_left, D_left, R1, P1, IMG_SIZE, cv2.CV_32FC1)
map1_right, map2_right = cv2.initUndistortRectifyMap(M_right, D_right, R2, P2, IMG_SIZE, cv2.CV_32FC1)

# OpenCV SGBM fallback (used when C++ lib not available)
NUM_DISP = 128
BLOCK_SIZE = 7

if not OMP_AVAILABLE:
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


# ============================================================
# DEPTH PIPELINE FUNCTIONS
# ============================================================

def compute_depth_map(left_gray, right_gray):
    """Full stereo pipeline: rectify → disparity → depth."""
    h, w = left_gray.shape

    if OMP_AVAILABLE:
        # OpenMP-parallelized rectification
        left_rect = np.zeros_like(left_gray)
        right_rect = np.zeros_like(right_gray)
        stereo_lib.rectify_remap(left_gray, left_rect, map1_left, map2_left, h, w)
        stereo_lib.rectify_remap(right_gray, right_rect, map1_right, map2_right, h, w)

        # OpenMP-parallelized disparity (8 threads)
        disparity = np.zeros((h, w), dtype=np.float32)
        stereo_lib.stereo_disparity_sgbm(left_rect, right_rect, disparity, h, w, NUM_DISP, BLOCK_SIZE)

        # OpenMP-parallelized depth conversion
        depth = np.zeros((h, w), dtype=np.float32)
        stereo_lib.disparity_to_depth(disparity, depth, h, w,
                                       ctypes.c_float(focal_length_px),
                                       ctypes.c_float(baseline_m))
    else:
        # OpenCV fallback
        left_rect = cv2.remap(left_gray, map1_left, map2_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_gray, map1_right, map2_right, cv2.INTER_LINEAR)
        disp_raw = sgbm.compute(left_rect, right_rect).astype(np.float32) / 16.0
        disparity = disp_raw
        depth = np.zeros_like(disparity)
        valid = disparity > 1.0
        depth[valid] = (focal_length_px * baseline_m) / disparity[valid]

    return depth, disparity


def get_object_depth_feet(depth_m, x1, y1, x2, y2):
    """Get distance in feet for a bounding box."""
    h, w = depth_m.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

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


def draw_yolo_boxes_with_depth(frame, results, depth_m):
    """Draw YOLO boxes with distance in feet."""
    r = results[0]
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)
        conf = float(box.conf)
        name = r.names[cls]

        dist_ft = get_object_depth_feet(depth_m, x1, y1, x2, y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if dist_ft is not None:
            label = f"{name} {conf:.2f} | {dist_ft:.1f} ft"
        else:
            label = f"{name} {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


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
    yolo_on = False
    depth_on = False
    show_disparity = False

    frame_count = 0
    start_time = time.time()
    fps_display = 0.0

    pipeline.start()
    print("\n========================================")
    print("  OAK-D Lite — Stereo Depth Pipeline")
    print("  OpenMP threads: 8")
    print("  Depth formula: Z = (f × B) / d")
    print(f"  f = {focal_length_px:.2f} px")
    print(f"  B = {baseline_m * 100:.2f} cm")
    print("========================================")
    print("Controls:")
    print("  e = toggle Sobel edges")
    print("  y = toggle YOLO detection")
    print("  d = toggle depth (auto-enables YOLO)")
    print("  m = toggle disparity heatmap")
    print("  q = quit\n")

    while pipeline.isRunning():
        color_frame = q_color.get().getCvFrame()
        left_frame = q_left.get().getCvFrame()
        right_frame = q_right.get().getCvFrame()

        # Keep raw grayscale for stereo
        left_gray = left_frame if len(left_frame.shape) == 2 else cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = right_frame if len(right_frame.shape) == 2 else cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Convert mono to BGR for display
        if len(left_frame.shape) == 2:
            left_frame = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR)
        if len(right_frame.shape) == 2:
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_GRAY2BGR)

        # Stereo depth computation (OpenMP parallelized)
        depth_m = None
        disparity = None
        if depth_on or show_disparity:
            t_depth = time.time()
            depth_m, disparity = compute_depth_map(left_gray, right_gray)
            depth_ms = (time.time() - t_depth) * 1000

        # Sobel edge overlay
        if sobel_on:
            left_frame = overlay_edges_on_rgb(left_frame, sobel_gpu_edges(left_frame))
            color_frame = overlay_edges_on_rgb(color_frame, sobel_gpu_edges(color_frame))
            right_frame = overlay_edges_on_rgb(right_frame, sobel_gpu_edges(right_frame))

        # YOLO + depth
        if yolo_on:
            results = model(color_frame, verbose=False)
            if depth_on and depth_m is not None:
                color_frame = draw_yolo_boxes_with_depth(color_frame, results, depth_m)
            else:
                r = results[0]
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls)
                    conf = float(box.conf)
                    cv2.rectangle(color_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(color_frame, f"{r.names[cls]} {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps_display = frame_count / elapsed

        # Resize for display
        small_size = (427, 320)
        left_disp = cv2.resize(left_frame, small_size)
        color_disp = cv2.resize(color_frame, small_size)
        right_disp = cv2.resize(right_frame, small_size)

        # Labels
        cv2.putText(left_disp, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(color_disp, "COLOR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(right_disp, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        panels = [left_disp, color_disp, right_disp]

        # Disparity heatmap panel
        if show_disparity and disparity is not None:
            disp_color = colorize_disparity(disparity)
            disp_color = cv2.resize(disp_color, small_size)
            cv2.putText(disp_color, "DISPARITY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            panels.append(disp_color)

        combined = np.hstack(panels)

        # Status bar
        status = (f"FPS: {fps_display:.1f} | Sobel[e]: {'ON' if sobel_on else 'OFF'} | "
                  f"YOLO[y]: {'ON' if yolo_on else 'OFF'} | Depth[d]: {'ON' if depth_on else 'OFF'} | "
                  f"Disparity[m]: {'ON' if show_disparity else 'OFF'} | "
                  f"Engine: {'OpenMP x8' if OMP_AVAILABLE else 'OpenCV'} | Quit[q]")
        cv2.putText(combined, status, (10, combined.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Depth timing
        if (depth_on or show_disparity) and depth_m is not None:
            cv2.putText(combined, f"Depth compute: {depth_ms:.1f} ms", (10, combined.shape[0] - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        cv2.imshow("OAK-D Lite Stereo Depth", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('e'):
            sobel_on = not sobel_on
            print(f"[Sobel {'ON' if sobel_on else 'OFF'}]")
        if key == ord('y'):
            yolo_on = not yolo_on
            print(f"[YOLO {'ON' if yolo_on else 'OFF'}]")
        if key == ord('d'):
            depth_on = not depth_on
            if depth_on:
                yolo_on = True
            print(f"[Depth {'ON' if depth_on else 'OFF'}]")
        if key == ord('m'):
            show_disparity = not show_disparity
            print(f"[Disparity map {'ON' if show_disparity else 'OFF'}]")