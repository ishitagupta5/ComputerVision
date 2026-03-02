# multicam/demo_stereo.py
# Stereo demo: LEFT | RIGHT | DISPARITY | "DEPTH PERCEPTION" overlays
#
# Uses YOUR GPU Sobel implementation via: ../GPU/sobel_gpu_api.py
#
# Controls:
#   q = quit
#   p = pause/resume
#   n = step one frame (when paused)
#   s = save screenshot (tiled output) to ./outputs/
#   e = toggle Sobel edge overlay ON TOP of RGB (per-view)
#   d = toggle disparity/depth panels
#   o = toggle disparity heatmap overlay on LEFT RGB (best for demos)
#   r = restart

import os
import time
import cv2
import traceback
import numpy as np

from stereo_loader import StereoKittiLoader

# ---- GPU Sobel import (same pattern as your triplet demo) ----
import sys
sys.path.append("../GPU")

GPU_AVAILABLE = True
try:
    from sobel_gpu_api import sobel_gpu_edges
except Exception as e:
    print("[GPU Sobel unavailable — using CPU Sobel fallback]")
    print("Reason:", e)
    GPU_AVAILABLE = False

    def sobel_gpu_edges(frame_bgr, threshold=80):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.convertScaleAbs(mag)
        _, edges = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
        return edges

# ---- Config ----
DATAROOT = "../data"          # expects ../data/image_2 and ../data/image_3
RESIZE = (640, 360)           # bump to (960,540) if your laptop can handle it
OUTDIR = "./outputs"

# For "approx metric" only (for perception + demo)
BASELINE_M = 0.075
FOCAL_PX = 700.0

# WLS filter support (opencv-contrib)
HAS_XIMGPROC = hasattr(cv2, "ximgproc")


def put_label(img, text, x=10, y=28):
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (0, 255, 0), 2, cv2.LINE_AA
    )


def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)


def tile_panels(panels):
    return cv2.hconcat(panels)


def overlay_edges_on_rgb(frame_bgr, edges_gray, alpha=0.45):
    """
    Overlays edges (green) on top of RGB instead of replacing the image.
    Looks WAY cleaner for demos.
    """
    overlay = frame_bgr.copy()
    edges_gray = cv2.erode(edges_gray, np.ones((2,2), np.uint8), iterations=1)
    overlay[edges_gray > 0] = (0, 255, 0)
    return cv2.addWeighted(frame_bgr, 1 - alpha, overlay, alpha, 0)


def colorize_disparity(disp_float):
    """
    Colorize disparity for visualization.
    """
    disp = disp_float.copy()
    disp[disp < 0] = 0
    mx = float(np.max(disp)) if disp.size else 0.0
    if mx > 1e-6:
        disp_norm = (disp / mx * 255.0).astype(np.uint8)
    else:
        disp_norm = np.zeros_like(disp, dtype=np.uint8)
    return cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)


def compute_disparity_basic(left_bgr, right_bgr):
    """
    Basic OpenCV SGBM disparity (fallback).
    """
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    min_disp = 0
    num_disp = 192   # divisible by 16
    block_size = 7   # odd

    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=12,
        speckleWindowSize=120,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disp


def compute_disparity_wls(left_bgr, right_bgr):
    """
    SGBM + WLS filtering (MUCH cleaner). Requires opencv-contrib-python.
    Falls back to basic if not available.
    """
    if not HAS_XIMGPROC:
        return compute_disparity_basic(left_bgr, right_bgr)

    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    min_disp = 0
    num_disp = 192
    block_size = 7

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=12,
        speckleWindowSize=120,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    dispL = left_matcher.compute(left_gray, right_gray).astype(np.int16)
    dispR = right_matcher.compute(right_gray, left_gray).astype(np.int16)

    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(8000)        # try 5000–12000
    wls.setSigmaColor(1.5)     # try 1.0–2.0

    filtered = wls.filter(dispL, left_gray, None, dispR)
    disp = filtered.astype(np.float32) / 16.0
    return disp


def disparity_to_depth_approx(disp, baseline_m=BASELINE_M, focal_px=FOCAL_PX):
    """
    Approx depth map (meters-ish) from disparity: Z = f*B/d
    Not fully calibrated unless you use real intrinsics/baseline.
    """
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 1.0
    depth[valid] = (focal_px * baseline_m) / disp[valid]
    depth[~valid] = 0.0
    return depth


def depth_to_grayscale(depth_m, max_m=30.0):
    """
    Near = bright, far = dark.
    """
    d = depth_m.copy()
    d[d < 0] = 0
    d[d > max_m] = max_m
    if max_m <= 0:
        return np.zeros_like(d, dtype=np.uint8)
    gray = (255.0 * (1.0 - (d / max_m))).astype(np.uint8)
    return gray


def overlay_disparity_on_rgb(rgb_bgr, disp_color_bgr, alpha=0.45):
    """
    Heatmap overlay on RGB for an intuitive "depth perception" view.
    """
    return cv2.addWeighted(rgb_bgr, 1 - alpha, disp_color_bgr, alpha, 0)


def main():
    ensure_outdir()
    loader = StereoKittiLoader(DATAROOT, size=RESIZE)
    print("[debug] stereo loader created (ximgproc/WLS =", HAS_XIMGPROC, ")")

    sobel_on = False
    show_disp = True
    overlay_on = True  # best demo mode

    paused = False
    step_once = False
    last_time = time.time()

    while True:
        if not paused or step_once:
            step_once = False

            result = loader.next()
            if result is None:
                print("[debug] end of sequence, restarting")
                loader.restart()
                continue

            t, left, right = result

            # Compute disparity (WLS if available)
            disp = compute_disparity_wls(left, right) if show_disp or overlay_on else None
            disp_color = colorize_disparity(disp) if disp is not None else None

            # Prepare left/right display
            left_view = left
            right_view = right

            # Optional Sobel overlay ON TOP of RGB
            if sobel_on:
                left_edges = sobel_gpu_edges(left)
                right_edges = sobel_gpu_edges(right)
                left_view = overlay_edges_on_rgb(left_view, left_edges, alpha=0.45)
                right_view = overlay_edges_on_rgb(right_view, right_edges, alpha=0.45)

            # Optional disparity heatmap overlay on left RGB (most intuitive)
            if overlay_on and disp_color is not None:
                left_view = overlay_disparity_on_rgb(left_view, disp_color, alpha=0.40)

            put_label(left_view, "LEFT (stereo)")
            put_label(right_view, "RIGHT (stereo)")

            panels = [left_view, right_view]

            center_depth = None

            if show_disp and disp_color is not None:
                put_label(disp_color, "DISPARITY (px)")

                depth = disparity_to_depth_approx(disp)
                depth_gray = depth_to_grayscale(depth, max_m=30.0)
                depth_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
                put_label(depth_bgr, "DEPTH approx (near=bright)")

                # Demo-friendly readout (approx)
                h, w = depth.shape
                center_depth = float(depth[h // 2, w // 2])

                panels += [disp_color, depth_bgr]

            tiled = tile_panels(panels)

            now = time.time()
            fps = 1.0 / max(1e-6, now - last_time)
            last_time = now

            put_label(
                tiled,
                f"frame={t}   FPS~{fps:.1f}   Sobel={'ON' if sobel_on else 'OFF'}   Disp={'ON' if show_disp else 'OFF'}   Overlay={'ON' if overlay_on else 'OFF'}",
                20, 60
            )

            if center_depth is not None and center_depth > 0:
                put_label(tiled, f"Center depth (approx) ≈ {center_depth:.2f} m", 20, 90)
            else:
                put_label(tiled, "Center depth (approx) ≈ N/A", 20, 90)

            cv2.imshow("Stereo Depth Demo", tiled)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("n"):
            paused = True
            step_once = True
        if key == ord("e"):
            if not GPU_AVAILABLE:
                print("[info] CPU Sobel active (GPU import failed)")
            sobel_on = not sobel_on
        if key == ord("d"):
            show_disp = not show_disp
        if key == ord("o"):
            overlay_on = not overlay_on
        if key == ord("r"):
            loader.restart()
        if key == ord("s"):
            ensure_outdir()
            outpath = os.path.join(OUTDIR, f"stereo_demo_{int(time.time())}.png")
            cv2.imwrite(outpath, tiled)
            print("[saved]", outpath)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("Press Enter to close...")