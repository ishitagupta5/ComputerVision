# multicam/demo_stereo.py
# Demo: Stereo overlap (LEFT | RIGHT) + disparity/depth visualization.
#
# Uses YOUR GPU Sobel implementation via: ../GPU/sobel_gpu_api.py
#
# Controls:
#   q = quit
#   p = pause/resume
#   n = step one frame (when paused)
#   s = save screenshot (tiled output) to ./outputs/
#   e = toggle GPU Sobel edge overlay (per-view)
#   d = toggle disparity/depth panels
#   r = restart

import os
import time
import cv2
import traceback
import numpy as np

from stereo_loader import StereoKittiLoader

# Try GPU Sobel import, fallback to CPU
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


# ✅ set this to your dataset root that contains image_2 and image_3
DATAROOT = "../data/drivingstereo"   # change if needed
RESIZE = (640, 360)
OUTDIR = "./outputs"


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


def compute_disparity_sgbm(left_bgr, right_bgr):
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    # Demo defaults
    min_disp = 0
    num_disp = 96      # divisible by 16
    block_size = 7     # odd

    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disp


def colorize_disparity(disp_float):
    disp = disp_float.copy()
    disp[disp < 0] = 0
    mx = float(np.max(disp)) if disp.size else 0.0
    if mx > 1e-6:
        disp_norm = (disp / mx * 255.0).astype(np.uint8)
    else:
        disp_norm = np.zeros_like(disp, dtype=np.uint8)
    return cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)


def disparity_to_depth_approx(disp, baseline_m=0.075, focal_px=700.0):
    """
    Approximate depth (meters): Z = f * B / disparity
    For your talk: call this "approximate metric depth" unless you use calibration.
    """
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 1.0
    depth[valid] = (focal_px * baseline_m) / disp[valid]
    depth[~valid] = 0.0
    return depth


def depth_to_grayscale(depth_m, max_m=30.0):
    d = depth_m.copy()
    d[d < 0] = 0
    d[d > max_m] = max_m
    if max_m <= 0:
        return np.zeros_like(d, dtype=np.uint8)
    gray = (255.0 * (1.0 - (d / max_m))).astype(np.uint8)  # near=bright
    return gray


def main():
    ensure_outdir()
    loader = StereoKittiLoader(DATAROOT, size=RESIZE)
    print("[debug] stereo loader created")

    sobel_on = False
    show_disp = True

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

            # Per-view Sobel overlay (like your triplet demo)
            if sobel_on:
                left_edges = sobel_gpu_edges(left)
                right_edges = sobel_gpu_edges(right)
                left_disp = cv2.cvtColor(left_edges, cv2.COLOR_GRAY2BGR)
                right_disp = cv2.cvtColor(right_edges, cv2.COLOR_GRAY2BGR)
            else:
                left_disp, right_disp = left, right

            put_label(left_disp, "LEFT (stereo)")
            put_label(right_disp, "RIGHT (stereo)")

            panels = [left_disp, right_disp]
            center_depth = None

            if show_disp:
                disp = compute_disparity_sgbm(left, right)
                disp_color = colorize_disparity(disp)
                put_label(disp_color, "DISPARITY (px)")

                depth = disparity_to_depth_approx(disp, baseline_m=0.075, focal_px=700.0)
                depth_gray = depth_to_grayscale(depth, max_m=30.0)
                depth_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
                put_label(depth_bgr, "DEPTH approx (near=bright)")

                # Simple distance readout: center pixel (demo-friendly)
                h, w = depth.shape
                center_depth = float(depth[h // 2, w // 2])

                panels += [disp_color, depth_bgr]

            tiled = tile_panels(panels)

            now = time.time()
            fps = 1.0 / max(1e-6, now - last_time)
            last_time = now

            put_label(
                tiled,
                f"frame={t}   FPS~{fps:.1f}   Sobel={'ON' if sobel_on else 'OFF'}   Disp={'ON' if show_disp else 'OFF'}",
                20, 60
            )
            if center_depth is not None and center_depth > 0:
                put_label(tiled, f"Center depth ≈ {center_depth:.2f} m", 20, 90)

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