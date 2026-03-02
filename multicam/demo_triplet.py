# multicam/demo_triplet.py
# Demo: synchronized multi-camera triplets (LEFT | FRONT | RIGHT) from nuScenes mini.
#
# Uses YOUR GPU Sobel implementation via: ../GPU/sobel_gpu_api.py
#
# Controls:
#   q = quit
#   p = pause/resume
#   n = step one frame (when paused)
#   s = save screenshot (tiled output) to ./outputs/
#   e = toggle GPU Sobel edge overlay (per-view)
#   r = restart (reopen first scene)

# multicam/demo_triplet.py

import os
import time
import cv2
import traceback

from triplet_loader import TripletLoader

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


DATAROOT = "../data/nuscenes"
RESIZE = (640, 360)


def put_label(img, text, x=10, y=28):
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (0, 255, 0), 2, cv2.LINE_AA
    )


def tile3(left, front, right):
    return cv2.hconcat([left, front, right])


def main():

    loader = TripletLoader(DATAROOT, size=RESIZE)
    print("[debug] loader created")

    sobel_on = False
    paused = False
    step_once = False

    last_time = time.time()

    while True:

        if not paused or step_once:
            step_once = False

            result = loader.next()
            if result is None:
                print("[debug] end of scene, restarting")
                loader = TripletLoader(DATAROOT, size=RESIZE)
                continue

            t, left, front, right = result

            if sobel_on:
                left_edges = sobel_gpu_edges(left)
                front_edges = sobel_gpu_edges(front)
                right_edges = sobel_gpu_edges(right)

                left_disp  = cv2.cvtColor(left_edges,  cv2.COLOR_GRAY2BGR)
                front_disp = cv2.cvtColor(front_edges, cv2.COLOR_GRAY2BGR)
                right_disp = cv2.cvtColor(right_edges, cv2.COLOR_GRAY2BGR)
            else:
                left_disp, front_disp, right_disp = left, front, right

            put_label(left_disp, "CAM_FRONT_LEFT")
            put_label(front_disp, "CAM_FRONT")
            put_label(right_disp, "CAM_FRONT_RIGHT")

            tiled = tile3(left_disp, front_disp, right_disp)

            now = time.time()
            fps = 1.0 / max(1e-6, now - last_time)
            last_time = now

            put_label(
                tiled,
                f"t={t}   FPS~{fps:.1f}   Sobel={'ON' if sobel_on else 'OFF'}",
                20, 60
            )

            cv2.imshow("nuScenes Triplet Sync", tiled)

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
                print("[info] CPU Sobel active (Intel GPU machine)")
            sobel_on = not sobel_on

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("Press Enter to close...")
