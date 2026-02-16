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

import os
import time
import cv2

from triplet_loader import TripletLoader

# Import your GPU Sobel API (keep GPU folder as-is; just import it)
import sys
sys.path.append("../GPU")
from sobel_gpu_api import sobel_gpu_edges, overlay_edges_red


DATAROOT = "../data/nuscenes"   # dataroot must contain samples/ and v1.0-mini/
OUTDIR = "./outputs"
RESIZE = (640, 360)            # (w, h)


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def put_label(img, text, x=10, y=28):
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (0, 255, 0), 2, cv2.LINE_AA
    )


def tile3(left, front, right):
    # left/front/right are same size => side-by-side
    return cv2.hconcat([left, front, right])


def main():
    ensure_outdir(OUTDIR)

    loader = TripletLoader(DATAROOT, size=RESIZE)

    sobel_on = False
    paused = False
    step_once = False

    last_time = time.time()
    fps = 0.0

    # For screenshot naming even when paused before first frame
    t = 0
    tiled = None

    while True:
        # Only pull a new synchronized triplet if not paused (or if stepping)
        if not paused or step_once:
            step_once = False

            result = loader.next()
            if result is None:
                # End of scene: restart for convenience
                loader = TripletLoader(DATAROOT, size=RESIZE)
                continue

            t, left, front, right = result

            # Optional: run YOUR GPU Sobel and overlay edges onto original frames
            if sobel_on:
                left_edges = sobel_gpu_edges(left)
                front_edges = sobel_gpu_edges(front)
                right_edges = sobel_gpu_edges(right)

                left_disp = overlay_edges_red(left, left_edges, alpha=0.70)
                front_disp = overlay_edges_red(front, front_edges, alpha=0.70)
                right_disp = overlay_edges_red(right, right_edges, alpha=0.70)
            else:
                left_disp, front_disp, right_disp = left, front, right

            # Label each panel
            put_label(left_disp, "CAM_FRONT_LEFT")
            put_label(front_disp, "CAM_FRONT")
            put_label(right_disp, "CAM_FRONT_RIGHT")

            # Tile into one image
            tiled = tile3(left_disp, front_disp, right_disp)

            # FPS estimate (loop timing)
            now = time.time()
            dt = max(1e-6, now - last_time)
            fps = 1.0 / dt
            last_time = now

            put_label(
                tiled,
                f"t={t}   FPS~{fps:.1f}   GPU Sobel={'ON' if sobel_on else 'OFF'}   (p pause, n step, e sobel, s save)",
                20, 60
            )

            cv2.imshow("nuScenes Triplet Sync (LEFT | FRONT | RIGHT)", tiled)

        # Key handling (always active)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("p"):
            paused = not paused

        if key == ord("n"):
            paused = True
            step_once = True

        if key == ord("e"):
            sobel_on = not sobel_on

        if key == ord("r"):
            loader = TripletLoader(DATAROOT, size=RESIZE)

        if key == ord("s"):
            if tiled is not None:
                out_path = os.path.join(OUTDIR, f"triplet_t{t:04d}_sobel{int(sobel_on)}.png")
                cv2.imwrite(out_path, tiled)
                print(f"[saved] {out_path}")
            else:
                print("[warn] No frame yet to save.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
