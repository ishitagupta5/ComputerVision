#!/usr/bin/env python3
# Live GPU-based Sobel edge detection for webcam video using CuPy

import cv2
import cupy as cp
import numpy as np
import sys
import time

# ---------------------------------------
# CUDA SOBEL KERNEL
# ---------------------------------------
sobel_kernel = cp.RawKernel(r"""
extern "C" __global__
void sobel(const unsigned char* src,
           unsigned char* dst,
           int width,
           int height,
           int threshold)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x; // column
    int x = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x >= height || y >= width) return;

    // borders
    if (x == 0 || x == height - 1 || y == 0 || y == width - 1) {
        dst[x * width + y] = 0;
        return;
    }

    int idx = x * width + y;

    int gx =
        src[(x+1)*width + (y+1)] -
        src[(x+1)*width + (y-1)] +
        2 * src[x*width + (y+1)] -
        2 * src[x*width + (y-1)] +
        src[(x-1)*width + (y+1)] -
        src[(x-1)*width + (y-1)];

    int gy =
        src[(x+1)*width + (y+1)] +
        2 * src[(x+1)*width + y] +
        src[(x+1)*width + (y-1)] -
        src[(x-1)*width + (y+1)] -
        2 * src[(x-1)*width + y] -
        src[(x-1)*width + (y-1)];

    float mag = sqrtf((float)(gx*gx + gy*gy));
    dst[idx] = (mag > threshold) ? 255 : 0;
}
""", "sobel")


def main():
    # --------- CLI args ---------
    camera_index = 0
    if len(sys.argv) >= 2:
        camera_index = int(sys.argv[1])

    threshold = 150
    if len(sys.argv) >= 3:
        threshold = int(sys.argv[2])

    verbose = 1
    if len(sys.argv) >= 4:
        verbose = int(sys.argv[3])

    print(f"Opening camera {camera_index}...")
    print("Press 'q' to quit, 's' to save screenshot, '+'/'-' to adjust threshold")
    print(f"Current threshold: {threshold}")

    # --------- Open camera ---------
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        print("Try a different camera index (0, 1, 2, ...)")
        sys.exit(1)

    # Set properties (best effort)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera opened: {width}x{height} @ {fps_cam:.2f} fps (reported)")

    # --------- GPU setup ---------
    block = (16, 16)
    grid = ((width + block[0] - 1) // block[0],
            (height + block[1] - 1) // block[1])

    d_src = cp.empty((height, width), dtype=cp.uint8)
    d_dst = cp.empty((height, width), dtype=cp.uint8)

    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    fps_display = 0.0

    cv2.namedWindow("GPU Sobel - Live", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("ERROR: Failed to read frame from camera")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Upload to GPU
            upload_start = time.time()
            d_src.set(gray)
            upload_ms = (time.time() - upload_start) * 1000.0

            # Run kernel
            gpustart = time.time()
            sobel_kernel(
                grid, block,
                (
                    d_src,
                    d_dst,
                    np.int32(width),
                    np.int32(height),
                    np.int32(threshold)
                )
            )
            cp.cuda.Stream.null.synchronize()
            gpu_ms = (time.time() - gpustart) * 1000.0

            # Download result
            download_start = time.time()
            edges = cp.asnumpy(d_dst)
            download_ms = (time.time() - download_start) * 1000.0

            # Convert to BGR for overlay
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # FPS calc
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps_display = frame_count / (now - start_time)
                last_fps_time = now

            # Overlay text
            cv2.putText(edges_bgr, f"FPS: {fps_display:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(edges_bgr, f"Threshold: {threshold}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(edges_bgr, f"GPU: {gpu_ms:.2f} ms", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(edges_bgr, f"Up: {upload_ms:.2f} ms Down: {download_ms:.2f} ms", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(edges_bgr, "Press 'q' to quit, 's' to save, +/- threshold",
                        (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

            cv2.imshow("GPU Sobel - Live", edges_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                filename = f"gpu_sobel_screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, edges)
                print(f"Screenshot saved: {filename}")
            elif key == ord('+') or key == ord('='):
                threshold = min(255, threshold + 10)
                print(f"Threshold increased to: {threshold}")
            elif key == ord('-') or key == ord('_'):
                threshold = max(0, threshold - 10)
                print(f"Threshold decreased to: {threshold}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        if frame_count > 0:
            avg_fps = frame_count / elapsed
            print(f"\nProcessed {frame_count} frames in {elapsed:.2f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")
        print("Done.")


if __name__ == "__main__":
    main()
