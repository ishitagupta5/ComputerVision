#!/usr/bin/env python3
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
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= height || y >= width) return;

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

# ---------------------------------------
# MAIN
# ---------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python3 sobel_video_gpu.py input.mp4 output.mp4 [verbose]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    verbose = int(sys.argv[3]) if len(sys.argv) >= 4 else 1

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {input_path}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 60.0

    if verbose:
        print(f"Processing video on GPU: {input_path} ({width}x{height} @ {fps:.2f} fps)")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
        False
    )

    if not writer.isOpened():
        print("ERROR: Cannot open output file:", output_path)
        sys.exit(1)

    block = (16, 16)
    grid = ((width + 15) // 16, (height + 15) // 16)

    frame_count = 0
    start_total = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Upload to GPU
        d_src = cp.asarray(gray)
        d_dst = cp.zeros_like(d_src)

        # ---- GPU timing ----
        gpustart = time.time()

        sobel_kernel(grid, block, (
            d_src, d_dst,
            np.int32(width), np.int32(height),
            np.int32(150)
        ))

        cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - gpustart) * 1000  # ms

        # Download
        edges = cp.asnumpy(d_dst)

        writer.write(edges)
        frame_count += 1

        if verbose and frame_count % 30 == 0:
            elapsed = time.time() - start_total
            fps_now = frame_count / elapsed
            print(f"[GPU {gpu_time:.3f} ms] Frame {frame_count}, Avg FPS={fps_now:.2f}")

    cap.release()
    writer.release()

    total = time.time() - start_total
    print(f"\n✅ Saved → {output_path}")
    print(f"Frames: {frame_count}, Total time: {total:.2f}s, Avg FPS = {frame_count/total:.2f}")

if __name__ == "__main__":
    main()
