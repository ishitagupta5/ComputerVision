# GPU/sobel_gpu_api.py
import cupy as cp
import numpy as np
import cv2

# Compile once at import time
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

def sobel_gpu_edges(frame_bgr, threshold=150):
    """
    Input: BGR uint8 image (H,W,3)
    Output: edges uint8 (H,W) on CPU
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    d_src = cp.asarray(gray)
    d_dst = cp.zeros_like(d_src)

    block = (16, 16)
    grid = ((w + 15) // 16, (h + 15) // 16)

    sobel_kernel(grid, block, (
        d_src, d_dst,
        np.int32(w), np.int32(h),
        np.int32(threshold)
    ))

    cp.cuda.Stream.null.synchronize()
    edges = cp.asnumpy(d_dst)
    return edges

def edges_to_bgr(edges_gray):
    return cv2.cvtColor(edges_gray, cv2.COLOR_GRAY2BGR)

def overlay_edges_red(frame_bgr, edges_gray, alpha=0.7):
    mask = edges_gray > 0
    overlay = frame_bgr.copy()
    overlay[mask] = (0, 0, 255)
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
