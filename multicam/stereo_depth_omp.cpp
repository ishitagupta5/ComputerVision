/*
 * stereo_depth_omp.cpp
 * OpenMP-parallelized stereo depth pipeline
 * 
 * Slab decomposition: each of 8 threads owns (height / 8) contiguous rows.
 * This maximizes cache locality — each thread works on a contiguous memory
 * block with no overlap or contention.
 *
 * Compile on Windows (MSVC):
 *   cl /O2 /openmp /LD /Fe:stereo_depth_omp.dll stereo_depth_omp.cpp
 *
 * Compile on Windows (MinGW/g++):
 *   g++ -O3 -fopenmp -shared -o stereo_depth_omp.dll stereo_depth_omp.cpp
 *
 * Compile on Linux:
 *   g++ -O3 -fopenmp -shared -fPIC -o stereo_depth_omp.so stereo_depth_omp.cpp
 *
 * Exports:
 *   census_transform()       - parallel census transform preprocessing
 *   stereo_disparity_sgbm()  - parallel SAD disparity with slab decomposition
 *   disparity_to_depth()     - parallel depth from disparity via Z = fB/d
 *   rectify_remap()          - parallel bilinear remap for rectification
 *   get_median_depth_roi()   - depth for a bounding box region
 */
 
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <omp.h>
 
#ifdef _WIN32
  #define EXPORT extern "C" __declspec(dllexport)
#else
  #define EXPORT extern "C"
#endif
 
static const int NUM_THREADS = 8;
 
// ============================================================
// Helper: compute slab boundaries for a given thread
// Thread tid owns rows [row_start, row_end)
// Last thread absorbs any remainder rows
// ============================================================
static inline void slab_bounds(int tid, int num_threads, int total_rows,
                               int& row_start, int& row_end)
{
    int rows_per_thread = total_rows / num_threads;
    row_start = tid * rows_per_thread;
    row_end = (tid == num_threads - 1) ? total_rows : row_start + rows_per_thread;
}
 
// ============================================================
// Census Transform — parallel slab decomposition
//
// Each pixel is encoded as a binary pattern of comparisons with
// its neighbors: 1 if center > neighbor, 0 otherwise.
// Captures local structure (edges + texture) while being robust
// to lighting/exposure differences between cameras.
//
// kernel_size must be odd (recommended: 7 or 9)
// ============================================================
EXPORT void census_transform(
    const uint8_t* src,       // input grayscale HxW
    uint8_t* dst,             // output census-transformed HxW
    int height,
    int width,
    int kernel_size           // e.g. 7
)
{
    omp_set_num_threads(NUM_THREADS);
 
    int half = kernel_size / 2;
 
    // Zero the output (border pixels stay 0)
    std::memset(dst, 0, height * width);
 
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
 
        // Each thread owns a slab of rows within the valid region
        int valid_start = half;
        int valid_end = height - half;
        int valid_rows = valid_end - valid_start;
 
        int slab_start, slab_end;
        slab_bounds(tid, NUM_THREADS, valid_rows, slab_start, slab_end);
        slab_start += valid_start;
        slab_end += valid_start;
 
        for (int y = slab_start; y < slab_end; y++) {
            for (int x = half; x < width - half; x++) {
                int center = (int)src[y * width + x];
                uint32_t census_val = 0;
                int bit = 0;
 
                for (int dy = -half; dy <= half && bit < 32; dy++) {
                    for (int dx = -half; dx <= half && bit < 32; dx++) {
                        if (dy == 0 && dx == 0) continue;
                        int neighbor = (int)src[(y + dy) * width + (x + dx)];
                        if (center > neighbor) {
                            census_val |= (1u << bit);
                        }
                        bit++;
                    }
                }
 
                // Map to uint8: use upper 8 bits of the 32-bit pattern
                // This preserves the most significant structural info
                dst[y * width + x] = (uint8_t)(census_val & 0xFF)
                                   ^ (uint8_t)((census_val >> 8) & 0xFF)
                                   ^ (uint8_t)((census_val >> 16) & 0xFF)
                                   ^ (uint8_t)((census_val >> 24) & 0xFF);
            }
        }
    }
}
 
// ============================================================
// SAD block matching — slab decomposition
//
// Image is divided into 8 horizontal slabs:
//   Thread 0: rows 0        .. height/8 - 1
//   Thread 1: rows height/8 .. 2*height/8 - 1
//   ...
//   Thread 7: rows 7*height/8 .. height - 1
//
// Each thread processes all columns within its slab, computing
// SAD for every disparity candidate. No inter-thread communication
// needed since each pixel's disparity depends only on its row
// (within the block window).
// ============================================================
EXPORT void stereo_disparity_sgbm(
    const uint8_t* left,      // left rectified grayscale HxW
    const uint8_t* right,     // right rectified grayscale HxW
    float* disparity,         // output disparity map HxW (float)
    int height,
    int width,
    int num_disparities,      // e.g. 128
    int block_size            // e.g. 7 (odd)
)
{
    omp_set_num_threads(NUM_THREADS);
 
    int half = block_size / 2;
 
    // Zero output so border pixels are 0
    std::memset(disparity, 0, height * width * sizeof(float));
 
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
 
        // Slab within the valid processing region
        int valid_start = half;
        int valid_end = height - half;
        int valid_rows = valid_end - valid_start;
 
        int slab_start, slab_end;
        slab_bounds(tid, NUM_THREADS, valid_rows, slab_start, slab_end);
        slab_start += valid_start;
        slab_end += valid_start;
 
        // Per-thread cost buffer (allocated once, reused per pixel)
        int* costs = new int[num_disparities];
 
        for (int y = slab_start; y < slab_end; y++) {
            for (int x = half + num_disparities; x < width - half; x++) {
                int best_disp = 0;
                int best_cost = INT32_MAX;
 
                std::memset(costs, 0, num_disparities * sizeof(int));
 
                // Compute SAD for each disparity
                for (int d = 0; d < num_disparities; d++) {
                    int sad = 0;
                    for (int wy = -half; wy <= half; wy++) {
                        const uint8_t* lrow = left + (y + wy) * width;
                        const uint8_t* rrow = right + (y + wy) * width;
                        for (int wx = -half; wx <= half; wx++) {
                            int lx = x + wx;
                            int rx = lx - d;
                            sad += std::abs((int)lrow[lx] - (int)rrow[rx]);
                        }
                    }
                    costs[d] = sad;
                    if (sad < best_cost) {
                        best_cost = sad;
                        best_disp = d;
                    }
                }
 
                // Sub-pixel refinement (parabola fit)
                float sub_disp = (float)best_disp;
                if (best_disp > 0 && best_disp < num_disparities - 1) {
                    int c_left  = costs[best_disp - 1];
                    int c_right = costs[best_disp + 1];
                    int c_center = costs[best_disp];
                    int denom = c_left + c_right - 2 * c_center;
                    if (denom != 0) {
                        sub_disp = (float)best_disp
                                 + 0.5f * (float)(c_left - c_right) / (float)denom;
                    }
                }
 
                // Uniqueness check: reject if second-best is too close
                int second_best = INT32_MAX;
                for (int d = 0; d < num_disparities; d++) {
                    if (d != best_disp && costs[d] < second_best) {
                        second_best = costs[d];
                    }
                }
                float uniqueness = (second_best > 0)
                    ? (float)(second_best - best_cost) / (float)second_best
                    : 1.0f;
 
                if (uniqueness < 0.05f) {
                    disparity[y * width + x] = 0.0f;
                } else {
                    disparity[y * width + x] = sub_disp;
                }
            }
        }
 
        delete[] costs;
    }
}
 
// ============================================================
// Disparity to depth: Z = (focal * baseline) / disparity
// Slab decomposition across rows
// ============================================================
EXPORT void disparity_to_depth(
    const float* disparity,   // HxW disparity map
    float* depth,             // HxW output depth in meters
    int height,
    int width,
    float focal_px,           // focal length in pixels
    float baseline_m          // baseline in meters
)
{
    omp_set_num_threads(NUM_THREADS);
 
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
 
        int slab_start, slab_end;
        slab_bounds(tid, NUM_THREADS, height, slab_start, slab_end);
 
        for (int y = slab_start; y < slab_end; y++) {
            for (int x = 0; x < width; x++) {
                int i = y * width + x;
                float d = disparity[i];
                if (d > 1.0f) {
                    depth[i] = (focal_px * baseline_m) / d;
                } else {
                    depth[i] = 0.0f;
                }
            }
        }
    }
}
 
// ============================================================
// Bilinear remap for stereo rectification
// Slab decomposition across rows
// ============================================================
EXPORT void rectify_remap(
    const uint8_t* src,       // source grayscale HxW
    uint8_t* dst,             // destination grayscale HxW
    const float* map_x,       // HxW float x-coordinates
    const float* map_y,       // HxW float y-coordinates
    int height,
    int width
)
{
    omp_set_num_threads(NUM_THREADS);
 
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
 
        int slab_start, slab_end;
        slab_bounds(tid, NUM_THREADS, height, slab_start, slab_end);
 
        for (int y = slab_start; y < slab_end; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                float fx = map_x[idx];
                float fy = map_y[idx];
 
                int x0 = (int)std::floor(fx);
                int y0 = (int)std::floor(fy);
                int x1 = x0 + 1;
                int y1 = y0 + 1;
 
                if (x0 < 0 || y0 < 0 || x1 >= width || y1 >= height) {
                    dst[idx] = 0;
                    continue;
                }
 
                float dx = fx - (float)x0;
                float dy = fy - (float)y0;
 
                float val = (1.0f - dx) * (1.0f - dy) * src[y0 * width + x0]
                          + dx * (1.0f - dy) * src[y0 * width + x1]
                          + (1.0f - dx) * dy * src[y1 * width + x0]
                          + dx * dy * src[y1 * width + x1];
 
                dst[idx] = (uint8_t)std::min(255.0f, std::max(0.0f, val));
            }
        }
    }
}
 
// ============================================================
// Median depth for a bounding box ROI
// Returns median of valid depths in feet
// ============================================================
EXPORT float get_median_depth_roi(
    const float* depth,       // HxW depth map in meters
    int img_width,
    int x1, int y1,           // top-left of bounding box
    int x2, int y2,           // bottom-right of bounding box
    float margin              // shrink margin (0.1 = 10%)
)
{
    int bw = x2 - x1;
    int bh = y2 - y1;
    int mx = (int)(bw * margin);
    int my = (int)(bh * margin);
 
    int cx1 = x1 + mx;
    int cy1 = y1 + my;
    int cx2 = x2 - mx;
    int cy2 = y2 - my;
 
    if (cx1 >= cx2 || cy1 >= cy2) return -1.0f;
 
    // Collect valid depth values
    int capacity = (cx2 - cx1) * (cy2 - cy1);
    float* valid = new float[capacity];
    int count = 0;
 
    for (int y = cy1; y < cy2; y++) {
        for (int x = cx1; x < cx2; x++) {
            float d = depth[y * img_width + x];
            if (d > 0.1f && d < 30.0f) {
                valid[count++] = d;
            }
        }
    }
 
    if (count < 10) {
        delete[] valid;
        return -1.0f;
    }
 
    // Partial sort for median
    std::nth_element(valid, valid + count / 2, valid + count);
    float median_m = valid[count / 2];
 
    delete[] valid;
 
    // Convert meters to feet
    return median_m * 3.28084f;
}