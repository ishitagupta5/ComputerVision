/*
 * stereo_depth_omp.cpp
 * OpenMP-parallelized stereo depth pipeline
 * 
 * Slab decomposition: each of 8 threads owns (height / 8) contiguous rows.
 *
 * Optimizations:
 *   1. Census transform preprocessing
 *   2. Left-right consistency check (reject mismatched disparities)
 *   3. Confidence filtering (uniqueness ratio)
 *   4. Slab-based OpenMP parallelism across all functions
 *
 * Compile on Windows (MinGW/g++):
 *   g++ -O3 -fopenmp -shared -o stereo_depth_omp.dll stereo_depth_omp.cpp
 *
 * Compile on Windows (MSVC):
 *   cl /O2 /openmp /LD /Fe:stereo_depth_omp.dll stereo_depth_omp.cpp
 *
 * Compile on Linux:
 *   g++ -O3 -fopenmp -shared -fPIC -o stereo_depth_omp.so stereo_depth_omp.cpp
 *
 * Exports:
 *   census_transform()         - parallel census transform preprocessing
 *   stereo_disparity_sgbm()    - parallel SAD disparity (left-to-right only)
 *   stereo_disparity_sgbm_lr() - L->R and R->L disparity + consistency check
 *   disparity_to_depth()       - parallel depth from disparity via Z = fB/d
 *   rectify_remap()            - parallel bilinear remap for rectification
 *   get_median_depth_roi()     - depth for a bounding box region
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
// ============================================================
EXPORT void census_transform(
    const uint8_t* src,
    uint8_t* dst,
    int height,
    int width,
    int kernel_size
)
{
    omp_set_num_threads(NUM_THREADS);
    int half = kernel_size / 2;
    std::memset(dst, 0, height * width);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
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

                dst[y * width + x] = (uint8_t)(census_val & 0xFF)
                                   ^ (uint8_t)((census_val >> 8) & 0xFF)
                                   ^ (uint8_t)((census_val >> 16) & 0xFF)
                                   ^ (uint8_t)((census_val >> 24) & 0xFF);
            }
        }
    }
}

// ============================================================
// Internal: compute disparity for one direction
// ============================================================
static void compute_disparity_direction(
    const uint8_t* left,
    const uint8_t* right,
    float* disparity,
    int height, int width,
    int num_disparities, int block_size,
    float uniqueness_thresh,
    bool left_to_right
)
{
    int half = block_size / 2;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int valid_start = half;
        int valid_end = height - half;
        int valid_rows = valid_end - valid_start;

        int slab_start, slab_end;
        slab_bounds(tid, NUM_THREADS, valid_rows, slab_start, slab_end);
        slab_start += valid_start;
        slab_end += valid_start;

        int* costs = new int[num_disparities];

        for (int y = slab_start; y < slab_end; y++) {

            int x_start, x_end;
            if (left_to_right) {
                x_start = half + num_disparities;
                x_end = width - half;
            } else {
                x_start = half;
                x_end = width - half - num_disparities;
            }

            for (int x = x_start; x < x_end; x++) {
                int best_disp = 0;
                int best_cost = INT32_MAX;
                std::memset(costs, 0, num_disparities * sizeof(int));

                for (int d = 0; d < num_disparities; d++) {
                    int sad = 0;
                    for (int wy = -half; wy <= half; wy++) {
                        const uint8_t* ref_row;
                        const uint8_t* match_row;
                        if (left_to_right) {
                            ref_row = left + (y + wy) * width;
                            match_row = right + (y + wy) * width;
                        } else {
                            ref_row = right + (y + wy) * width;
                            match_row = left + (y + wy) * width;
                        }
                        for (int wx = -half; wx <= half; wx++) {
                            int ref_x = x + wx;
                            int match_x = left_to_right ? (ref_x - d) : (ref_x + d);
                            sad += std::abs((int)ref_row[ref_x] - (int)match_row[match_x]);
                        }
                    }
                    costs[d] = sad;
                    if (sad < best_cost) {
                        best_cost = sad;
                        best_disp = d;
                    }
                }

                // Sub-pixel refinement
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

                // Uniqueness check
                int second_best = INT32_MAX;
                for (int d = 0; d < num_disparities; d++) {
                    if (d != best_disp && costs[d] < second_best) {
                        second_best = costs[d];
                    }
                }
                float uniqueness = (second_best > 0)
                    ? (float)(second_best - best_cost) / (float)second_best
                    : 1.0f;

                if (uniqueness < uniqueness_thresh) {
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
// Simple left-to-right disparity (backward compatible)
// ============================================================
EXPORT void stereo_disparity_sgbm(
    const uint8_t* left,
    const uint8_t* right,
    float* disparity,
    int height, int width,
    int num_disparities, int block_size
)
{
    omp_set_num_threads(NUM_THREADS);
    std::memset(disparity, 0, height * width * sizeof(float));
    compute_disparity_direction(left, right, disparity,
                                height, width, num_disparities, block_size,
                                0.05f, true);
}

// ============================================================
// LEFT-RIGHT CONSISTENCY CHECK
//
// Computes disparity in both directions (L->R and R->L).
// For each pixel, if |disp_LR(x) - disp_RL(x - disp_LR(x))| > threshold,
// the pixel is rejected (set to 0).
// ============================================================
EXPORT void stereo_disparity_sgbm_lr(
    const uint8_t* left,
    const uint8_t* right,
    float* disparity,
    int height, int width,
    int num_disparities, int block_size,
    float lr_threshold
)
{
    omp_set_num_threads(NUM_THREADS);

    int total = height * width;

    float* disp_lr = new float[total]();
    float* disp_rl = new float[total]();

    compute_disparity_direction(left, right, disp_lr,
                                height, width, num_disparities, block_size,
                                0.05f, true);
    compute_disparity_direction(left, right, disp_rl,
                                height, width, num_disparities, block_size,
                                0.05f, false);

    std::memset(disparity, 0, total * sizeof(float));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int slab_start, slab_end;
        slab_bounds(tid, NUM_THREADS, height, slab_start, slab_end);

        for (int y = slab_start; y < slab_end; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                float d_lr = disp_lr[idx];

                if (d_lr <= 0.0f) {
                    continue;
                }

                int x_right = x - (int)std::round(d_lr);
                if (x_right < 0 || x_right >= width) {
                    continue;
                }

                float d_rl = disp_rl[y * width + x_right];

                if (d_rl > 0.0f && std::fabs(d_lr - d_rl) <= lr_threshold) {
                    disparity[idx] = d_lr;
                }
            }
        }
    }

    delete[] disp_lr;
    delete[] disp_rl;
}

// ============================================================
// Disparity to depth: Z = (focal * baseline) / disparity
// ============================================================
EXPORT void disparity_to_depth(
    const float* disparity,
    float* depth,
    int height, int width,
    float focal_px, float baseline_m
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
// ============================================================
EXPORT void rectify_remap(
    const uint8_t* src,
    uint8_t* dst,
    const float* map_x, const float* map_y,
    int height, int width
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
// Median depth for a bounding box ROI (in feet)
// ============================================================
EXPORT float get_median_depth_roi(
    const float* depth,
    int img_width,
    int x1, int y1,
    int x2, int y2,
    float margin
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

    std::nth_element(valid, valid + count / 2, valid + count);
    float median_m = valid[count / 2];

    delete[] valid;
    return median_m * 3.28084f;
}