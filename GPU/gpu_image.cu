#include <cstdio>
#include <cstdint>
#include <cmath>

#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

__global__
void sobelKernel(const uint8_t* __restrict src,
                 uint8_t* __restrict dst,
                 int width, int height,
                 int threshold)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x; // column
    int x = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x >= height || y >= width) return;

    // Zero out border pixels
    if (x == 0 || x == height - 1 || y == 0 || y == width - 1) {
        dst[x * width + y] = 0;
        return;
    }

    int idx = x * width + y;

    int gx =
        src[(x + 1) * width + (y + 1)] -
        src[(x + 1) * width + (y - 1)] +
        2 * src[x * width + (y + 1)] -
        2 * src[x * width + (y - 1)] +
        src[(x - 1) * width + (y + 1)] -
        src[(x - 1) * width + (y - 1)];

    int gy =
        src[(x + 1) * width + (y + 1)] +
        2 * src[(x + 1) * width + y] +
        src[(x + 1) * width + (y - 1)] -
        src[(x - 1) * width + (y + 1)] -
        2 * src[(x - 1) * width + y] -
        src[(x - 1) * width + (y - 1)];

    float mag = sqrtf((float)gx * gx + (float)gy * gy);

    dst[idx] = (mag > threshold) ? 255 : 0;
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::printf("Usage: %s input.png output.png\n", argv[0]);
        return 1;
    }

    const char* input_path  = argv[1];
    const char* output_path = argv[2];

     int verbose = 1;              // default: print
    if (argc >= 4) {
        verbose = std::atoi(argv[3]);   // 0 = quiet, 1 = verbose
    }

    int width, height, bpp;
    // force 1 channel (grayscale)
    uint8_t* h_img = stbi_load(input_path, &width, &height, &bpp, 1);
    if (!h_img) {
        std::printf("ERROR: Could not load image: %s\n", input_path);
        return 1;
    }

if (verbose) {
    std::printf("Loaded %s (%dx%d)\n", input_path, width, height);
}
    size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    size_t num_bytes  = num_pixels * sizeof(uint8_t);

    uint8_t* h_out = (uint8_t*)std::calloc(num_pixels, sizeof(uint8_t));
    if (!h_out) {
        std::printf("ERROR: Could not allocate host output buffer\n");
        stbi_image_free(h_img);
        return 1;
    }

    uint8_t *d_src = nullptr, *d_dst = nullptr;
    cudaError_t err;

    //declare everything BEFORE any goto cleanup 
    dim3 block(16, 16);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    int   threshold = 150;          // same as CPU version
    const int N     = 50;           // number of timed runs
    float ms        = 0.0f;         // for timing

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // device alloc
    err = cudaMalloc((void**)&d_src, num_bytes);
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_src failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMalloc((void**)&d_dst, num_bytes);
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_dst failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // copy input image to device
    err = cudaMemcpy(d_src, h_img, num_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::printf("cudaMemcpy H→D failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    //warmup run 
    sobelKernel<<<grid, block>>>(d_src, d_dst, width, height, threshold);
    cudaDeviceSynchronize();

    //timed runs 
    cudaEventRecord(start);
    for (int i = 0; i < N; ++i) {
        sobelKernel<<<grid, block>>>(d_src, d_dst, width, height, threshold);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    ms /= N; // average per kernel

if (verbose) {
    std::printf("[GPU] Sobel kernel avg time over %d runs: %.3f ms\n", N, ms);
}
    // copy result of last run back
    err = cudaMemcpy(h_out, d_dst, num_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::printf("cudaMemcpy D→H failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    if (!stbi_write_png(output_path, width, height, 1, h_out, width)) {
        std::printf("ERROR: Could not write output image: %s\n", output_path);
    } else {
if (verbose) {
    std::printf("Saved result to %s\n", output_path);
}    }

cleanup:
    cudaFree(d_src);
    cudaFree(d_dst);
    stbi_image_free(h_img);
    std::free(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
