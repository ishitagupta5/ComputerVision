#include <cstdio>
#include <cstdint>
#include <cmath>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

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

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::printf("Usage: %s input_video output_video\n", argv[0]);
        return 1;
    }

    const char* input_path  = argv[1];
    const char* output_path = argv[2];

    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::printf("ERROR: Cannot open video file: %s\n", input_path);
        return 1;
    }

    int width  = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0; // fallback if FPS is not set

    std::printf("Processing video: %s (%dx%d @ %.2f fps)\n",
                input_path, width, height, fps);

    // We’ll write grayscale output (single channel) but OpenCV VideoWriter
    // often expects 3-channel BGR, so we’ll convert edges to BGR for writing.
    int fourcc = VideoWriter::fourcc('M','J','P','G');   // or 'X','V','I','D'
    VideoWriter writer(output_path, fourcc, fps, Size(width, height), true);
    if (!writer.isOpened()) {
        std::printf("ERROR: Cannot open output video: %s\n", output_path);
        return 1;
    }

    size_t num_pixels = (size_t)width * (size_t)height;
    size_t num_bytes  = num_pixels * sizeof(uint8_t);

    // Host Mats
    Mat frame, gray, sobel_gray(height, width, CV_8UC1), sobel_bgr;

    // Device buffers
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    cudaError_t err;

    dim3 block(16, 16);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    int threshold = 150;

    err = cudaMalloc((void**)&d_src, num_bytes);
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_src failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&d_dst, num_bytes);
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_dst failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_src);
        return 1;
    }

    // Optional: timing per frame (kernel only)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int frame_idx = 0;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;  // end of video

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Ensure data is continuous (usually true for video frames)
        if (!gray.isContinuous()) {
            gray = gray.clone();
        }

        // Copy frame to GPU
        err = cudaMemcpy(d_src, gray.data, num_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::printf("cudaMemcpy H->D failed: %s\n", cudaGetErrorString(err));
            break;
        }

        // Run Sobel on GPU (timed)
        cudaEventRecord(start);
        sobelKernel<<<grid, block>>>(d_src, d_dst, width, height, threshold);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        // Copy result back
        err = cudaMemcpy(sobel_gray.data, d_dst, num_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::printf("cudaMemcpy D->H failed: %s\n", cudaGetErrorString(err));
            break;
        }

        // Convert grayscale edges to BGR for video writer
        cvtColor(sobel_gray, sobel_bgr, COLOR_GRAY2BGR);
        writer.write(sobel_bgr);

        std::printf("Frame %d: kernel time = %.3f ms\n", frame_idx++, ms);
    }

    cap.release();
    writer.release();
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::printf("Saved GPU Sobel video to %s\n", output_path);
    return 0;
}
