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
        std::printf("Usage: %s input_video output_video [verbose]\n", argv[0]);
        std::printf("  verbose: 0 = quiet, 1 = verbose (default: 1)\n");
        return 1;
    }

    const char* input_path  = argv[1];
    const char* output_path = argv[2];
    
    int verbose = 1;  // default: print
    if (argc >= 4) {
        verbose = std::atoi(argv[3]);  // 0 = quiet, 1 = verbose
    }

    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::printf("ERROR: Cannot open video file: %s\n", input_path);
        return 1;
    }

    int width  = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0; // fallback if FPS is not set

    if (verbose) {
        std::printf("Processing video: %s (%dx%d @ %.2f fps)\n",
                    input_path, width, height, fps);
    }

    // Try H264 codec first (better compression), fallback to MJPEG if unavailable
    int fourcc = VideoWriter::fourcc('H','2','6','4');  // H264 codec
    VideoWriter writer(output_path, fourcc, fps, Size(width, height), true);
    if (!writer.isOpened()) {
        // Fallback to MJPEG if H264 not available
        fourcc = VideoWriter::fourcc('M','J','P','G');
        writer.open(output_path, fourcc, fps, Size(width, height), true);
    }
    if (!writer.isOpened()) {
        std::printf("ERROR: Cannot open output video: %s\n", output_path);
        return 1;
    }

    size_t num_pixels = (size_t)width * (size_t)height;
    size_t num_bytes  = num_pixels * sizeof(uint8_t);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    int threshold = 150;

    // Create CUDA stream for async operations
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        std::printf("cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Allocate pinned (page-locked) memory for faster transfers
    uint8_t *h_src_pinned = nullptr, *h_dst_pinned = nullptr;
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    
    err = cudaMallocHost((void**)&h_src_pinned, num_bytes);
    if (err != cudaSuccess) {
        std::printf("cudaMallocHost h_src_pinned failed: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        return 1;
    }

    err = cudaMallocHost((void**)&h_dst_pinned, num_bytes);
    if (err != cudaSuccess) {
        std::printf("cudaMallocHost h_dst_pinned failed: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_src_pinned);
        cudaStreamDestroy(stream);
        return 1;
    }

    err = cudaMalloc((void**)&d_src, num_bytes);
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_src failed: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_src_pinned);
        cudaFreeHost(h_dst_pinned);
        cudaStreamDestroy(stream);
        return 1;
    }

    err = cudaMalloc((void**)&d_dst, num_bytes);
    if (err != cudaSuccess) {
        std::printf("cudaMalloc d_dst failed: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_src_pinned);
        cudaFreeHost(h_dst_pinned);
        cudaFree(d_src);
        cudaStreamDestroy(stream);
        return 1;
    }

    Mat frame, gray;
    Mat sobel_gray(height, width, CV_8UC1);
    Mat sobel_bgr;

    int frame_idx = 0;
    
    // Read first frame
    cap >> frame;
    
    while (!frame.empty()) {
        // Convert to grayscale
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // Ensure data is continuous
        if (!gray.isContinuous()) {
            gray = gray.clone();
        }
        
        // Copy to pinned memory
        memcpy(h_src_pinned, gray.data, num_bytes);
        
        // Asynchronous copy to GPU (non-blocking!)
        err = cudaMemcpyAsync(d_src, h_src_pinned, num_bytes, 
                             cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            std::printf("cudaMemcpyAsync H->D failed: %s\n", cudaGetErrorString(err));
            break;
        }
        
        // Launch kernel asynchronously
        sobelKernel<<<grid, block, 0, stream>>>(d_src, d_dst, width, height, threshold);
        
        // Asynchronous copy back (non-blocking!)
        err = cudaMemcpyAsync(h_dst_pinned, d_dst, num_bytes, 
                             cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            std::printf("cudaMemcpyAsync D->H failed: %s\n", cudaGetErrorString(err));
            break;
        }
        
        // While GPU is working, read next frame (overlaps with GPU work!)
        cap >> frame;
        
        // Now wait for GPU to finish current frame
        cudaStreamSynchronize(stream);
        
        // Copy result to OpenCV Mat
        memcpy(sobel_gray.data, h_dst_pinned, num_bytes);
        
        // Convert and write
        cvtColor(sobel_gray, sobel_bgr, COLOR_GRAY2BGR);
        writer.write(sobel_bgr);
        
        if (verbose && frame_idx % 30 == 0) {
            std::printf("Processed frame %d\n", frame_idx);
        }
        frame_idx++;
    }

    cap.release();
    writer.release();
    
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFreeHost(h_src_pinned);
    cudaFreeHost(h_dst_pinned);
    cudaStreamDestroy(stream);

    if (verbose) {
        std::printf("Saved GPU Sobel video to %s (%d frames)\n", output_path, frame_idx);
    }
    return 0;
}
