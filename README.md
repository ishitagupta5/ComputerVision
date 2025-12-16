# Sobel Edge Detection for Self-Driving Car Vision  
### A Parallel Computing Pipeline from CPU to GPU Acceleration

This repository contains a complete Sobel edge-detection pipeline implemented across multiple computational platforms, progressing from serial CPU execution to multicore OpenMP parallelism and finally to GPU acceleration using CUDA and CuPy. The project evaluates performance, scalability, and real-time behavior on static images, prerecorded videos, and live camera streams, and integrates YOLO-based object detection for hybrid perception experiments.

---

## Repository Structure

```
ComputerVision/
├── CPU/                # Serial and OpenMP CPU implementations
├── GPU/                # CUDA and CuPy GPU implementations
├── foxImageSize/       # Test images at multiple resolutions
├── README.md
├── *.png, *.mp4        # Output images, videos, and benchmarks
```

---

## CPU Implementations

The `CPU/` directory contains:
- Serial Sobel edge detection (C/C++)
- OpenMP-parallel Sobel implementations (static and dynamic scheduling)
- CPU-based video processing pipelines
- Makefile for building all CPU binaries

### Build CPU Code

```bash
cd CPU
make
```

This builds the following executables:
- `sobel_serial`
- `sobel_static`
- `sobel_dynamic`
- `parallelized_sobel_filter`

### Run CPU Sobel on Static Images

**Serial**
```bash
./sobel_serial fox.png output.png
```

**OpenMP (static scheduling)**
```bash
OMP_NUM_THREADS=4 ./sobel_static fox.png output.png
```

**OpenMP (dynamic scheduling)**
```bash
OMP_NUM_THREADS=4 ./sobel_dynamic fox.png output.png
```

### Run CPU Sobel on Video

```bash
python3 cpu_video_live.py kidcrossing.mp4 output_cpu.mp4
```

---

## GPU Implementations

The `GPU/` directory contains:
- CUDA C implementation for Sobel on static images
- CuPy-based GPU Sobel for video and live camera streams
- YOLO11x integration for object detection overlays
- Example input images and videos

### CUDA Sobel on Static Images

**Compile:**
```bash
cd GPU
nvcc gpu_image.cu -o gpu_image
```

**Run:**
```bash
./gpu_image fox.png out_gpu_fox.png
```

### GPU Video & Live Camera Sobel (CuPy)

This implementation runs Sobel edge detection on the GPU using CuPy.

**Single-line GPU command (video input):**
```bash
python3 gpu_video_live.py input.mp4 output.mp4 1
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `input.mp4` | Input video file |
| `output.mp4` | Output video with Sobel edges |
| `1` | Verbose |

**Live camera (device 0):**
```bash
python3 gpu_video_live.py 0 output_live.mp4 1
```

---

## Datasets

The `foxImageSize/` directory contains test images at multiple resolutions:
- 256×256
- 512×512
- 1024×1024
- 2048×2048

These images are used to evaluate scalability and performance across CPU and GPU implementations.

---

## Reproducing Results

1. Build and run CPU implementations using the commands above.
2. Compile and execute the CUDA Sobel kernel for static image benchmarks.
3. Run the CuPy-based GPU pipeline for real-time video processing.
4. Use the provided images and videos to replicate output artifacts and performance measurements reported in the paper.

---

## Environment Requirements

**CPU**
- GCC/Clang with OpenMP support

**GPU**
- NVIDIA CUDA-capable GPU
- CUDA Toolkit 12.x

**Python**
- Python 3.8+
- CuPy
- PyTorch (CUDA-enabled)
- Ultralytics YOLO
- OpenCV

---

## Authors

- **Ishita Gupta** – CPU pipeline, OpenMP parallelism, CUDA image kernel, GPU video pipeline, testing, performance analysis, and final report
- **Jayant Dulani** – GPU experimentation, YOLO integration, object detection overlays, and presentation development

---
