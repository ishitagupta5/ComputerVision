# GPU-Accelerated Computer Vision Pipeline
### Sobel Edge Detection · Stereo Depth Estimation · Real-Time Object Detection

> A full-stack parallel computing research project spanning serial CPU execution on Raspberry Pi, OpenMP multicore parallelism, CUDA GPU kernels, CuPy live-video pipelines, OAK-D Lite stereo depth estimation, and YOLO11x object detection — built for autonomous vehicle perception research.

**Presented at Virginia Tech CS Symposium, April 2026**

---

## Results at a Glance

| Implementation | Runtime (2048×2048) | Speedup vs Serial |
|---|---|---|
| Serial CPU (Raspberry Pi) | 13.79 ms | 1× (baseline) |
| OpenMP 4-thread CPU | 0.428 ms | ~32× |
| CUDA GPU | **0.010 ms** | **~1,380×** |
| CuPy GPU (live video) | 0.063 ms/frame | **~135 FPS** |

> GPU achieves **20–30× speedup** over the best CPU configuration across all image sizes, with the gap widening as resolution increases.

---

## Repository Structure

```
ComputerVision/
├── CPU/                        # Serial C and OpenMP parallel implementations
│   ├── sobel_filter.c          # Core Sobel convolution logic (serial)
│   ├── sobel_serial.c          # Raspberry Pi serial baseline
│   ├── sobel_static.c          # OpenMP static scheduling
│   ├── sobel_dynamic.c         # OpenMP dynamic scheduling
│   ├── parallelized_sobel_filter.c  # Generalized OpenMP filter
│   ├── sobel_video.cpp         # CPU video processing pipeline
│   ├── sobel_video_headless.cpp     # Headless CPU video (no display)
│   ├── sobel_video_profile.cpp      # CPU video with profiling/timing
│   ├── sobel_video_cpu_headless.py  # Python wrapper for headless CPU video
│   ├── cpu_video_live.py       # CPU real-time video processing
│   ├── cpu_video_live_ai.py    # CPU video + YOLO object detection
│   ├── sobel_lib.cpp           # Shared Sobel library (.so)
│   ├── resize.c                # Image resizing utility
│   ├── stb_image.h             # Header-only image loader
│   ├── stb_image_write.h       # Header-only image writer
│   ├── Makefile                # Build all CPU binaries
│   ├── commands.txt            # Quick-reference run commands
│   ├── fox.png                 # Test image (fox)
│   ├── road.png                # Test image (road scene)
│   └── kidcrossing.mp4         # Benchmark video (640×360, 30 FPS, 631 frames)
│
├── GPU/                        # CUDA and CuPy GPU implementations
│   ├── gpu_image.cu            # CUDA C kernel — static image Sobel (16×16 thread blocks)
│   ├── sobel_video_cuda.cu     # CUDA C kernel — video frame processing
│   ├── gpu_video_live.py       # CuPy GPU real-time video pipeline
│   ├── sobel_video_gpu.py      # CuPy Sobel on prerecorded video
│   ├── sobel_gpu_api.py        # GPU Sobel API + multicam triplet integration
│   ├── aitest.py               # YOLO11x + Sobel hybrid inference test
│   ├── stb_image.h / .h        # Header-only image I/O
│   ├── commands.txt            # Quick-reference GPU commands
│   ├── fox.png / road.png      # Test images
│   ├── out_gpu_fox*.png        # GPU Sobel output at 256/512/1024/2048 resolution
│   ├── out_gpu_road.png        # GPU output on road scene
│   ├── kidcrossing.mp4         # Benchmark video
│   └── output_gpu_kidcrossing.mp4  # GPU-processed output video
│
├── multicam/                   # OAK-D Lite stereo depth + multi-camera pipeline
│   ├── stereo_depth_omp.cpp    # C++ stereo depth: SGBM + WLS filtering + OpenMP + temporal smoothing
│   ├── collect_depth_data.py   # Controlled depth measurement collection (15-point accuracy experiment)
│   ├── analyze_depth_data.py   # Regression analysis + R² computation (R²=0.9956)
│   ├── depth_accuracy_data.csv # Raw depth accuracy measurements
│   ├── check_calib.py          # OAK-D stereo calibration validation
│   ├── demo_stereo.py          # Stereo depth + Sobel overlay live demo
│   ├── demo_triplet.py         # Triplet camera demo (left/right/depth)
│   ├── threecam.py             # Three-camera synchronized stream
│   ├── oak_test.py             # OAK-D hardware connection test
│   ├── stereo_loader.py        # Stereo frame loading utility
│   └── triplet_loader.py       # Triplet frame loading utility
│
├── foxImageSize/               # Benchmark images at 256², 512², 1024², 2048²
├── fox_parallel_output.png     # OpenMP Sobel output (fox)
├── road_parallel_output.png    # OpenMP Sobel output (road)
├── outputnvccfox.png           # CUDA Sobel output (fox)
├── outputnvccroad.png          # CUDA Sobel output (road)
├── out_serial_fox_pi.png       # Serial Raspberry Pi output
└── out_serial_road_pi.png      # Serial Raspberry Pi output (road)
```

---

## Pipeline Overview

```
Raw Input (image / video / live camera)
        │
        ▼
  Grayscale Conversion
        │
        ├──────────────────────────────────────────────────┐
        │                                                  │
   [CPU PATH]                                        [GPU PATH]
   Serial C (Raspberry Pi)                     CUDA kernel (16×16 blocks)
   OpenMP (1/2/4/8 threads)                   CuPy RawKernel (live video)
        │                                                  │
        └──────────────┬───────────────────────────────────┘
                       │
                  Sobel Edge Map
                       │
              ┌────────┴────────┐
              │                 │
         Output File      YOLO11x Overlay
         (image/video)   (bounding boxes on
                          Sobel visualization)
```

**Stereo Depth Pipeline (OAK-D Lite):**
```
Left + Right Camera Frames (OAK-D Lite, 75mm baseline)
        │
        ▼
   SGBM Stereo Matching (OpenMP-parallelized)
        │
        ▼
   WLS Filter (disparity refinement)
        │
        ▼
   Temporal Smoothing (inter-frame stability)
        │
        ▼
   Depth Map → Sobel Overlay
   R² = 0.9956 | 15-point controlled accuracy experiment
```

---

## CPU Implementations

### Build All CPU Binaries

```bash
cd CPU
make
```

Builds: `sobel_serial`, `sobel_static`, `sobel_dynamic`, `parallelized_sobel_filter`, `resize`

### Static Image Sobel

```bash
# Serial baseline
./sobel_serial fox.png output.png

# OpenMP static scheduling (4 threads)
OMP_NUM_THREADS=4 ./sobel_static fox.png output.png

# OpenMP dynamic scheduling (8 threads)
OMP_NUM_THREADS=8 ./sobel_dynamic fox.png output.png
```

### Video Processing

```bash
# CPU video (with display)
python3 cpu_video_live.py kidcrossing.mp4 output_cpu.mp4

# CPU video (headless, no display required)
python3 sobel_video_cpu_headless.py kidcrossing.mp4 output_headless.mp4

# CPU video + YOLO object detection
python3 cpu_video_live_ai.py kidcrossing.mp4 output_ai.mp4
```

### CPU Files Reference

| File | What it does |
|---|---|
| `sobel_filter.c` | Core 3×3 Sobel convolution kernel used by all CPU variants |
| `sobel_serial.c` | Baseline serial implementation — nested loop pixel traversal |
| `sobel_static.c` | OpenMP with static row distribution — best for uniform workloads |
| `sobel_dynamic.c` | OpenMP with dynamic scheduling — tested for load-balance comparison |
| `sobel_video_profile.cpp` | Per-frame timing + CSV output for all benchmarking |
| `sobel_lib.cpp` | Compiled as shared library for Python interop |
| `resize.c` | Resizes images to target resolution before processing |

---

## GPU Implementations

### CUDA Static Image (C Kernel)

```bash
cd GPU

# Compile
nvcc gpu_image.cu -o gpu_image

# Run
./gpu_image fox.png out_gpu_fox.png
```

The CUDA kernel uses **16×16 thread blocks**, one thread per pixel. Averaged over 50 runs with CUDA event timing for precision.

### CuPy Live Video Pipeline

```bash
# Prerecorded video
python3 gpu_video_live.py input.mp4 output.mp4 1

# Live camera (device 0)
python3 gpu_video_live.py 0 output_live.mp4 1
```

| Argument | Description |
|---|---|
| `input.mp4` or `0` | Input video file or camera device ID |
| `output.mp4` | Output path |
| `1` | Verbose mode — prints FPS + kernel timing per frame |

### YOLO11x + Sobel Hybrid

```bash
python3 aitest.py kidcrossing.mp4 output_yolo_sobel.mp4
```

Runs YOLO11x object detection on the original RGB frame, then overlays bounding boxes and class labels onto the Sobel edge visualization. Demonstrates classical + deep learning perception in a single pipeline.

### GPU Files Reference

| File | What it does |
|---|---|
| `gpu_image.cu` | CUDA C Sobel kernel for static images — 16×16 thread blocks, 50-run averaged timing |
| `sobel_video_cuda.cu` | CUDA C video frame kernel |
| `gpu_video_live.py` | CuPy RawKernel — compiles CUDA at runtime, processes live/prerecorded video |
| `sobel_video_gpu.py` | CuPy Sobel on prerecorded video with frame-level logging |
| `sobel_gpu_api.py` | GPU Sobel as callable API — integrates with multicam pipeline |
| `aitest.py` | YOLO11x inference + Sobel overlay, end-to-end hybrid pipeline |

---

## Stereo Depth Pipeline (OAK-D Lite)

The `multicam/` directory contains the second major stage of this project: real-world stereo depth estimation using the OAK-D Lite camera with Sobel overlay integration.

### Hardware
- **OAK-D Lite** — dual 800p stereo cameras + Intel Myriad X neural inference chip
- 75mm stereo baseline

### Run the Stereo Pipeline

```bash
cd multicam

# Live stereo depth + Sobel edge overlay
python3 demo_stereo.py

# Three-camera synchronized stream (left / right / depth)
python3 demo_triplet.py

# Collect depth accuracy measurements (controlled experiment)
python3 collect_depth_data.py

# Run regression analysis + compute R²
python3 analyze_depth_data.py

# Validate OAK-D stereo calibration
python3 check_calib.py

# Test hardware connection
python3 oak_test.py
```

### Depth Accuracy Results

Controlled 15-measurement experiment across 0.3m–3.0m:

| Metric | Result |
|---|---|
| R² (measured vs ground truth depth) | **0.9956** |
| Algorithm | SGBM + WLS filter + temporal smoothing |
| Parallelism | OpenMP-parallelized disparity computation |
| Raw data | `depth_accuracy_data.csv` |

### Multicam Files Reference

| File | What it does |
|---|---|
| `stereo_depth_omp.cpp` | Core depth engine: SGBM + WLS filter + temporal smoothing + OpenMP |
| `collect_depth_data.py` | 15-point controlled accuracy experiment — logs measured vs ground truth |
| `analyze_depth_data.py` | Linear regression on depth data, computes R²=0.9956 |
| `depth_accuracy_data.csv` | Raw experimental measurements |
| `check_calib.py` | Validates OAK-D stereo calibration matrix |
| `demo_stereo.py` | Live stereo depth + Sobel overlay — combines both pipelines |
| `demo_triplet.py` | Triplet view: left cam / right cam / depth map |
| `threecam.py` | Synchronized three-camera stream handler |
| `oak_test.py` | Hardware connectivity and frame capture test |

---

## Performance Benchmarks

### CPU Thread Scaling (Video, Raspberry Pi 4)

| Threads | Avg Frame Time | Speedup |
|---|---|---|
| 1 | 13.325 ms | 1× |
| 2 | 6.374 ms | 2.09× |
| 4 | 4.168 ms | 3.20× |
| 8 | 4.447 ms | 3.00× ← bandwidth-limited |

> Speedup plateaus at 4 threads. Sobel is **memory-bandwidth-bound**, not compute-bound — all threads compete for the same DRAM bus.

### GPU vs CPU Across Image Resolutions (Fox Image)

| Resolution | GPU (ms) | CPU 4-thread (ms) | GPU Speedup |
|---|---|---|---|
| 256×256 | 0.000082 | 0.000195 | ~2.4× |
| 512×512 | 0.000193 | 0.000467 | ~2.4× |
| 1024×1024 | 0.000047 | 0.003650 | ~78× |
| 2048×2048 | 0.000190 | 0.005970 | ~31× |

### Head-to-Head Single Benchmark (Fox Image)

| Implementation | Runtime | Speedup |
|---|---|---|
| Serial CPU (Raspberry Pi) | 13.792 ms | 1× |
| OpenMP 4-thread | 0.428 ms | 32× |
| CUDA GPU | **0.010 ms** | **~1,380×** |

### Live Video (640×360, 631 frames)

| Implementation | Avg Frame Time | Achieved FPS |
|---|---|---|
| CPU 1-thread | 13.325 ms | ~75 |
| CPU 4-thread | 4.168 ms | ~240 |
| GPU (CuPy kernel) | **0.063 ms** | **~135** |

> Real bottleneck at GPU speeds is OpenCV frame capture + BGR→grayscale conversion + PCIe transfer — not the Sobel kernel itself.

---

## Environment Setup

### CPU

```bash
# Ubuntu/Debian
sudo apt install gcc g++ libomp-dev

cd CPU && make
```

### GPU

```bash
# CUDA Toolkit 12.x
# https://developer.nvidia.com/cuda-downloads

pip install cupy-cuda12x torch torchvision ultralytics opencv-python
```

### Stereo Depth (OAK-D Lite)

```bash
pip install depthai opencv-python numpy scipy
```

### Hardware Requirements

| Component | Minimum |
|---|---|
| GPU | NVIDIA CUDA-capable GPU |
| CUDA | 12.x |
| Python | 3.8+ |
| Stereo camera | OAK-D Lite (DepthAI SDK) |
| CPU (optional) | Raspberry Pi 4 or x86 with OpenMP |

---

## Key Findings

**GPU acceleration delivers ~1,380× speedup over serial CPU** for large images. The CUDA kernel runs at microsecond-level latencies because Sobel's per-pixel independence maps perfectly to GPU thread parallelism.

**CPU scaling is memory-bandwidth-limited.** OpenMP provides 3–5× speedup, but performance plateaus at 4 threads as all cores saturate the DRAM bus simultaneously.

**Real-time GPU pipelines are I/O-bound, not compute-bound.** Once Sobel moves to GPU, the kernel itself becomes the fastest part. Bottlenecks shift to camera capture, color conversion, and PCIe data transfer.

**Classical edge detection and deep learning are complementary.** YOLO11x provides semantic object detection; Sobel reveals structural boundaries (lane markings, curbs, edges). Combined, they produce richer scene understanding — mirroring production AV perception stacks.

**Stereo depth estimation achieves R²=0.9956** using SGBM + WLS filtering + temporal smoothing on the OAK-D Lite, validated across 15 controlled measurements from 0.3m to 3.0m.


---

## Authors

**Ishita Gupta** — CPU pipeline (serial + OpenMP), CUDA image kernel, CuPy GPU video pipeline, object detection overlays, GPU video experimentation across NVIDIA hardware, OAK-D Lite stereo depth pipeline (SGBM, WLS, temporal smoothing), depth accuracy experiment + regression analysis, end-to-end debugging, performance benchmarking, final report

**Jayant Dulani** — YOLO11x integration, intermediate reports, benchmark graphs

---

## References

- NVIDIA Corporation. [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- Ultralytics. [YOLO11x Documentation](https://docs.ultralytics.com)
- OpenCV. [Documentation](https://docs.opencv.org)
- CuPy Developers. [CuPy: NumPy & SciPy for CUDA](https://docs.cupy.dev)
- Kirk, D. & Hwu, W. *Programming Massively Parallel Processors*
- Gonzalez, R. C. & Woods, R. E. *Digital Image Processing*
- Sobel, I. (2014). *An Isotropic 3×3 Image Gradient Operator*
