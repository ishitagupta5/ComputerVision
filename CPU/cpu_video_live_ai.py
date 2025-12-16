#!/usr/bin/env python3
"""
Live webcam Sobel edge detection + YOLO bounding boxes
Uses:
 - OpenMP-accelerated Sobel (C++ library: sobel.so)
 - CPU-based pipeline (YOLO can run CPU/GPU depending on environment)
"""

import cv2
import numpy as np
import sys
import time
from ultralytics import YOLO
import ctypes
import numpy.ctypeslib as npct

model = YOLO("yolo11x.pt") 

sobel_lib = ctypes.CDLL("./sobel.so")

sobel_func = sobel_lib.sobel_filter
sobel_func.argtypes = [
    npct.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int  
]
sobel_func.restype = None


def sobel_openmp(gray, threshold=150):
    height, width = gray.shape
    dest = np.zeros_like(gray, dtype=np.uint8)

    sobel_func(gray, dest,
               ctypes.c_int(height),
               ctypes.c_int(width),
               ctypes.c_int(threshold),
               ctypes.c_int(1))

    return dest


def main():

    camera_index = 0
    if len(sys.argv) >= 2:
        camera_index = int(sys.argv[1])

    threshold = 150
    if len(sys.argv) >= 3:
        threshold = int(sys.argv[2])

    print(f"Opening camera {camera_index}...")
    print("Controls:")
    print("  q  -> quit")
    print("  s  -> save screenshot")
    print("  +  -> raise Sobel threshold")
    print("  -  -> lower Sobel threshold")
    print(f"Initial threshold: {threshold}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera opened: {width}x{height} @ {fps:.2f} fps")

    frame_count = 0
    start_time = time.time()
    last_update_time = start_time
    fps_display = 0.0

    cv2.namedWindow("Sobel Edge Detection - Live", cv2.WINDOW_NORMAL)

    try:
        while True:

            ret, frame = cap.read()
            if not ret:
                print("ERROR: Could not read frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            t0 = time.time()
            edges = sobel_openmp(gray, threshold)
            process_time = (time.time() - t0) * 1000  # ms

            results = model(frame, verbose=False)
            r = results[0]

            frame_count += 1
            now = time.time()
            if now - last_update_time >= 1.0:
                fps_display = frame_count / (now - start_time)
                last_update_time = now

            sobel_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            cv2.putText(sobel_3ch, f"FPS: {fps_display:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(sobel_3ch, f"Threshold: {threshold}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(sobel_3ch, f"Sobel: {process_time:.1f} ms", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = float(box.conf)

                cv2.rectangle(sobel_3ch, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.putText(sobel_3ch,
                            f"{r.names[cls]} {conf:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

            cv2.imshow("Sobel Edge Detection - Live", sobel_3ch)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting…")
                break
            elif key == ord('s'):
                fname = f"sobel_frame_{int(time.time())}.png"
                cv2.imwrite(fname, edges)
                print(f"Saved: {fname}")
            elif key in (ord('+'), ord('=')):
                threshold = min(255, threshold + 10)
                print(f"Threshold -> {threshold}")
            elif key in (ord('-'), ord('_')):
                threshold = max(0, threshold - 10)
                print(f"Threshold -> {threshold}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        total_time = time.time() - start_time
        if frame_count > 0:
            avg_fps = frame_count / total_time
            print(f"\nProcessed {frame_count} frames")
            print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
