#!/usr/bin/env python3
"""
CPU-based Sobel edge detection for video processing
Works without GPU - uses OpenCV and NumPy
"""

import cv2
import numpy as np
import sys
import time

def sobel_filter_cpu(src, threshold=150):
    """
    Apply Sobel edge detection filter on CPU
    """
    # Convert to grayscale if needed
    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()
    
    # Apply Sobel operators
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Threshold
    edges = (magnitude > threshold).astype(np.uint8) * 255
    
    # Zero out border pixels
    edges[0, :] = 0
    edges[-1, :] = 0
    edges[:, 0] = 0
    edges[:, -1] = 0
    
    return edges

def main():
    if len(sys.argv) < 3:
        print("Usage: python sobel_video_cpu.py input_video output_video [verbose]")
        print("  verbose: 0 = quiet, 1 = verbose (default: 1)")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    verbose = int(sys.argv[3]) if len(sys.argv) >= 4 else 1
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {input_path}")
        sys.exit(1)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    if verbose:
        print(f"Processing video: {input_path} ({width}x{height} @ {fps:.2f} fps)")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)
    
    if not writer.isOpened():
        print(f"ERROR: Cannot open output video: {output_path}")
        cap.release()
        sys.exit(1)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        # Process frame
        edges = sobel_filter_cpu(frame, threshold=150)
        
        # Write frame
        writer.write(edges)
        frame_count += 1
        
        if verbose and frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            print(f"Processed frame {frame_count} (FPS: {fps_actual:.2f})")
    
    cap.release()
    writer.release()
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"Done! Saved GPU Sobel video to {output_path}")
        print(f"Processed {frame_count} frames in {elapsed:.2f} seconds")
        print(f"Average FPS: {frame_count/elapsed:.2f}" if elapsed > 0 else "")

if __name__ == "__main__":
    main()

