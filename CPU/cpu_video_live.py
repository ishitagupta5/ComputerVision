#!/usr/bin/env python3
"""
Live video Sobel edge detection - processes webcam feed in real-time
Works without GPU - uses OpenCV and NumPy
"""

import cv2
import numpy as np
import sys
import time
from ultralytics import YOLO
model = YOLO("yolo11x.pt")


def sobel_filter_cpu(src, threshold=150):
    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    edges = (magnitude > threshold).astype(np.uint8) * 255
    
    edges[0, :] = 0
    edges[-1, :] = 0
    edges[:, 0] = 0
    edges[:, -1] = 0
    
    return edges

def main():
    camera_index = 0
    if len(sys.argv) >= 2:
        camera_index = int(sys.argv[1])
    
    threshold = 150
    if len(sys.argv) >= 3:
        threshold = int(sys.argv[2])
    
    print(f"Opening camera {camera_index}...")
    print("Press 'q' to quit, 's' to save screenshot, '+'/'-' to adjust threshold")
    print(f"Current threshold: {threshold}")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        print("Try a different camera index (0, 1, 2, etc.)")
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
    last_fps_time = start_time
    fps_display = 0.0
    
    cv2.namedWindow('Sobel Edge Detection - Live', cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("ERROR: Failed to read frame from camera")
                break
            
            process_start = time.time()
            edges = sobel_filter_cpu(frame, threshold=threshold)
            process_time = time.time() - process_start
            
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            results = model(frame, verbose=False)          
            r = results[0]

            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0: 
                fps_display = frame_count / (current_time - start_time)
                last_fps_time = current_time
            
            cv2.putText(edges_bgr, f"FPS: {fps_display:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(edges_bgr, f"Threshold: {threshold}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(edges_bgr, f"Process: {process_time*1000:.1f}ms", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(edges_bgr, "Press 'q' to quit", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            sobel_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = float(box.conf)

                cv2.rectangle(sobel_3ch, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    sobel_3ch,
                    f"{r.names[cls]} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            cv2.imshow('Sobel Edge Detection - Live', sobel_3ch)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                filename = f"sobel_screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, edges)
                print(f"Screenshot saved: {filename}")
            elif key == ord('+') or key == ord('='):
                threshold = min(255, threshold + 10)
                print(f"Threshold increased to: {threshold}")
            elif key == ord('-') or key == ord('_'):
                threshold = max(0, threshold - 10)
                print(f"Threshold decreased to: {threshold}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        if frame_count > 0:
            avg_fps = frame_count / elapsed
            print(f"\nProcessed {frame_count} frames in {elapsed:.2f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()

