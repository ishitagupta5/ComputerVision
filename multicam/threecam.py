import cv2
import numpy as np
import depthai as dai
import sys
import time
from ultralytics import YOLO

# ---- YOLO model ----
model = YOLO("yolo11x.pt")

# ---- GPU Sobel import ----
sys.path.append("../GPU")

GPU_AVAILABLE = True
try:
    from sobel_gpu_api import sobel_gpu_edges
except Exception as e:
    print("[GPU Sobel unavailable — using CPU Sobel fallback]")
    GPU_AVAILABLE = False

    def sobel_gpu_edges(frame_bgr, threshold=80):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 else frame_bgr
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.convertScaleAbs(mag)
        _, edges = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
        return edges


def overlay_edges_on_rgb(frame_bgr, edges_gray, alpha=0.45):
    overlay = frame_bgr.copy()
    edges_gray = cv2.erode(edges_gray, np.ones((2, 2), np.uint8), iterations=1)
    overlay[edges_gray > 0] = (0, 255, 0)
    return cv2.addWeighted(frame_bgr, 1 - alpha, overlay, alpha, 0)


def draw_yolo_boxes(frame, results):
    r = results[0]
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)
        conf = float(box.conf)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{r.names[cls]} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


device = dai.Device(dai.UsbSpeed.HIGH)
with dai.Pipeline(device) as pipeline:
    cam_color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    q_color = cam_color.requestOutput((640, 480)).createOutputQueue()

    cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    q_left = cam_left.requestOutput((640, 480)).createOutputQueue()

    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    q_right = cam_right.requestOutput((640, 480)).createOutputQueue()

    sobel_on = False
    yolo_on = False

    frame_count = 0
    start_time = time.time()
    fps_display = 0.0

    pipeline.start()
    while pipeline.isRunning():
        color_frame = q_color.get().getCvFrame()
        left_frame = q_left.get().getCvFrame()
        right_frame = q_right.get().getCvFrame()

        # Convert mono to BGR
        if len(left_frame.shape) == 2:
            left_frame = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR)
        if len(right_frame.shape) == 2:
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_GRAY2BGR)

        # Sobel edge overlay
        if sobel_on:
            left_frame = overlay_edges_on_rgb(left_frame, sobel_gpu_edges(left_frame))
            color_frame = overlay_edges_on_rgb(color_frame, sobel_gpu_edges(color_frame))
            right_frame = overlay_edges_on_rgb(right_frame, sobel_gpu_edges(right_frame))

        # YOLO detection on all three
        if yolo_on:
            left_frame = draw_yolo_boxes(left_frame, model(left_frame, verbose=False))
            color_frame = draw_yolo_boxes(color_frame, model(color_frame, verbose=False))
            right_frame = draw_yolo_boxes(right_frame, model(right_frame, verbose=False))

        # FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps_display = frame_count / elapsed

        # Resize to fit screen
        small_size = (427, 320)
        left_frame = cv2.resize(left_frame, small_size)
        color_frame = cv2.resize(color_frame, small_size)
        right_frame = cv2.resize(right_frame, small_size)

        # Labels
        cv2.putText(left_frame, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(color_frame, "COLOR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(right_frame, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined = np.hstack([left_frame, color_frame, right_frame])

        # Status bar
        status = f"FPS: {fps_display:.1f} | Sobel: {'ON' if sobel_on else 'OFF'} [e] | YOLO: {'ON' if yolo_on else 'OFF'} [y] | Quit [q]"
        cv2.putText(combined, status, (10, combined.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("OAK-D Lite - Left | Color | Right", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('e'):
            sobel_on = not sobel_on
            print(f"[Sobel {'ON' if sobel_on else 'OFF'}]")
        if key == ord('y'):
            yolo_on = not yolo_on
            print(f"[YOLO {'ON' if yolo_on else 'OFF'}]")