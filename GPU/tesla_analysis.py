"""
Tesla Crash Analysis - CS4624 Progress Report 5
Processes ENTIRE video: outputs a single side-by-side video showing
[Tesla frame | Sobel edges | YOLO detections] for every frame.

USAGE:
    python tesla_analysis.py

    Output: tesla_analysis_output.mp4 in the same folder.
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time

# ============ EDIT THESE IF YOU NEED TO ============
VIDEO_FILE = "yourvideo.mp4"
OUTPUT_FILE = "tesla_analysis_output.mp4"
MODEL_FILE = "yolo11x.pt"
SOBEL_THRESHOLD = 150
CONF_THRESHOLD = 0.15   # YOLO confidence threshold (lower = more detections)
# ===================================================


def sobel_edges(gray, threshold=SOBEL_THRESHOLD):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return np.where(mag > threshold, 255, 0).astype(np.uint8)


def label_panel(img, text):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    print(f"Loading YOLO model: {MODEL_FILE}")
    model = YOLO(MODEL_FILE)

    cap = cv2.VideoCapture(VIDEO_FILE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input: {VIDEO_FILE} - {total} frames @ {fps:.1f} FPS ({w}x{h})")

    # Output video: 3 panels wide + 40px footer
    out_w = w * 3
    out_h = h + 40
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        print("ERROR: Could not open output video for writing")
        return

    print(f"Output: {OUTPUT_FILE} ({out_w}x{out_h} @ {fps:.1f} FPS)")
    print("Processing frames...")

    frame_idx = 0
    t_start = time.time()
    total_detections = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Sobel edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = sobel_edges(gray)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # YOLO detection on GPU
        results = model(frame, verbose=False, conf=CONF_THRESHOLD)
        det = results[0].plot()
        total_detections += len(results[0].boxes.cls)

        # Label each panel
        frame_labeled = frame.copy()
        label_panel(frame_labeled, "TESLA CAMERA")
        label_panel(edges_bgr, "SOBEL EDGES (threshold=150)")
        label_panel(det, "YOLO DETECTIONS")

        # Build horizontal strip
        panel = np.hstack([frame_labeled, edges_bgr, det])

        # Footer with stats
        footer = np.zeros((40, panel.shape[1], 3), dtype=np.uint8)
        footer[:] = (30, 30, 30)
        ts = frame_idx / fps
        cv2.putText(footer,
                    f"Frame {frame_idx}/{total}  |  t={ts:.2f}s  |  Sobel thresh={SOBEL_THRESHOLD}  |  YOLO conf>{CONF_THRESHOLD}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        output_frame = np.vstack([panel, footer])

        writer.write(output_frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            elapsed = time.time() - t_start
            fps_proc = frame_idx / elapsed
            pct = 100 * frame_idx / total
            print(f"  {frame_idx}/{total} ({pct:.0f}%)  |  {fps_proc:.1f} FPS processing")

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    print(f"\n=== DONE ===")
    print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS)")
    print(f"Total detections across all frames: {total_detections}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()