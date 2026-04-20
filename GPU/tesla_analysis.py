"""
Tesla Crash Analysis - CS4624 Progress Report 5
Generates 4 side-by-side images: [Tesla frame | Sobel edges | YOLO detections]

USAGE:
    Put this file and your tesla video in the same folder, then run:
        python tesla_analysis.py

    Output PNG files go into a new folder called "tesla_frames".
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ============ EDIT THESE IF YOU NEED TO ============
VIDEO_FILE = "yourvideo.mp4"   # rename your video to this, OR change this to match
MODEL_FILE = "yolo11x.pt"      # your existing YOLO model
SOBEL_THRESHOLD = 150           # matches your project
KEY_TIMESTAMPS_SEC = [1.0, 3.5, 6.0, 7.5]  # your video is 8s long
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
    out = Path("tesla_frames"); out.mkdir(exist_ok=True)
    model = YOLO(MODEL_FILE)

    cap = cv2.VideoCapture(VIDEO_FILE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Loaded {VIDEO_FILE}: {total} frames @ {fps:.1f} FPS ({total/fps:.1f}s)")

    for i, ts in enumerate(KEY_TIMESTAMPS_SEC):
        fn = int(ts * fps)
        if fn >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = sobel_edges(gray)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        results = model(frame, verbose=False, conf=0.15)
        det = results[0].plot()

        label_panel(frame, "TESLA CAMERA")
        label_panel(edges_bgr, "SOBEL EDGES (threshold=150)")
        label_panel(det, "YOLO DETECTIONS")

        panel = np.hstack([frame, edges_bgr, det])

        footer = np.zeros((40, panel.shape[1], 3), dtype=np.uint8)
        footer[:] = (30, 30, 30)
        cv2.putText(footer, f"t = {ts:.1f}s  |  frame {fn}  |  Sobel threshold = 150",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        panel = np.vstack([panel, footer])

        fname = out / f"panel_{i:02d}_t{ts:.1f}s.png"
        cv2.imwrite(str(fname), panel)
        detected = [model.names[int(c)] for c in results[0].boxes.cls]
        print(f"  {fname.name}  -> detected: {detected}")

    cap.release()
    print(f"\nDone. Open the 'tesla_frames' folder and drag the PNGs into your slides.")


if __name__ == "__main__":
    main()