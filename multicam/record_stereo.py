"""
record_stereo.py
================
Records synchronized left + right mono camera streams from the OAK-D Lite
as two MP4 files, so you can run your pipeline on the recording later.

This is intentionally minimal: it does NOT run SGBM, YOLO, rectification,
or any of your pipeline. It's *just* a recorder, so it can run at the
camera's native frame rate without CPU-side bottlenecks.

Usage:
    python record_stereo.py

Controls:
    r = start/stop recording   (toggle)
    q = quit

Output files (written to ./recordings/<timestamp>/):
    left.mp4            - grayscale left mono camera
    right.mp4           - grayscale right mono camera
    metadata.json       - fps, frame count, resolution, start/stop time

Then feed those into `process_recording.py` to produce the annotated
output video you can play at the symposium.

NOTE: this records the RAW RECTIFIED-BEFORE frames (straight from the
sensors). `process_recording.py` will do rectification itself, matching
what your live pipeline does. Don't pre-rectify here.
"""

import cv2
import depthai as dai
import json
import os
import time
from datetime import datetime


# -----------------------------------------------------------------------------
# Config — keep these in sync with threecam.py so SGBM sees identical pixels
# -----------------------------------------------------------------------------
IMG_W, IMG_H = 640, 480   # OAK-D Lite mono resolution (THE_480_P)
FPS = 30                  # Camera FPS
OUTPUT_ROOT = "recordings"


def build_pipeline():
    """Minimal DepthAI pipeline: just left + right mono, no color, no stereo node."""
    pipeline = dai.Pipeline()

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    xout_left = pipeline.create(dai.node.XLinkOut)
    xout_right = pipeline.create(dai.node.XLinkOut)

    xout_left.setStreamName("left")
    xout_right.setStreamName("right")

    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    mono_left.setFps(FPS)

    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    mono_right.setFps(FPS)

    mono_left.out.link(xout_left.input)
    mono_right.out.link(xout_right.input)

    return pipeline


def open_writer(path):
    """Open an MP4 writer. Uses mp4v which is broadly supported and lossless-enough
    for grayscale at 480p. If you want truly lossless, change fourcc to 'FFV1' and
    rename the extension to .avi — but mp4v is fine for SGBM input in practice."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # isColor=False because we record grayscale
    return cv2.VideoWriter(path, fourcc, FPS, (IMG_W, IMG_H), isColor=False)


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    pipeline = build_pipeline()

    recording = False
    writer_left = None
    writer_right = None
    session_dir = None
    frame_count = 0
    start_time = None

    print("=" * 60)
    print("  OAK-D Lite stereo recorder")
    print("  r = start/stop recording")
    print("  q = quit")
    print("=" * 60)

    with dai.Device(pipeline) as device:
        q_left = device.getOutputQueue("left", maxSize=4, blocking=False)
        q_right = device.getOutputQueue("right", maxSize=4, blocking=False)

        while True:
            left_pkt = q_left.tryGet()
            right_pkt = q_right.tryGet()

            if left_pkt is None or right_pkt is None:
                # Wait for both streams to have a frame; preserves pairing
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            left_frame = left_pkt.getCvFrame()
            right_frame = right_pkt.getCvFrame()

            # Sanity: force grayscale, correct size
            if len(left_frame.shape) == 3:
                left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            if len(right_frame.shape) == 3:
                right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            if left_frame.shape[:2] != (IMG_H, IMG_W):
                left_frame = cv2.resize(left_frame, (IMG_W, IMG_H))
            if right_frame.shape[:2] != (IMG_H, IMG_W):
                right_frame = cv2.resize(right_frame, (IMG_W, IMG_H))

            # Build a small side-by-side preview so you can see what you're recording
            preview = cv2.hconcat([left_frame, right_frame])
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

            if recording:
                cv2.circle(preview_bgr, (20, 20), 10, (0, 0, 255), -1)
                cv2.putText(preview_bgr, f"REC  frame {frame_count}",
                            (40, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if writer_left is not None:
                    writer_left.write(left_frame)
                if writer_right is not None:
                    writer_right.write(right_frame)
                frame_count += 1
            else:
                cv2.putText(preview_bgr, "press 'r' to record",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("OAK-D Lite Recorder  (left | right)", preview_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                if not recording:
                    # start a new recording
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    session_dir = os.path.join(OUTPUT_ROOT, ts)
                    os.makedirs(session_dir, exist_ok=True)
                    writer_left = open_writer(os.path.join(session_dir, "left.mp4"))
                    writer_right = open_writer(os.path.join(session_dir, "right.mp4"))
                    frame_count = 0
                    start_time = time.time()
                    recording = True
                    print(f"[REC] started -> {session_dir}")
                else:
                    # stop the current recording
                    recording = False
                    elapsed = time.time() - start_time
                    if writer_left is not None:
                        writer_left.release()
                        writer_left = None
                    if writer_right is not None:
                        writer_right.release()
                        writer_right = None
                    metadata = {
                        "timestamp": os.path.basename(session_dir),
                        "frame_count": frame_count,
                        "fps": FPS,
                        "width": IMG_W,
                        "height": IMG_H,
                        "duration_seconds": round(elapsed, 2),
                        "note": "raw unrectified left+right grayscale from OAK-D Lite",
                    }
                    with open(os.path.join(session_dir, "metadata.json"), "w") as f:
                        json.dump(metadata, f, indent=2)
                    print(f"[REC] stopped. {frame_count} frames in {elapsed:.1f}s")
                    print(f"      saved: {session_dir}/left.mp4")
                    print(f"      saved: {session_dir}/right.mp4")
                    print(f"      saved: {session_dir}/metadata.json")

    # Clean shutdown
    if writer_left is not None:
        writer_left.release()
    if writer_right is not None:
        writer_right.release()
    cv2.destroyAllWindows()
    print("done.")


if __name__ == "__main__":
    main()