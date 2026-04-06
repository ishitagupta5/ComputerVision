import depthai as dai
import numpy as np

device = dai.Device(dai.UsbSpeed.HIGH)
calib = device.readCalibration()

model_left = calib.getDistortionModel(dai.CameraBoardSocket.CAM_B)
model_right = calib.getDistortionModel(dai.CameraBoardSocket.CAM_C)
print(f"Left distortion model: {model_left}")
print(f"Right distortion model: {model_right}")

dist_left = calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B)
dist_right = calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C)

print(f"Left distortion ({len(dist_left)} coeffs):")
print(dist_left)
print(f"\nRight distortion ({len(dist_right)} coeffs):")
print(dist_right)

# Test with full coefficients
M_left = np.array([
    [458.79290771484375, 0.0, 330.3569641113281],
    [0.0, 458.92999267578125, 248.63180541992188],
    [0.0, 0.0, 1.0]
])
M_right = np.array([
    [449.5007629394531, 0.0, 317.2297058105469],
    [0.0, 449.496826171875, 251.83892822265625],
    [0.0, 0.0, 1.0]
])
R = np.array([
    [0.9999504685401917, -0.004337015096098185, -0.00894914660602808],
    [0.004344655200839043, 0.9999902248382568, 0.0008343947120010853],
    [0.008945440873503685, -0.0008732345886528492, 0.9999595880508423]
])
T = np.array([-7.492964744567871, 0.002622523345053196, -0.01669374108314514]) / 100.0

import cv2

D_left_full = np.array(dist_left)
D_right_full = np.array(dist_right)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    M_left, D_left_full, M_right, D_right_full, (640, 480), R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

print(f"\nFocal length from P1: {P1[0,0]:.2f} px")
print(f"Baseline: {abs(T[0]) * 100:.2f} cm")
print(f"\nP1:\n{P1}")
print(f"\nP2:\n{P2}")

device.close()