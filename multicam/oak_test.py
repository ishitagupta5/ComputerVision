import cv2
import depthai as dai

device = dai.Device()
device = dai.Device(dai.UsbSpeed.HIGH)
with dai.Pipeline(device) as pipeline:
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    queue = cam.requestOutput((640, 480)).createOutputQueue()

    pipeline.start()
    while pipeline.isRunning():
        frame = queue.get()
        cv2.imshow("OAK-D Lite RGB", frame.getCvFrame())
        if cv2.waitKey(1) == ord('q'):
            break