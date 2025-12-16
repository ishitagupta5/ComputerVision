from ultralytics import YOLO
import torch
#test file to run ai detector
model = YOLO("yolo11x.pt")
model.to("cuda")

print("CUDA available:", torch.cuda.is_available())
print("YOLO device:", model.device)
