from ultralytics import YOLO

model = YOLO("yolo11x.pt") 

results = model("kidcrossing.mp4", save=True)
