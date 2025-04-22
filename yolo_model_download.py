from ultralytics import YOLO

model = YOLO('yolov8s.pt')  
model.save('yolov8s_local.pt') 