from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("Ping-Pong-Detection-3-best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict(source=1, show=True, imgsz=640, conf=0.5)