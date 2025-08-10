!unzip "/content/crack.v1i.yolov9.zip" -d "/content/crack"

from ultralytics import YOLO

model = YOLO('yolov9c-seg.pt')
model.info()

results = model.train(data='/content/crack/data.yaml', epochs=10, imgsz=640, batch=8)