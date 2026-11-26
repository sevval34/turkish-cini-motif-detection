import sys
from ultralytics import YOLO

image_path = sys.argv[1]

model = YOLO("runs/detect/train/weights/best.pt")

model.predict(
    source=image_path,
    save=True,
    project="outputs",
    name="result",
    conf=0.4
)

print("Bitti. Sonuç outputs/result klasöründe.")
