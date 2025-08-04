import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('yolo11n-obb.yaml')
    model = YOLO('yolo11n-qbb.yaml')
    results = model.train(data='dota8.yaml', epochs=10, imgsz=640, fliplr=0.0, batch=1, workers=0)
