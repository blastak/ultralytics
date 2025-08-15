import os
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('yolo11n-obb.yaml')
    # model.train(data='ultralytics/webpm_obb8.yaml', epochs=10, imgsz=640, fliplr=0.0, batch=1, workers=0)

    # os.system("rm -f /workspace/repo/ultralytics/ultralytics/assets/good_all_obb8/labels/*.cache")
    model = YOLO('yolov8n-qbb.yaml')
    results = model.train(name='debug_by_user', data='webpm_obb8.yaml', epochs=2, imgsz=640, fliplr=0.0, batch=1, workers=0, plots=False)
