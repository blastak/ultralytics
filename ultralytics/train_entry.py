from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n-obb.yaml')
    # model = YOLO('yolo11n-qbb.yaml')
    # model.train(data='ultralytics/webpm_obb8.yaml', epochs=10, imgsz=640, fliplr=0.0, batch=1, workers=0)
    results = model.train(data='dota8.yaml', epochs=10, imgsz=640, fliplr=0.0, batch=1, workers=0)
