from datetime import date

from ultralytics import YOLO


if __name__ == '__main__':
    today_str = date.today().strftime("%m%d")  # "2025-mm-dd"

    # os.system("rm -f /workspace/repo/ultralytics/ultralytics/assets/good_all_obb1944/labels/*.cache")
    model = YOLO('yolov8n-qbb.yaml')
    # results = model.train(name='debug%s_' % today_str, data='webpm_obb1944.yaml', epochs=2, fliplr=0.0, batch=4, workers=0,
    #                       imgsz=640, plots=True, device="0", deterministic=True, seed=42,)# dfl=0.0)
    results = model.train(name='train%s_' % today_str, data='webpm_obb1944.yaml', epochs=200, fliplr=0.0, batch=16, workers=8,
                          imgsz=640, plots=True, device="0,1", dfl=5.0)#, dfl=0.0) # deterministic=True, seed=42,
