import sys
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_images
import torch
from PIL import Image

if __name__ == '__main__':
    # os.system("rm -f /workspace/repo/ultralytics/ultralytics/assets/good_all_obb1944/labels/*.cache")
    model = YOLO('yolov8n-qbb.yaml')
    results = model.train(name='debug0827_', data='webpm_obb1944.yaml', epochs=300, fliplr=0.0, batch=16, workers=8,
                          imgsz=640, plots=True, device="0,1")  # deterministic=True, seed=42,
    # results = model.train(
    #     name='backward_debug',
    #     data='webpm_obb1944.yaml',
    #     imgsz=640,
    #     epochs=20,  # 에폭 증가
    #     batch=8,  # 배치 사이즈 증가 (GPU 2개 * 8 each)
    #     device="0",  # Multi-GPU 활성화
    #     workers=2,  # CPU 코어 수 활용
    #     fliplr=0.0,  # Flip augmentation 활성화
    #     plots=True,  # JPG visualization 활성화
    #     # cache=True,  # 데이터 캐싱으로 속도 향상
    #     # amp=True  # Automatic Mixed Precision
    # )

    # results = model.train(name='debug_by_user', data='webpm_bb8.yaml', epochs=2, imgsz=640, fliplr=0.0, batch=1, workers=0, plots=True)
    # model = YOLO('yolov8n.yaml')
    # results = model.train(name='debug_by_user', data='webpm_bb8.yaml', epochs=20, imgsz=640, fliplr=0.0, batch=2, workers=0, plots=True)
