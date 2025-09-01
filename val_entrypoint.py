from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':
    model_path = '/workspace/repo/ultralytics/runs/qbb/debug0901_7/weights/best.pt'
    model = YOLO(model_path)
    results = model.val(name='val0901_', data='webpm_obb1944.yaml', imgsz=640, batch=8, workers=0,
                        plots=True, device="0",# deterministic=True, seed=42,
                        save_json=False, save_txt=False, conf=0.001, iou=0.6, max_det=300, split='val')

    print(f"\n검증 완료!")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")