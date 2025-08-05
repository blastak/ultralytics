import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO


def val_visualization_callback(trainer):
    """20에폭마다 validation 이미지의 OBB 예측 결과를 JPG로 저장"""
    if (trainer.epoch+1) % 20 == 0 and trainer.epoch > 0:
        val_path = trainer.data.get('val', '')
        if val_path:
            # trainer에서 모델 weight를 가져와 새 YOLO 객체 생성
            from ultralytics import YOLO
            model = YOLO(trainer.best if trainer.best.exists() else trainer.last)
            results = model.predict(
                val_path,
                save=True,
                project='/workspace/repo/ultralytics/runs/val_vis',
                name=f'epoch_{trainer.epoch}',
                conf=0.25,
                imgsz=trainer.args.imgsz
            )
    return True

if __name__ == '__main__':
    # os.system("rm -f /workspace/repo/ultralytics/ultralytics/assets/good_all\(obb8\)/labels/*.cache")
    # os.system("rm -f /workspace/repo/ultralytics/ultralytics/assets/good_all\(obb_full\)/labels/*.cache")

    # model = YOLO('yolo11n-obb.yaml')
    # model = YOLO('yolo11n-qbb.yaml')

    # results = model.train(data='dota8.yaml', epochs=10, imgsz=640, fliplr=0.0, batch=1, workers=0)
    # results = model.train(data='ultralytics/webpm_obb8.yaml', epochs=500, imgsz=640, fliplr=0.0, batch=1, workers=0, patience=0)
    # results = model.train(data='ultralytics/webpm_obb_full.yaml', epochs=10, imgsz=640, fliplr=0.0, batch=1, workers=0)

    model = YOLO('yolo11n.yaml')
    model.add_callback('on_train_epoch_end', val_visualization_callback)
    results = model.train(data='ultralytics/webpm_bb8.yaml', epochs=200, imgsz=640, fliplr=0.0, batch=1, workers=0, patience=0)
