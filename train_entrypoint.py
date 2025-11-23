from datetime import date

from ultralytics import YOLO

if __name__ == '__main__':
    today_str = date.today().strftime("%m%d")  # "2025-mm-dd"

    ################# 기존 코드 (주석 처리)
    # ################# CCPD Over60 - AABB Training (YOLOv11n)
    # print("\n" + "="*80)
    # print("Starting AABB Training (YOLOv11n) - CCPD Over60")
    # print("="*80 + "\n")
    # model = YOLO('yolo11n.pt')
    # results = model.train(
    #     name='ccpd_over60_yolov11n_aabb',
    #     data='dataset_ccpd_over60_xywh.yaml',
    #     epochs=100,
    #     batch=64,
    #     workers=32,
    #     imgsz=640,
    #     plots=True,
    #     device="0,1,2,3,4,5,6,7"
    # )

    # ################# CCPD Over60 - OBB Training (YOLOv11n)
    # print("\n" + "="*80)
    # print("Starting OBB Training (YOLOv11n) - CCPD Over60")
    # print("="*80 + "\n")
    # model = YOLO('yolo11n-obb.pt')
    # results = model.train(
    #     name='ccpd_over60_yolov11n_obb',
    #     data='dataset_ccpd_over60_xyxyxyxy.yaml',
    #     epochs=100,
    #     batch=64,
    #     workers=32,
    #     imgsz=640,
    #     plots=True,
    #     device="0,1,2,3,4,5,6,7"
    # )

    # ################# CCPD 1/10 - AABB Training (YOLOv8n)
    # print("\n" + "="*80)
    # print("Starting AABB Training (YOLOv8n) - CCPD 1/10")
    # print("="*80 + "\n")
    # model = YOLO('yolov8n.pt')
    # results = model.train(
    #     name='ccpd_1over10_yolov8n_aabb',
    #     data='dataset_ccpd_1over10_xywh.yaml',
    #     epochs=100,
    #     batch=64,
    #     workers=32,
    #     imgsz=640,
    #     plots=True,
    #     device="0,1,2,3,4,5,6,7"
    # )

    # ################# CCPD 1/10 - OBB Training (YOLOv8n)
    # print("\n" + "="*80)
    # print("Starting OBB Training (YOLOv8n) - CCPD 1/10")
    # print("="*80 + "\n")
    # model = YOLO('yolov8n-obb.pt')
    # results = model.train(
    #     name='ccpd_1over10_yolov8n_obb',
    #     data='dataset_ccpd_1over10_xyxyxyxy.yaml',
    #     epochs=100,
    #     batch=64,
    #     workers=32,
    #     imgsz=640,
    #     plots=True,
    #     device="0,1,2,3,4,5,6,7"
    # )

    # ################# CCPD 1/10 - QBB Training (YOLOv8n)
    # print("\n" + "="*80)
    # print("Starting QBB Training (YOLOv8n) - CCPD 1/10")
    # print("="*80 + "\n")
    # model = YOLO('yolov8n-qbb.yaml')
    # results = model.train(
    #     name='ccpd_1over10_yolov8n_qbb',
    #     data='dataset_ccpd_1over10_xyxyxyxy.yaml',
    #     epochs=100,
    #     batch=64,
    #     workers=32,
    #     imgsz=640,
    #     plots=True,
    #     device="0,1,2,3,4,5,6,7"
    # )

    # ################# CCPD 1/10 - AABB Training (YOLOv11n)
    # print("\n" + "="*80)
    # print("Starting AABB Training (YOLOv11n) - CCPD 1/10")
    # print("="*80 + "\n")
    # model = YOLO('yolo11n.pt')
    # results = model.train(
    #     name='ccpd_1over10_yolo11n_aabb',
    #     data='dataset_ccpd_1over10_xywh.yaml',
    #     epochs=100,
    #     batch=64,
    #     workers=32,
    #     imgsz=640,
    #     plots=True,
    #     device="0,1,2,3,4,5,6,7"
    # )

    # ################# CCPD 1/10 - OBB Training (YOLOv11n)
    # print("\n" + "="*80)
    # print("Starting OBB Training (YOLOv11n) - CCPD 1/10")
    # print("="*80 + "\n")
    # model = YOLO('yolo11n-obb.pt')
    # results = model.train(
    #     name='ccpd_1over10_yolo11n_obb',
    #     data='dataset_ccpd_1over10_xyxyxyxy.yaml',
    #     epochs=100,
    #     batch=64,
    #     workers=32,
    #     imgsz=640,
    #     plots=True,
    #     device="0,1,2,3,4,5,6,7"
    # )

    # ################# CCPD Over60 - QBB Training (YOLOv11n) - Full Training
    # print("\n" + "="*80)
    # print("Starting QBB Training (YOLOv11n) - CCPD Over60")
    # print("="*80 + "\n")
    # model = YOLO('yolo11n-qbb.yaml')
    # results = model.train(
    #     name='ccpd_over60_yolov11n_qbb',
    #     data='dataset_ccpd_over60_xyxyxyxy.yaml',
    #     epochs=100,
    #     batch=64,
    #     workers=32,
    #     imgsz=640,
    #     plots=True,
    #     device="0,1,2,3,4,5,6,7",
    #     dfl=5.0,
    #     fliplr=0.0
    # )

    # ################# CCPD Over60 - QBB Training (YOLOv11m) - Full Training
    # print("\n" + "="*80)
    # print("Starting QBB Training (YOLOv11m) - CCPD Over60")
    # print("="*80 + "\n")
    # model = YOLO('yolo11m-qbb.yaml')
    # results = model.train(
    #     name='ccpd_over60_yolo11m_qbb',
    #     data='dataset_ccpd_over60_xyxyxyxy.yaml',
    #     epochs=100,
    #     batch=64,
    #     workers=32,
    #     imgsz=640,
    #     plots=True,
    #     device="0,1,2,3,4,5,6,7",
    #     dfl=5.0,
    #     fliplr=0.0
    # )

    # ################# QBB with Differentiable Polygon IoU - 100 Epoch Full Training
    # print("\n" + "="*80)
    # print("🚀 QBB with Differentiable Polygon IoU - 100 Epoch Full Training")
    # print("Experiment: ccpd_1over10_yolov8n_qbb_polyiou")
    # print("="*80 + "\n")
    # model = YOLO('yolov8n-qbb.yaml')
    # results = model.train(
    #     name='ccpd_1over10_yolov8n_qbb_polyiou',
    #     data='dataset_ccpd_1over10_xyxyxyxy.yaml',
    #     epochs=100,
    #     batch=64,
    #     workers=32,
    #     imgsz=640,
    #     plots=True,
    #     device="0,1,2,3,4,5,6,7",
    #     dfl=5.0,
    #     fliplr=0.0
    # )

    ################# QBB reg_max=1 Full Training - 100 Epoch
    print("\n" + "="*80)
    print("🚀 QBB reg_max=1 Full Training - 100 Epoch")
    print("Experiment: ccpd_1over10_yolov8n_qbb_regmax1")
    print("="*80 + "\n")
    model = YOLO('yolov8n-qbb.yaml')
    results = model.train(
        name='ccpd_1over10_yolov8n_qbb_regmax1',
        data='dataset_ccpd_1over10_xyxyxyxy.yaml',
        epochs=100,
        batch=64,
        workers=32,
        imgsz=640,
        plots=True,
        device="0,1,2,3,4,5,6,7",
        dfl=5.0,
        fliplr=0.0
    )

    # ################# CCPD Over60 - QBB Training (YOLOv8m) - Full Training
    # print("\n" + "="*80)
    # print("Starting QBB Training (YOLOv8m) - CCPD Over60")
    # print("="*80 + "\n")
    # model = YOLO('yolov8m-qbb.yaml')
    # results = model.train(
    #     name='ccpd_over60_yolov8m_qbb',
    #     data='dataset_ccpd_over60_xyxyxyxy.yaml',
    #     epochs=100,
    #     batch=64,
    #     workers=32,
    #     imgsz=640,
    #     plots=True,
    #     device="0,1,2,3,4,5,6,7",
    #     dfl=5.0,
    #     fliplr=0.0
    # )
