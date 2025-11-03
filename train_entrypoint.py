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

    ################# CCPD Over60 - QBB Training (YOLOv11n) - 2 Epoch Test (Debug)
    print("\n" + "="*80)
    print("Starting QBB Training (YOLOv11n) - CCPD Over60 - 2 Epochs (Debug)")
    print("="*80 + "\n")
    model = YOLO('yolo11n-qbb.yaml')
    results = model.train(
        name='ccpd_over60_yolov11n_qbb_2epoch_debug',
        data='dataset_ccpd_over60_xyxyxyxy.yaml',
        epochs=2,
        batch=4,
        workers=0,
        imgsz=640,
        plots=True,
        device="0"
    )
