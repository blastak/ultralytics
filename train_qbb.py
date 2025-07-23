from ultralytics import YOLO

def train_qbb():
    """
    YOLOv11-QBB 모델을 사용하여 번호판 QBB(Quadrilateral Bounding Box) 감지 모델을 학습합니다.
    """
    # 모델 구성 파일로부터 YOLO 모델을 로드합니다.
    # 'yolo11-qbb.yaml'은 모델 구조를 정의합니다.
    model = YOLO('ultralytics/cfg/models/11/yolo11-qbb.yaml')

    # 모델 학습을 시작합니다.
    # 'task=qbb'는 Quadrilateral Bounding Box 작업을 명시적으로 지정합니다.
    # 'data'는 데이터셋 설정 파일의 경로입니다.
    # 'epochs'는 총 학습 반복 횟수입니다.
    # 'imgsz'는 학습 이미지의 크기입니다.
    # 'device'는 학습에 사용할 장치를 지정합니다 (예: 'cpu', '0', '0,1').
    results = model.train(
        task='qbb',
        data='license_plate_qbb.yaml',
        epochs=100,
        imgsz=640,
        device='0'  # GPU 사용을 위해 '0'으로 설정, CPU를 사용하려면 'cpu'로 변경
    )

    # 학습 결과를 출력합니다.
    print(results)

if __name__ == '__main__':
    train_qbb()
