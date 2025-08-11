#!/usr/bin/env python
"""
OBB vs QBB 모델 학습 테스트 스크립트
DOTA8 데이터셋을 사용하여 두 모델의 성능을 비교합니다.
"""

from ultralytics import YOLO
import os
from pathlib import Path

# 데이터셋 경로를 assets 폴더로 설정
DATASET_DIR = Path("ultralytics/assets/datasets")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def train_obb():
    """OBB 모델 학습"""
    print("=" * 50)
    print("OBB 모델 학습 시작...")
    print("=" * 50)
    
    # OBB 모델 초기화
    model = YOLO('ultralytics/cfg/models/v8/yolov8-obb.yaml')  # 새 모델 생성
    
    # 학습 실행 (테스트용으로 짧게)
    results = model.train(
        data='ultralytics/cfg/datasets/dota8.yaml',
        epochs=20,  # 20 에포크로 증가
        imgsz=640,
        batch=4,
        device='cpu',  # GPU가 없으면 CPU 사용
        project='runs/obb',
        name='test',
        exist_ok=True,
        verbose=True
    )
    
    print("\nOBB 학습 완료!")
    return results

def train_qbb():
    """QBB 모델 학습"""
    print("=" * 50)
    print("QBB 모델 학습 시작...")
    print("=" * 50)
    
    # QBB 모델 초기화
    model = YOLO('ultralytics/cfg/models/v8/yolov8-qbb.yaml')  # 새 모델 생성
    
    # 학습 실행 (테스트용으로 짧게)
    results = model.train(
        data='ultralytics/cfg/datasets/dota8.yaml',
        epochs=20,  # 20 에포크로 증가
        imgsz=640,
        batch=4,
        device='cpu',  # GPU가 없으면 CPU 사용
        project='runs/qbb',
        name='test',
        exist_ok=True,
        verbose=True
    )
    
    print("\nQBB 학습 완료!")
    return results

if __name__ == "__main__":
    print("OBB vs QBB 학습 테스트")
    print("현재 작업 디렉토리:", os.getcwd())
    print(f"데이터셋 디렉토리: {DATASET_DIR.absolute()}")
    
    # 환경 변수로 데이터셋 경로 설정
    os.environ['YOLO_DATASETS_DIR'] = str(DATASET_DIR.absolute())
    
    # 데이터셋 경로 확인
    dota8_path = DATASET_DIR / "dota8"
    if not dota8_path.exists():
        print(f"\n주의: DOTA8 데이터셋이 {DATASET_DIR}에 자동으로 다운로드됩니다.")
    
    try:
        # OBB 모델 학습
        print("\n1. OBB 모델 학습")
        obb_results = train_obb()
        
        # QBB 모델 학습
        print("\n2. QBB 모델 학습")
        qbb_results = train_qbb()
        
        print("\n" + "=" * 50)
        print("학습 완료! 결과는 runs/ 폴더에 저장되었습니다.")
        print("OBB 결과: runs/obb/test/")
        print("QBB 결과: runs/qbb/test/")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()