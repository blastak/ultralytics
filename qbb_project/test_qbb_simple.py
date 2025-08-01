#!/usr/bin/env python3
"""
간단한 QBB 모델 테스트
"""

import sys
sys.path.insert(0, '/workspace/repo/ultralytics')

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def test_qbb_inference():
    """QBB 모델 추론 테스트"""
    
    print("=== QBB 간단 추론 테스트 ===\n")
    
    # QBB 모델 로드 (task 자동 인식 확인)
    print("1. QBB 모델 로드 중...")
    model = YOLO('yolo11n-qbb.yaml')
    print(f"✓ QBB 모델 로드 성공! Task: {model.task}")
    
    # 테스트 이미지 생성
    print("\n2. 테스트 이미지 생성...")
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    test_path = Path("test_qbb_image.jpg")
    cv2.imwrite(str(test_path), test_img)
    print("✓ 테스트 이미지 생성됨")
    
    # 추론 실행
    print("\n3. QBB 추론 실행...")
    results = model(test_path, verbose=False)
    print("✓ 추론 실행 성공!")
    
    # 결과 확인
    for r in results:
        print(f"\n결과:")
        print(f"- 이미지 크기: {r.orig_shape}")
        print(f"- QBB 결과 타입: {type(r.obb)}")
        if r.obb is not None:
            print(f"- 검출 수: {len(r.obb)}")
        else:
            print("- 검출된 객체 없음 (정상 - 랜덤 이미지)")
    
    # 정리
    test_path.unlink()
    print("\n✅ QBB 모델이 정상적으로 작동합니다!")

def test_obb_vs_qbb_comparison():
    """OBB vs QBB 비교"""
    
    print("\n=== OBB vs QBB 비교 ===")
    
    # 모델들 로드
    obb_model = YOLO('yolo11n-obb.yaml')
    qbb_model = YOLO('yolo11n-qbb.yaml')
    
    print(f"OBB 모델 task: {obb_model.task}")
    print(f"QBB 모델 task: {qbb_model.task}")
    
    # 파라미터 수 비교
    obb_params = sum(p.numel() for p in obb_model.model.parameters())
    qbb_params = sum(p.numel() for p in qbb_model.model.parameters())
    
    print(f"OBB 파라미터 수: {obb_params:,}")
    print(f"QBB 파라미터 수: {qbb_params:,}")
    
    if obb_params == qbb_params:
        print("✅ OBB와 QBB가 동일한 파라미터 수를 가집니다!")
    else:
        print("❌ 파라미터 수가 다릅니다.")

if __name__ == "__main__":
    test_qbb_inference()
    test_obb_vs_qbb_comparison()