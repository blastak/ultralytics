#!/usr/bin/env python3
"""
QBB task 자동 인식 테스트
"""

import sys
from pathlib import Path

# Ultralytics 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

def test_qbb_auto_task():
    """QBB task가 자동으로 인식되는지 테스트"""
    
    print("=== QBB Task 자동 인식 테스트 ===\n")
    
    # 1. task 명시 없이 QBB 모델 로드
    print("1. task 파라미터 없이 QBB 모델 로드 테스트:")
    try:
        model = YOLO('yolo11n-qbb.yaml')
        print(f"✓ 모델 로드 성공!")
        print(f"  - 모델 타입: {model.model.__class__.__name__}")
        print(f"  - 자동 인식된 task: {model.task}")
        print(f"  - Head 타입: {model.model.model[-1].__class__.__name__}")
        
        if model.task == 'qbb':
            print("\n✅ QBB task가 자동으로 인식되었습니다!")
        else:
            print(f"\n❌ task가 '{model.task}'로 잘못 인식되었습니다.")
            
    except Exception as e:
        print(f"✗ 모델 로드 실패: {e}")
        
    # 2. OBB와 비교
    print("\n2. OBB 모델과 비교:")
    try:
        obb_model = YOLO('yolo11n-obb.yaml')
        print(f"  - OBB task: {obb_model.task} (자동 인식)")
        print(f"  - QBB task: {model.task} (자동 인식)")
        
    except Exception as e:
        print(f"✗ OBB 모델 로드 실패: {e}")

if __name__ == "__main__":
    test_qbb_auto_task()