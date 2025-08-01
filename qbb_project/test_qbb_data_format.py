#!/usr/bin/env python3
"""
QBB 데이터 형식 요구사항 테스트
"""

import sys
sys.path.insert(0, '/workspace/repo/ultralytics')

from ultralytics import YOLO
import torch
from pathlib import Path

def test_qbb_data_format():
    """QBB가 실제로 기대하는 데이터 형식 확인"""
    
    print("🔍 Testing QBB data format requirements...")
    
    # QBB 모델 로드
    model = YOLO('yolo11n-qbb.yaml')
    print(f"✅ QBB model loaded successfully")
    
    # 모델 정보 출력
    print(f"📊 Model task: {model.task}")
    print(f"📊 Model type: {type(model.model)}")
    
    # 헤드 정보 확인
    head = model.model.model[-1]  # 마지막 레이어가 헤드
    print(f"📊 Head type: {type(head)}")
    print(f"📊 Head class name: {head.__class__.__name__}")
    
    # QBB 헤드가 어떤 형식을 기대하는지 확인
    if hasattr(head, 'ne'):
        print(f"📊 Head extra parameters (ne): {head.ne}")
    
    if hasattr(head, 'cv4'):
        print(f"📊 Head has cv4 (angle prediction): Yes")
    else:
        print(f"📊 Head has cv4 (angle prediction): No")
    
    # QBB 데이터 로더 테스트
    try:
        # 5컬럼 데이터 테스트 (detection 형식)
        test_5col_data()
        
        # 6컬럼 데이터 테스트 (OBB 형식)  
        test_6col_data()
        
    except Exception as e:
        print(f"❌ Data format test failed: {e}")

def test_5col_data():
    """5컬럼 데이터 형식 테스트"""
    print(f"\n🧪 Testing 5-column data format (detection style)...")
    
    # 5컬럼 테스트 데이터 생성
    test_dir = Path('test_5col_data')
    test_dir.mkdir(exist_ok=True)
    
    (test_dir / 'images').mkdir(exist_ok=True)
    (test_dir / 'labels').mkdir(exist_ok=True)
    
    # 더미 이미지 생성 (640x640 검은색)
    import cv2
    import numpy as np
    
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(test_dir / 'images' / 'test.jpg'), img)
    
    # 5컬럼 라벨 생성 (class x y w h)
    with open(test_dir / 'labels' / 'test.txt', 'w') as f:
        f.write('0 0.5 0.5 0.2 0.1\n')
    
    # YAML 파일 생성
    yaml_content = f"""
path: {test_dir.absolute()}
train: images
val: images
test: images

names:
  0: test_object

nc: 1
"""
    
    yaml_file = test_dir / 'test_5col.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    try:
        model = YOLO('yolo11n-qbb.yaml')
        # 1 epoch 단축 테스트
        results = model.train(
            data=str(yaml_file),
            epochs=1,
            imgsz=640,
            batch=1,
            device='cpu',
            workers=0,
            verbose=False,
            plots=False,
            val=False,
            save=False,
            project='test_runs',
            name='test_5col',
            exist_ok=True
        )
        print(f"✅ 5-column data format: SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ 5-column data format: FAILED - {e}")
        return False
    
    finally:
        # 정리
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

def test_6col_data():
    """6컬럼 데이터 형식 테스트"""
    print(f"\n🧪 Testing 6-column data format (OBB style)...")
    
    # 6컬럼 테스트 데이터 생성
    test_dir = Path('test_6col_data')
    test_dir.mkdir(exist_ok=True)
    
    (test_dir / 'images').mkdir(exist_ok=True)
    (test_dir / 'labels').mkdir(exist_ok=True)
    
    # 더미 이미지 생성 (640x640 검은색)
    import cv2
    import numpy as np
    
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(test_dir / 'images' / 'test.jpg'), img)
    
    # 6컬럼 라벨 생성 (class x y w h angle)
    with open(test_dir / 'labels' / 'test.txt', 'w') as f:
        f.write('0 0.5 0.5 0.2 0.1 0.785\n')  # 45도 = 0.785 라디안
    
    # YAML 파일 생성
    yaml_content = f"""
path: {test_dir.absolute()}
train: images
val: images
test: images

names:
  0: test_object

nc: 1
"""
    
    yaml_file = test_dir / 'test_6col.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    try:
        model = YOLO('yolo11n-qbb.yaml')
        # 1 epoch 단축 테스트
        results = model.train(
            data=str(yaml_file),
            epochs=1,
            imgsz=640,
            batch=1,
            device='cpu',
            workers=0,
            verbose=False,
            plots=False,
            val=False,
            save=False,
            project='test_runs',
            name='test_6col',
            exist_ok=True
        )
        print(f"✅ 6-column data format: SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ 6-column data format: FAILED - {e}")
        return False
    
    finally:
        # 정리
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

def main():
    print("🔍 QBB Data Format Compatibility Test")
    print("=" * 40)
    
    test_qbb_data_format()
    
    print(f"\n📊 Test Results Summary:")
    print(f"This test will help determine the correct data format for QBB models.")

if __name__ == "__main__":
    main()