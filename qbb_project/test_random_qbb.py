#!/usr/bin/env python3
"""
무작위 이미지로 QBB 모델 테스트
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def test_random_images():
    """무작위 이미지들로 QBB 테스트"""
    
    model_path = 'runs/qbb/train5/weights/best.pt'
    if not Path(model_path).exists():
        print(f"모델 파일이 없습니다: {model_path}")
        return
    
    model = YOLO(model_path)
    print(f"QBB 모델 로드됨: {model_path}")
    
    # 테스트용 이미지들 찾기
    test_dir = Path('/workspace/DB/01_LicensePlate/55_WebPlatemania_1944/all')
    test_images = list(test_dir.glob('*.jpg'))[300:310]  # 300~309번 이미지
    
    vis_dir = Path('random_qbb_test')
    vis_dir.mkdir(exist_ok=True)
    
    print(f"\n{len(test_images)}개 무작위 이미지 테스트...")
    
    for i, img_file in enumerate(test_images):
        print(f"\n테스트 {i+1}: {img_file.name}")
        
        # 다양한 신뢰도로 테스트
        for conf in [0.1, 0.25, 0.5]:
            results = model(img_file, conf=conf, verbose=False)
            
            img = cv2.imread(str(img_file))
            detection_count = 0
            
            for r in results:
                if r.qbb is not None and len(r.qbb) > 0:
                    detection_count = len(r.qbb)
                    for j in range(len(r.qbb)):
                        qbb_coords = r.qbb.xyxyxyxy[j].cpu().numpy().reshape(-1, 2)
                        conf_score = r.qbb.conf[j]
                        
                        points = np.array(qbb_coords, dtype=np.int32)
                        cv2.polylines(img, [points], True, (0, 255, 0), 3)
                        
                        center = np.mean(points, axis=0).astype(int)
                        cv2.putText(img, f'QBB: {conf_score:.3f}', (center[0]-50, center[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            print(f"  Conf {conf}: {detection_count}개 검출")
            
            # 가장 좋은 결과 저장 (conf=0.25)
            if conf == 0.25:
                save_path = vis_dir / f'test_{i+1:02d}_{detection_count}det_{img_file.name}'
                cv2.imwrite(str(save_path), img)
                print(f"  저장: {save_path}")


def test_specific_images():
    """특정 이미지들로 상세 테스트"""
    
    model_path = 'runs/qbb/train5/weights/best.pt'
    model = YOLO(model_path)
    
    # 특정 이미지들 테스트
    test_dir = Path('/workspace/DB/01_LicensePlate/55_WebPlatemania_1944/all')
    specific_images = [
        '14113295_P2_79가3535.jpg',  # 학습에 사용된 이미지
        '14112898_P2_71거1377.jpg',  # 첫 번째 이미지
        '14627726_P2_84버1104.jpg',  # 다른 타입
    ]
    
    print(f"\n특정 이미지 상세 테스트:")
    
    for img_name in specific_images:
        img_file = test_dir / img_name
        if not img_file.exists():
            print(f"이미지 없음: {img_name}")
            continue
        
        print(f"\n=== {img_name} ===")
        
        # 여러 신뢰도로 테스트
        for conf in [0.1, 0.2, 0.3, 0.4, 0.5]:
            results = model(img_file, conf=conf, verbose=False)
            
            detection_count = 0
            max_conf = 0.0
            
            if results[0].qbb is not None and len(results[0].qbb) > 0:
                detection_count = len(results[0].qbb)
                max_conf = float(results[0].qbb.conf.max())
            
            print(f"  Conf {conf}: {detection_count}개 (최고 신뢰도: {max_conf:.3f})")
            
            # 0.25에서 이미지 저장
            if conf == 0.25 and detection_count > 0:
                img = cv2.imread(str(img_file))
                
                for r in results:
                    if r.qbb is not None and len(r.qbb) > 0:
                        for j in range(len(r.qbb)):
                            qbb_coords = r.qbb.xyxyxyxy[j].cpu().numpy().reshape(-1, 2)
                            conf_score = r.qbb.conf[j]
                            
                            points = np.array(qbb_coords, dtype=np.int32)
                            cv2.polylines(img, [points], True, (0, 255, 0), 3)
                            
                            center = np.mean(points, axis=0).astype(int)
                            cv2.putText(img, f'QBB: {conf_score:.3f}', (center[0]-50, center[1]), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                save_path = f'specific_test_{img_name}'
                cv2.imwrite(save_path, img)
                print(f"  저장됨: {save_path}")


def main():
    print("=== QBB 모델 무작위 테스트 ===")
    
    # 1. 무작위 이미지들 테스트
    test_random_images()
    
    # 2. 특정 이미지들 상세 테스트
    test_specific_images()
    
    print("\n=== 테스트 완료 ===")


if __name__ == '__main__':
    main()