#!/usr/bin/env python3
"""
DOTA8 데이터셋으로 OBB 학습 테스트
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


def main():
    print("DOTA8 데이터셋으로 OBB 학습 테스트 시작...")
    
    try:
        # 사용자가 제공한 방식으로 모델 생성 및 학습
        model = YOLO('yolo11n-obb.pt')
        
        results = model.train(
            data='ultralytics/cfg/datasets/dota8.yaml',
            epochs=10,
            imgsz=640,
            fliplr=0.0,
            batch=1,
            workers=0
        )
        
        print("✅ DOTA8 OBB 학습 완료!")
        print(f"Results saved in: runs/obb/train/")
        
        # 학습된 모델로 테스트
        best_model_path = 'runs/obb/train/weights/best.pt'
        if Path(best_model_path).exists():
            print(f"Best model saved at: {best_model_path}")
            test_dota8_model(best_model_path)
        
        return True
        
    except Exception as e:
        print(f"❌ DOTA8 학습 실패: {e}")
        return False


def test_dota8_model(model_path):
    """DOTA8 모델 테스트"""
    
    print(f"\nDOTA8 모델 테스트: {model_path}")
    
    try:
        model = YOLO(model_path)
        
        # 테스트 이미지 경로 확인
        test_dir = Path('../datasets/dota8/images/val')
        if not test_dir.exists():
            print(f"테스트 디렉토리 없음: {test_dir}")
            return
        
        # 테스트 이미지들
        test_images = list(test_dir.glob('*.jpg'))[:3]
        
        if not test_images:
            print("테스트 이미지 없음")
            return
        
        vis_dir = Path('dota8_test_results')
        vis_dir.mkdir(exist_ok=True)
        
        for i, img_path in enumerate(test_images):
            print(f"Testing: {img_path.name}")
            
            # 예측 수행
            results = model(img_path, verbose=False)
            
            # 결과 시각화
            img = cv2.imread(str(img_path))
            
            # OBB 결과 표시
            detection_count = 0
            for r in results:
                if r.obb is not None and len(r.obb) > 0:
                    detection_count = len(r.obb)
                    for j in range(len(r.obb)):
                        # OBB 좌표 추출
                        obb_coords = r.obb.xyxyxyxy[j].cpu().numpy().reshape(-1, 2)
                        conf = r.obb.conf[j]
                        cls = int(r.obb.cls[j])
                        
                        # OBB 그리기
                        points = np.array(obb_coords, dtype=np.int32)
                        cv2.polylines(img, [points], True, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        # 클래스와 신뢰도 표시
                        center = np.mean(points, axis=0).astype(int)
                        cv2.putText(img, f'cls:{cls} {conf:.3f}', (center[0]-50, center[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"  Detected {detection_count} objects")
            
            # 저장
            save_path = vis_dir / f'dota8_test_{i+1:02d}_{img_path.name}'
            cv2.imwrite(str(save_path), img)
            print(f"  Saved: {save_path}")
        
        print(f"\nDOTA8 테스트 결과 저장: {vis_dir}")
        
    except Exception as e:
        print(f"DOTA8 테스트 실패: {e}")


if __name__ == '__main__':
    main()