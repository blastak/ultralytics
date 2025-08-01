#!/usr/bin/env python3
"""
인하대 테스트 이미지로 OBB 모델 테스트
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def test_inha_images():
    """인하대 테스트 이미지들로 OBB 테스트"""
    
    model_path = 'runs/obb/train5/weights/best.pt'
    if not Path(model_path).exists():
        print(f"모델 파일이 없습니다: {model_path}")
        return
    
    model = YOLO(model_path)
    print(f"OBB 모델 로드됨: {model_path}")
    
    # 인하대 테스트 이미지들
    test_dir = Path('/workspace/DB/01_LicensePlate/inha_test')
    test_images = list(test_dir.glob('*.jpg'))
    
    vis_dir = Path('inha_obb_test_results')
    vis_dir.mkdir(exist_ok=True)
    
    print(f"\n{len(test_images)}개 인하대 테스트 이미지 분석...")
    
    for i, img_file in enumerate(test_images):
        print(f"\n=== 테스트 {i+1}: {img_file.name} ===")
        
        # 다양한 신뢰도로 테스트
        confidence_thresholds = [0.1, 0.25, 0.5]
        
        for conf_thresh in confidence_thresholds:
            results = model(img_file, conf=conf_thresh, verbose=False)
            
            detection_count = 0
            max_conf = 0.0
            
            if results[0].obb is not None and len(results[0].obb) > 0:
                detection_count = len(results[0].obb)
                max_conf = float(results[0].obb.conf.max())
            
            print(f"  Conf {conf_thresh}: {detection_count}개 검출 (최고 신뢰도: {max_conf:.3f})")
        
        # 0.25 신뢰도로 시각화
        results = model(img_file, conf=0.25, verbose=False)
        
        img = cv2.imread(str(img_file))
        detection_count = 0
        
        for r in results:
            if r.obb is not None and len(r.obb) > 0:
                detection_count = len(r.obb)
                for j in range(len(r.obb)):
                    obb_coords = r.obb.xyxyxyxy[j].cpu().numpy().reshape(-1, 2)
                    conf = r.obb.conf[j]
                    
                    points = np.array(obb_coords, dtype=np.int32)
                    cv2.polylines(img, [points], True, (0, 255, 0), 3)
                    
                    center = np.mean(points, axis=0).astype(int)
                    cv2.putText(img, f'OBB: {conf:.3f}', (center[0]-50, center[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 저장
        save_path = vis_dir / f'inha_test_{i+1:02d}_{detection_count}det_{img_file.name}'
        cv2.imwrite(str(save_path), img)
        print(f"  저장: {save_path}")


def detailed_inha_analysis():
    """인하대 이미지 상세 분석"""
    
    model_path = 'runs/obb/train5/weights/best.pt'
    model = YOLO(model_path)
    
    test_dir = Path('/workspace/DB/01_LicensePlate/inha_test')
    test_images = list(test_dir.glob('*.jpg'))
    
    print(f"\n=== 인하대 이미지 상세 분석 ===")
    
    total_detections = 0
    
    for i, img_file in enumerate(test_images):
        print(f"\n{i+1}. {img_file.name}")
        
        # 이미지 정보
        img = cv2.imread(str(img_file))
        height, width = img.shape[:2]
        print(f"   크기: {width}x{height}")
        
        # 다양한 신뢰도로 상세 분석
        for conf in [0.1, 0.2, 0.3, 0.4, 0.5]:
            results = model(img_file, conf=conf, verbose=False)
            
            detection_count = 0
            confidences = []
            
            if results[0].obb is not None and len(results[0].obb) > 0:
                detection_count = len(results[0].obb)
                confidences = [float(c) for c in results[0].obb.conf]
            
            if detection_count > 0:
                conf_str = ", ".join([f"{c:.3f}" for c in confidences])
                print(f"   Conf {conf}: {detection_count}개 [{conf_str}]")
            else:
                print(f"   Conf {conf}: 0개")
        
        # 최적 신뢰도(0.25)에서 검출 수 카운트
        results = model(img_file, conf=0.25, verbose=False)
        if results[0].obb is not None and len(results[0].obb) > 0:
            total_detections += len(results[0].obb)
    
    print(f"\n총 검출된 번호판: {total_detections}개 (신뢰도 0.25 기준)")
    print(f"평균 검출률: {total_detections/len(test_images):.2f}개/이미지")


def compare_with_different_models():
    """다른 신뢰도와 비교 분석"""
    
    model_path = 'runs/obb/train5/weights/best.pt'
    model = YOLO(model_path)
    
    test_dir = Path('/workspace/DB/01_LicensePlate/inha_test')
    test_images = list(test_dir.glob('*.jpg'))
    
    print(f"\n=== 신뢰도별 성능 비교 ===")
    
    confidence_levels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    for conf in confidence_levels:
        total_detections = 0
        images_with_detections = 0
        
        for img_file in test_images:
            results = model(img_file, conf=conf, verbose=False)
            
            if results[0].obb is not None and len(results[0].obb) > 0:
                total_detections += len(results[0].obb)
                images_with_detections += 1
        
        detection_rate = images_with_detections / len(test_images) * 100
        avg_detections = total_detections / len(test_images)
        
        print(f"Conf {conf}: {total_detections}개 검출, {images_with_detections}/{len(test_images)}장 ({detection_rate:.1f}%), 평균 {avg_detections:.2f}개/이미지")


def main():
    print("=== 인하대 테스트 이미지 OBB 분석 ===")
    
    # 1. 기본 테스트 및 시각화
    test_inha_images()
    
    # 2. 상세 분석
    detailed_inha_analysis()
    
    # 3. 신뢰도별 성능 비교
    compare_with_different_models()
    
    print("\n=== 인하대 테스트 완료 ===")


if __name__ == '__main__':
    main()