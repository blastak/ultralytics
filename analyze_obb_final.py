#!/usr/bin/env python3
"""
번호판 OBB 모델로 false alarm과 miss detection 분석
"""

import cv2
import numpy as np
from pathlib import Path
import json
from ultralytics import YOLO
import shutil


def analyze_obb_performance():
    """OBB 모델 성능 분석"""
    
    # 학습된 OBB 모델 로드
    model_path = 'runs/obb/train5/weights/best.pt'
    if not Path(model_path).exists():
        print(f"모델 파일이 없습니다: {model_path}")
        return
    
    model = YOLO(model_path)
    print(f"OBB 모델 로드됨: {model_path}")
    
    # 테스트 데이터 경로
    test_dir = Path('/workspace/DB/01_LicensePlate/55_WebPlatemania_1944/all')
    
    # 결과 저장 디렉토리
    vis_dir = Path('obb_analysis_results')
    if vis_dir.exists():
        shutil.rmtree(vis_dir)
    vis_dir.mkdir()
    
    false_alarm_dir = vis_dir / 'false_alarms'
    miss_detection_dir = vis_dir / 'miss_detections'
    correct_detection_dir = vis_dir / 'correct_detections'
    
    false_alarm_dir.mkdir()
    miss_detection_dir.mkdir()
    correct_detection_dir.mkdir()
    
    # 테스트 이미지들 (처음 200개만)
    json_files = list(test_dir.glob('*.json'))[:200]
    
    false_alarms = []
    miss_detections = []
    correct_detections = []
    
    print(f"\n{len(json_files)}개 이미지 분석 시작...")
    
    for i, json_file in enumerate(json_files):
        if i % 50 == 0:
            print(f"진행률: {i}/{len(json_files)}")
        
        img_file = json_file.with_suffix('.jpg')
        if not img_file.exists():
            continue
        
        try:
            # Ground truth 로드
            with open(json_file, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            
            has_gt = any(
                shape['shape_type'] == 'polygon' and len(shape['points']) >= 4
                for shape in gt_data['shapes']
            )
            
            # YOLO 예측
            results = model(img_file, conf=0.25, verbose=False)
            has_detection = False
            
            if results[0].obb is not None and len(results[0].obb) > 0:
                has_detection = True
            
            # 분류
            if has_gt and has_detection:
                correct_detections.append(img_file)
            elif has_gt and not has_detection:
                miss_detections.append(img_file)
            elif not has_gt and has_detection:
                false_alarms.append(img_file)
            
            # 시각화 (일부만)
            if len(correct_detections) <= 10 and has_gt and has_detection:
                visualize_obb_result(img_file, json_file, results, correct_detection_dir, "correct")
            elif len(miss_detections) <= 10 and has_gt and not has_detection:
                visualize_obb_result(img_file, json_file, results, miss_detection_dir, "miss")
            elif len(false_alarms) <= 10 and not has_gt and has_detection:
                visualize_obb_result(img_file, json_file, results, false_alarm_dir, "false_alarm")
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    # 결과 출력
    print(f"\n=== OBB 모델 성능 분석 결과 ===")
    print(f"총 테스트 이미지: {len(json_files)}")
    print(f"False Alarms: {len(false_alarms)}")
    print(f"Miss Detections: {len(miss_detections)}")
    print(f"Correct Detections: {len(correct_detections)}")
    
    total_gt = len(miss_detections) + len(correct_detections)
    total_pred = len(false_alarms) + len(correct_detections)
    
    precision = len(correct_detections) / total_pred if total_pred > 0 else 0
    recall = len(correct_detections) / total_gt if total_gt > 0 else 0
    
    print(f"Precision: {precision:.3f} ({len(correct_detections)}/{total_pred})")
    print(f"Recall: {recall:.3f} ({len(correct_detections)}/{total_gt})")
    
    print(f"\n시각화 결과 저장: {vis_dir}")


def visualize_obb_result(img_file, json_file, results, save_dir, result_type):
    """OBB 결과 시각화"""
    
    img = cv2.imread(str(img_file))
    if img is None:
        return
    
    try:
        # Ground truth 그리기
        with open(json_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        for shape in gt_data['shapes']:
            if shape['shape_type'] == 'polygon' and len(shape['points']) >= 4:
                points = np.array(shape['points'], dtype=np.int32)
                cv2.polylines(img, [points], True, (0, 255, 0), 2)  # 초록색: GT
                cv2.putText(img, 'GT', (points[0][0], points[0][1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 예측 결과 그리기
        for r in results:
            if r.obb is not None and len(r.obb) > 0:
                for j in range(len(r.obb)):
                    obb_coords = r.obb.xyxyxyxy[j].cpu().numpy().reshape(-1, 2)
                    conf = r.obb.conf[j]
                    
                    points = np.array(obb_coords, dtype=np.int32)
                    cv2.polylines(img, [points], True, (0, 0, 255), 2)  # 빨간색: 예측
                    
                    center = np.mean(points, axis=0).astype(int)
                    cv2.putText(img, f'OBB:{conf:.2f}', (center[0]-30, center[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 저장
        save_path = save_dir / f'{result_type}_{img_file.name}'
        cv2.imwrite(str(save_path), img)
        
    except Exception as e:
        print(f"Error visualizing {img_file}: {e}")


def test_navertaxi_obb():
    """네이버택시 이미지로 OBB 테스트"""
    
    model_path = 'runs/obb/train5/weights/best.pt'
    if not Path(model_path).exists():
        print(f"모델 파일이 없습니다: {model_path}")
        return
    
    model = YOLO(model_path)
    
    # 네이버택시 이미지
    taxi_img = Path('@license_plate_navertaxi/61.jpg')
    if not taxi_img.exists():
        print(f"택시 이미지 없습니다: {taxi_img}")
        return
    
    print(f"\n네이버택시 이미지 OBB 테스트: {taxi_img}")
    
    # 다양한 신뢰도로 테스트
    confidence_thresholds = [0.1, 0.25, 0.5]
    
    for conf_thresh in confidence_thresholds:
        print(f"\nConfidence threshold: {conf_thresh}")
        
        results = model(taxi_img, conf=conf_thresh, verbose=False)
        
        img = cv2.imread(str(taxi_img))
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
        
        print(f"검출된 번호판: {detection_count}개")
        
        # 저장
        save_path = f'navertaxi_obb_conf_{conf_thresh}.jpg'
        cv2.imwrite(save_path, img)
        print(f"결과 저장: {save_path}")


def main():
    print("=== OBB 모델 최종 분석 ===")
    
    # 1. False alarm / Miss detection 분석
    analyze_obb_performance()
    
    # 2. 네이버택시 테스트
    test_navertaxi_obb()
    
    print("\n=== 분석 완료 ===")


if __name__ == '__main__':
    main()