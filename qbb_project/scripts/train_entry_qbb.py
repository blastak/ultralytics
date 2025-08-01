#!/usr/bin/env python3
"""
사용자 방식에 따른 QBB 학습 진입점
"""

import cv2
import numpy as np
from pathlib import Path
import json
import shutil
from ultralytics import YOLO


def create_8point_qbb_dataset():
    """8-point QBB 형식 데이터셋 생성"""
    
    source_dir = Path('/workspace/DB/01_LicensePlate/55_WebPlatemania_1944/all')
    target_dir = Path('webpm_qbb8_dataset')
    
    # 기존 디렉토리 삭제 후 재생성
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    # 디렉토리 구조 생성 (사용자 형식에 맞춤)
    (target_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (target_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (target_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (target_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # 작은 데이터셋으로 시작 (100개)
    json_files = list(source_dir.glob('*.json'))[:150]
    
    # 분할 (test 제거, train/val만 사용)
    train_files = json_files[:120]  # 80%
    val_files = json_files[120:]    # 20% 
    
    splits_data = {
        'train': train_files,
        'val': val_files
    }
    
    total_converted = 0
    
    for split_name, files in splits_data.items():
        print(f"Processing {split_name}: {len(files)} files")
        
        for json_file in files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    labelme_data = json.load(f)
                
                img_width = labelme_data['imageWidth']
                img_height = labelme_data['imageHeight']
                
                # 이미지 파일 복사
                img_name = json_file.stem + '.jpg'
                src_img = source_dir / img_name
                dst_img = target_dir / 'images' / split_name / img_name
                
                if not src_img.exists():
                    continue
                
                shutil.copy2(src_img, dst_img)
                
                # 라벨 파일 생성 (8-point 형식)
                label_file = target_dir / 'labels' / split_name / (json_file.stem + '.txt')
                
                labels = []
                for shape in labelme_data['shapes']:
                    if shape['shape_type'] == 'polygon' and len(shape['points']) >= 4:
                        points = shape['points']
                        
                        # 4개 점이 있는 경우만 처리
                        if len(points) == 4:
                            # 8-point 형식: class x1 y1 x2 y2 x3 y3 x4 y4
                            normalized_points = []
                            for x, y in points:
                                norm_x = x / img_width
                                norm_y = y / img_height
                                
                                # 범위 체크
                                norm_x = max(0, min(1, norm_x))
                                norm_y = max(0, min(1, norm_y))
                                
                                normalized_points.extend([norm_x, norm_y])
                            
                            # 9개 컬럼: class + 8개 좌표
                            class_idx = 0
                            coord_str = " ".join(f"{p:.6f}" for p in normalized_points)
                            label = f"{class_idx} {coord_str}"
                            labels.append(label)
                            total_converted += 1
                
                # 라벨이 있는 경우에만 파일 생성
                if labels:
                    with open(label_file, 'w') as f:
                        for label in labels:
                            f.write(label + '\n')
                
            except Exception as e:
                print(f"Error: {json_file}: {e}")
                continue
    
    print(f"Total converted: {total_converted}")
    
    # YAML 생성 (사용자 제공 형식 적용)
    yaml_content = f"""path: {target_dir.absolute()}
train: images/train
val: images/val
nc: 1
names: ['license_plate']
"""
    
    yaml_file = target_dir / 'webpm_qbb8.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"YAML saved: {yaml_file}")
    return target_dir, yaml_file


def main():
    print("Creating 8-point QBB dataset...")
    dataset_dir, yaml_file = create_8point_qbb_dataset()
    
    # 라벨 검증
    train_labels = list((dataset_dir / 'labels' / 'train').glob('*.txt'))
    print(f"\nVerifying 8-point labels...")
    
    for i, label_file in enumerate(train_labels[:3]):
        print(f"Sample {i+1}: {label_file.name}")
        with open(label_file, 'r') as f:
            content = f.read().strip()
            lines = content.split('\n')
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    print(f"  {len(parts)} columns: {parts[0]} {float(parts[1]):.3f} {float(parts[2]):.3f} ... {float(parts[-2]):.3f} {float(parts[-1]):.3f}")
    
    print(f"\nStarting QBB training with user's method...")
    
    # 사용자 방식으로 학습
    try:
        model = YOLO('yolo11n-qbb.pt')  # 사용자는 .yaml을 썼지만 .pt가 더 안전
        results = model.train(
            data=str(yaml_file), 
            epochs=10, 
            imgsz=640, 
            fliplr=0.0, 
            batch=1, 
            workers=0
        )
        
        print("✅ QBB training completed successfully!")
        print(f"Results saved in: runs/qbb/train/")
        
        # 모델 테스트
        best_model_path = 'runs/qbb/train/weights/best.pt'
        if Path(best_model_path).exists():
            print(f"Best model saved at: {best_model_path}")
            test_qbb_model(best_model_path, dataset_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def test_qbb_model(model_path, dataset_dir):
    """QBB 모델 테스트"""
    
    print(f"\nTesting QBB model: {model_path}")
    
    try:
        model = YOLO(model_path)
        
        # 테스트 이미지들
        test_images = list((dataset_dir / 'images' / 'val').glob('*.jpg'))[:5]
        
        vis_dir = Path('qbb_test_results')
        vis_dir.mkdir(exist_ok=True)
        
        for i, img_path in enumerate(test_images):
            print(f"Testing: {img_path.name}")
            
            # 예측 수행
            results = model(img_path, verbose=False)
            
            # 결과 시각화
            img = cv2.imread(str(img_path))
            
            # QBB 결과 표시
            detection_count = 0
            for r in results:
                if r.qbb is not None and len(r.qbb) > 0:
                    detection_count = len(r.qbb)
                    for j in range(len(r.qbb)):
                        # QBB 좌표 추출
                        qbb_coords = r.qbb.xyxyxyxy[j].cpu().numpy().reshape(-1, 2)
                        conf = r.qbb.conf[j]
                        
                        # QBB 그리기
                        points = np.array(qbb_coords, dtype=np.int32)
                        cv2.polylines(img, [points], True, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        # 신뢰도 표시
                        center = np.mean(points, axis=0).astype(int)
                        cv2.putText(img, f'QBB: {conf:.3f}', (center[0]-50, center[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"  Detected {detection_count} license plates")
            
            # 저장
            save_path = vis_dir / f'qbb_test_{i+1:02d}_{img_path.name}'
            cv2.imwrite(str(save_path), img)
            print(f"  Saved: {save_path}")
        
        print(f"\nTest results saved in: {vis_dir}")
        
    except Exception as e:
        print(f"Testing failed: {e}")


if __name__ == '__main__':
    main()