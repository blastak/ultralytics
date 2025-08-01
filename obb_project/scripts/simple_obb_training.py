#!/usr/bin/env python3
"""
간단하고 확실한 OBB 모델 학습
"""

import cv2
import numpy as np
from pathlib import Path
import json
import math
import shutil
from ultralytics import YOLO
import torch


def create_simple_obb_dataset():
    """간단한 OBB 데이터셋 생성"""
    
    source_dir = Path('../../DB/01_LicensePlate/55_WebPlatemania_1944/all')
    target_dir = Path('simple_obb_dataset')
    
    # 기존 디렉토리 삭제 후 재생성
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    # 디렉토리 구조 생성
    splits = ['train', 'val', 'test']
    for split in splits:
        (target_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (target_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # JSON 파일들 수집 (처음 100개만 사용해서 빠르게 테스트)
    json_files = list(source_dir.glob('*.json'))[:300]  # 작은 데이터셋으로 시작
    total_files = len(json_files)
    
    print(f"Using {total_files} JSON files for quick test")
    
    # 7:2:1 비율로 분할
    train_end = int(total_files * 0.7)
    val_end = int(total_files * 0.9)
    
    train_files = json_files[:train_end]
    val_files = json_files[train_end:val_end]
    test_files = json_files[val_end:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    splits_data = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    total_converted = 0
    
    for split_name, files in splits_data.items():
        print(f"\nProcessing {split_name} split...")
        
        for i, json_file in enumerate(files):
            try:
                # JSON 파일 읽기
                with open(json_file, 'r', encoding='utf-8') as f:
                    labelme_data = json.load(f)
                
                img_width = labelme_data['imageWidth']
                img_height = labelme_data['imageHeight']
                
                # 이미지 파일 복사
                img_name = json_file.stem + '.jpg'
                src_img = source_dir / img_name
                dst_img = target_dir / split_name / 'images' / img_name
                
                if not src_img.exists():
                    continue
                
                shutil.copy2(src_img, dst_img)
                
                # 라벨 파일 생성
                label_file = target_dir / split_name / 'labels' / (json_file.stem + '.txt')
                
                with open(label_file, 'w', encoding='utf-8') as f:
                    for shape in labelme_data['shapes']:
                        if shape['shape_type'] == 'polygon' and len(shape['points']) >= 3:
                            points = np.array(shape['points'], dtype=np.float32)
                            
                            # YOLOv11 방식: cv2.minAreaRect 사용
                            try:
                                (cx, cy), (w, h), angle = cv2.minAreaRect(points)
                                
                                # 기본 검증
                                if w <= 0 or h <= 0:
                                    continue
                                if not (0 <= cx < img_width and 0 <= cy < img_height):
                                    continue
                                
                                # 정규화
                                norm_cx = cx / img_width
                                norm_cy = cy / img_height
                                norm_w = w / img_width
                                norm_h = h / img_height
                                
                                # 범위 체크
                                if not (0 < norm_cx < 1 and 0 < norm_cy < 1):
                                    continue
                                if not (0 < norm_w < 1 and 0 < norm_h < 1):
                                    continue
                                
                                # 각도를 라디안으로 변환 (YOLOv11 방식)
                                angle_rad = angle / 180 * np.pi
                                
                                # 클래스는 0 (번호판)
                                class_idx = 0
                                
                                # YOLOv11 OBB 형식: class cx cy w h rotation
                                line = f"{class_idx} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f} {angle_rad:.6f}\n"
                                f.write(line)
                                total_converted += 1
                                
                            except Exception as e:
                                print(f"Error processing shape in {json_file}: {e}")
                                continue
                        
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
    
    print(f"\nConversion completed!")
    print(f"Total objects converted: {total_converted}")
    
    # YAML 설정 파일 생성
    yaml_content = f"""# Simple YOLOv11 OBB dataset
path: {target_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: license_plate

# Number of classes
nc: 1
"""
    
    yaml_file = target_dir / 'simple_obb.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"YAML configuration saved: {yaml_file}")
    
    return target_dir, yaml_file


def train_simple_obb():
    """간단한 OBB 모델 학습"""
    
    print("Creating simple OBB dataset...")
    dataset_dir, yaml_file = create_simple_obb_dataset()
    
    print("Starting simple OBB training...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # YOLOv11n-obb 모델 로드
    model = YOLO('yolo11n-obb.pt')
    
    # 간단한 학습 설정
    try:
        results = model.train(
            data=str(yaml_file),
            epochs=5,                       # 짧은 학습으로 시작
            imgsz=640,
            batch=4,                        # 작은 배치
            device=0 if torch.cuda.is_available() else 'cpu',
            workers=1,                      # 워커 1개
            patience=5,
            save=True,
            plots=True,
            val=True,
            project='runs/obb',
            name='simple_train',
            exist_ok=True,
            verbose=True,
            lr0=0.01,                       # 표준 학습률
            cache=False                     # 캐시 비활성화
        )
        
        print("Simple OBB training completed successfully!")
        print(f"Results saved in: runs/obb/simple_train/")
        
        # 모델 테스트
        best_model_path = 'runs/obb/simple_train/weights/best.pt'
        if Path(best_model_path).exists():
            print(f"Best model saved at: {best_model_path}")
            test_trained_model(best_model_path, dataset_dir)
            return True
        else:
            print("Warning: Best model not found")
            return False
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False


def test_trained_model(model_path, dataset_dir):
    """학습된 모델 테스트"""
    
    print(f"\nTesting trained OBB model: {model_path}")
    
    try:
        model = YOLO(model_path)
        
        # 테스트 이미지들 가져오기
        test_images = list((dataset_dir / 'test' / 'images').glob('*.jpg'))[:5]  # 처음 5개만
        
        vis_dir = Path('simple_obb_test_results')
        vis_dir.mkdir(exist_ok=True)
        
        for i, img_path in enumerate(test_images):
            print(f"Testing image {i+1}: {img_path.name}")
            
            # 예측 수행
            results = model(img_path, verbose=False)
            
            # 결과 시각화
            img = cv2.imread(str(img_path))
            
            # OBB 결과 표시
            for r in results:
                if r.obb is not None and len(r.obb) > 0:
                    for j in range(len(r.obb)):
                        # OBB 좌표 추출
                        obb_coords = r.obb.xyxyxyxy[j].cpu().numpy().reshape(-1, 2)
                        conf = r.obb.conf[j]
                        
                        # OBB 그리기
                        points = np.array(obb_coords, dtype=np.int32)
                        cv2.polylines(img, [points], True, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        # 신뢰도 표시
                        center = np.mean(points, axis=0).astype(int)
                        cv2.putText(img, f'OBB: {conf:.3f}', (center[0]-50, center[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    print(f"  Detected {len(r.obb)} license plates")
                else:
                    print(f"  No detections")
            
            # 저장
            save_path = vis_dir / f'test_result_{i+1:02d}_{img_path.name}'
            cv2.imwrite(str(save_path), img)
            print(f"  Saved: {save_path}")
        
        print(f"\nTest results saved in: {vis_dir}")
        
    except Exception as e:
        print(f"Testing failed: {e}")


def main():
    print("Starting simple OBB training approach...")
    
    success = train_simple_obb()
    
    if success:
        print("\n✅ Simple OBB training completed successfully!")
        print("Next steps:")
        print("1. Check results in runs/obb/simple_train/")
        print("2. Review test results in simple_obb_test_results/")
        print("3. Scale up with more data if needed")
    else:
        print("\n❌ Simple OBB training failed")
        print("Check error messages above for debugging")


if __name__ == "__main__":
    main()