#!/usr/bin/env python3
"""
간단한 QBB 라벨 형식 수정
"""

from pathlib import Path
import cv2
import numpy as np
import json
import math
import shutil


def fix_qbb_labels():
    """QBB 라벨 형식 수정"""
    
    source_dir = Path('../../DB/01_LicensePlate/55_WebPlatemania_1944/all')
    target_dir = Path('fixed_qbb_dataset')
    
    # 기존 디렉토리 삭제 후 재생성
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    # 디렉토리 구조 생성
    splits = ['train', 'val', 'test']
    for split in splits:
        (target_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (target_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 작은 데이터셋으로 시작 (50개만)
    json_files = list(source_dir.glob('*.json'))[:100]
    
    # 분할
    train_files = json_files[:70]
    val_files = json_files[70:85]
    test_files = json_files[85:]
    
    splits_data = {
        'train': train_files,
        'val': val_files,
        'test': test_files
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
                dst_img = target_dir / split_name / 'images' / img_name
                
                if not src_img.exists():
                    continue
                
                shutil.copy2(src_img, dst_img)
                
                # 라벨 파일 생성
                label_file = target_dir / split_name / 'labels' / (json_file.stem + '.txt')
                
                labels = []
                for shape in labelme_data['shapes']:
                    if shape['shape_type'] == 'polygon' and len(shape['points']) >= 3:
                        points = np.array(shape['points'], dtype=np.float32)
                        
                        try:
                            # cv2.minAreaRect 사용
                            (cx, cy), (w, h), angle = cv2.minAreaRect(points)
                            
                            # 검증
                            if w <= 0 or h <= 0:
                                continue
                            if not (0 <= cx < img_width and 0 <= cy < img_height):
                                continue
                            
                            # 정규화
                            norm_cx = cx / img_width
                            norm_cy = cy / img_height  
                            norm_w = w / img_width
                            norm_h = h / img_height
                            
                            if not (0 < norm_cx < 1 and 0 < norm_cy < 1 and 0 < norm_w < 1 and 0 < norm_h < 1):
                                continue
                            
                            # 각도를 라디안으로 (정확한 범위로)
                            angle_rad = angle / 180.0 * np.pi
                            # 0 to pi/2 범위로 제한
                            angle_rad = max(0, min(np.pi/2, abs(angle_rad)))
                            
                            # 정확히 5개 값만
                            label = f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f} {angle_rad:.6f}"
                            labels.append(label)
                            total_converted += 1
                            
                        except Exception as e:
                            continue
                
                # 라벨이 있는 경우에만 파일 생성
                if labels:
                    with open(label_file, 'w') as f:
                        for label in labels:
                            f.write(label + '\n')
                
            except Exception as e:
                print(f"Error: {json_file}: {e}")
                continue
    
    print(f"Total converted: {total_converted}")
    
    # YAML 생성
    yaml_content = f"""path: {target_dir.absolute()}
train: train/images
val: val/images  
test: test/images

names:
  0: license_plate

nc: 1
"""
    
    yaml_file = target_dir / 'fixed_qbb.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"YAML saved: {yaml_file}")
    return target_dir, yaml_file


def main():
    dataset_dir, yaml_file = fix_qbb_labels()
    
    # 라벨 검증
    train_labels = list((dataset_dir / 'train' / 'labels').glob('*.txt'))
    print(f"\nVerifying labels...")
    
    for i, label_file in enumerate(train_labels[:3]):
        print(f"Sample {i+1}: {label_file.name}")
        with open(label_file, 'r') as f:
            content = f.read().strip()
            lines = content.split('\n')
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    print(f"  {len(parts)} columns: {line}")
    
    print(f"\nDataset ready: {dataset_dir}")
    print(f"YAML: {yaml_file}")


if __name__ == "__main__":
    main()