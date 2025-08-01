#!/usr/bin/env python3
"""
완전 새로운 통합 QBB 번호판 데이터셋 생성
소스: /workspace/DB/01_LicensePlate/55_WebPlatemania_1944/all/
1944개 LabelMe JSON 파일을 QBB 형식으로 변환
"""

import cv2
import numpy as np
from pathlib import Path
import json
import math
import shutil
import sys
from typing import List, Tuple
import random

sys.path.insert(0, '/workspace/repo/ultralytics')


def create_unified_qbb_dataset():
    """1944개 파일로 통합 QBB 데이터셋 생성"""
    
    source_dir = Path('/workspace/DB/01_LicensePlate/55_WebPlatemania_1944/all')
    target_dir = Path('datasets/unified_license_plates')
    
    print(f"🚀 Creating unified QBB dataset from {source_dir}")
    print(f"📁 Target directory: {target_dir}")
    
    # 기존 디렉토리 삭제 후 재생성
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    # 디렉토리 구조 생성
    splits = ['train', 'val', 'test']
    for split in splits:
        (target_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (target_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # JSON 파일들 수집 및 정렬 (일관된 분할을 위해)
    json_files = sorted(list(source_dir.glob('*.json')))
    total_files = len(json_files)
    
    print(f"📊 Found {total_files} JSON files")
    
    if total_files != 1944:
        print(f"⚠️  Warning: Expected 1944 files, found {total_files}")
    
    # 7:2:1 비율로 분할 (1361:389:194)
    train_end = int(total_files * 0.7)  # 1361
    val_end = int(total_files * 0.9)    # 1750 (389 val files)
    # test files: 194
    
    train_files = json_files[:train_end]
    val_files = json_files[train_end:val_end]
    test_files = json_files[val_end:]
    
    print(f"📋 Dataset split:")
    print(f"  Train: {len(train_files)} files ({len(train_files)/total_files*100:.1f}%)")
    print(f"  Val:   {len(val_files)} files ({len(val_files)/total_files*100:.1f}%)")
    print(f"  Test:  {len(test_files)} files ({len(test_files)/total_files*100:.1f}%)")
    
    splits_data = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    total_converted = 0
    conversion_stats = {
        'successful': 0,
        'failed_json': 0,
        'failed_image': 0,
        'failed_shape': 0,
        'failed_bbox': 0
    }
    
    for split_name, files in splits_data.items():
        print(f"\n🔄 Processing {split_name} split ({len(files)} files)...")
        
        for i, json_file in enumerate(files):
            if (i + 1) % 100 == 0:
                print(f"  Processing {i+1}/{len(files)} files...")
            
            try:
                # JSON 파일 읽기
                with open(json_file, 'r', encoding='utf-8') as f:
                    labelme_data = json.load(f)
                
                img_width = labelme_data['imageWidth']
                img_height = labelme_data['imageHeight']
                
                # 이미지 파일 확인 및 복사
                img_name = json_file.stem + '.jpg'
                src_img = source_dir / img_name
                dst_img = target_dir / split_name / 'images' / img_name
                
                if not src_img.exists():
                    conversion_stats['failed_image'] += 1
                    continue
                
                # 이미지 복사
                shutil.copy2(src_img, dst_img)
                
                # 라벨 파일 생성
                label_file = target_dir / split_name / 'labels' / (json_file.stem + '.txt')
                
                valid_objects = 0
                
                with open(label_file, 'w', encoding='utf-8') as f:
                    for shape in labelme_data['shapes']:
                        if shape['shape_type'] == 'polygon' and len(shape['points']) >= 4:
                            points = np.array(shape['points'], dtype=np.float32)
                            
                            try:
                                # OpenCV minAreaRect으로 OBB 계산
                                (cx, cy), (w, h), angle = cv2.minAreaRect(points)
                                
                                # 기본 유효성 검사
                                if w <= 0 or h <= 0:
                                    conversion_stats['failed_bbox'] += 1
                                    continue
                                
                                if not (0 <= cx < img_width and 0 <= cy < img_height):
                                    conversion_stats['failed_bbox'] += 1
                                    continue
                                
                                # 정규화 (0-1 범위)
                                norm_cx = cx / img_width
                                norm_cy = cy / img_height
                                norm_w = w / img_width
                                norm_h = h / img_height
                                
                                # 정규화된 좌표 범위 검사
                                if not (0 < norm_cx < 1 and 0 < norm_cy < 1):
                                    conversion_stats['failed_bbox'] += 1
                                    continue
                                if not (0 < norm_w < 1 and 0 < norm_h < 1):
                                    conversion_stats['failed_bbox'] += 1
                                    continue
                                
                                # 각도를 라디안으로 변환
                                angle_rad = math.radians(angle)
                                
                                # 클래스는 0 (번호판)
                                class_idx = 0
                                
                                # YOLO QBB 형식: class cx cy w h rotation
                                line = f"{class_idx} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f} {angle_rad:.6f}\n"
                                f.write(line)
                                
                                valid_objects += 1
                                total_converted += 1
                                
                            except Exception as e:
                                conversion_stats['failed_shape'] += 1
                                continue
                
                if valid_objects > 0:
                    conversion_stats['successful'] += 1
                else:
                    # 유효한 객체가 없는 경우 파일 삭제
                    dst_img.unlink(missing_ok=True)
                    label_file.unlink(missing_ok=True)
                    
            except Exception as e:
                conversion_stats['failed_json'] += 1
                continue
    
    print(f"\n✅ Conversion completed!")
    print(f"📊 Conversion Statistics:")
    print(f"  ✅ Successful files: {conversion_stats['successful']}")
    print(f"  ❌ Failed JSON read: {conversion_stats['failed_json']}")
    print(f"  ❌ Failed image copy: {conversion_stats['failed_image']}")
    print(f"  ❌ Failed shape processing: {conversion_stats['failed_shape']}")
    print(f"  ❌ Failed bbox validation: {conversion_stats['failed_bbox']}")
    print(f"  📦 Total objects converted: {total_converted}")
    
    # YAML 설정 파일 생성
    yaml_content = f"""# Unified QBB License Plate Dataset
# Source: WebPlatemania 1944 images
# Created: {Path.cwd()}

path: {target_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: license_plate

# Number of classes
nc: 1

# Dataset Statistics
# Total files processed: {total_files}
# Successful conversions: {conversion_stats['successful']}
# Train: {len(train_files)} images
# Val: {len(val_files)} images  
# Test: {len(test_files)} images
"""
    
    yaml_file = target_dir / 'unified_license_plates.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"📝 YAML configuration saved: {yaml_file}")
    
    # 최종 통계 출력
    print(f"\n📈 Final Dataset Statistics:")
    for split_name in splits:
        images_dir = target_dir / split_name / 'images'
        labels_dir = target_dir / split_name / 'labels'
        
        img_count = len(list(images_dir.glob('*.jpg')))
        label_count = len(list(labels_dir.glob('*.txt')))
        
        print(f"  {split_name.capitalize()}: {img_count} images, {label_count} labels")
    
    return target_dir, yaml_file, conversion_stats


def verify_dataset(dataset_dir: Path):
    """데이터셋 검증"""
    print(f"\n🔍 Verifying dataset: {dataset_dir}")
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = dataset_dir / split / 'images'
        labels_dir = dataset_dir / split / 'labels'
        
        images = list(images_dir.glob('*.jpg'))
        labels = list(labels_dir.glob('*.txt'))
        
        print(f"\n📁 {split} split:")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")
        
        # 몇 개 라벨 파일 샘플 확인
        if labels:
            sample_label = labels[0]
            with open(sample_label, 'r') as f:
                lines = f.readlines()
                print(f"  Sample label ({sample_label.name}): {len(lines)} objects")
                if lines:
                    print(f"    First line: {lines[0].strip()}")
    
    print(f"✅ Dataset verification completed!")


def main():
    print("🚀 Starting unified QBB dataset creation...")
    print("📊 Source: WebPlatemania 1944 license plate images")
    
    dataset_dir, yaml_file, stats = create_unified_qbb_dataset()
    
    # 데이터셋 검증
    verify_dataset(dataset_dir)
    
    print(f"\n🎉 Unified QBB dataset creation completed!")
    print(f"📁 Dataset directory: {dataset_dir}")
    print(f"📝 YAML file: {yaml_file}")
    print(f"📊 Success rate: {stats['successful']}/{stats['successful'] + stats['failed_json'] + stats['failed_image']} files")
    
    print(f"\n📋 Next steps:")
    print(f"1. Review dataset structure and statistics")
    print(f"2. Start QBB training with: python long_qbb_training.py")
    print(f"3. Monitor training progress and results")


if __name__ == "__main__":
    main()