"""
JSON to YOLO txt converter
LabelMe JSON 형식을 YOLO 학습용 txt로 변환 (BB/OBB 지원)
Train/Val/Test를 70/20/10으로 분할
"""

import json
import random
from pathlib import Path
import argparse
import shutil
from typing import List, Tuple


def read_json(json_path: Path) -> dict:
    """JSON 파일 읽기"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_image_size(json_data: dict) -> Tuple[int, int]:
    """이미지 크기 추출 (width, height)"""
    return json_data.get('imageWidth', 0), json_data.get('imageHeight', 0)


def points_to_bbox(points: List[List[float]], img_w: int, img_h: int) -> str:
    """
    4점 좌표를 YOLO bbox 형식으로 변환 (클래스 + normalized xywh)

    Args:
        points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        img_w, img_h: 이미지 크기

    Returns:
        "class x_center y_center width height" (normalized)
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Center, width, height
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    # Normalize
    x_center /= img_w
    y_center /= img_h
    width /= img_w
    height /= img_h

    return f"{x_center} {y_center} {width} {height}"


def points_to_obb(points: List[List[float]], img_w: int, img_h: int) -> str:
    """
    4점 좌표를 YOLO OBB 형식으로 변환 (클래스 + normalized xyxyxyxy)

    Args:
        points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        img_w, img_h: 이미지 크기

    Returns:
        "class x1 y1 x2 y2 x3 y3 x4 y4" (normalized)
    """
    normalized_points = []
    for x, y in points:
        normalized_points.extend([x / img_w, y / img_h])

    return " ".join(f"{p:.6f}" for p in normalized_points)


def convert_json_to_yolo(
    json_folder: Path,
    output_folder: Path,
    format_type: str = "bbox",
    class_id: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1
):
    """
    JSON 폴더를 YOLO txt로 변환 및 train/val/test 분할

    Args:
        json_folder: JSON 파일이 있는 폴더 (하위 폴더 포함 검색)
        output_folder: 출력 폴더 (ultralytics/assets/dataset_name)
        format_type: "bbox" (xywh) 또는 "obb" (xyxyxyxy)
        class_id: 클래스 ID (단일 클래스인 경우 0)
        train_ratio, val_ratio, test_ratio: 분할 비율
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율의 합은 1이어야 합니다"

    # JSON 파일 찾기 (하위 폴더 포함)
    json_files = list(json_folder.rglob("*.json"))
    print(f"발견된 JSON 파일: {len(json_files)}개")

    if len(json_files) == 0:
        print(f"경고: {json_folder}에 JSON 파일이 없습니다")
        return

    # 랜덤 셔플
    random.shuffle(json_files)

    # Train/Val/Test 분할
    n_total = len(json_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = json_files[:n_train]
    val_files = json_files[n_train:n_train + n_val]
    test_files = json_files[n_train + n_val:]

    print(f"분할: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # 출력 폴더 구조 생성
    for split in ['train', 'val', 'test']:
        (output_folder / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_folder / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 변환 함수 선택
    convert_func = points_to_obb if format_type == "obb" else points_to_bbox

    # 각 split별 처리
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    total_converted = 0
    total_skipped = 0

    for split_name, file_list in splits.items():
        print(f"\n{split_name.upper()} 처리 중...")
        converted = 0
        skipped = 0

        for json_file in file_list:
            try:
                # JSON 읽기
                json_data = read_json(json_file)
                img_w, img_h = get_image_size(json_data)

                if img_w == 0 or img_h == 0:
                    print(f"  경고: {json_file.name} - 이미지 크기 정보 없음")
                    skipped += 1
                    continue

                # 이미지 파일 찾기 (JSON과 같은 디렉토리에서)
                image_name = json_data.get('imagePath', json_file.stem)
                if not any(image_name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']):
                    # 확장자가 없으면 추가
                    image_exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                    image_file = None
                    for ext in image_exts:
                        potential_file = json_file.parent / (json_file.stem + ext)
                        if potential_file.exists():
                            image_file = potential_file
                            break
                else:
                    image_file = json_file.parent / image_name

                if image_file is None or not image_file.exists():
                    print(f"  경고: {json_file.name} - 이미지 파일 없음")
                    skipped += 1
                    continue

                # 라벨 변환
                txt_lines = []
                shapes = json_data.get('shapes', [])

                for shape in shapes:
                    if shape['shape_type'] != 'polygon':
                        continue

                    points = shape['points']
                    if len(points) != 4:
                        continue

                    # 좌표 변환
                    coords_str = convert_func(points, img_w, img_h)
                    txt_lines.append(f"{class_id} {coords_str}")

                if len(txt_lines) == 0:
                    skipped += 1
                    continue

                # 이미지 복사
                dst_image = output_folder / 'images' / split_name / image_file.name
                shutil.copy2(image_file, dst_image)

                # 라벨 저장
                txt_file = output_folder / 'labels' / split_name / f"{image_file.stem}.txt"
                with open(txt_file, 'w') as f:
                    f.write('\n'.join(txt_lines))

                converted += 1

            except Exception as e:
                print(f"  오류: {json_file.name} - {e}")
                skipped += 1

        print(f"  변환: {converted}개, 스킵: {skipped}개")
        total_converted += converted
        total_skipped += skipped

    print(f"\n총 변환: {total_converted}개, 총 스킵: {total_skipped}개")
    print(f"출력 폴더: {output_folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JSON to YOLO txt converter')
    parser.add_argument(
        '--json_folder',
        type=str,
        required=True,
        help='JSON 파일이 있는 폴더 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='출력 폴더 경로 (예: ultralytics/assets/dataset_name)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['bbox', 'obb'],
        default='obb',
        help='출력 형식: bbox (xywh) 또는 obb (xyxyxyxy)'
    )
    parser.add_argument(
        '--class_id',
        type=int,
        default=0,
        help='클래스 ID (단일 클래스인 경우 0)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Train 비율'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='Val 비율'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.1,
        help='Test 비율'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Random seed 설정
    random.seed(args.seed)

    convert_json_to_yolo(
        json_folder=Path(args.json_folder),
        output_folder=Path(args.output),
        format_type=args.format,
        class_id=args.class_id,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
