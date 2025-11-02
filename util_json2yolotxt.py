"""
JSON to YOLO txt converter
LabelMe JSON 형식을 YOLO 학습용 txt로 변환 (BB/OBB 지원)
Train/Val/Test를 70/20/10으로 분할
멀티클래스 지원 (폴더명에서 클래스 추출)

## 사용법

### 1. 단일 클래스 모드 (기존 방식)
하나의 폴더에 있는 모든 JSON을 단일 클래스로 변환

```bash
python util_json2yolotxt.py \
    --json_folder /path/to/json/folder \
    --output ultralytics/assets/my_dataset \
    --format obb \
    --class_id 0
```

**옵션 설명:**
- `--json_folder`: JSON 파일이 있는 폴더 (하위 폴더 포함 재귀 검색)
- `--output`: 출력 폴더 경로
- `--format`: 'bbox' (xywh) 또는 'obb' (xyxyxyxy)
- `--class_id`: 클래스 ID (기본값: 0)

### 2. 멀티클래스 모드 (확장 기능)
여러 폴더를 각각 다른 클래스로 변환 (폴더명에서 클래스 이름 추출)

```bash
python util_json2yolotxt.py \
    --json_folder /workspace/DB/01_LicensePlate/55_WebPlatemania_1944 \
    --output ultralytics/assets/kor_all_multicls_xyxyxyxy \
    --format obb \
    --multiclass \
    --folder_pattern "GoodMatches*" \
    --class_prefix "GoodMatches_"
```

**필수 옵션 (멀티클래스 모드):**
- `--multiclass`: 멀티클래스 모드 활성화
- `--folder_pattern`: 처리할 폴더 패턴 (예: "GoodMatches*")
- `--class_prefix`: 클래스 이름 추출용 접두사 (예: "GoodMatches_")

**동작 방식:**
1. `json_folder` 아래에서 `folder_pattern`과 매칭되는 폴더들을 찾음
   예: GoodMatches_P1-1, GoodMatches_P2, GoodMatches_P3 등

2. 각 폴더명에서 `class_prefix` 이후의 문자열을 클래스 이름으로 추출
   예: "GoodMatches_P1-1" → 클래스 이름 "P1-1"

3. 클래스 이름을 알파벳순으로 정렬하여 클래스 ID 자동 할당
   예: P1-1 → 0, P1-2 → 1, P2 → 2, P3 → 3, ...

4. 모든 폴더의 JSON을 랜덤 셔플 후 train/val/test로 분할

5. 각 JSON을 해당 클래스 ID로 라벨링하여 변환

**출력 예시:**
```
클래스 매핑:
  0: P1-1
  1: P1-2
  2: P1-3
  ...

클래스별 분포:
Class      Train      Val        Test       Total
--------------------------------------------------
P1-1       382        124        60         566
P1-2       54         15         4          73
...
```

### 3. 공통 옵션

```bash
--train_ratio 0.7    # Train 비율 (기본값: 0.7)
--val_ratio 0.2      # Val 비율 (기본값: 0.2)
--test_ratio 0.1     # Test 비율 (기본값: 0.1)
--seed 42            # Random seed (기본값: 42)
```

### 4. 실제 사용 예시

**예시 1: 단일 클래스 xywh 데이터셋**
```bash
python util_json2yolotxt.py \
    --json_folder /data/license_plates \
    --output ultralytics/assets/license_plate_xywh \
    --format bbox \
    --class_id 0
```

**예시 2: 멀티클래스 xyxyxyxy 데이터셋**
```bash
python util_json2yolotxt.py \
    --json_folder /workspace/DB/01_LicensePlate/55_WebPlatemania_1944 \
    --output ultralytics/assets/kor_all_multicls_xyxyxyxy \
    --format obb \
    --multiclass \
    --folder_pattern "GoodMatches*" \
    --class_prefix "GoodMatches_" \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --test_ratio 0.1
```

### 5. 출력 폴더 구조

```
output_folder/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 6. 주의사항

- 멀티클래스 모드에서는 `--class_id`가 무시됩니다
- JSON 파일과 같은 디렉토리에 이미지 파일이 있어야 합니다
- 지원 이미지 확장자: .jpg, .jpeg, .png, .JPG, .JPEG, .PNG
- 4점 polygon만 처리됩니다 (다른 shape_type이나 점 개수는 스킵)
- 비율의 합은 반드시 1.0이어야 합니다

"""

import json
import random
from pathlib import Path
import argparse
import shutil
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


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


def extract_class_from_folder(folder_path: Path, class_prefix: str) -> str:
    """
    폴더명에서 클래스 이름 추출

    Args:
        folder_path: 폴더 경로
        class_prefix: 클래스 접두사 (예: "GoodMatches_")

    Returns:
        클래스 이름 (예: "P1-1", "P2" 등)
    """
    folder_name = folder_path.name
    if class_prefix in folder_name:
        return folder_name.split(class_prefix)[1]
    return folder_name


def extract_class_from_label(label: str, label_separator: str = "_") -> str:
    """
    JSON label 필드에서 클래스 이름 추출

    Args:
        label: label 문자열 (예: "P1-1_53마9030")
        label_separator: 클래스와 나머지를 구분하는 구분자 (기본값: "_")

    Returns:
        클래스 이름 (예: "P1-1")
    """
    return label.split(label_separator)[0]


def build_class_mapping(json_folders: List[Path], class_prefix: str) -> Dict[str, int]:
    """
    폴더 목록에서 클래스 매핑 생성

    Args:
        json_folders: JSON 파일이 있는 폴더 리스트
        class_prefix: 클래스 접두사

    Returns:
        {class_name: class_id} 딕셔너리
    """
    class_names = sorted(set([
        extract_class_from_folder(folder, class_prefix)
        for folder in json_folders
    ]))

    class_mapping = {name: idx for idx, name in enumerate(class_names)}

    print(f"\n클래스 매핑:")
    for name, idx in class_mapping.items():
        print(f"  {idx}: {name}")

    return class_mapping


def build_class_mapping_from_labels(json_files: List[Path], label_separator: str = "_") -> Dict[str, int]:
    """
    JSON 파일들의 label 필드에서 클래스 매핑 생성

    Args:
        json_files: JSON 파일 리스트
        label_separator: 클래스와 나머지를 구분하는 구분자

    Returns:
        {class_name: class_id} 딕셔너리
    """
    class_names_set = set()

    for json_file in json_files:
        try:
            data = read_json(json_file)
            shapes = data.get('shapes', [])
            if len(shapes) > 0 and 'label' in shapes[0]:
                label = shapes[0]['label']
                class_name = extract_class_from_label(label, label_separator)
                class_names_set.add(class_name)
        except Exception:
            continue

    class_names = sorted(class_names_set)
    class_mapping = {name: idx for idx, name in enumerate(class_names)}

    print(f"\n클래스 매핑:")
    for name, idx in class_mapping.items():
        print(f"  {idx}: {name}")

    return class_mapping


def convert_json_to_yolo(
    json_folder: Path,
    output_folder: Path,
    format_type: str = "bbox",
    class_id: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    multiclass: bool = False,
    folder_pattern: Optional[str] = None,
    class_prefix: Optional[str] = None,
    label_based_class: bool = False,
    label_separator: str = "_",
    fraction: float = 1.0
):
    """
    JSON 폴더를 YOLO txt로 변환 및 train/val/test 분할

    Args:
        json_folder: JSON 파일이 있는 폴더 (하위 폴더 포함 검색)
        output_folder: 출력 폴더 (ultralytics/assets/dataset_name)
        format_type: "bbox" (xywh) 또는 "obb" (xyxyxyxy)
        class_id: 클래스 ID (단일 클래스인 경우 0, multiclass=False일 때만 사용)
        train_ratio, val_ratio, test_ratio: 분할 비율
        multiclass: 멀티클래스 모드 활성화 (폴더별로 클래스 할당)
        folder_pattern: 멀티클래스 모드에서 처리할 폴더 패턴 (예: "GoodMatches*")
        class_prefix: 멀티클래스 모드에서 클래스 이름 추출용 접두사 (예: "GoodMatches_")
        label_based_class: JSON label 필드에서 클래스 추출 (multiclass=True일 때만 유효)
        label_separator: label 기반 클래스 추출 시 구분자 (기본값: "_")
        fraction: 사용할 데이터의 비율 (0.0 ~ 1.0, 기본값: 1.0)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율의 합은 1이어야 합니다"

    if multiclass:
        if label_based_class:
            # Label 기반 멀티클래스 모드
            print("Label 기반 멀티클래스 모드")

            if folder_pattern:
                # 폴더 패턴이 지정된 경우: 여러 폴더에서 JSON 수집
                json_folders = sorted(json_folder.glob(folder_pattern))
                if len(json_folders) == 0:
                    print(f"오류: {json_folder}/{folder_pattern} 패턴과 일치하는 폴더가 없습니다")
                    return

                print(f"발견된 폴더: {len(json_folders)}개")
                for folder in json_folders:
                    print(f"  - {folder.name}")

                all_json_files = []
                for folder in json_folders:
                    json_files_in_folder = list(folder.glob("*.json"))
                    all_json_files.extend(json_files_in_folder)
                    print(f"{folder.name}: {len(json_files_in_folder)}개 JSON 파일")

                json_files = all_json_files
            else:
                # 폴더 패턴 없음: 단일 폴더에서 재귀 검색
                json_files = list(json_folder.rglob("*.json"))

            print(f"\n총 JSON 파일: {len(json_files)}개")

            # Label에서 클래스 매핑 생성
            class_mapping = build_class_mapping_from_labels(json_files, label_separator)

            # 각 JSON 파일의 클래스 ID 미리 추출
            file_class_map = {}
            for json_file in json_files:
                try:
                    data = read_json(json_file)
                    shapes = data.get('shapes', [])
                    if len(shapes) > 0 and 'label' in shapes[0]:
                        label = shapes[0]['label']
                        class_name = extract_class_from_label(label, label_separator)
                        file_class_map[json_file] = class_mapping[class_name]
                except Exception:
                    continue

            folder_class_map = file_class_map
        else:
            # 폴더 기반 멀티클래스 모드 (기존 방식)
            if folder_pattern is None:
                raise ValueError("multiclass=True이고 label_based_class=False일 때 folder_pattern을 지정해야 합니다")
            if class_prefix is None:
                raise ValueError("multiclass=True이고 label_based_class=False일 때 class_prefix를 지정해야 합니다")

            json_folders = sorted(json_folder.glob(folder_pattern))
            if len(json_folders) == 0:
                print(f"오류: {json_folder}/{folder_pattern} 패턴과 일치하는 폴더가 없습니다")
                return

            print(f"발견된 폴더: {len(json_folders)}개")
            for folder in json_folders:
                print(f"  - {folder.name}")

            # 클래스 매핑 생성
            class_mapping = build_class_mapping(json_folders, class_prefix)

            # 각 폴더에서 JSON 파일 수집
            all_json_files = []
            folder_class_map = {}  # {json_file: class_id}

            for folder in json_folders:
                class_name = extract_class_from_folder(folder, class_prefix)
                cls_id = class_mapping[class_name]

                json_files = list(folder.glob("*.json"))
                print(f"\n{folder.name}: {len(json_files)}개 JSON 파일 발견 (클래스={class_name}, ID={cls_id})")

                for json_file in json_files:
                    folder_class_map[json_file] = cls_id

                all_json_files.extend(json_files)

            json_files = all_json_files
            print(f"\n총 JSON 파일: {len(json_files)}개")
    else:
        # 단일 클래스 모드: 기존 방식
        json_files = list(json_folder.rglob("*.json"))
        folder_class_map = None
        print(f"발견된 JSON 파일: {len(json_files)}개")

    if len(json_files) == 0:
        print(f"경고: {json_folder}에 JSON 파일이 없습니다")
        return

    # 랜덤 셔플
    random.shuffle(json_files)

    # Fraction 적용 (데이터 샘플링)
    if fraction < 1.0:
        n_sample = int(len(json_files) * fraction)
        json_files = json_files[:n_sample]
        print(f"Fraction {fraction:.1%} 적용: {n_sample}개 파일 사용")

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
    class_counts = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0}) if multiclass else None

    for split_name, file_list in splits.items():
        print(f"\n{split_name.upper()} 처리 중...")
        converted = 0
        skipped = 0

        for json_file in file_list:
            try:
                # 클래스 ID 결정
                if multiclass:
                    current_class_id = folder_class_map[json_file]
                    class_name = [k for k, v in class_mapping.items() if v == current_class_id][0]
                else:
                    current_class_id = class_id

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
                    txt_lines.append(f"{current_class_id} {coords_str}")

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
                if multiclass:
                    class_counts[class_name][split_name] += 1

            except Exception as e:
                print(f"  오류: {json_file.name} - {e}")
                skipped += 1

        print(f"  변환: {converted}개, 스킵: {skipped}개")
        total_converted += converted
        total_skipped += skipped

    # 멀티클래스 통계 출력
    if multiclass:
        print(f"\n클래스별 분포:")
        print(f"{'Class':<10} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
        print("-" * 50)
        for cls_name in sorted(class_counts.keys()):
            counts = class_counts[cls_name]
            total = counts['train'] + counts['val'] + counts['test']
            print(f"{cls_name:<10} {counts['train']:<10} {counts['val']:<10} {counts['test']:<10} {total:<10}")

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
        help='클래스 ID (단일 클래스인 경우 0, multiclass 모드에서는 무시됨)'
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
    parser.add_argument(
        '--multiclass',
        action='store_true',
        help='멀티클래스 모드 활성화 (폴더별로 클래스 할당)'
    )
    parser.add_argument(
        '--folder_pattern',
        type=str,
        default=None,
        help='멀티클래스 모드에서 처리할 폴더 패턴 (예: "GoodMatches*")'
    )
    parser.add_argument(
        '--class_prefix',
        type=str,
        default=None,
        help='멀티클래스 모드에서 클래스 이름 추출용 접두사 (예: "GoodMatches_")'
    )
    parser.add_argument(
        '--label_based_class',
        action='store_true',
        help='JSON label 필드에서 클래스 추출 (multiclass=True일 때만 유효)'
    )
    parser.add_argument(
        '--label_separator',
        type=str,
        default='_',
        help='label 기반 클래스 추출 시 구분자 (기본값: "_")'
    )
    parser.add_argument(
        '--fraction',
        type=float,
        default=1.0,
        help='사용할 데이터의 비율 (0.0 ~ 1.0, 기본값: 1.0)'
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
        test_ratio=args.test_ratio,
        multiclass=args.multiclass,
        folder_pattern=args.folder_pattern,
        class_prefix=args.class_prefix,
        label_based_class=args.label_based_class,
        label_separator=args.label_separator,
        fraction=args.fraction
    )
