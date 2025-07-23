

import json
import shutil
from pathlib import Path
import random
from tqdm import tqdm

from ultralytics.data.converter import convert_labelme_to_yolo_qbb

# 1. 설정
SOURCE_DATA_DIR = Path("/mnt/d/Dataset/Web_Crawling/20241014_from_안민찬/P1-1_single_row plate(2)_with_json")
OUTPUT_DATA_DIR = Path("/home/raykim_srv/projects/ultralytics/license_plate_qbb_dataset")
TRAIN_RATIO = 0.8

CLASS_MAP = {"P1-1": 0, "P1-2": 1, "P1-3": 2, "P1-4": 3, "P2": 4, "P3": 5, "P4": 6, "P6": 7}

# 2. 출력 디렉토리 생성
if OUTPUT_DATA_DIR.exists():
    shutil.rmtree(OUTPUT_DATA_DIR)
print(f"Creating output directory: {OUTPUT_DATA_DIR}")
img_train_dir = OUTPUT_DATA_DIR / "images/train"
img_val_dir = OUTPUT_DATA_DIR / "images/val"
lbl_train_dir = OUTPUT_DATA_DIR / "labels/train"
lbl_val_dir = OUTPUT_DATA_DIR / "labels/val"

img_train_dir.mkdir(parents=True, exist_ok=True)
img_val_dir.mkdir(parents=True, exist_ok=True)
lbl_train_dir.mkdir(parents=True, exist_ok=True)
lbl_val_dir.mkdir(parents=True, exist_ok=True)

# 3. 데이터 분할
print("Scanning and splitting dataset...")
json_files = list(SOURCE_DATA_DIR.glob("*.json"))
random.shuffle(json_files)
split_index = int(len(json_files) * TRAIN_RATIO)
train_files = json_files[:split_index]
val_files = json_files[split_index:]

print(f"Found {len(json_files)} total samples.")
print(f"Splitting into {len(train_files)} training samples and {len(val_files)} validation samples.")

# 4. 파일 복사 및 변환 준비
def process_files(file_list, img_dir, desc):
    for json_path in tqdm(file_list, desc=f"Processing {desc} set"):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # LabelMe JSON의 label에서 클래스 정보 추출
        for shape in data.get("shapes", []):
            shape["label"] = shape["label"].split('_')[0]

        # 수정된 JSON 데이터를 이미지와 동일한 디렉토리에 임시 저장
        temp_json_path = img_dir / json_path.name
        with open(temp_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        # 원본 이미지 복사
        image_path = SOURCE_DATA_DIR / data["imagePath"]
        if image_path.exists():
            shutil.copy(image_path, img_dir / image_path.name)
        else:
            print(f"Warning: Image not found for {json_path.name}, skipping.")


process_files(train_files, img_train_dir, "train")
process_files(val_files, img_val_dir, "val")

# 5. YOLO QBB 형식으로 변환
print("\nConverting annotations to YOLO QBB format...")
convert_labelme_to_yolo_qbb(source_dir=str(img_train_dir), class_map=CLASS_MAP)
convert_labelme_to_yolo_qbb(source_dir=str(img_val_dir), class_map=CLASS_MAP)

# 6. 변환된 .txt 파일들을 labels 폴더로 이동 및 임시 json 삭제
def finalize_labels(img_dir, lbl_dir):
    for txt_file in tqdm(list(img_dir.glob("*.txt")), desc=f"Finalizing labels in {lbl_dir}"):
        shutil.move(str(txt_file), str(lbl_dir / txt_file.name))
    for json_file in img_dir.glob("*.json"):
        json_file.unlink()

finalize_labels(img_train_dir, lbl_train_dir)
finalize_labels(img_val_dir, lbl_val_dir)

print("\nDataset preparation complete!")
print(f"Dataset ready at: {OUTPUT_DATA_DIR}")

