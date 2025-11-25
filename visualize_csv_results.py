#!/usr/bin/env python3
"""
CSV 결과를 이미지에 시각화
각 CSV 파일을 읽어서 이미지에 경계 상자와 confidence를 그려서 저장
"""

import csv
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def visualize_csv_results(
    csv_folder: str,
    image_folder: str,
    output_prefix: str = "box_",
    line_thickness: int = 2,
    font_scale: float = 0.6,
    font_thickness: int = 2
):
    """
    CSV 결과를 이미지에 시각화하여 저장

    Args:
        csv_folder: CSV 파일들이 있는 폴더 경로
        image_folder: 원본 이미지 폴더 경로
        output_prefix: 출력 파일명 prefix (기본값: "box_")
        line_thickness: 경계 상자 선 두께
        font_scale: 텍스트 크기
        font_thickness: 텍스트 두께
    """
    csv_path = Path(csv_folder)
    image_path = Path(image_folder)

    # CSV 파일 목록 가져오기
    csv_files = sorted(list(csv_path.glob("*.csv")))

    if len(csv_files) == 0:
        print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_folder}")
        return

    print(f"\n{'='*80}")
    print(f"📂 CSV 폴더: {csv_folder}")
    print(f"🖼️  이미지 폴더: {image_folder}")
    print(f"💾 출력 경로: {csv_folder} (prefix: {output_prefix})")
    print(f"📊 총 CSV 파일 수: {len(csv_files)}")
    print(f"{'='*80}\n")

    total_detections = 0
    processed_images = 0
    skipped_images = 0

    # 각 CSV 파일에 대해 처리
    for csv_file in tqdm(csv_files, desc="시각화 중"):
        # 이미지 파일명 (CSV 파일명에서 .csv 제거)
        image_name = csv_file.stem

        # 이미지 파일 찾기 (여러 확장자 시도)
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = image_path / f"{image_name}{ext}"
            if candidate.exists():
                image_file = candidate
                break

        if image_file is None:
            skipped_images += 1
            continue

        # 이미지 로드
        img = cv2.imread(str(image_file))
        if img is None:
            skipped_images += 1
            continue

        # CSV 파일 읽기
        detections = []
        try:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 10:  # class + 8 coords + confidence
                        detections.append(row)
        except Exception as e:
            print(f"⚠️  CSV 읽기 실패: {csv_file.name} - {e}")
            continue

        # 검출 결과 그리기
        for detection in detections:
            try:
                class_name = detection[0]
                coords = [float(x) for x in detection[1:9]]  # 8개 좌표
                confidence = float(detection[9])

                # 4개 점 좌표 추출
                points = np.array([
                    [coords[0], coords[1]],  # (x1, y1)
                    [coords[2], coords[3]],  # (x2, y2)
                    [coords[4], coords[5]],  # (x3, y3)
                    [coords[6], coords[7]]   # (x4, y4)
                ], dtype=np.int32)

                # 경계 상자 그리기 (4개 선분)
                color = (0, 255, 0)  # 초록색
                for i in range(4):
                    pt1 = tuple(points[i])
                    pt2 = tuple(points[(i + 1) % 4])
                    cv2.line(img, pt1, pt2, color, line_thickness)

                total_detections += 1

            except Exception as e:
                print(f"⚠️  검출 결과 그리기 실패: {csv_file.name} - {e}")
                continue

        # 출력 파일명 생성
        output_filename = csv_path / f"{output_prefix}{image_name}.jpg"

        # 이미지 저장
        cv2.imwrite(str(output_filename), img)
        processed_images += 1

    print(f"\n{'='*80}")
    print(f"✅ 시각화 완료!")
    print(f"  처리된 이미지: {processed_images}개")
    print(f"  건너뛴 이미지: {skipped_images}개")
    print(f"  총 검출 객체: {total_detections}개")
    print(f"  출력 경로: {csv_folder}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    import sys

    # 3개의 CSV 폴더 경로
    csv_folders = [
        "/workspace/repo/ultralytics/runs/qbb/ccpd_over60_max_b64_gpu8/inference_csv",
        "/workspace/repo/ultralytics/runs/obb/ccpd_over60_yolov8n_obb2/inference_csv",
        "/workspace/repo/ultralytics/runs/detect/ccpd_over60_yolov8n_aabb/inference_csv"
    ]

    # 이미지 폴더 경로
    image_folder = "/workspace/repo/ultralytics/ultralytics/assets/ccpd_over60_xyxyxyxy/images/test"

    # 명령행 인자로 경로를 받을 수도 있음
    if len(sys.argv) >= 3:
        csv_folders = [sys.argv[1]]
        image_folder = sys.argv[2]

    # 각 CSV 폴더에 대해 시각화 수행
    for csv_folder in csv_folders:
        if Path(csv_folder).exists():
            visualize_csv_results(csv_folder, image_folder)
        else:
            print(f"⚠️  폴더가 존재하지 않습니다: {csv_folder}")
