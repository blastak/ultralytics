"""
Universal Inference to CSV
학습된 YOLO 모델(AABB/OBB/QBB)로 inference를 수행하고 결과를 CSV 형식으로 저장
step2_frontalization.py와 호환되는 형식 출력
"""

import csv
import time
from pathlib import Path
import argparse

from ultralytics import YOLO


def inference_to_csv(
    model_path: str,
    image_folder: str,
    output_csv_folder: str = None,
    dataset_name: str = None
):
    """
    YOLO 모델(AABB/OBB/QBB)로 inference를 수행하고 각 이미지마다 CSV 파일 생성

    Args:
        model_path: 학습된 YOLO 모델 경로 (.pt 파일)
        image_folder: 입력 이미지 폴더 경로
        output_csv_folder: CSV 파일 저장 폴더 경로 (None이면 모델 경로 기준 자동 설정)
        dataset_name: 데이터셋 이름 (None이면 기본값 'inference_csv' 사용)

    CSV 형식 (모두 10개 항목으로 통일):
        - AABB: class, x1, y1, x2, y1, x2, y2, x1, y2, confidence (좌상단부터 시계방향 4점)
        - OBB: class, x1, y1, x2, y2, x3, y3, x4, y4, confidence (회전된 4점 좌표)
        - QBB: class, x1, y1, x2, y2, x3, y3, x4, y4, confidence (자유 형태 4점 좌표)

    고정 설정:
        - Confidence threshold: 0.25
        - IoU threshold for NMS: 0.7
    """
    # 고정 threshold 값
    conf_threshold = 0.25
    iou_threshold = 0.7
    # 모델 로드
    print(f"모델 로드 중: {model_path}")
    model = YOLO(model_path)

    # 모델 타입 자동 감지
    model_task = model.task
    print(f"모델 타입: {model_task}")

    # 출력 폴더 설정: None이면 모델 파일이 있는 상위 폴더에 데이터셋 이름 포함한 폴더 생성
    if output_csv_folder is None:
        model_parent_dir = Path(model_path).parent.parent  # weights 폴더의 상위 폴더
        if dataset_name:
            folder_name = f"inference_csv_{dataset_name}"
        else:
            folder_name = "inference_csv"
        output_path = model_parent_dir / folder_name
    else:
        output_path = Path(output_csv_folder)

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"CSV 저장 경로: {output_path}")

    # 이미지 폴더에서 이미지 파일 목록 가져오기
    image_path = Path(image_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_path.glob(f'*{ext}')))

    image_files = sorted(image_files)
    print(f"\n총 {len(image_files)}개 이미지 처리 시작...")

    total_detections = 0
    timing_data = []  # 이미지당 처리 시간 저장

    # 각 이미지에 대해 inference 수행
    for idx, img_file in enumerate(image_files):
        # Inference 시작 시간 측정
        start_time = time.time()

        # Inference 실행
        results = model.predict(
            source=str(img_file),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        # Inference 종료 시간 측정
        end_time = time.time()
        inference_time = end_time - start_time
        timing_data.append({
            'image': img_file.name,
            'time_ms': inference_time * 1000  # 밀리초로 변환
        })

        # 결과 파싱
        result = results[0]

        # CSV 파일명 생성 (이미지 파일명.csv)
        csv_filename = output_path / f"{img_file.stem}.csv"

        # CSV 데이터 준비
        csv_rows = []

        # QBB 결과 처리
        if hasattr(result, 'qbb') and result.qbb is not None and len(result.qbb) > 0:
            # QBB 결과가 있는 경우
            # result.qbb.xyxyxyxy: (N, 4, 2) 형태 - 좌상단부터 시계방향
            xyxyxyxy = result.qbb.xyxyxyxy  # (N, 4, 2)
            confs = result.qbb.conf  # (N,)
            clss = result.qbb.cls  # (N,)

            for i in range(len(result.qbb)):
                # 클래스 이름
                class_name = result.names[int(clss[i].item())]

                # 8개 좌표 (절대 좌표) - (4, 2) → flatten to [x1, y1, x2, y2, x3, y3, x4, y4]
                coords = xyxyxyxy[i].reshape(-1).tolist()

                # Confidence
                conf = float(confs[i].item())

                # CSV 행: class, x1, y1, x2, y2, x3, y3, x4, y4, confidence
                row = [class_name] + coords + [conf]
                csv_rows.append(row)
                total_detections += 1

        # OBB 결과 처리
        elif hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
            # OBB 결과가 있는 경우
            xyxyxyxy = result.obb.xyxyxyxy  # (N, 4, 2)
            confs = result.obb.conf  # (N,)
            clss = result.obb.cls  # (N,)

            for i in range(len(result.obb)):
                # 클래스 이름
                class_name = result.names[int(clss[i].item())]

                # 8개 좌표 (절대 좌표)
                coords = xyxyxyxy[i].reshape(-1).tolist()

                # Confidence
                conf = float(confs[i].item())

                # CSV 행: class, x1, y1, x2, y2, x3, y3, x4, y4, confidence
                row = [class_name] + coords + [conf]
                csv_rows.append(row)
                total_detections += 1

        # AABB (일반 Detection) 결과 처리
        elif result.boxes is not None and len(result.boxes) > 0:
            # AABB 결과가 있는 경우
            xyxy = result.boxes.xyxy  # (N, 4)
            confs = result.boxes.conf  # (N,)
            clss = result.boxes.cls  # (N,)

            for i in range(len(result.boxes)):
                # 클래스 이름
                class_name = result.names[int(clss[i].item())]

                # 4개 좌표 (x1, y1, x2, y2)를 xyxyxyxy 형식으로 변환
                # 좌상단부터 시계방향: (x1,y1), (x2,y1), (x2,y2), (x1,y2)
                x1, y1, x2, y2 = xyxy[i].tolist()
                coords = [x1, y1, x2, y1, x2, y2, x1, y2]

                # Confidence
                conf = float(confs[i].item())

                # CSV 행: class, x1, y1, x2, y1, x2, y2, x1, y2, confidence (10개 항목)
                row = [class_name] + coords + [conf]
                csv_rows.append(row)
                total_detections += 1

        # CSV 파일 저장
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)

        # 진행률 출력
        if (idx + 1) % 50 == 0:
            print(f"  진행률: {idx + 1}/{len(image_files)}, 총 {total_detections}개 검출")

    # 타이밍 통계 계산
    if timing_data:
        times_ms = [t['time_ms'] for t in timing_data]
        avg_time = sum(times_ms) / len(times_ms)
        min_time = min(times_ms)
        max_time = max(times_ms)
        total_time = sum(times_ms) / 1000  # 초 단위

        # 타이밍 정보를 TXT 파일로 저장
        timing_txt = output_path / "inference_timing.txt"
        with open(timing_txt, 'w') as f:
            f.write("image,inference_time_ms\n")
            for item in timing_data:
                f.write(f"{item['image']},{item['time_ms']:.2f}\n")

    print(f"\nInference 완료!")
    print(f"  처리된 이미지: {len(image_files)}개")
    print(f"  검출된 객체: {total_detections}개")
    print(f"  CSV 저장 경로: {output_path}")

    if timing_data:
        print(f"\n⏱️  처리 시간 통계:")
        print(f"  평균: {avg_time:.2f} ms/image")
        print(f"  최소: {min_time:.2f} ms")
        print(f"  최대: {max_time:.2f} ms")
        print(f"  총 시간: {total_time:.2f} 초")
        print(f"  타이밍 정보 저장: {timing_txt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO 모델 inference 및 CSV 저장 (AABB/OBB/QBB 지원)')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='학습된 YOLO 모델 경로 (.pt 파일)'
    )
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='입력 이미지 폴더 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='CSV 파일 저장 폴더 경로 (지정하지 않으면 모델 경로 기준 자동 설정)'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default=None,
        help='데이터셋 이름 (폴더명에 포함됨, 예: ccpd_over60_xyxyxyxy)'
    )

    args = parser.parse_args()

    inference_to_csv(
        model_path=args.model,
        image_folder=args.images,
        output_csv_folder=args.output,
        dataset_name=args.dataset_name
    )
