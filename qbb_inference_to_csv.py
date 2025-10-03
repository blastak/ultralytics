"""
QBB Inference to CSV
학습된 QBB 모델로 inference를 수행하고 결과를 CSV 형식으로 저장
step2_frontalization.py와 호환되는 형식 출력
"""

import csv
from pathlib import Path
import argparse

from ultralytics import YOLO


def inference_qbb_to_csv(
    model_path: str,
    image_folder: str,
    output_csv_folder: str = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7
):
    """
    QBB 모델로 inference를 수행하고 각 이미지마다 CSV 파일 생성

    Args:
        model_path: 학습된 QBB 모델 경로 (.pt 파일)
        image_folder: 입력 이미지 폴더 경로
        output_csv_folder: CSV 파일 저장 폴더 경로 (None이면 모델 경로 기준 자동 설정)
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS

    CSV 형식:
        class, x1, y1, x2, y2, x3, y3, x4, y4, confidence
    """
    # 모델 로드
    print(f"모델 로드 중: {model_path}")
    model = YOLO(model_path)

    # 출력 폴더 설정: None이면 모델 파일이 있는 상위 폴더에 inference_csv 생성
    if output_csv_folder is None:
        model_parent_dir = Path(model_path).parent.parent  # weights 폴더의 상위 폴더
        output_path = model_parent_dir / "inference_csv"
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

    # 각 이미지에 대해 inference 수행
    for idx, img_file in enumerate(image_files):
        # Inference 실행
        results = model.predict(
            source=str(img_file),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        # 결과 파싱
        result = results[0]

        # CSV 파일명 생성 (이미지 파일명.csv)
        csv_filename = output_path / f"{img_file.stem}.csv"

        # CSV 데이터 준비
        csv_rows = []

        if result.qbb is not None and len(result.qbb) > 0:
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

        # CSV 파일 저장
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)

        # 진행률 출력
        if (idx + 1) % 50 == 0:
            print(f"  진행률: {idx + 1}/{len(image_files)}, 총 {total_detections}개 검출")

    print(f"\nInference 완료!")
    print(f"  처리된 이미지: {len(image_files)}개")
    print(f"  검출된 객체: {total_detections}개")
    print(f"  CSV 저장 경로: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QBB 모델 inference 및 CSV 저장')
    parser.add_argument(
        '--model',
        type=str,
        default='runs/qbb/train0902_2/weights/best.pt',
        help='학습된 QBB 모델 경로'
    )
    parser.add_argument(
        '--images',
        type=str,
        default='./ultralytics/assets/good_all_obb1944/images/test',
        help='입력 이미지 폴더 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='CSV 파일 저장 폴더 경로 (지정하지 않으면 모델 경로 기준 자동 설정)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='IoU threshold for NMS'
    )

    args = parser.parse_args()

    inference_qbb_to_csv(
        model_path=args.model,
        image_folder=args.images,
        output_csv_folder=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
