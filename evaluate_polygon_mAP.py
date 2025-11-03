#!/usr/bin/env python3
"""
Polygon IoU 기반 mAP 평가
AABB, OBB, QBB 세 모델의 성능을 비교
"""

import csv
import numpy as np
import argparse
from pathlib import Path
from shapely.geometry import Polygon
from shapely.validation import make_valid
from tqdm import tqdm


def load_ground_truth(label_dir):
    """
    Ground truth 레이블 로드 (normalized xyxyxyxy format)
    """
    gt_dict = {}
    label_path = Path(label_dir)
    
    for label_file in label_path.glob("*.txt"):
        image_name = label_file.stem
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 9:  # class + 8 coords
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:9]]
                boxes.append({
                    'class': class_id,
                    'coords': coords,  # normalized
                    'matched': False
                })
        
        gt_dict[image_name] = boxes
    
    return gt_dict


def load_predictions_from_csv(csv_dir):
    """
    CSV 파일에서 예측 결과 로드
    """
    pred_dict = {}
    csv_path = Path(csv_dir)
    
    for csv_file in csv_path.glob("*.csv"):
        image_name = csv_file.stem
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            detections = []
            
            for row in reader:
                if len(row) >= 10:  # class + 8 coords + confidence
                    class_name = row[0]
                    coords = [float(x) for x in row[1:9]]
                    confidence = float(row[9])
                    
                    detections.append({
                        'class': 0,  # license_plate
                        'coords': coords,  # absolute coords
                        'confidence': confidence
                    })
        
        pred_dict[image_name] = detections
    
    return pred_dict


def denormalize_coords(coords, img_width, img_height):
    """
    Normalized 좌표를 absolute 좌표로 변환
    """
    return [
        coords[0] * img_width, coords[1] * img_height,
        coords[2] * img_width, coords[3] * img_height,
        coords[4] * img_width, coords[5] * img_height,
        coords[6] * img_width, coords[7] * img_height,
    ]


def coords_to_polygon(coords):
    """
    8개 좌표를 Shapely Polygon으로 변환
    coords: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    points = [
        (coords[0], coords[1]),
        (coords[2], coords[3]),
        (coords[4], coords[5]),
        (coords[6], coords[7])
    ]
    poly = Polygon(points)
    
    # Invalid polygon 처리
    if not poly.is_valid:
        poly = make_valid(poly)
    
    return poly


def calculate_polygon_iou(poly1, poly2):
    """
    두 polygon의 IoU 계산
    """
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        if union == 0:
            return 0.0
        
        return intersection / union
    except:
        return 0.0


def calculate_ap(precisions, recalls):
    """
    11-point interpolation으로 AP 계산
    """
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def evaluate_map(gt_dict, pred_dict, iou_threshold=0.5, img_width=720, img_height=1160):
    """
    mAP 계산
    """
    # 모든 예측을 confidence 순으로 정렬
    all_predictions = []
    
    for image_name, detections in pred_dict.items():
        for det in detections:
            all_predictions.append({
                'image': image_name,
                'confidence': det['confidence'],
                'coords': det['coords']
            })
    
    # Confidence 내림차순 정렬
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    # TP, FP 계산
    tp = np.zeros(len(all_predictions))
    fp = np.zeros(len(all_predictions))
    
    # GT 총 개수
    total_gt = sum(len(boxes) for boxes in gt_dict.values())
    
    for idx, pred in enumerate(tqdm(all_predictions, desc="Calculating IoU")):
        image_name = pred['image']
        pred_poly = coords_to_polygon(pred['coords'])
        
        if image_name not in gt_dict:
            fp[idx] = 1
            continue
        
        gt_boxes = gt_dict[image_name]
        
        max_iou = 0.0
        max_gt_idx = -1
        
        # 모든 GT와 비교하여 최대 IoU 찾기
        for gt_idx, gt_box in enumerate(gt_boxes):
            # Denormalize GT coords
            gt_coords = denormalize_coords(gt_box['coords'], img_width, img_height)
            gt_poly = coords_to_polygon(gt_coords)
            
            iou = calculate_polygon_iou(pred_poly, gt_poly)
            
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        # IoU threshold 이상이고 아직 매칭되지 않은 GT
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            if not gt_boxes[max_gt_idx]['matched']:
                tp[idx] = 1
                gt_boxes[max_gt_idx]['matched'] = True
            else:
                fp[idx] = 1  # 이미 매칭된 GT
        else:
            fp[idx] = 1
    
    # Cumulative sum
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Precision과 Recall 계산
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # AP 계산
    ap = calculate_ap(precisions, recalls)
    
    return ap, precisions, recalls


def main():
    """
    메인 평가 함수
    """
    parser = argparse.ArgumentParser(description='Polygon IoU 기반 mAP 평가')
    parser.add_argument('--gt-labels', type=str, required=True,
                        help='Ground truth labels 디렉토리 경로')
    parser.add_argument('--pred-csv', type=str, nargs='+', required=True,
                        help='예측 결과 CSV 디렉토리 경로 (여러 개 가능)')
    parser.add_argument('--model-names', type=str, nargs='+', required=True,
                        help='모델 이름 (pred-csv와 동일한 순서)')
    parser.add_argument('--img-width', type=int, default=720,
                        help='이미지 너비 (기본값: 720)')
    parser.add_argument('--img-height', type=int, default=1160,
                        help='이미지 높이 (기본값: 1160)')
    
    args = parser.parse_args()
    
    # 모델 이름과 CSV 경로 개수 확인
    if len(args.pred_csv) != len(args.model_names):
        print("❌ --pred-csv와 --model-names의 개수가 일치해야 합니다!")
        return
    
    # IoU threshold
    iou_thresholds = [0.5, 0.75, 0.95]
    
    # Ground truth 로드
    print("Loading ground truth...")
    gt_dict = load_ground_truth(args.gt_labels)
    print(f"Loaded {len(gt_dict)} ground truth images")
    
    # 결과 저장
    results = {}
    
    # 각 모델에 대해 평가
    for model_name, csv_dir in zip(args.model_names, args.pred_csv):
        print(f"\n{'='*80}")
        print(f"Evaluating {model_name}")
        print(f"{'='*80}")
        
        if not Path(csv_dir).exists():
            print(f"⚠️  CSV directory not found: {csv_dir}")
            continue
        
        # 예측 결과 로드
        pred_dict = load_predictions_from_csv(csv_dir)
        print(f"Loaded {len(pred_dict)} predictions")
        
        model_results = {}
        
        for iou_thresh in iou_thresholds:
            print(f"\nEvaluating at IoU threshold: {iou_thresh}")
            
            # GT 매칭 상태 초기화
            for boxes in gt_dict.values():
                for box in boxes:
                    box['matched'] = False
            
            ap, precisions, recalls = evaluate_map(
                gt_dict.copy(), 
                pred_dict, 
                iou_threshold=iou_thresh,
                img_width=args.img_width,
                img_height=args.img_height
            )
            
            model_results[f'AP@{iou_thresh}'] = ap
            print(f"  AP@{iou_thresh}: {ap:.4f}")
        
        # mAP 계산 (0.5:0.95)
        ap_values = []
        for thresh in np.linspace(0.5, 0.95, 10):
            # GT 매칭 상태 초기화
            for boxes in gt_dict.values():
                for box in boxes:
                    box['matched'] = False
            
            ap, _, _ = evaluate_map(
                gt_dict.copy(), 
                pred_dict, 
                iou_threshold=thresh,
                img_width=args.img_width,
                img_height=args.img_height
            )
            ap_values.append(ap)
        
        mAP = np.mean(ap_values)
        model_results['mAP@0.5:0.95'] = mAP
        
        results[model_name] = model_results
    
    # 결과 출력 순서 정의 (v8AABB, v11AABB, v8OBB, v11OBB, IWPOD, v8QBB)
    model_order = ['v8AABB', 'v11AABB', 'v8OBB', 'v11OBB', 'IWPOD', 'v8QBB']
    # 결과에 있는 모델만 순서대로 정렬
    sorted_model_names = [name for name in model_order if name in results]
    # 순서에 없는 모델이 있다면 뒤에 추가
    for name in results.keys():
        if name not in sorted_model_names:
            sorted_model_names.append(name)

    # 결과 출력
    print(f"\n{'='*80}")
    print("📊 Final Results - Polygon IoU based mAP")
    print(f"{'='*80}\n")

    print(f"{'Model':<15} {'AP@0.5':<10} {'AP@0.75':<10} {'AP@0.95':<10} {'mAP@0.5:0.95':<15}")
    print("-" * 80)

    for model_name in sorted_model_names:
        model_results = results[model_name]
        print(f"{model_name:<15} ", end="")
        print(f"{model_results.get('AP@0.5', 0):<10.4f} ", end="")
        print(f"{model_results.get('AP@0.75', 0):<10.4f} ", end="")
        print(f"{model_results.get('AP@0.95', 0):<10.4f} ", end="")
        print(f"{model_results.get('mAP@0.5:0.95', 0):<15.4f}")

    print(f"\n{'='*80}\n")

    # 결과를 TXT 파일로 저장 - runs 폴더에 저장
    # 첫 번째 pred-csv 경로에서 runs 디렉토리를 찾아 그곳에 저장
    first_csv_path = Path(args.pred_csv[0])
    # runs 디렉토리까지 올라가기
    runs_dir = None
    for parent in first_csv_path.parents:
        if parent.name == 'runs':
            runs_dir = parent
            break

    if runs_dir is None:
        # runs 디렉토리를 찾지 못한 경우 현재 디렉토리에 저장
        base_dir = Path('.')
    else:
        base_dir = runs_dir

    # 파일명에 번호 추가 (01부터 시작)
    counter = 1
    while True:
        output_file = base_dir / f'polygon_iou_evaluation_results_{counter:02d}.txt'
        if not output_file.exists():
            break
        counter += 1

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Polygon IoU based mAP Evaluation Results\n")
        f.write("="*80 + "\n\n")

        # 평가 설정 정보
        f.write("Evaluation Configuration:\n")
        f.write("-"*80 + "\n")
        f.write(f"Ground Truth Labels: {args.gt_labels}\n")
        f.write(f"Image Width: {args.img_width}\n")
        f.write(f"Image Height: {args.img_height}\n")
        f.write(f"Total GT Images: {len(gt_dict)}\n")
        f.write("\n")

        # 모델별 예측 경로 (정렬된 순서로)
        f.write("Model Prediction Paths:\n")
        f.write("-"*80 + "\n")
        # 모델 이름과 CSV 경로 매핑
        model_csv_map = dict(zip(args.model_names, args.pred_csv))
        for model_name in sorted_model_names:
            csv_dir = model_csv_map.get(model_name, "N/A")
            f.write(f"{model_name}: {csv_dir}\n")
        f.write("\n")

        # 결과 테이블
        f.write("="*80 + "\n")
        f.write("Results:\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Model':<15} {'AP@0.5':<10} {'AP@0.75':<10} {'AP@0.95':<10} {'mAP@0.5:0.95':<15}\n")
        f.write("-" * 80 + "\n")

        for model_name in sorted_model_names:
            model_results = results[model_name]
            f.write(f"{model_name:<15} ")
            f.write(f"{model_results.get('AP@0.5', 0):<10.4f} ")
            f.write(f"{model_results.get('AP@0.75', 0):<10.4f} ")
            f.write(f"{model_results.get('AP@0.95', 0):<10.4f} ")
            f.write(f"{model_results.get('mAP@0.5:0.95', 0):<15.4f}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"\n💾 Results saved to: {output_file}")


if __name__ == '__main__':
    main()
