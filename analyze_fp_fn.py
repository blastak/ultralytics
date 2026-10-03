"""
오검출(False Positive) 및 미검출(False Negative) 분석 스크립트

- GT와 예측 결과를 비교하여 FP/FN 검출
- 큰 이미지와 crop 이미지를 함께 시각화
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from shapely.geometry import Polygon


# ============ 설정 ============
MODEL_PATH = "/workspace/repo/ultralytics/runs/qbb/crpd_multi_qbb_polyiou_regmax1/weights/best.pt"
TEST_IMAGES_DIR = "/workspace/repo/ultralytics/ultralytics/assets/crpd_multi_xyxyxyxy6/images/test"
TEST_LABELS_DIR = "/workspace/repo/ultralytics/ultralytics/assets/crpd_multi_xyxyxyxy6/labels/test"
OUTPUT_DIR = "/workspace/repo/ultralytics/analysis_fp_fn"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5  # GT와 예측 매칭 기준 IoU
CLASS_NAMES = ['blue', 'green', 'yellow', 'white']
CLASS_COLORS = {
    0: (255, 0, 0),    # blue - BGR
    1: (0, 255, 0),    # green
    2: (0, 255, 255),  # yellow
    3: (255, 255, 255) # white
}


def polygon_iou(poly1_pts, poly2_pts):
    """두 다각형 사이의 IoU 계산"""
    try:
        poly1 = Polygon(poly1_pts)
        poly2 = Polygon(poly2_pts)
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        if union == 0:
            return 0.0
        return inter / union
    except:
        return 0.0


def load_gt_labels(label_path, img_w, img_h):
    """GT 라벨 로드 (normalized -> pixel 좌표)"""
    gts = []
    if not os.path.exists(label_path):
        return gts

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:9]))
            # normalized -> pixel
            pts = []
            for i in range(0, 8, 2):
                x = coords[i] * img_w
                y = coords[i+1] * img_h
                pts.append([x, y])
            gts.append({
                'cls': cls,
                'pts': np.array(pts, dtype=np.float32),
                'matched': False
            })
    return gts


def get_predictions(results, conf_thresh):
    """예측 결과 파싱"""
    preds = []
    if results[0].qbb is None:
        return preds

    qbb = results[0].qbb
    boxes = qbb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
    confs = qbb.conf.cpu().numpy()
    classes = qbb.cls.cpu().numpy().astype(int)

    for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
        if conf >= conf_thresh:
            preds.append({
                'cls': cls,
                'conf': conf,
                'pts': box,  # (4, 2)
                'matched': False
            })
    return preds


def match_gt_pred(gts, preds, iou_thresh):
    """GT와 예측 매칭, FP/FN 분류"""
    # 각 GT에 대해 가장 높은 IoU의 예측 매칭
    for gt in gts:
        best_iou = 0
        best_pred_idx = -1
        for idx, pred in enumerate(preds):
            if pred['matched']:
                continue
            # 클래스가 같아야 매칭
            if gt['cls'] != pred['cls']:
                continue
            iou = polygon_iou(gt['pts'], pred['pts'])
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = idx

        if best_iou >= iou_thresh and best_pred_idx >= 0:
            gt['matched'] = True
            preds[best_pred_idx]['matched'] = True
            preds[best_pred_idx]['iou'] = best_iou

    # FN: 매칭되지 않은 GT
    fn_list = [gt for gt in gts if not gt['matched']]
    # FP: 매칭되지 않은 예측
    fp_list = [pred for pred in preds if not pred['matched']]
    # TP: 매칭된 예측
    tp_list = [pred for pred in preds if pred['matched']]

    return tp_list, fp_list, fn_list


def draw_polygon(img, pts, color, thickness=2, label=None):
    """사각형 그리기"""
    pts_int = pts.astype(np.int32)
    cv2.polylines(img, [pts_int], isClosed=True, color=color, thickness=thickness)
    if label:
        x, y = pts_int[0]
        cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_crop_region(pts, img_shape, margin=30):
    """crop 영역 계산 (margin 포함)"""
    h, w = img_shape[:2]
    x_min = max(0, int(pts[:, 0].min()) - margin)
    y_min = max(0, int(pts[:, 1].min()) - margin)
    x_max = min(w, int(pts[:, 0].max()) + margin)
    y_max = min(h, int(pts[:, 1].max()) + margin)
    return x_min, y_min, x_max, y_max


def create_visualization(img_path, gts, preds, tp_list, fp_list, fn_list, output_dir, img_name):
    """시각화 이미지 생성"""
    img = cv2.imread(img_path)
    if img is None:
        return

    h, w = img.shape[:2]

    # 오류가 있는 경우에만 저장
    if len(fp_list) == 0 and len(fn_list) == 0:
        return None

    # 전체 이미지에 GT와 예측 그리기
    img_full = img.copy()

    # GT (초록색 점선)
    for gt in gts:
        color = CLASS_COLORS.get(gt['cls'], (0, 255, 0))
        pts_int = gt['pts'].astype(np.int32)
        # 점선 효과
        for i in range(4):
            pt1 = tuple(pts_int[i])
            pt2 = tuple(pts_int[(i+1)%4])
            cv2.line(img_full, pt1, pt2, color, 2, cv2.LINE_AA)

    # 예측 (실선)
    for pred in preds:
        if pred['matched']:
            color = (0, 255, 0)  # TP: 초록
        else:
            color = (0, 0, 255)  # FP: 빨강
        pts_int = pred['pts'].astype(np.int32)
        cv2.polylines(img_full, [pts_int], True, color, 3)
        label = f"{CLASS_NAMES[pred['cls']]} {pred['conf']:.2f}"
        cv2.putText(img_full, label, (pts_int[0][0], pts_int[0][1]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # FN 표시 (GT만 있고 예측 없음) - 파란색
    for fn in fn_list:
        pts_int = fn['pts'].astype(np.int32)
        cv2.polylines(img_full, [pts_int], True, (255, 0, 0), 3)
        label = f"FN:{CLASS_NAMES[fn['cls']]}"
        cv2.putText(img_full, label, (pts_int[0][0], pts_int[0][1]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    results_info = {
        'img_name': img_name,
        'fp_count': len(fp_list),
        'fn_count': len(fn_list),
        'tp_count': len(tp_list),
        'crops': []
    }

    # 오류별 crop 이미지 생성
    crop_imgs = []

    # FP crops
    for i, fp in enumerate(fp_list):
        x1, y1, x2, y2 = get_crop_region(fp['pts'], img.shape)
        crop = img[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        # crop 내 좌표 조정
        pts_adj = fp['pts'] - np.array([x1, y1])
        cv2.polylines(crop, [pts_adj.astype(np.int32)], True, (0, 0, 255), 2)
        label = f"FP:{CLASS_NAMES[fp['cls']]} {fp['conf']:.2f}"
        cv2.putText(crop, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        crop_imgs.append(('FP', crop, fp))

    # FN crops (GT 기준)
    for i, fn in enumerate(fn_list):
        x1, y1, x2, y2 = get_crop_region(fn['pts'], img.shape)
        crop = img[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        pts_adj = fn['pts'] - np.array([x1, y1])
        cv2.polylines(crop, [pts_adj.astype(np.int32)], True, (255, 0, 0), 2)
        label = f"FN:{CLASS_NAMES[fn['cls']]}"
        cv2.putText(crop, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        crop_imgs.append(('FN', crop, fn))

    # 결과 이미지 조합 (전체 이미지 + crops)
    # 전체 이미지 리사이즈 (가로 800 기준)
    scale = 800 / w
    img_full_resized = cv2.resize(img_full, (800, int(h * scale)))

    # crop 이미지들을 오른쪽에 배치
    if crop_imgs:
        # crop 이미지 리사이즈 (가로 200 기준)
        crops_resized = []
        for err_type, crop, _ in crop_imgs:
            crop_h, crop_w = crop.shape[:2]
            crop_scale = 200 / max(crop_w, 1)
            new_h = max(int(crop_h * crop_scale), 50)
            new_w = 200
            crop_resized = cv2.resize(crop, (new_w, new_h))
            crops_resized.append(crop_resized)

        # 세로로 crop 이미지 연결
        max_crop_height = sum(c.shape[0] for c in crops_resized) + 10 * len(crops_resized)
        canvas_height = max(img_full_resized.shape[0], max_crop_height)

        # 캔버스 생성
        canvas = np.zeros((canvas_height, 800 + 220, 3), dtype=np.uint8)
        canvas[:img_full_resized.shape[0], :800] = img_full_resized

        # crops 배치
        y_offset = 10
        for crop_resized in crops_resized:
            ch = crop_resized.shape[0]
            if y_offset + ch > canvas_height:
                break
            canvas[y_offset:y_offset+ch, 810:1010] = crop_resized
            y_offset += ch + 10
    else:
        canvas = img_full_resized

    return canvas, results_info


def main():
    print("=" * 60)
    print("오검출(FP) / 미검출(FN) 분석 시작")
    print("=" * 60)

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fp_dir = os.path.join(OUTPUT_DIR, "false_positive")
    fn_dir = os.path.join(OUTPUT_DIR, "false_negative")
    combined_dir = os.path.join(OUTPUT_DIR, "combined")
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    # 모델 로드
    print(f"\n모델 로딩: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 테스트 이미지 목록
    img_files = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
    print(f"테스트 이미지 수: {len(img_files)}")

    # 통계
    total_tp, total_fp, total_fn = 0, 0, 0
    fp_images = []
    fn_images = []

    for idx, img_file in enumerate(img_files):
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(TEST_LABELS_DIR, label_file)

        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # GT 로드
        gts = load_gt_labels(label_path, w, h)

        # 예측
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
        preds = get_predictions(results, CONF_THRESHOLD)

        # 매칭 및 분류
        tp_list, fp_list, fn_list = match_gt_pred(gts, preds, IOU_THRESHOLD)

        total_tp += len(tp_list)
        total_fp += len(fp_list)
        total_fn += len(fn_list)

        # 시각화 생성 (오류가 있는 경우만)
        if fp_list or fn_list:
            result = create_visualization(img_path, gts, preds, tp_list, fp_list, fn_list,
                                         OUTPUT_DIR, img_file)
            if result:
                canvas, info = result

                # 저장
                if fp_list:
                    fp_images.append(info)
                    out_path = os.path.join(fp_dir, img_file)
                    cv2.imwrite(out_path, canvas)

                if fn_list:
                    fn_images.append(info)
                    out_path = os.path.join(fn_dir, img_file)
                    cv2.imwrite(out_path, canvas)

                # combined에도 저장
                out_path = os.path.join(combined_dir, img_file)
                cv2.imwrite(out_path, canvas)

        if (idx + 1) % 50 == 0:
            print(f"진행: {idx+1}/{len(img_files)}")

    # 결과 출력
    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)
    print(f"Total TP: {total_tp}")
    print(f"Total FP (오검출): {total_fp}")
    print(f"Total FN (미검출): {total_fn}")

    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
        print(f"Precision: {precision:.4f}")

    if total_tp + total_fn > 0:
        recall = total_tp / (total_tp + total_fn)
        print(f"Recall: {recall:.4f}")

    print(f"\n오검출 이미지 수: {len(fp_images)}")
    print(f"미검출 이미지 수: {len(fn_images)}")
    print(f"\n결과 저장 위치: {OUTPUT_DIR}")
    print(f"  - false_positive/: 오검출 포함 이미지")
    print(f"  - false_negative/: 미검출 포함 이미지")
    print(f"  - combined/: 모든 오류 이미지")


if __name__ == "__main__":
    main()
