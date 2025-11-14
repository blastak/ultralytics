#!/usr/bin/env python3
"""
추론 속도 평가
여러 모델의 추론 속도를 비교 분석
"""

import csv
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import pytz


def load_timing_from_csv(csv_dir):
    """
    inference_timing.txt 파일에서 타이밍 데이터 로드

    Args:
        csv_dir: inference_csv 디렉토리 경로

    Returns:
        timings: 타이밍 데이터 리스트 (ms 단위)
    """
    timing_file = Path(csv_dir) / 'inference_timing.txt'

    if not timing_file.exists():
        print(f"⚠️  Timing file not found: {timing_file}")
        return None

    timings = []
    with open(timing_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # 헤더 스킵

        # 첫 번째 데이터 행(warm-up) 스킵
        try:
            next(reader)
        except StopIteration:
            return None

        # 나머지 데이터 읽기
        for row in reader:
            try:
                if len(row) >= 2:
                    timing = float(row[1])
                    timings.append(timing)
            except:
                continue

    return timings if timings else None


def calculate_statistics(timings):
    """
    타이밍 데이터의 통계 계산

    Args:
        timings: 타이밍 데이터 리스트

    Returns:
        stats: 통계 딕셔너리
    """
    timings_array = np.array(timings)

    return {
        'mean': np.mean(timings_array),
        'std': np.std(timings_array),
        'min': np.min(timings_array),
        'max': np.max(timings_array),
        'median': np.median(timings_array),
        'p95': np.percentile(timings_array, 95),
        'p99': np.percentile(timings_array, 99),
        'count': len(timings_array),
        'fps': 1000.0 / np.mean(timings_array)  # ms to FPS
    }


def main():
    """
    메인 평가 함수
    """
    parser = argparse.ArgumentParser(description='추론 속도 평가')
    parser.add_argument('--pred-csv', type=str, nargs='+', required=True,
                        help='예측 결과 CSV 디렉토리 경로 (여러 개 가능)')
    parser.add_argument('--model-names', type=str, nargs='+', required=True,
                        help='모델 이름 (pred-csv와 동일한 순서)')
    parser.add_argument('--dataset-name', type=str, default='test',
                        help='데이터셋 이름 (결과 파일명에 사용)')

    args = parser.parse_args()

    # 모델 이름과 CSV 경로 개수 확인
    if len(args.pred_csv) != len(args.model_names):
        print("❌ --pred-csv와 --model-names의 개수가 일치해야 합니다!")
        return

    # 결과 저장
    results = {}

    # 각 모델에 대해 타이밍 분석
    for model_name, csv_dir in zip(args.model_names, args.pred_csv):
        print(f"\n{'='*80}")
        print(f"Analyzing {model_name}")
        print(f"{'='*80}")

        if not Path(csv_dir).exists():
            print(f"⚠️  CSV directory not found: {csv_dir}")
            continue

        # 타이밍 데이터 로드
        timings = load_timing_from_csv(csv_dir)

        if timings is None:
            print(f"⚠️  No timing data found for {model_name}")
            continue

        # 통계 계산
        stats = calculate_statistics(timings)
        results[model_name] = stats

        print(f"Loaded {stats['count']} timing measurements")
        print(f"  Mean: {stats['mean']:.2f} ms")
        print(f"  FPS:  {stats['fps']:.1f}")

    if not results:
        print("\n❌ No valid timing data found!")
        return

    # 결과 출력 순서 정의 (v8AABB, v11AABB, v8OBB, v11OBB, IWPOD, v8QBB)
    model_order = ['v8AABB', 'v11AABB', 'v8OBB', 'v11OBB', 'IWPOD', 'v8QBB']
    # 결과에 있는 모델만 순서대로 정렬
    sorted_model_names = [name for name in model_order if name in results]
    # 순서에 없는 모델이 있다면 뒤에 추가
    for name in results.keys():
        if name not in sorted_model_names:
            sorted_model_names.append(name)

    # FPS 기준으로도 정렬된 버전 생성
    sorted_by_fps = sorted(results.items(), key=lambda x: x[1]['fps'], reverse=True)

    # 결과 출력
    print(f"\n{'='*100}")
    print(f"📊 Final Results - Inference Speed (sorted by predefined order)")
    print(f"{'='*100}\n")

    print(f"{'Model':<30} {'Mean(ms)':<12} {'Std(ms)':<12} {'Median(ms)':<12} {'FPS':<10} {'Count':<8}")
    print("-"*100)

    for model_name in sorted_model_names:
        stats = results[model_name]
        print(f"{model_name:<30} {stats['mean']:<12.2f} {stats['std']:<12.2f} {stats['median']:<12.2f} {stats['fps']:<10.1f} {stats['count']:<8}")

    print(f"\n{'='*100}")
    print(f"📊 Results sorted by FPS (fastest first)")
    print(f"{'='*100}\n")

    print(f"{'Model':<30} {'Mean(ms)':<12} {'Std(ms)':<12} {'Median(ms)':<12} {'FPS':<10} {'Count':<8}")
    print("-"*100)

    for model_name, stats in sorted_by_fps:
        print(f"{model_name:<30} {stats['mean']:<12.2f} {stats['std']:<12.2f} {stats['median']:<12.2f} {stats['fps']:<10.1f} {stats['count']:<8}")

    print(f"\n{'='*100}\n")

    # 결과를 TXT 파일로 저장
    base_dir = Path('runs/analysis/evaluation_results/inference_speed')
    base_dir.mkdir(parents=True, exist_ok=True)

    # KST 타임스탬프로 파일명 생성
    kst = pytz.timezone('Asia/Seoul')
    now_kst = datetime.now(kst)
    timestamp = now_kst.strftime('%Y%m%d_%H%M%S')
    output_file = base_dir / f'inference_speed_comparison_{timestamp}.txt'

    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"Inference Speed Evaluation Results - {args.dataset_name}\n")
        f.write("="*100 + "\n\n")

        # 평가 설정 정보
        f.write("Evaluation Configuration:\n")
        f.write("-"*100 + "\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Note: First inference (warm-up) excluded from statistics\n")
        f.write(f"Total Models Evaluated: {len(results)}\n")
        f.write("\n")

        # 모델별 예측 경로
        f.write("Model Prediction Paths:\n")
        f.write("-"*100 + "\n")
        model_csv_map = dict(zip(args.model_names, args.pred_csv))
        for model_name in sorted_model_names:
            csv_dir = model_csv_map.get(model_name, "N/A")
            f.write(f"{model_name}: {csv_dir}\n")
        f.write("\n")

        # 결과 테이블 (정의된 순서)
        f.write("="*100 + "\n")
        f.write("Results (predefined order):\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Model':<30} {'Mean(ms)':<12} {'Std(ms)':<12} {'Median(ms)':<12} {'P95(ms)':<12} {'P99(ms)':<12} {'FPS':<10} {'Count':<8}\n")
        f.write("-"*100 + "\n")

        for model_name in sorted_model_names:
            stats = results[model_name]
            f.write(f"{model_name:<30} ")
            f.write(f"{stats['mean']:<12.2f} ")
            f.write(f"{stats['std']:<12.2f} ")
            f.write(f"{stats['median']:<12.2f} ")
            f.write(f"{stats['p95']:<12.2f} ")
            f.write(f"{stats['p99']:<12.2f} ")
            f.write(f"{stats['fps']:<10.1f} ")
            f.write(f"{stats['count']:<8}\n")

        f.write("\n" + "="*100 + "\n")

        # 결과 테이블 (FPS 순서)
        f.write("Results (sorted by FPS):\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Model':<30} {'Mean(ms)':<12} {'Std(ms)':<12} {'Median(ms)':<12} {'P95(ms)':<12} {'P99(ms)':<12} {'FPS':<10} {'Count':<8}\n")
        f.write("-"*100 + "\n")

        for model_name, stats in sorted_by_fps:
            f.write(f"{model_name:<30} ")
            f.write(f"{stats['mean']:<12.2f} ")
            f.write(f"{stats['std']:<12.2f} ")
            f.write(f"{stats['median']:<12.2f} ")
            f.write(f"{stats['p95']:<12.2f} ")
            f.write(f"{stats['p99']:<12.2f} ")
            f.write(f"{stats['fps']:<10.1f} ")
            f.write(f"{stats['count']:<8}\n")

        f.write("\n" + "="*100 + "\n")

        # 상대 속도 비교
        fastest_model = sorted_by_fps[0]
        fastest_fps = fastest_model[1]['fps']

        f.write("\nRelative Speed (compared to fastest model):\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Model':<30} {'FPS':<12} {'Relative Speed':<20}\n")
        f.write("-"*100 + "\n")

        for model_name, stats in sorted_by_fps:
            relative = (stats['fps'] / fastest_fps) * 100
            f.write(f"{model_name:<30} {stats['fps']:<12.1f} {relative:<20.1f}%\n")

        f.write("\n" + "="*100 + "\n")

    print(f"\n💾 Results saved to: {output_file}")


if __name__ == '__main__':
    main()
