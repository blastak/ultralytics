#!/usr/bin/env python
"""
OBB vs QBB 모델 학습 테스트 스크립트 (개선된 버전)
DOTA8 데이터셋을 사용하여 두 모델의 성능을 비교합니다.
GPU를 최대치로 활용하고 충분한 타임아웃으로 20에폭 학습을 수행합니다.
"""

from ultralytics import YOLO
import os
import torch
from pathlib import Path
import time
from datetime import datetime

# 데이터셋 경로를 assets 폴더로 설정
DATASET_DIR = Path("ultralytics/assets/datasets")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def setup_gpu_environment():
    """GPU 환경 설정 및 정보 출력"""
    print("=" * 60)
    print("GPU 환경 설정")
    print("=" * 60)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA 사용 가능: {gpu_count}개의 GPU 감지됨")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        device = 'cuda'  # 모든 GPU 사용
        
        # 최적의 배치 사이즈 계산 (GPU 메모리에 따라)
        total_memory = sum(torch.cuda.get_device_properties(i).total_memory 
                          for i in range(gpu_count)) / 1024**3
        
        if total_memory > 20:
            batch_size = 16  # 고성능 GPU
        elif total_memory > 10:
            batch_size = 8   # 중급 GPU
        else:
            batch_size = 4   # 저급 GPU
            
    else:
        print("❌ CUDA 사용 불가능, CPU 사용")
        device = 'cpu'
        batch_size = 2  # CPU는 작은 배치 사이즈
    
    print(f"사용할 디바이스: {device}")
    print(f"배치 사이즈: {batch_size}")
    return device, batch_size

def train_obb(device='cuda', batch_size=8):
    """OBB 모델 학습 (개선된 버전)"""
    print("=" * 60)
    print("OBB 모델 학습 시작...")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    
    # OBB 모델 초기화
    model = YOLO('ultralytics/cfg/models/v8/yolov8n-obb.yaml')  # 작은 모델로 빠른 학습
    
    # 학습 실행 (최적화된 설정)
    results = model.train(
        data='ultralytics/cfg/datasets/dota8.yaml',
        epochs=20,
        imgsz=640,
        batch=batch_size,
        device=device,
        project='runs/obb',
        name='performance_test',
        exist_ok=True,
        verbose=True,
        patience=50,  # 조기 종료 방지
        save=True,
        plots=True,
        cache=True,  # 데이터 캐싱으로 속도 향상
        workers=8,   # 데이터 로더 워커 수 증가
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ OBB 학습 완료!")
    print(f"학습 시간: {training_time/60:.2f}분")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results, training_time

def train_qbb(device='cuda', batch_size=8):
    """QBB 모델 학습 (개선된 버전)"""
    print("=" * 60)
    print("QBB 모델 학습 시작...")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    
    # QBB 모델 초기화
    model = YOLO('ultralytics/cfg/models/v8/yolov8n-qbb.yaml')  # 작은 모델로 빠른 학습
    
    # 학습 실행 (최적화된 설정)
    results = model.train(
        data='ultralytics/cfg/datasets/dota8.yaml',
        epochs=20,
        imgsz=640,
        batch=batch_size,
        device=device,
        project='runs/qbb',
        name='performance_test',
        exist_ok=True,
        verbose=True,
        patience=50,  # 조기 종료 방지
        save=True,
        plots=True,
        cache=True,  # 데이터 캐싱으로 속도 향상
        workers=8,   # 데이터 로더 워커 수 증가
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ QBB 학습 완료!")
    print(f"학습 시간: {training_time/60:.2f}분")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results, training_time

def compare_results(obb_results, obb_time, qbb_results, qbb_time):
    """두 모델의 결과 비교"""
    print("=" * 80)
    print("🔍 학습 결과 비교 분석")
    print("=" * 80)
    
    print(f"\n⏱️  학습 시간 비교:")
    print(f"  OBB: {obb_time/60:.2f}분")
    print(f"  QBB: {qbb_time/60:.2f}분")
    print(f"  시간 차이: {abs(obb_time-qbb_time)/60:.2f}분")
    
    # 최종 메트릭 비교 (가능한 경우)
    try:
        if hasattr(obb_results, 'metrics') and hasattr(qbb_results, 'metrics'):
            print(f"\n📊 성능 메트릭 비교:")
            if hasattr(obb_results.metrics, 'box'):
                obb_map = obb_results.metrics.box.map
                print(f"  OBB mAP: {obb_map:.4f}")
            if hasattr(qbb_results.metrics, 'box'):
                qbb_map = qbb_results.metrics.box.map
                print(f"  QBB mAP: {qbb_map:.4f}")
    except:
        print("\n📊 메트릭 정보를 가져올 수 없습니다. 로그 파일을 확인해주세요.")
    
    print(f"\n📁 결과 저장 위치:")
    print(f"  OBB: runs/obb/performance_test/")
    print(f"  QBB: runs/qbb/performance_test/")
    print(f"\n💡 자세한 결과는 위 디렉토리의 results.png와 로그 파일들을 확인하세요.")
    print("=" * 80)

if __name__ == "__main__":
    print("🚀 OBB vs QBB 모델 성능 비교 테스트 (개선된 버전)")
    print("=" * 80)
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print(f"데이터셋 디렉토리: {DATASET_DIR.absolute()}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # GPU 환경 설정
    device, batch_size = setup_gpu_environment()
    
    # 환경 변수로 데이터셋 경로 설정
    os.environ['YOLO_DATASETS_DIR'] = str(DATASET_DIR.absolute())
    
    # 데이터셋 경로 확인
    dota8_path = DATASET_DIR / "dota8"
    if not dota8_path.exists():
        print(f"\n📥 주의: DOTA8 데이터셋이 {DATASET_DIR}에 자동으로 다운로드됩니다.")
        print("첫 실행시 다운로드에 시간이 소요될 수 있습니다.")
    
    total_start_time = time.time()
    
    try:
        print(f"\n🎯 20에폭 학습 시작 (디바이스: {device}, 배치사이즈: {batch_size})")
        
        # 1. OBB 모델 학습
        print("\n" + "🟦" * 20 + " OBB 모델 학습 " + "🟦" * 20)
        obb_results, obb_time = train_obb(device, batch_size)
        
        # GPU 메모리 정리
        if device == 'cuda':
            torch.cuda.empty_cache()
            print("🧹 GPU 메모리 정리 완료")
        
        # 2. QBB 모델 학습
        print("\n" + "🟨" * 20 + " QBB 모델 학습 " + "🟨" * 20)
        qbb_results, qbb_time = train_qbb(device, batch_size)
        
        # 결과 비교
        compare_results(obb_results, obb_time, qbb_results, qbb_time)
        
        total_time = time.time() - total_start_time
        print(f"\n🎉 전체 학습 완료!")
        print(f"⏱️  총 소요 시간: {total_time/60:.2f}분")
        print(f"📅 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 성공 메시지
        print("\n" + "✅" * 40)
        print("🏆 QBB 2단계 성능 비교 테스트 성공적으로 완료!")
        print("✅" * 40)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 문제 해결 팁:")
        print("- GPU 메모리 부족시 배치 사이즈를 줄여보세요")
        print("- 데이터셋 다운로드 실패시 인터넷 연결을 확인해보세요") 
        print("- CUDA 오류시 PyTorch 설치를 확인해보세요")