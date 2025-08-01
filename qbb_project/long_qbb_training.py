#!/usr/bin/env python3
"""
QBB 장기 학습 스크립트
1944개 통합 번호판 데이터셋으로 100 epochs 학습
"""

import sys
import torch
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/workspace/repo/ultralytics')
from ultralytics import YOLO


def setup_training_environment():
    """학습 환경 설정"""
    print("🔧 Setting up training environment...")
    
    # CUDA 확인
    print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 작업 디렉토리 확인
    work_dir = Path.cwd()
    print(f"📁 Working directory: {work_dir}")
    
    # 데이터셋 확인
    dataset_yaml = Path('datasets/unified_license_plates/unified_license_plates.yaml')
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    
    print(f"📊 Dataset YAML: {dataset_yaml}")
    
    return dataset_yaml


def train_qbb_long():
    """QBB 장기 학습 실행"""
    
    print("🚀 Starting QBB Long Training Session")
    print("=" * 50)
    
    # 환경 설정
    dataset_yaml = setup_training_environment()
    
    # 모델 로드
    print(f"\n📦 Loading QBB model...")
    model = YOLO('yolo11n-qbb.yaml')
    
    print(f"✅ QBB model loaded successfully")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # 학습 시작 시간 기록
    start_time = time.time()
    
    # 고급 학습 설정
    training_config = {
        'data': str(dataset_yaml),
        'epochs': 100,              # 장기 학습
        'imgsz': 640,               # 표준 이미지 크기
        'batch': 8,                 # GPU 메모리 고려
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 4,               # 데이터 로딩 워커
        'patience': 20,             # 조기 중단 임계값
        'save': True,               # 모델 저장
        'plots': True,              # 플롯 생성
        'val': True,                # 검증 실행
        'project': 'runs/qbb',      # 결과 저장 위치
        'name': 'unified_training', # 실험 이름
        'exist_ok': True,           # 기존 실험 덮어쓰기
        'verbose': True,            # 상세 로그
        
        # 하이퍼파라미터 최적화
        'lr0': 0.01,                # 초기 학습률
        'lrf': 0.01,                # 최종 학습률 (cosine decay)
        'momentum': 0.937,          # SGD 모멘텀
        'weight_decay': 0.0005,     # 가중치 감소
        'warmup_epochs': 3.0,       # 워밍업 에포크
        'warmup_momentum': 0.8,     # 워밍업 모멘텀
        'warmup_bias_lr': 0.1,      # 워밍업 편향 학습률
        
        # 데이터 증강
        'hsv_h': 0.015,             # 색조 증강
        'hsv_s': 0.7,               # 채도 증강
        'hsv_v': 0.4,               # 명도 증강
        'degrees': 0.0,             # 회전 증강 (번호판은 회전 최소화)
        'translate': 0.1,           # 이동 증강
        'scale': 0.5,               # 크기 증강
        'shear': 0.0,               # 전단 증강 (번호판 형태 보존)
        'perspective': 0.0,         # 원근 증강 (형태 보존)
        'flipud': 0.0,              # 상하 반전 (번호판 특성상 비활성화)
        'fliplr': 0.5,              # 좌우 반전
        'mosaic': 1.0,              # 모자이크 증강
        'mixup': 0.0,               # 믹스업 (OBB에서는 비활성화)
        'copy_paste': 0.0,          # 복사-붙여넣기 (OBB에서는 비활성화)
        
        # 정규화 및 안정성
        'box': 7.5,                 # 박스 손실 가중치
        'cls': 0.5,                 # 분류 손실 가중치
        'dfl': 1.5,                 # 분포 초점 손실 가중치
        
        # 평가 설정
        'iou': 0.7,                 # NMS IoU 임계값
        'save_period': 10,          # 모델 저장 주기
        'cache': False,             # 캐시 비활성화 (메모리 절약)
        'amp': True,                # 자동 혼합 정밀도
        'fraction': 1.0,            # 데이터셋 사용 비율
        'profile': False,           # 프로파일링 비활성화
        'freeze': None,             # 레이어 동결 없음
        'multi_scale': False,       # 다중 스케일 비활성화
        'overlap_mask': True,       # 마스크 오버랩 허용
        'mask_ratio': 4,            # 마스크 다운샘플링 비율
        'dropout': 0.0,             # 드롭아웃 비율
    }
    
    print(f"\n⚙️  Training Configuration:")
    print(f"  📊 Dataset: unified_license_plates (1944 images)")
    print(f"  🎯 Epochs: {training_config['epochs']}")
    print(f"  📦 Batch size: {training_config['batch']}")
    print(f"  🖼️  Image size: {training_config['imgsz']}")
    print(f"  🧠 Learning rate: {training_config['lr0']} → {training_config['lrf']}")
    print(f"  ⏰ Patience: {training_config['patience']}")
    print(f"  🎲 Device: {training_config['device']}")
    
    try:
        print(f"\n🚀 Starting training...")
        results = model.train(**training_config)
        
        # 학습 완료 시간 계산
        end_time = time.time()
        training_duration = end_time - start_time
        hours = int(training_duration // 3600)
        minutes = int((training_duration % 3600) // 60)
        
        print(f"\n🎉 QBB Long Training Completed Successfully!")
        print(f"⏰ Training duration: {hours}h {minutes}m")
        print(f"📁 Results saved in: runs/qbb/unified_training/")
        
        # 결과 모델 경로
        best_model_path = Path('runs/qbb/unified_training/weights/best.pt')
        last_model_path = Path('runs/qbb/unified_training/weights/last.pt')
        
        if best_model_path.exists():
            print(f"🏆 Best model: {best_model_path}")
        if last_model_path.exists():
            print(f"📊 Last model: {last_model_path}")
        
        # 학습 결과 요약
        print_training_summary(results, training_duration)
        
        return results, best_model_path
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def print_training_summary(results, duration):
    """학습 결과 요약 출력"""
    print(f"\n📊 Training Summary:")
    print(f"=" * 30)
    
    try:
        # 결과 경로에서 메트릭 정보 읽기
        results_dir = Path('runs/qbb/unified_training')
        
        if (results_dir / 'results.csv').exists():
            import pandas as pd
            df = pd.read_csv(results_dir / 'results.csv')
            
            if not df.empty:
                last_epoch = df.iloc[-1]
                
                print(f"📈 Final Metrics:")
                if 'metrics/mAP50(B)' in df.columns:
                    print(f"  🎯 mAP@0.5: {last_epoch.get('metrics/mAP50(B)', 'N/A'):.4f}")
                if 'metrics/mAP50-95(B)' in df.columns:
                    print(f"  🎯 mAP@0.5:0.95: {last_epoch.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
                if 'train/box_loss' in df.columns:
                    print(f"  📦 Box Loss: {last_epoch.get('train/box_loss', 'N/A'):.4f}")
                if 'train/cls_loss' in df.columns:
                    print(f"  🏷️  Class Loss: {last_epoch.get('train/cls_loss', 'N/A'):.4f}")
                if 'val/box_loss' in df.columns:
                    print(f"  ✅ Val Box Loss: {last_epoch.get('val/box_loss', 'N/A'):.4f}")
                if 'val/cls_loss' in df.columns:
                    print(f"  ✅ Val Class Loss: {last_epoch.get('val/cls_loss', 'N/A'):.4f}")
    
    except Exception as e:
        print(f"  📊 Metrics summary not available: {e}")
    
    print(f"⏰ Total training time: {duration/3600:.1f} hours")
    print(f"📁 All results saved in: runs/qbb/unified_training/")


def validate_trained_model(model_path):
    """학습된 모델 검증"""
    print(f"\n🔍 Validating trained model: {model_path}")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        model = YOLO(model_path)
        
        # 검증 실행
        dataset_yaml = Path('datasets/unified_license_plates/unified_license_plates.yaml')
        results = model.val(data=str(dataset_yaml), verbose=True)
        
        print(f"✅ Model validation completed")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")


def main():
    print("🚀 QBB Long Training Session")
    print("📊 Dataset: Unified License Plates (1944 images)")
    print("🎯 Target: 100 epochs comprehensive training")
    print("=" * 60)
    
    # 학습 실행
    results, best_model = train_qbb_long()
    
    if results and best_model:
        # 모델 검증
        validate_trained_model(best_model)
        
        print(f"\n🎊 QBB Long Training Session Completed!")
        print(f"📈 Next steps:")
        print(f"1. Review training plots in runs/qbb/unified_training/")
        print(f"2. Run visualization script: python visualize_qbb_results.py")
        print(f"3. Test model performance on test set")
        print(f"4. Compare with OBB baseline performance")
        
    else:
        print(f"\n❌ Training failed. Please check error messages above.")
        print(f"💡 Troubleshooting tips:")
        print(f"1. Check GPU memory availability")
        print(f"2. Verify dataset paths and formats")
        print(f"3. Review training configuration")


if __name__ == "__main__":
    main()