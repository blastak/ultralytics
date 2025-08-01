# YOLO11 OBB (Oriented Bounding Box) 번호판 검출 프로젝트

이 프로젝트는 YOLO11 OBB 모델을 사용하여 번호판 검출을 위한 연구 및 개발 결과물입니다.

## 📁 프로젝트 구조

```
obb_project/
├── README.md              # 프로젝트 설명서
├── datasets/               # 학습 데이터셋들
│   ├── webpm_obb8_dataset/    # 최종 학습용 데이터셋 (8-point OBB)
│   ├── fixed_obb_dataset/     # 수정된 OBB 데이터셋
│   └── simple_obb_dataset/    # 단순 테스트 데이터셋
├── results/                # 학습 결과들
│   └── runs/
│       ├── detect/            # Detection 모델 결과
│       └── obb/
│           ├── train3/        # DOTA8 학습 결과
│           └── train5/        # 번호판 최종 학습 결과 ⭐
├── visualizations/         # 시각화 결과들
│   ├── obb_analysis_results/  # False alarm/Miss detection 분석
│   ├── inha_obb_test_results/ # 인하대 테스트 이미지 결과
│   └── random_obb_test/       # 무작위 테스트 결과
└── scripts/                # 학습 및 분석 스크립트들
    ├── train_entry_obb.py     # 메인 학습 스크립트
    ├── analyze_obb_final.py   # 성능 분석 스크립트
    ├── test_inha_obb.py       # 인하대 테스트 스크립트
    ├── test_random_obb.py     # 무작위 테스트 스크립트
    ├── test_dota8_obb.py      # DOTA8 테스트 스크립트
    ├── simple_obb_training.py # 단순 학습 스크립트
    └── fix_simple_obb.py      # 데이터셋 수정 스크립트
```

## 🎯 주요 성과

### 최종 모델 성능 (results/runs/obb/train5/)
- **Precision: 100%** (False alarm: 0개)
- **Recall: 95%** (Miss detection: 10/200개)
- **검출률**: 웹플레이트마니아 데이터 95%, 인하대 테스트 80%

### 학습 데이터
- **소스**: 웹플레이트마니아 1944장 중 150장 사용
- **형식**: 8-point OBB (x1,y1,x2,y2,x3,y3,x4,y4)
- **분할**: train 120장, val 30장

## 🚀 사용법

### 1. 학습 실행
```bash
cd scripts/
python train_entry_obb.py
```

### 2. 성능 분석
```bash
python analyze_obb_final.py
```

### 3. 테스트 실행
```bash
# 인하대 테스트 이미지
python test_inha_obb.py

# 무작위 테스트
python test_random_obb.py
```

## 📊 결과 해석

### False Alarm 분석
- **결과**: 0개 (완벽한 정밀도)
- **위치**: visualizations/obb_analysis_results/false_alarms/

### Miss Detection 분석  
- **결과**: 10개 (95% 재현율)
- **원인**: 작은 번호판, 복잡한 배경
- **위치**: visualizations/obb_analysis_results/miss_detections/

### 성공 케이스
- **다양한 차종**: 승용차, 트럭, 버스, 택시
- **다양한 각도**: 정면, 측면, 기울어진 번호판
- **위치**: visualizations/obb_analysis_results/correct_detections/

## 🔧 기술적 특징

### OBB vs Detection
- **Detection 모델**: Precision 100%, Recall 1.5%
- **OBB 모델**: Precision 100%, Recall 95%
- **63배 성능 향상** 달성

### 데이터 형식
- **입력**: LabelMe JSON (polygon)
- **변환**: 8-point OBB 좌표
- **정규화**: 0-1 범위로 정규화

### 학습 설정
```python
model = YOLO('yolo11n-obb.pt')
model.train(
    data='webpm_obb8.yaml',
    epochs=10,
    imgsz=640,
    fliplr=0.0,
    batch=1,
    workers=0
)
```

## 📈 개발 히스토리

1. **DOTA8 검증**: 기본 OBB 기능 확인
2. **번호판 데이터 적용**: 8-point 형식 구현
3. **성능 분석**: False alarm/Miss detection 분석
4. **실제 테스트**: 인하대 이미지로 검증

## 🎉 결론

YOLO11 OBB 모델이 번호판 검출에서 뛰어난 성능을 보여주었으며, 
기존 Detection 모델 대비 획기적인 성능 향상을 달성했습니다.