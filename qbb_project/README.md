# YOLO11 QBB (Quadrilateral Bounding Box) 번호판 검출 프로젝트

이 프로젝트는 YOLO11 QBB 모델을 사용하여 번호판 검출을 위한 연구 및 개발 결과물입니다.

## 📁 프로젝트 구조

```
qbb_project/
├── README.md              # 프로젝트 설명서
├── datasets/               # 학습 데이터셋들
│   ├── webpm_qbb8_dataset/    # 최종 학습용 데이터셋 (8-point QBB)
│   ├── fixed_qbb_dataset/     # 수정된 QBB 데이터셋
│   └── simple_qbb_dataset/    # 단순 테스트 데이터셋
├── visualizations/         # QBB 시각화 결과들 (학습 후 생성)
├── train_entry_qbb.py     # 메인 학습 스크립트
├── analyze_qbb_final.py   # 성능 분석 스크립트
├── test_inha_qbb.py       # 인하대 테스트 스크립트
├── test_random_qbb.py     # 무작위 테스트 스크립트
├── test_dota8_qbb.py      # DOTA8 테스트 스크립트
├── simple_qbb_training.py # 단순 학습 스크립트
├── fix_simple_qbb.py      # 데이터셋 수정 스크립트
└── test_qbb_simple.py     # QBB 간단 테스트
```

## 🎯 프로젝트 개요

### QBB 개발 목표
- **1단계**: QBB를 OBB와 동일하게 작동하도록 구현 (현재 완료)
- **2단계**: 향후 실제 8-point QBB 알고리즘 구현 예정
- **현재 상태**: QBB = OBB 복사본으로 완벽 동작

### 학습 데이터
- **소스**: 웹플레이트마니아 번호판 데이터
- **형식**: 8-point QBB (실제로는 OBB 형식)
- **데이터셋**: webpm_qbb8_dataset 활용

## 🚀 사용법

### 1. 간단한 QBB 테스트
```bash
python test_qbb_simple.py
```

### 2. QBB 학습 실행
```bash
python simple_qbb_training.py
```

### 3. 메인 학습 스크립트
```bash
python train_entry_qbb.py
```

### 4. 성능 분석
```bash
python analyze_qbb_final.py
```

### 5. 테스트 실행
```bash
# 인하대 테스트 이미지
python test_inha_qbb.py

# 무작위 테스트
python test_random_qbb.py
```

## 📊 QBB 모델 특성

### QBB vs OBB 동일성
- **파라미터 수**: 동일 (2,695,747개)
- **모델 구조**: 100% 동일
- **추론 결과**: 동일한 형식의 bounding box 출력
- **성능**: OBB와 동일한 성능 기대

### QBB 출력 형식
- **형식**: xyxyxyxy (8-point coordinates)
- **실제**: OBB 알고리즘으로 계산된 4개 모서리 좌표
- **호환성**: OBB 시각화 도구와 완전 호환

## 🔧 기술적 특징

### QBB 구현 방식
- **기반**: OBB (Oriented Bounding Box) 알고리즘
- **인터페이스**: QBB라는 이름으로 래핑
- **호환성**: OBB와 100% 호환

### 데이터 형식
- **입력**: LabelMe JSON (polygon)
- **변환**: 8-point QBB 좌표 (실제로는 OBB)
- **정규화**: 0-1 범위로 정규화

### 학습 설정
```python
model = YOLO('yolo11n-qbb.yaml')  # QBB 모델 사용
model.train(
    data='webpm_qbb8.yaml',
    epochs=10,
    imgsz=640,
    batch=4,
    workers=1
)
```

## 📈 QBB 개발 히스토리

1. **Phase 1-4**: QBB 기본 구조 및 시스템 통합 완료
2. **QBB = OBB**: QBB를 OBB 복사본으로 완벽 구현
3. **Task 자동 인식**: QBB task 자동 인식 기능 추가
4. **프로젝트 구성**: qbb_project 폴더 생성 및 정리

## 🎯 현재 상태 및 다음 단계

### ✅ 완료된 작업
- QBB 모델 클래스 구현 (OBB 기반)
- QBB 시스템 통합 완료
- QBB 프로젝트 구조 정리

### 🚀 다음 단계
1. **QBB 학습**: 실제 데이터로 QBB 모델 학습
2. **성능 검증**: OBB vs QBB 동일성 확인
3. **시각화**: QBB 결과 시각화 및 분석
4. **향후 계획**: 실제 8-point QBB 알고리즘 구현 준비