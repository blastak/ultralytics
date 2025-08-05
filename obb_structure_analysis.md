# YOLOv8-OBB 구조 분석 - QBB 구현을 위한 참고 자료

## 개요
이 문서는 Ultralytics YOLOv8-OBB (Oriented Bounding Box) 구조를 분석하여 QBB (Quadrilateral Bounding Box) 구현을 위한 참고 자료로 작성되었습니다.

## 1. 핵심 OBB 모듈 구조

### 1.1 메인 OBB 모듈 디렉토리
```
ultralytics/models/yolo/obb/
├── __init__.py          # OBBPredictor, OBBTrainer, OBBValidator 내보내기
├── predict.py           # OBBPredictor 클래스 - 추론 파이프라인
├── train.py             # OBBTrainer 클래스 - 훈련 파이프라인
└── val.py               # OBBValidator 클래스 - 검증 파이프라인
```

### 1.2 모델 설정 파일
```
ultralytics/cfg/models/
├── 11/yolo11-obb.yaml   # YOLO11 OBB 모델 아키텍처
├── 12/yolo12-obb.yaml   # YOLO12 OBB 모델 아키텍처
└── v8/yolov8-obb.yaml   # YOLOv8 OBB 모델 아키텍처
```

### 1.3 데이터셋 설정 파일
```
ultralytics/cfg/datasets/
├── dota8.yaml                    # DOTA8 데이터셋 설정
├── dota8-multispectral.yaml      # DOTA8 다중스펙트럼 설정
├── DOTAv1.yaml                   # DOTAv1 데이터셋 설정
└── DOTAv1.5.yaml                 # DOTAv1.5 데이터셋 설정
```

## 2. 핵심 클래스 및 함수

### 2.1 신경망 컴포넌트
- **파일**: `ultralytics/nn/modules/head.py:213`
  - **클래스**: `OBB` - Detect를 확장한 방향성 바운딩 박스 검출 헤드
  
- **파일**: `ultralytics/nn/tasks.py:436`
  - **클래스**: `OBBModel` - OBB 작업을 위한 DetectionModel 확장

### 2.2 손실 함수 및 메트릭
- **파일**: `ultralytics/utils/loss.py:620`
  - **클래스**: `v8OBBLoss` - OBB 전용 손실 계산
  
- **파일**: `ultralytics/utils/metrics.py`
  - **클래스**: `OBBMetrics` (라인 1257) - OBB 평가 메트릭
  - **함수**: `probiou()` (라인 198) - 방향성 박스 Probabilistic IoU 계산
  - **함수**: `batch_probiou()` (라인 242) - 배치 방향성 IoU 계산

### 2.3 결과 처리 및 데이터 핸들링
- **파일**: `ultralytics/engine/results.py:1619`
  - **클래스**: `OBB` - 방향성 바운딩 박스 저장 및 조작
  
- **파일**: `ultralytics/data/dataset.py:84`
  - **변수**: `use_obb` 플래그 - OBB 데이터셋 처리 로직
  
- **파일**: `ultralytics/data/converter.py:426`
  - **함수**: `convert_dota_to_yolo_obb()` - DOTA to YOLO OBB 형식 변환

## 3. QBB 구현을 위한 복사 대상 파일들

### 3.1 최우선 복사 대상 (필수)
1. **`ultralytics/models/yolo/obb/`** 전체 디렉토리
   - `__init__.py` → `qbb/__init__.py`
   - `predict.py` → `qbb/predict.py`
   - `train.py` → `qbb/train.py` 
   - `val.py` → `qbb/val.py`

2. **모델 설정 파일**
   - `ultralytics/cfg/models/v8/yolov8-obb.yaml` → `yolov8-qbb.yaml`

3. **핵심 클래스들**
   - `ultralytics/nn/modules/head.py` 내 `OBB` 클래스 → `QBB` 클래스
   - `ultralytics/nn/tasks.py` 내 `OBBModel` 클래스 → `QBBModel` 클래스
   - `ultralytics/utils/loss.py` 내 `v8OBBLoss` 클래스 → `v8QBBLoss` 클래스

### 3.2 두번째 우선순위 복사 대상
1. **메트릭 및 유틸리티**
   - `ultralytics/utils/metrics.py` 내 `OBBMetrics` 클래스 → `QBBMetrics` 클래스
   - `ultralytics/engine/results.py` 내 `OBB` 클래스 → `QBB` 클래스

2. **설정 및 초기화**
   - `ultralytics/cfg/__init__.py` 내 TASKS에 "qbb" 추가
   - `ultralytics/models/yolo/__init__.py` 내 qbb 모듈 import 추가

### 3.3 세번째 우선순위 (선택적)
1. **데이터 처리**
   - `ultralytics/data/dataset.py` 내 `use_qbb` 플래그 추가
   - `ultralytics/data/augment.py` 내 `return_qbb` 파라미터 추가

2. **엔진 컴포넌트**
   - `ultralytics/engine/trainer.py` 내 QBB 배치 크기 처리
   - `ultralytics/engine/exporter.py` 내 QBB 모델 내보내기 기능

## 4. 구현 전략

### 4.1 1단계: 기본 구조 복사
1. `ultralytics/models/yolo/obb/` → `ultralytics/models/yolo/qbb/` 복사
2. 모든 파일 내 "obb" → "qbb", "OBB" → "QBB" 문자열 치환
3. 클래스명 변경: `OBBPredictor` → `QBBPredictor` 등

### 4.2 2단계: 핵심 컴포넌트 복사 및 수정
1. `head.py` 내 `OBB` 클래스 → `QBB` 클래스 복사
2. `tasks.py` 내 `OBBModel` 클래스 → `QBBModel` 클래스 복사
3. `loss.py` 내 `v8OBBLoss` 클래스 → `v8QBBLoss` 클래스 복사

### 4.3 3단계: 설정 및 통합
1. 모델 설정 파일 생성: `yolov8-qbb.yaml`
2. `cfg/__init__.py`에 "qbb" 작업 추가
3. 필요한 import 문 추가

### 4.4 4단계: 테스트 및 검증
1. QBB 모델 로딩 테스트
2. 기본 훈련 파이프라인 테스트
3. 추론 파이프라인 테스트

## 5. 주요 파일별 수정 포인트

### 5.1 ultralytics/models/yolo/qbb/train.py
```python
# 수정할 주요 부분
from ultralytics.utils.loss import v8QBBLoss  # v8OBBLoss에서 변경
class QBBTrainer(DetectionTrainer):  # OBBTrainer에서 변경
    def build_loss(self, ...):
        return v8QBBLoss(...)  # v8OBBLoss에서 변경
```

### 5.2 ultralytics/nn/modules/head.py
```python
# QBB 클래스 추가 (OBB 클래스 복사 후 수정)
class QBB(Detect):
    # OBB 클래스와 동일한 구조를 유지하되 클래스명만 변경
```

### 5.3 ultralytics/nn/tasks.py
```python
# QBBModel 클래스 추가
class QBBModel(DetectionModel):
    # task = 'qbb'로 설정
```

## 6. 디렉토리 구조 요약

### 복사 후 예상 QBB 구조:
```
ultralytics/
├── models/yolo/qbb/           # 새로 생성할 QBB 모듈
│   ├── __init__.py
│   ├── predict.py
│   ├── train.py
│   └── val.py
├── cfg/models/v8/
│   └── yolov8-qbb.yaml        # 새로 생성할 QBB 모델 설정
├── nn/modules/head.py         # QBB 클래스 추가
├── nn/tasks.py                # QBBModel 클래스 추가
├── utils/loss.py              # v8QBBLoss 클래스 추가
├── utils/metrics.py           # QBBMetrics 클래스 추가 (선택적)
└── engine/results.py          # QBB 결과 클래스 추가 (선택적)
```

이 구조를 기반으로 단계별로 OBB를 복사하여 QBB 모듈을 구축하면, 기존 OBB와 동일하게 작동하는 QBB 검출기를 만들 수 있습니다.