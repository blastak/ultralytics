# YOLOv8-QBB (Quadrilateral Bounding Box) 개발 로그

## 프로젝트 개요
OBB(Oriented Bounding Box)를 기반으로 QBB(Quadrilateral Bounding Box) 검출기를 구현하는 프로젝트입니다.
QBB는 4개의 꼭짓점 좌표를 직접 예측하여 더 정확한 객체 경계를 검출할 수 있습니다.

## 버전 히스토리

### v0.1.0-qbb-baseline (2025-01-XX)
**OBB 복사본 기반 기본 구조 완성**

#### ✅ 완료된 작업
1. **기본 QBB 모듈 구조 생성**
   - `ultralytics/models/yolo/qbb/` 디렉토리 생성
   - `__init__.py`, `predict.py`, `train.py`, `val.py` 구현
   - OBB 모듈을 복사하여 QBB로 이름 변경

2. **핵심 QBB 클래스 구현**
   - `nn/modules/head.py`: QBB 검출 헤드 클래스 추가
   - `nn/tasks.py`: QBBModel 클래스 구현
   - `utils/loss.py`: v8QBBLoss 클래스 구현 (현재는 OBB와 동일)
   - `utils/metrics.py`: QBBMetrics 클래스 구현
   - `engine/results.py`: QBB 결과 처리 클래스 구현

3. **모델 설정 파일 생성**
   - `cfg/models/v8/yolov8-qbb.yaml`
   - `cfg/models/11/yolo11-qbb.yaml` 
   - `cfg/models/12/yolo12-qbb.yaml`

4. **QBB 모듈 통합 및 초기화**
   - `cfg/__init__.py`: "qbb" 작업 추가
   - `models/yolo/__init__.py`: qbb 모듈 import 추가
   - `models/yolo/model.py`: QBB 작업 지원 추가
   - `nn/tasks.py`: QBB 작업 인식 로직 추가

5. **데이터 처리 파이프라인 수정**
   - `data/dataset.py`: `use_qbb` 플래그 추가, `return_qbb` 파라미터 추가
   - `data/augment.py`: QBB 형식 데이터 변환 지원 추가
   - `nn/tasks.py`: QBB 모델의 stride 계산 및 parse_model 지원

6. **테스트 및 검증**
   - `train_entry.py`: QBB 모델 학습 엔트리 포인트 구현
   - DOTA8 데이터셋으로 학습 성공 확인

#### 📝 주요 구현 파일
```
ultralytics/
├── models/yolo/qbb/
│   ├── __init__.py          # QBBPredictor, QBBTrainer, QBBValidator
│   ├── predict.py           # QBB 추론 파이프라인
│   ├── train.py             # QBB 훈련 파이프라인
│   └── val.py               # QBB 검증 파이프라인
├── cfg/models/
│   ├── v8/yolov8-qbb.yaml   # YOLOv8 QBB 모델 설정
│   ├── 11/yolo11-qbb.yaml   # YOLO11 QBB 모델 설정
│   └── 12/yolo12-qbb.yaml   # YOLO12 QBB 모델 설정
├── nn/
│   ├── modules/head.py      # QBB 검출 헤드 (라인 243-280)
│   └── tasks.py             # QBBModel 클래스 (라인 458-476)
├── utils/
│   ├── loss.py              # v8QBBLoss 클래스 (라인 736-849)
│   └── metrics.py           # QBBMetrics 클래스
├── engine/results.py        # QBB 결과 처리 클래스
├── data/
│   ├── dataset.py           # QBB 데이터셋 지원
│   └── augment.py           # QBB 데이터 변환
└── train_entry.py           # QBB 학습 테스트 스크립트
```

#### 💡 현재 상태
- QBB는 현재 OBB와 동일한 로직으로 동작 (5개 값: x, y, width, height, angle)
- 모든 파이프라인이 정상적으로 작동하며 학습 가능
- OBB 데이터셋(DOTA8)과 호환

---

## 🚀 향후 개발 계획

### Phase 1: QBB 핵심 로직 구현
#### 1.1 QBB 전용 데이터로더 구현
- **목표**: 4개 꼭짓점 좌표 (8개 값: x1,y1,x2,y2,x3,y3,x4,y4) 처리
- **파일**: `data/dataset.py`, `data/augment.py`
- **작업**: 
  - QBB 형식 라벨 파싱 로직 구현
  - 데이터 증강 시 4개 꼭짓점 변환 로직
  - 정규화 및 좌표 변환 함수

#### 1.2 QBB 전용 손실 함수 구현  
- **목표**: Quadrilateral 형태에 맞는 손실 계산
- **파일**: `utils/loss.py`
- **작업**:
  - 4개 꼭짓점 기반 IoU 계산
  - Quadrilateral-specific 손실 함수 설계
  - 회전 불변성을 고려한 손실 계산

#### 1.3 QBB 전용 모델 헤드 구현
- **목표**: 4개 꼭짓점을 직접 예측하는 헤드 구조
- **파일**: `nn/modules/head.py`
- **작업**:
  - 8개 출력값 (x1,y1,x2,y2,x3,y3,x4,y4) 예측 헤드
  - 꼭짓점 순서 일관성 보장 로직
  - 기존 anchor 기반 시스템과의 호환성

### Phase 2: QBB 데이터셋 지원
#### 2.1 QBB 형식 데이터셋 포맷 정의
- QBB 라벨 형식 표준화
- 기존 polygon/segmentation 데이터의 QBB 변환
- 데이터셋 검증 도구 구현

#### 2.2 QBB 전용 평가 메트릭
- **파일**: `utils/metrics.py`
- **작업**:
  - Quadrilateral IoU 계산
  - QBB mAP 평가 지표
  - 시각화 도구 개선

### Phase 3: 성능 최적화 및 확장
#### 3.1 QBB 후처리 최적화
- NMS 알고리즘 QBB 대응
- 추론 속도 최적화
- 메모리 사용량 최적화

#### 3.2 QBB 내보내기 및 배포
- ONNX, TensorRT 등 모델 내보내기
- QBB 결과 시각화 개선
- 문서화 및 예제 코드

---

## 📋 작업 우선순위

### 🔴 High Priority
1. QBB 전용 모델 헤드 구현 (8개 출력값)
2. QBB 전용 손실 함수 구현
3. QBB 데이터로더 4개 꼭짓점 처리

### 🟡 Medium Priority  
4. QBB 형식 데이터셋 변환 도구
5. QBB 전용 평가 메트릭 구현
6. QBB 후처리 및 NMS 최적화

### 🟢 Low Priority
7. 성능 최적화 및 속도 개선
8. 문서화 및 예제 코드 작성
9. 모델 내보내기 기능 구현

---

## 🐛 알려진 이슈 및 해결책

### 현재 이슈 없음
- 모든 기본 파이프라인이 정상 작동
- DOTA8 데이터셋으로 학습 성공 확인

---

*마지막 업데이트: 2025-01-XX*
*작성자: Claude Code Assistant*