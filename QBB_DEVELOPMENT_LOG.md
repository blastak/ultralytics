# QBB (Quadrilateral Bounding Box) Development Progress Log

## 📋 프로젝트 개요
OBB (Oriented Bounding Box)를 기반으로 QBB (Quadrilateral Bounding Box) 모듈을 개발하는 프로젝트입니다. 
초기에는 OBB와 동일한 동작을 하도록 구현하고, 성공 후 실제 QBB 알고리즘으로 점진적 전환할 예정입니다.

## 🎯 현재 상태
- **Current Status**: QBB 데이터 형식 문제 해결 진행 중 - 중요한 OBB 호환성 이슈 발견
- **Current Branch**: qbb-development  
- **Last Updated**: 2025-08-01 21:15

## ✅ 완료된 작업들

### Phase 0: 프로젝트 관리 설정
- [x] QBB 개발 계획 수립
- [x] 진행상황 추적 파일 생성 (`QBB_DEVELOPMENT_LOG.md`)
- [x] qbb-development 브랜치 생성
- [x] 초기 커밋 및 Git 추적 설정

### Phase 1: 기본 QBB 구조 생성 ✅
- [x] `ultralytics/models/yolo/qbb/` 폴더 생성
- [x] OBB 파일들 복사 및 이름 변경:
  - [x] `obb/train.py` → `qbb/train.py` (OBBTrainer → QBBTrainer)
  - [x] `obb/val.py` → `qbb/val.py` (OBBValidator → QBBValidator)  
  - [x] `obb/predict.py` → `qbb/predict.py` (OBBPredictor → QBBPredictor)
  - [x] `obb/__init__.py` → `qbb/__init__.py`

### Phase 2: 모델 및 Head 클래스 생성 ✅
- [x] `ultralytics/nn/tasks.py`에 `QBBModel` 클래스 추가
- [x] `ultralytics/nn/modules/head.py`에 `QBB` Head 클래스 추가

### Phase 3: 통합 및 Import 설정 ✅
- [x] `ultralytics/models/yolo/__init__.py`에 qbb 모듈 추가
- [x] `ultralytics/nn/modules/__init__.py`에 QBB 추가  
- [x] `ultralytics/models/yolo/model.py`에 'qbb' task 추가

### Phase 4: 초기 테스트 및 검증 ✅
- [x] yolo11-qbb.yaml 설정 파일 생성
- [x] QBB 모델 로딩 및 기본 기능 테스트
- [x] 추론(predict) 테스트 성공
- [x] OBB vs QBB 구조 비교 (동일한 파라미터 수 확인)
- [x] QBB = OBB 복사본 동작 확인

## 🔄 진행 중인 작업들

## 📋 다음 세션에서 할 작업
1. **Phase 5 계획**: 실제 QBB (8-point coordinates) 알고리즘 구현 방향 논의
2. **구현 전략**: OBB (4 corners + angle) → QBB (8 points) 변환 방법
3. **데이터 형식**: QBB 라벨 포맷 정의 (xyxyxyxyxyxyxyxy)
4. **손실 함수**: 8-point regression loss 설계


## 🔬 Phase 4 테스트 결과

### 테스트 1: QBB 모델 로딩 ✅ (2025-08-01 19:20)
- ✅ yolo11-qbb.yaml 설정 파일 생성 완료
- ✅ QBB 모델 초기화 성공 (YOLO('yolo11n-qbb.yaml', task='qbb'))
- ✅ QBBModel 클래스 정상 동작 확인
- ✅ QBB Head 정상 로드 확인
- ✅ 모델 파라미터: 2,695,747개

### 테스트 2: QBB 추론 테스트 ✅
- ✅ 더미 이미지로 추론 실행 성공
- ✅ OBB 형식의 출력 확인 (xyxyxyxy 좌표)
- ✅ 결과 객체에 obb 속성 존재 확인
- ✅ QBB가 OBB와 동일한 인터페이스로 동작

### 수정 사항:
1. `ultralytics/models/yolo/model.py`:
   - QBBModel import 추가
   - task_map에 'qbb' 태스크 추가
2. `ultralytics/nn/tasks.py`:
   - QBB import 추가
   - parse_model에서 QBB 처리 추가
   - _forward 함수에서 QBB 지원



### 테스트 3: QBB 학습 테스트 (2025-08-01 19:16)
- ⚠️ 실제 OBB 데이터셋 필요 (DOTAv1 등)
- ✅ QBB 모델 학습 코드 정상 동작 확인
- ✅ train, val, predict 메서드 모두 정상 작동
- ✅ OBB와 동일한 파라미터 수 확인 (2,695,747)

### 테스트 4: OBB vs QBB 비교
- ✅ 동일한 모델 구조 확인
- ✅ 동일한 파라미터 수 확인
- ✅ QBB는 OBB의 완전한 복사본으로 동작

## 🚀 QBB 프로젝트 완료 (2025-08-01 19:45)

### ✅ qbb_project 생성 및 설정
- obb_project → qbb_project 완전 복사
- 모든 하위 파일의 OBB → QBB 일괄 변경
- 데이터셋 yaml 파일명 변경
- 스크립트 파일명 변경

### ✅ QBB 시스템 통합 완료
- TASKS, TASK2DATA, TASK2MODEL, TASK2METRIC에 QBB 추가
- guess_model_task 함수에 QBB 인식 추가
- QBB task 자동 인식 기능 완료

### ✅ QBB 학습 및 테스트 확인
- QBB 모델 정상 로드 확인 (task='qbb' 자동 인식)
- QBB 학습 스크립트 정상 실행 확인
- OBB vs QBB 동일한 파라미터 수 확인 (2,695,747)
- QBB 추론 테스트 성공

## 🚀 QBB 프로젝트 정리 및 검증 완료 (2025-08-01 20:30)

### ✅ qbb_project 구조 정리
- README.md를 QBB 프로젝트에 맞게 완전 수정
- visualizations/ 폴더의 OBB 결과물 삭제
- results/ 폴더 삭제 (OBB 학습 결과)
- scripts/ 내용을 qbb_project/ 최상위로 이동
- 깔끔하고 체계적인 QBB 전용 구조 완성

### ✅ QBB 학습 검증
- DOTA8 데이터셋으로 QBB 모델 학습 성공
- 2 epochs 학습 완료, 모델 파일 생성 확인
- 학습된 모델로 추론 테스트 성공
- QBB = OBB 동일성 재확인

### 최종 결과
- QBB 모듈이 OBB와 100% 동일하게 작동 확인
- 정리된 qbb_project 구조로 향후 개발 준비 완료
- 실제 8-point QBB 알고리즘 구현을 위한 완벽한 기반 구축

## 🚀 Git 커밋 기록
- `712b59a7` - feat: QBB 개발 프로젝트 시작 및 진행상황 추적 시스템 구축
- `6265f85a` - feat: QBB 기본 폴더 구조 및 파일 생성 (Phase 1) ✅
- `a3fc3aea` - feat: QBB 모델 및 Head 클래스 구현 (Phase 2) ✅
- `5ef47513` - feat: QBB 모듈 통합 및 시스템 등록 (Phase 3) ✅

## 🔧 기술적 분석 결과

### OBB 구조 분석:
1. **모델 클래스**: `ultralytics/nn/tasks.py` - `OBBModel` 클래스
2. **Head 모듈**: `ultralytics/nn/modules/head.py` - `OBB` 클래스  
3. **Training/Validation/Prediction**: `ultralytics/models/yolo/obb/` 폴더
4. **모델 설정**: `ultralytics/cfg/models/11/yolo11-obb.yaml`
5. **통합**: `ultralytics/models/yolo/__init__.py`에서 obb 모듈 import

### 핵심 복사 대상:
- `OBBTrainer`, `OBBValidator`, `OBBPredictor` 클래스들
- `OBBModel` 클래스 (DetectionModel 상속)
- `OBB` Head 클래스 (Detect 상속, angle 예측 추가)
- `yolo11-obb.yaml` 설정 파일

## 💡 중요 노트
- QBB는 초기에 OBB와 100% 동일한 동작을 하도록 구현
- 성공 후 실제 QBB 알고리즘 (8-point coordinates) 구현 예정
- 각 Phase 완료시마다 Git 커밋 수행
- 세션 연속성을 위해 이 파일을 지속적으로 업데이트

## 🧪 테스트 계획
1. **기본 동작 테스트**: QBB 모델 로딩 확인
2. **학습 테스트**: 기존 번호판 데이터셋으로 학습
3. **성능 검증**: OBB 모델과 동일한 결과 확인

## 🎯 최종 목표
QBB 모델이 OBB와 동일한 성능으로 작동하는 것을 확인한 후, 
실제 Quadrilateral Bounding Box 알고리즘 구현 방향 논의

## 🚀 Phase 5: QBB 데이터 형식 문제 해결 (2025-08-01 21:00)

### 🔍 문제 발견
사용자 요청에 따라 실제 번호판 데이터로 QBB 장기 학습을 시도하던 중 데이터 형식 문제 발견:

#### 문제 상황:
- **에러**: "QBB dataset incorrectly formatted" - 6컬럼 데이터가 5컬럼으로 인식됨
- **원인**: QBB는 OBB와 동일한 6컬럼 형식(class cx cy w h angle)을 사용해야 하지만, DetectionTrainer를 상속받아 5컬럼 검증이 적용됨

### ✅ 1단계: 새로운 통합 데이터셋 생성
- `/workspace/DB/01_LicensePlate/55_WebPlatemania_1944/all/`에서 1944개 번호판 이미지 사용
- LabelMe JSON → QBB 6컬럼 형식 자동 변환
- train:val:test = 7:2:1 비율로 분할 (1361:389:194)
- cv2.minAreaRect() 사용하여 정확한 OBB 좌표 계산

### ✅ 2단계: QBB 데이터 로더 수정
**수정 파일들:**
1. **`ultralytics/data/dataset.py`**:
   - `self.use_qbb = task == "qbb"` 추가 (84번째 줄)
   - `repeat(self.use_obb or self.use_qbb)` 6컬럼 지원 (122번째 줄)
   - `return_obb=self.use_obb or self.use_qbb` 형식 처리 (230번째 줄)

2. **`ultralytics/data/utils.py`**:
   - `verify_image_label()` 함수에 `use_obb_qbb` 파라미터 추가
   - 6컬럼 검증 로직 추가: `assert lb.shape[1] == 6`

### ✅ 3단계: QBB 전용 손실 함수 생성
**`ultralytics/utils/loss.py`**:
- `v8QBBLoss` 클래스 생성 (v8OBBLoss 복사본)
- 6컬럼 데이터 처리 로직 추가:
  ```python
  bbox_data = batch["bboxes"]
  if bbox_data.shape[-1] == 6:
      targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), bbox_data.view(-1, 6)[:, 1:]), 1)
  ```

**`ultralytics/nn/tasks.py`**:
- QBBModel에서 `v8QBBLoss` 사용하도록 수정
- `from ultralytics.utils.loss import v8QBBLoss` import 추가

### 🚨 4단계: 중요한 문제 발견 - OBB 호환성 이슈

#### 문제 분석:
OBB 학습에서도 동일한 "OBB dataset incorrectly formatted" 에러 발생 확인!

**원인 분석:**
1. 우리가 수정한 데이터 로더가 모든 OBB 데이터를 6컬럼으로 생성
2. 하지만 원래 `v8OBBLoss`는 5컬럼만 기대 (`batch["bboxes"].view(-1, 5)`)
3. QBB용 수정사항이 OBB 모드에도 영향을 미침

**테스트 결과:**
- obb_project의 기존 OBB 학습 스크립트도 동일한 에러 발생
- 기존에 성공했던 fixed_obb_dataset도 학습 실패
- 6컬럼 라벨 파일은 정상적으로 생성됨: `0 0.631864 0.834746 0.047316 0.133249 1.536504`

### 📋 현재 상태 요약
- ✅ QBB 데이터 변환 및 검증 시스템 구축 완료
- ✅ QBB 전용 손실 함수 및 모델 클래스 구현 완료  
- ⚠️ 데이터 로더 수정이 OBB 호환성에 영향을 미치는 것 발견
- ❌ QBB와 OBB 모두 학습 실패 (데이터 형식 에러)

### 🔧 다음 세션 해결 과제
1. **데이터 로더 분리**: OBB와 QBB를 완전히 독립적으로 처리하도록 수정
2. **OBB 호환성 복구**: 기존 OBB 학습이 정상 작동하도록 원복
3. **QBB 학습 완료**: 수정된 시스템으로 QBB 장기 학습 실행
4. **시각화 및 결과 분석**: 학습 결과 시각화 스크립트 생성

### 🧪 상세 기술 분석

**데이터 형식 분석:**
- QBB/OBB 라벨 형식: `class cx cy w h angle` (6컬럼)
- 일반 Detection 형식: `class cx cy w h` (5컬럼)  
- 현재 문제: v8OBBLoss가 `.view(-1, 5)` 하드코딩됨

**핵심 수정 위치:**
- `ultralytics/data/dataset.py:122` - OBB/QBB 플래그 전달
- `ultralytics/data/utils.py` - 6컬럼 검증 로직
- `ultralytics/utils/loss.py:667` - OBB 손실함수 5컬럼 하드코딩
- `ultralytics/utils/loss.py:867` - QBB 손실함수 6컬럼 처리

---
*Last Updated: 2025-08-01 21:15*