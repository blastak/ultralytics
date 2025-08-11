# QBB (Quadrilateral Bounding Box) 개발 로드맵

## 프로젝트 개요
OBB (Oriented Bounding Box) 모델을 기반으로 QBB (Quadrilateral Bounding Box) 모델을 개발합니다.
QBB는 4개의 꼭짓점(xyxyxyxy)을 사용하여 더 유연한 사각형 경계 상자를 표현할 수 있는 모델입니다.

**중요**: YOLOv8 기반으로만 개발 (v11, v12는 사용하지 않음)

## 개발 단계

### 1단계: OBB 구조 복제 및 QBB 생성 ✅
- [x] OBB 관련 모든 파일 검색 및 분석
- [x] 각 파일에서 OBB 관련 코드를 복사하여 QBB 버전 생성
  - [x] 폴더 구조: `ultralytics/models/yolo/qbb/` 생성
  - [x] 파일명: `train.py`, `predict.py`, `val.py`, `__init__.py`
  - [x] 클래스명: QBBTrainer, QBBPredictor, QBBValidator, QBBMetrics
  - [x] 함수명: probiou_quad, batch_probiou_quad
  - [x] 변수명: is_qbb, use_qbb, return_qbb
  - [x] 설정 플래그: TASKS, TASK2DATA, TASK2MODEL에 "qbb" 추가
  - [x] 주석: OBB/QBB 동등 지원
- [x] 기본적인 QBB 구조 동작 확인

### 2단계: 성능 검증 및 코드 정리 🔄
- [x] DOTA8.yaml 데이터셋 준비
- [x] OBB 모델로 학습 실행 (20 에포크 완료)
- [x] QBB 모델 파일들의 OBB 참조 제거 및 수정
- [x] OBB/QBB 단어 개수 패리티 작업 (총 OBB:248, QBB:243개)
- [x] 주요 파일들 OBB/QBB 지원 완전 통합
- [ ] QBB 모델로 학습 실행 (OBB와 동일한 로직)
- [ ] 두 모델의 성능 비교 및 유사성 확인

### 3단계: QBB 전용 데이터로더 구현
- [ ] xyxyxyxy (8개 좌표) 형식 지원
- [ ] xywhr 변환 로직 제거/수정
- [ ] QBB용 loss function 설계 및 구현
  - IoU 계산 방식 수정
  - 회귀 손실 조정

### 4단계: QBB 모델 최적화
- [ ] QBB 모델 아키텍처 수정
- [ ] Head 클래스 QBB용으로 커스터마이징
- [ ] 후처리(NMS 등) 로직 수정

### 5단계: 추가 작업 (TBD)
- 성능 최적화
- 벤치마크
- 문서화

## 주요 파일 위치 (예상)
```
ultralytics/
├── cfg/
│   ├── models/       # 모델 설정 YAML
│   └── datasets/     # 데이터셋 설정
├── models/
│   └── yolo/
│       ├── obb/      # OBB 구현 → QBB 복제 대상
│       └── qbb/      # QBB 구현 (새로 생성)
├── data/             # 데이터로더
├── nn/               # 신경망 모듈
└── utils/            # 유틸리티 함수
```

## 개발 노트
- OBB는 회전된 사각형 (xywhr: 중심점, 너비, 높이, 회전각)
- QBB는 자유로운 사각형 (xyxyxyxy: 4개 꼭짓점의 x,y 좌표)
- 주요 차이점: 더 유연한 형태 표현 가능, 계산 복잡도 증가

## 진행 상태
- ✅ 완료: 1단계 - OBB 구조 분석 및 복제 완료
- 🔄 진행 중: 2단계 - 성능 검증 및 코드 정리 (거의 완료)
- ⏳ 대기: 3-4단계

## 최근 작업 완료 사항
- QBB 모델 파일들에서 OBB 참조 완전 제거
- batch_probiou → batch_probiou_quad 전환
- obb=qbb → qbb=qbb 매개변수 수정
- OBB/QBB 단어 개수 패리티 작업으로 일관성 확보
- 전체 코드베이스에서 OBB/QBB 동등 지원 구현

---
*마지막 업데이트: 2025-08-11 (코드 정리 및 패리티 작업 완료)*