# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 작업 규칙

1. **한글 답변**: 모든 답변과 설명은 한글로 작성합니다.
2. **코드 작성 규칙**: 
   - Python 코드는 영어로 작성
   - 코드 내 주석은 한글로 상세히 작성
3. **세션 시작**: Claude Code 세션이 시작될 때 항상 이 CLAUDE.md 파일을 먼저 읽어주세요.
4. **Auto Compacting 후**: 메모리 정리(auto compacting) 이후에는 이전에 하던 작업을 리마인드해주세요.
5. **날짜 시간 기록**: 작업내역을 업데이트할 때는 항상 현재 날짜와 시간을 조사하여 정확히 기입합니다.

## 📅 특별 조건 (2025-08-12 ~ 2025-08-31)

**⚠️ 중요: 이 조건은 2025년 8월 12일부터 8월 31일까지 적용됩니다**

5. **이번 주 특별 작업 모드**: 
   - 사용자가 QBB 구현을 직접 수정하려고 합니다
   - Claude는 **분석과 가이드만** 제공하고 **직접 코드 수정은 하지 않습니다**
   - 파일 위치 안내, 코드 구조 설명, 수정 방향 제시에 집중
   - 사용자가 명시적으로 요청하는 경우에만 코드 예시 제공
   - **Read, Grep, LS 도구는 자유롭게 사용** (분석 목적)
   - **Edit, Write, MultiEdit 도구는 사용 금지** (사용자가 직접 수정)
   - **예외: Git 관련 명령어 (commit, push, pull 등)는 허용**

## 프로젝트 개요

Ultralytics YOLO는 YOLO (You Only Look Once) 모델 패밀리를 구현한 최첨단 컴퓨터 비전 프레임워크입니다. 객체 감지, 인스턴스 분할, 포즈 추정, 분류, 회전된 경계 상자(OBB), 다중 객체 추적을 위한 통합 API를 제공합니다.

### ✨ 추가 개발 기능: QBB (Quadrilateral Bounding Box)
본 저장소에는 표준 YOLO 기능 외에 **QBB (Quadrilateral Bounding Box)** 모델이 추가로 개발되었습니다:

- **8좌표 시스템**: `xyxyxyxy` 형식으로 자유로운 사각형 표현
- **OBB 확장**: 회전된 사각형을 넘어 비틀린 사각형까지 감지
- **응용 분야**: 항공/위성 이미지, 문서 OCR, 왜곡된 객체 감지
- **개발 상태**: Phase 5+ 완료, 성능 최적화 진행 중

## 아키텍처 개요

### 핵심 구성 요소

1. **모델** (`ultralytics/models/`)
   - **YOLO**: 주요 탐지 모델 계열 (v3, v5, v8, v9, v10, v11, v12)
   - **SAM/FastSAM**: 고급 분할을 위한 Segment Anything 모델
   - **RT-DETR**: 실시간 Detection Transformer
   - **YOLO-World**: 개방형 어휘 탐지 모델
   - **NAS**: Neural Architecture Search 모델
   - 각 모델은 작업별 모듈 보유: detect, segment, classify, pose, obb, **qbb**

2. **엔진** (`ultralytics/engine/`)
   - **Model**: 모든 작업을 위한 통합 API를 제공하는 기본 클래스
   - **Trainer**: 분산 학습 지원을 포함한 모델 학습 처리
   - **Validator**: 모델 검증 및 메트릭 계산
   - **Predictor**: 스트리밍 지원을 포함한 추론 엔진
   - **Exporter**: 다중 형식 모델 내보내기 (ONNX, TensorRT, CoreML 등)
   - **Results**: 모든 작업에 대한 표준화된 결과 객체

3. **신경망 구성 요소** (`ultralytics/nn/`)
   - **tasks.py**: 모델 아키텍처 정의 및 로딩
   - **modules/**: 빌딩 블록 (conv, blocks, heads, transformers)
     - **head.py**: Detection, OBB, **QBB** Head 클래스들
   - **autobackend.py**: 여러 프레임워크를 위한 통합 추론 백엔드

4. **데이터 파이프라인** (`ultralytics/data/`)
   - **dataset.py**: 모든 작업을 위한 핵심 데이터셋 클래스 (QBB 8좌표 처리 포함)
   - **augment.py**: 데이터 증강 파이프라인 (QBB 전용 처리 추가)
   - **loaders.py**: 다양한 데이터 소스 로더 (이미지, 비디오, 스트림)
   - **build.py**: 데이터셋 빌더 및 데이터로더 생성

5. **유틸리티** (`ultralytics/utils/`)
   - **ops.py**: NMS, 좌표 변환 등 (QBB 통합 NMS 포함)
   - **metrics.py**: IoU 계산, 평가 메트릭 (QBB AABB IoU 최적화)
   - **tal.py**: Task Aligned Learning (QBB 벡터 기반 Assigner 포함)
   - **loss.py**: Loss 함수들 (QBB Loss 시스템 추가)

6. **구성** (`ultralytics/cfg/`)
   - 모델 및 데이터셋을 위한 YAML 구성
   - 기본 학습/추론 매개변수
   - 작업별 구성 (QBB 포함)

7. **솔루션** (`ultralytics/solutions/`)
   - 고수준 애플리케이션: 객체 카운팅, 추적 영역, 히트맵
   - 컴퓨터 비전 유틸리티: 거리 계산, 속도 추정
   - 일반적인 사용 사례를 위한 즉시 사용 가능한 워크플로우

### 작업 유형

프레임워크는 각각 특화된 구현을 가진 여러 컴퓨터 비전 작업을 지원합니다:
- **Detect**: 경계 상자를 사용한 객체 탐지
- **Segment**: 인스턴스 분할
- **Classify**: 이미지 분류
- **Pose**: 키포인트 탐지 및 포즈 추정
- **OBB**: 회전된 객체를 위한 방향이 있는 경계 상자
- **QBB**: 자유로운 형태의 사각형 경계 상자 ⭐ (추가 개발)
- **Track**: 비디오 프레임 전체의 다중 객체 추적

### 주요 설계 패턴

1. **통합 모델 API**: 모든 모델은 일관된 train/val/predict/export 인터페이스를 제공하는 기본 `Model` 클래스에서 상속

2. **작업 다형성**: 작업별 구현(detect/segment/qbb 등)은 모델 구성에 따라 동적으로 로드

3. **AutoBackend**: 다양한 형식(PyTorch, ONNX, TensorRT 등)의 모델 자동 감지 및 로딩

4. **콜백 시스템**: 학습 이벤트, 통합(W&B, ClearML, Comet)을 위한 확장 가능한 콜백

5. **구성 상속**: CLI 또는 Python API를 통해 재정의할 수 있는 기본값이 있는 YAML 구성

## 중요 사항

- 프로젝트는 오픈 소스용 AGPL-3.0 라이선스 사용; 상업적 사용은 Ultralytics Enterprise 라이선스 필요
- CI/CD는 테스트, 포맷팅(Ruff), 문서화를 위해 GitHub Actions 사용
- 테스트는 다양한 종속성 필요; `pip install -e ".[export]"`로 설치
- GPU 테스트는 특수 러너에서 별도로 실행
- 문서는 MkDocs로 빌드되어 docs.ultralytics.com에 배포

### QBB 추가 개발 현황

- **Phase 5+ 완료**: NMS 최적화 및 통합 시스템 구현
- **주요 성취**: 100-1000배 NMS 성능 향상, 벡터 기반 TAL Assigner
- **현재 상태**: 기본 기능 완성, 성능 Gap 해결 진행 중
- **개발 문서**: `QBB_ROADMAP.md` 참조

### 학습 진입점

- **학습 스크립트**: 모든 학습은 `train_entrypoint.py` 파일을 통해 실행
- **통합 진입점**: 새로운 진입점을 만들지 않고 기존 파일 활용
- **설정 관리**: `train_entrypoint.py` 내에서 모든 학습 파라미터 조정
- **예시**:
  ```python
  # QBB 학습 실행
  model = YOLO('yolov8n-qbb.yaml')
  results = model.train(
      name='qbb_experiment',
      data='webpm_obb8.yaml',
      epochs=100,
      batch=16,
      device="0,1"
  )
  ```