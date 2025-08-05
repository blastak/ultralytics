# QBB 데이터로더 개발 작업 로그

## 작업 일자: 2025-08-05

### 작업 목표
QBB(Quadrilateral Bounding Box) 데이터로더를 구현하여 8개 좌표값(4개 꼭짓점)을 직접 처리하도록 수정

### 현재 상황 분석
1. **라벨 형식**: 
   - OBB와 QBB 모두 동일한 형식 사용: `class_id x1 y1 x2 y2 x3 y3 x4 y4` (총 9개 값)
   - 예시: `0 0.5914119720458985 0.5539623119212963 0.8060232374403212 0.4974509910300926 0.8089341057671441 0.5660311098451968 0.6013332790798611 0.6284126564308449`

2. **OBB 현재 처리 과정**:
   - `data/utils.py:205`: xyxyxyxy 형식을 segments로 읽음
   - `data/augment.py:2179`: `xyxyxyxy2xywhr()` 함수로 5개 값(x,y,w,h,angle)으로 변환
   - `utils/ops.py:560-580`: cv2.minAreaRect()를 사용하여 변환

3. **캐시 처리**:
   - 라벨 파일이 `.cache` 파일로 캐시됨
   - 캐시 파일 위치: `/workspace/repo/ultralytics/ultralytics/assets/good_all(obb8)/labels/train.cache`
   - 캐시 삭제 완료 (2025-08-05)

### 계획된 수정 사항

#### 1. 디버깅 환경 설정
- [ ] `NUM_THREADS = 1` 설정 (`/workspace/repo/ultralytics/ultralytics/utils/__init__.py` 라인 45)
- 목적: ThreadPool 비활성화로 디버깅 용이성 확보

#### 2. data/utils.py 
- 수정 불필요
- 이유: OBB와 QBB가 동일한 9개 값 형식을 사용하므로 `verify_image_label`에서 구분 불가
- 현재 코드가 이미 올바르게 segments로 처리 중

#### 3. data/augment.py 수정 필요 (핵심)
**현재 코드** (라인 2181-2184):
- QBB도 OBB와 동일하게 `xyxyxyxy2xywhr`로 변환하고 있음
- 이를 8개 좌표값을 그대로 유지하도록 수정 필요

**계획된 수정**:
- QBB의 경우 8개 좌표값을 직접 사용 (변환하지 않음)
- segments (N, 4, 2) → bboxes (N, 8) 형태로 변환
- 정규화 시 8개 좌표 모두 처리

#### 4. utils/ops.py 
- 새로운 QBB 전용 함수 추가 고려
- `validate_qbb_points()`: 꼭짓점 순서 정렬
- `qbb_iou()`: QBB IoU 계산 (추후 구현)

### 작업 진행 상황
- [x] 프로젝트 분석 완료
- [x] 캐시 파일 삭제
- [ ] NUM_THREADS 설정
- [ ] data/augment.py 수정
- [ ] 테스트 및 검증

### 다음 단계
1. NUM_THREADS=1로 설정
2. data/augment.py의 return_qbb 부분 수정
3. train_entry.py로 테스트
4. labels["bboxes"] shape 확인 (기대값: (N, 8))

---
*작성 시작: 2025-08-05 10:17*
*마지막 업데이트: 2025-08-05 10:17*