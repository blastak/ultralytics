import os
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_images
import torch
from PIL import Image


def save_val_images(validator):
    """
    매 validation batch마다 GT와 예측 이미지를 좌우로 붙여서 저장하는 콜백

    Args:
        validator: YOLO validator 객체
    """
    try:
        # 현재 배치 정보 가져오기
        batch_i = validator.batch_i

        if batch_i < 5:  # 처음 5개 배치만 저장
            # 임시 경로
            gt_path = validator.save_dir / f"temp_gt_{batch_i}.jpg"
            pred_path = validator.save_dir / f"temp_pred_{batch_i}.jpg"

            # GT 이미지 생성
            plot_images(
                images=validator.batch["img"],
                labels=validator.batch,
                fname=gt_path,
                names=validator.names,
            )
            print(f"📝 Generated GT image for batch {batch_i}")

            # Pred 이미지 생성 (preds가 있을 때)
            if hasattr(validator, 'preds') and validator.preds is not None:
                # 예측 결과를 batch 형식으로 변환
                preds = validator.preds

                # batched_preds 생성
                if isinstance(preds, list) and len(preds) > 0:  # 빈 리스트 체크
                    # 각 이미지의 예측에 batch_idx 추가
                    for i, pred in enumerate(preds):
                        if pred is not None and len(pred) > 0:  # None과 빈 예측 체크
                            if "conf" in pred:
                                pred["batch_idx"] = torch.ones_like(pred["conf"]) * i
                            elif "scores" in pred:
                                pred["batch_idx"] = torch.ones_like(pred["scores"]) * i
                            else:
                                # conf나 scores가 없으면 기본값 사용
                                if "bboxes" in pred:
                                    pred["batch_idx"] = torch.zeros(len(pred["bboxes"])) * i

                    # 모든 예측을 하나로 합치기
                    if len(preds[0]) > 0:  # 첫 번째 예측이 비어있지 않은지 확인
                        keys = preds[0].keys()
                        batched_preds = {}
                        for k in keys:
                            # 각 키에 대해 안전하게 concatenate
                            valid_tensors = [x[k] for x in preds if k in x and x[k].numel() > 0]
                            if valid_tensors:
                                batched_preds[k] = torch.cat(valid_tensors, dim=0)
                    else:
                        batched_preds = None
                        print(f"⚠️ Empty predictions for batch {batch_i}")

                elif isinstance(preds, dict):
                    # 이미 batched 형태면 그대로 사용
                    batched_preds = preds
                else:
                    batched_preds = None
                    print(f"⚠️ Unexpected preds format: {type(preds)}")

                # 예측 이미지 생성
                if batched_preds is not None and len(batched_preds) > 0:
                    plot_images(
                        images=validator.batch["img"],
                        labels=batched_preds,
                        fname=pred_path,
                        names=validator.names,
                    )
                    print(f"📝 Generated prediction image for batch {batch_i}")

                    # 이미지 합치기
                    if gt_path.exists() and pred_path.exists():
                        try:
                            gt_img = Image.open(gt_path)
                            pred_img = Image.open(pred_path)

                            # 새 캔버스 생성
                            total_width = gt_img.width + pred_img.width
                            max_height = max(gt_img.height, pred_img.height)
                            combined = Image.new('RGB', (total_width, max_height))

                            # 이미지 붙이기
                            combined.paste(gt_img, (0, 0))
                            combined.paste(pred_img, (gt_img.width, 0))

                            # 텍스트 추가 (선택사항)
                            from PIL import ImageDraw, ImageFont
                            try:
                                draw = ImageDraw.Draw(combined)
                                # 기본 폰트 사용 (시스템 폰트가 없을 경우 대비)
                                draw.text((10, 10), "Ground Truth", fill=(0, 255, 0))
                                draw.text((gt_img.width + 10, 10), "Predictions", fill=(255, 0, 0))
                            except:
                                pass  # 폰트 에러는 무시

                            # 저장
                            final_path = validator.save_dir / f"val_batch{batch_i}_comparison.jpg"
                            combined.save(final_path)

                            # 임시 파일 삭제
                            gt_path.unlink(missing_ok=True)
                            pred_path.unlink(missing_ok=True)

                            print(f"✅ Saved comparison: {final_path}")

                        except Exception as e:
                            print(f"❌ Error combining images: {e}")
                            # 합치기 실패해도 개별 파일은 유지
                            if gt_path.exists():
                                gt_path.rename(validator.save_dir / f"val_batch{batch_i}_gt.jpg")
                            if pred_path.exists():
                                pred_path.rename(validator.save_dir / f"val_batch{batch_i}_pred.jpg")
                else:
                    print(f"⚠️ No valid predictions to plot for batch {batch_i}")
                    # GT만 저장
                    if gt_path.exists():
                        final_path = validator.save_dir / f"val_batch{batch_i}_gt_only.jpg"
                        gt_path.rename(final_path)
                        print(f"💾 Saved GT only: {final_path}")
            else:
                print(f"ℹ️ No predictions available for batch {batch_i} (might be first epoch)")
                # GT만 저장
                if gt_path.exists():
                    final_path = validator.save_dir / f"val_batch{batch_i}_gt_only.jpg"
                    gt_path.rename(final_path)
                    print(f"💾 Saved GT only: {final_path}")

    except Exception as e:
        print(f"❌ Error in save_val_images: {e}")
        import traceback
        traceback.print_exc()
        # 에러가 나도 학습은 계속되도록 함


if __name__ == '__main__':
    # os.system("rm -f /workspace/repo/ultralytics/ultralytics/assets/good_all_obb8/labels/*.cache")
    model = YOLO('yolov8n-qbb.yaml')
    # model.add_callback("on_val_batch_end", save_val_images)
    # results = model.train(name='debug_by_user', data='webpm_obb1944.yaml', epochs=20, imgsz=640, fliplr=0.0, batch=16, workers=0, plots=True)
    results = model.train(
        name='multi_gpu_train',
        data='webpm_obb1944.yaml',
        imgsz=640,
        epochs=5,  # 에폭 증가
        batch=8,  # 배치 사이즈 증가 (GPU 2개 * 8 each)
        device="0",  # Multi-GPU 활성화
        workers=2,  # CPU 코어 수 활용
        fliplr=0.0,  # Flip augmentation 활성화
        plots=True,  # JPG visualization 활성화
        # cache=True,  # 데이터 캐싱으로 속도 향상
        # amp=True  # Automatic Mixed Precision
    )

    # results = model.train(name='debug_by_user', data='webpm_bb8.yaml', epochs=2, imgsz=640, fliplr=0.0, batch=1, workers=0, plots=True)
    # model = YOLO('yolov8n.yaml')
    # results = model.train(name='debug_by_user', data='webpm_bb8.yaml', epochs=20, imgsz=640, fliplr=0.0, batch=2, workers=0, plots=True)
