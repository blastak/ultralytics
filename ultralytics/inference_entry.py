from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r"..\runs\obb\train_baseline_LP\weights\best.pt")  # or "yolo11n-obb.pt"

    test_path = Path(r'assets\good_all(obb)\images\test')
    jpg_filepaths = list(test_path.glob('*.jpg'))

    for jpg_filepath in jpg_filepaths:
        # Run prediction
        results = model(jpg_filepath)  # or local path

        # Visualize results with OpenCV
        for result in results:
            img = result.orig_img.copy()
            polygons = result.obb.xyxyxyxy.cpu().numpy()  # (N, 8)
            class_ids = result.obb.cls.cpu().int().numpy()
            names = [result.names[int(cls)] for cls in class_ids]
            confs = result.obb.conf.cpu().numpy()

            for i, polygon in enumerate(polygons):
                pts = polygon.reshape((4, 2)).astype(np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                label = f"{names[i]} {confs[i]:.2f}"
                text_pos = tuple(pts[0])
                cv2.putText(img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Show the image
            cv2.imshow("OBB Detection", img)
            cv2.waitKey(0)
            # cv2.destroyAllWindows()
