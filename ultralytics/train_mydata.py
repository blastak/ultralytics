import os
import re
import shutil

import matplotlib

from ultralytics import YOLO

matplotlib.use('Agg')


def save_pt(dataset, list):
    path = '../runs/detect/'
    destination_path = f'./pt/runs/{dataset}/'  # 복사할 폴더 지정
    if list[4:][0] == 'v':
        m = list[4:][1:]
    else:
        m = list[4:]
    if m.split('.')[1]=='yaml':
        m = m.split('.')[0] + '.pt'
    new_filename = f'{dataset}_{m}'  # 새로운 파일 이름
    os.makedirs(destination_path, exist_ok=True)

    # 폴더 목록 가져오기
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    train_folders = [f for f in folders if re.match(r'train\d+', f)]

    # 숫자 부분을 추출하여 가장 큰 값 찾기
    if train_folders:
        latest_train = max(train_folders, key=lambda x: int(re.search(r'\d+', x).group()))
        latest_train_path = os.path.join(path, latest_train)
        best_pt_path = os.path.join(latest_train_path, 'weights/best.pt')
        new_file_path = os.path.join(destination_path, new_filename)
        shutil.copy(best_pt_path, new_file_path)
    else:
        print("train 폴더가 존재하지 않습니다.")


def train_YOLO(dataset, model_list):
    prefix = './pt/yolo'
    for list in model_list:
        pt = os.path.join(prefix, list)
        model = YOLO(pt)  # load a pretrained model (recommended for training)
        model.train(data=f"{dataset}/{dataset}.yaml", epochs=100, imgsz=640)
        save_pt(dataset, list)


if __name__ == "__main__":
    pt = 'ultralytics/cfg/models/v10/yolov10n-quad.yaml'
    model = YOLO(pt)
    model.train(data='ultralytics/qyolo_data.yaml', epochs=100, imgsz=640)

#
# if __name__ == '__main__':
#
#
#     dataset_list = [
#         # 'min838',
#         # 'min838_2cls',
#         # 'min838_3cls',
#         # 'min838_3cls_2',
#         'min838_3cls_3',
#         # 'min838_4cls',
#         # 'min838_4cls_2',
#         # 'min838_4cls_3',
#         # 'min838_5cls',
#         # 'min838_6cls',
#         # 'min838_6cls_2',
#         # 'min838_multi',
#     ]
#     model_list1 = [
#         "yolo11n.pt",
#         "yolo11s.pt",
#         "yolov10n.pt",
#         "yolov10s.pt",
#         "yolov9t.pt",
#         "yolov9s.pt",
#         "yolov8n.pt",
#         "yolov8s.pt"
#     ]
#     model_list2 = [
#         "yolov6n.yaml",
#         "yolov6s.yaml",
#         "yolov5n.pt",
#         "yolov5s.pt",
#     ]
#
#     for d in dataset_list:
#         train_YOLO(d, model_list1)
#         train_YOLO(d, model_list2)
