from datetime import date
import argparse

from ultralytics import YOLO


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Training Entrypoint')
    parser.add_argument('--model', type=str, default='yolov8n-qbb.yaml', help='Model config file')
    parser.add_argument('--data', type=str, default='webpm_obb1944.yaml', help='Data config file')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--device', type=str, default='0,1', help='Device (e.g., 0 or 0,1)')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--fliplr', type=float, default=0.0, help='Horizontal flip probability')
    parser.add_argument('--dfl', type=float, default=5.0, help='DFL loss weight')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--save_period', type=int, default=-1, help='Save checkpoint every x epochs')

    args = parser.parse_args()

    today_str = date.today().strftime("%m%d")
    exp_name = args.name if args.name else f'train{today_str}_'

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        name=exp_name,
        data=args.data,
        epochs=args.epochs,
        fliplr=args.fliplr,
        batch=args.batch,
        workers=args.workers,
        imgsz=args.imgsz,
        plots=True,
        device=args.device,
        dfl=args.dfl,
        patience=args.patience,
        save_period=args.save_period
    )
