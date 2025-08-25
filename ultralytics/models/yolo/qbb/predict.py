# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class QBBPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a Quadrilateral Bounding Box (QBB) model.

    This predictor handles quadrilateral bounding box detection tasks, processing images and returning results with quadrilateral
    bounding boxes.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO QBB model.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.qbb import QBBPredictor
        >>> args = dict(model="yolov8n-qbb.pt", source=ASSETS)
        >>> predictor = QBBPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize QBBPredictor with optional model and data configuration overrides.

        Args:
            cfg (dict, optional): Default configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over the default config.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.

        Examples:
            >>> from ultralytics.utils import ASSETS
            >>> from ultralytics.models.yolo.qbb import QBBPredictor
            >>> args = dict(model="yolov8n-qbb.pt", source=ASSETS)
            >>> predictor = QBBPredictor(overrides=args)
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "qbb"

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """
        Post-processes predictions and returns Results objects with quadrilateral bounding boxes.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list[Results]): A list of Results objects with quadrilateral bounding box results.
        """
        save_feats = getattr(self, "_feats", None) is not None

        # QBB 전용 NMS 사용
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            return_idxs=save_feats,
            quad=True
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    def construct_results(self, preds, img, orig_imgs):
        """
        Construct a list of Results objects from model predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted quadrilateral bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.

        Returns:
            (List[Results]): List of Results objects containing quadrilateral detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted quadrilateral boxes and scores with shape (N, 10) where N is the number
                of detections. Each row contains [x1, y1, x2, y2, x3, y3, x4, y4, conf, cls].
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled quadrilateral
                bounding boxes.
        """
        # QBB는 8개 좌표를 스케일링
        pred[:, :8] = ops.scale_boxes(img.shape[2:], pred[:, :8], orig_img.shape, padding=False)

        # Results 객체 생성 - QBB는 10개 값 전체 전달 (8 coords + conf + cls)
        return Results(orig_img, path=img_path, names=self.model.names, qbb=pred)