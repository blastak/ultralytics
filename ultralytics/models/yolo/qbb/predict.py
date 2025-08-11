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

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct the result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and rotation angles with shape (N, 7) where
                the last dimension contains [x, y, w, h, confidence, class_id, angle].
            img (torch.Tensor): The image after preprocessing with shape (B, C, H, W).
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.

        Returns:
            (Results): The result object containing the original image, image path, class names, and quadrilateral bounding
                boxes.
        """
        rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        # QBB 처리를 위한 임시 코드 - 추후 수정 필요
        qbb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
        return Results(orig_img, path=img_path, names=self.model.names, obb=qbb)  # TODO: obb를 qbb로 변경 필요
