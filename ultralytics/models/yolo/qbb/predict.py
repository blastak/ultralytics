# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class QBBPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on an Quadrilateral Bounding Box (QBB) model.

    This predictor handles quadrilateral bounding box detection tasks, processing images and returning results with rotated
    bounding boxes.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO QBB model.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.qbb import QBBPredictor
        >>> args = dict(model="yolo11n-qbb.pt", source=ASSETS)
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
            >>> args = dict(model="yolo11n-qbb.pt", source=ASSETS)
            >>> predictor = QBBPredictor(overrides=args)
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "qbb"

    def postprocess(self, preds, img, orig_imgs):
        """
        Extends the post-processing functionality of the DetectionPredictor to handle QBB outputs.
        """
        preds = super().postprocess(preds, img, orig_imgs)
        for i, pred in enumerate(preds):
            if not pred.obb is None:
                pred.qbb = pred.obb
                pred.obb = None
        return preds