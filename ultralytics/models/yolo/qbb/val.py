# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import QBBMetrics, batch_probiou


class QBBValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on an Quadrilateral Bounding Box (QBB) model.

    This validator specializes in evaluating models that predict rotated bounding boxes, commonly used for aerial and
    satellite imagery where objects can appear at various orientations.

    Attributes:
        args (dict): Configuration arguments for the validator.
        metrics (QBBMetrics): Metrics object for evaluating QBB model performance.
        is_dota (bool): Flag indicating whether the validation dataset is in DOTA format.

    Methods:
        init_metrics: Initialize evaluation metrics for YOLO.
        _process_batch: Process batch of detections and ground truth boxes to compute IoU matrix.
        _prepare_batch: Prepare batch data for QBB validation.
        _prepare_pred: Prepare predictions with scaled and padded bounding boxes.
        plot_predictions: Plot predicted bounding boxes on input images.
        pred_to_json: Serialize YOLO predictions to COCO json format.
        save_one_txt: Save YOLO detections to a txt file in normalized coordinates.
        eval_json: Evaluate YOLO output in JSON format and return performance statistics.

    Examples:
        >>> from ultralytics.models.yolo.qbb import QBBValidator
        >>> args = dict(model="yolo11n-qbb.pt", data="dota8.yaml")
        >>> validator = QBBValidator(args=args)
        >>> validator(model=args["model"])
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize QBBValidator and set task to 'qbb', metrics to QBBMetrics.

        This constructor initializes an QBBValidator instance for validating Quadrilateral Bounding Box (QBB) models.
        It extends the DetectionValidator class and configures it specifically for the QBB task.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (str | Path, optional): Directory to save results.
            args (dict | SimpleNamespace, optional): Arguments containing validation parameters.
            _callbacks (list, optional): List of callback functions to be called during validation.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "qbb"
        self.metrics = QBBMetrics()

    def get_desc(self):
        """Return a formatted string summarizing detection metrics."""
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )