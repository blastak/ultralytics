# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import QBBModel
from ultralytics.utils import DEFAULT_CFG, RANK


class QBBTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on an Oriented Bounding Box (QBB) model.

    Attributes:
        loss_names (Tuple[str]): Names of the loss components used during training.

    Methods:
        get_model: Return QBBModel initialized with specified config and weights.
        get_validator: Return an instance of QBBValidator for validation of YOLO model.

    Examples:
        >>> from ultralytics.models.yolo.qbb import QBBTrainer
        >>> args = dict(model="yolo11n-qbb.pt", data="dota8.yaml", epochs=3)
        >>> trainer = QBBTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize an QBBTrainer object for training Oriented Bounding Box (QBB) models.

        This trainer extends the DetectionTrainer class to specialize in training models that detect oriented
        bounding boxes. It automatically sets the task to 'qbb' in the configuration.

        Args:
            cfg (dict, optional): Configuration dictionary for the trainer. Contains training parameters and
                model configuration.
            overrides (dict, optional): Dictionary of parameter overrides for the configuration. Any values here
                will take precedence over those in cfg.
            _callbacks (list, optional): List of callback functions to be invoked during training.

        Examples:
            >>> from ultralytics.models.yolo.qbb import QBBTrainer
            >>> args = dict(model="yolo11n-qbb.pt", data="dota8.yaml", epochs=3)
            >>> trainer = QBBTrainer(overrides=args)
            >>> trainer.train()
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "qbb"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return QBBModel initialized with specified config and weights.

        Args:
            cfg (str | dict | None): Model configuration. Can be a path to a YAML config file, a dictionary
                containing configuration parameters, or None to use default configuration.
            weights (str | Path | None): Path to pretrained weights file. If None, random initialization is used.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (QBBModel): Initialized QBBModel with the specified configuration and weights.

        Examples:
            >>> trainer = QBBTrainer()
            >>> model = trainer.get_model(cfg="yolo11n-qbb.yaml", weights="yolo11n-qbb.pt")
        """
        model = QBBModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of QBBValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.qbb.QBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
