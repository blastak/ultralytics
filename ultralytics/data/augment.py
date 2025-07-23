# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils import LOGGER, IterableSimpleNamespace, colorstr
from ultralytics.utils.checks import check_version
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.ops import segment2box, xywh2xyxy, xyxyxyxy2xywhr
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)


class BaseTransform:
    """
    Base class for image transformations in the Ultralytics library.

    This class serves as a foundation for implementing various image processing operations, designed to be
    compatible with both classification and semantic segmentation tasks.

    Methods:
        apply_image: Apply image transformations to labels.
        apply_instances: Apply transformations to object instances in labels.
        apply_semantic: Apply semantic segmentation to an image.
        __call__: Apply all label transformations to an image, instances, and semantic masks.

    Examples:
        >>> transform = BaseTransform()
        >>> labels = {"image": np.array(...), "instances": [...], "semantic": np.array(...)}
        >>> transformed_labels = transform(labels)
    """

    def __init__(self) -> None:
        """
        Initialize the BaseTransform object.

        This constructor sets up the base transformation object, which can be extended for specific image
        processing tasks. It is designed to be compatible with both classification and semantic segmentation.

        Examples:
            >>> transform = BaseTransform()
        """
        pass

    def apply_image(self, labels):
        """
        Apply image transformations to labels.

        This method is intended to be overridden by subclasses to implement specific image transformation
        logic. In its base form, it returns the input labels unchanged.

        Args:
            labels (Any): The input labels to be transformed. The exact type and structure of labels may
                vary depending on the specific implementation.

        Returns:
            (Any): The transformed labels. In the base implementation, this is identical to the input.

        Examples:
            >>> transform = BaseTransform()
            >>> original_labels = [1, 2, 3]
            >>> transformed_labels = transform.apply_image(original_labels)
            >>> print(transformed_labels)
            [1, 2, 3]
        """
        pass

    def apply_instances(self, labels):
        """
        Apply transformations to object instances in labels.

        This method is responsible for applying various transformations to object instances within the given
        labels. It is designed to be overridden by subclasses to implement specific instance transformation
        logic.

        Args:
            labels (dict): A dictionary containing label information, including object instances.

        Returns:
            (dict): The modified labels dictionary with transformed object instances.

        Examples:
            >>> transform = BaseTransform()
            >>> labels = {"instances": Instances(xyxy=torch.rand(5, 4), cls=torch.randint(0, 80, (5,)))}
            >>> transformed_labels = transform.apply_instances(labels)
        """
        pass

    def apply_semantic(self, labels):
        """
        Apply semantic segmentation transformations to an image.

        This method is intended to be overridden by subclasses to implement specific semantic segmentation
        transformations. In its base form, it does not perform any operations.

        Args:
            labels (Any): The input labels or semantic segmentation mask to be transformed.

        Returns:
            (Any): The transformed semantic segmentation mask or labels.

        Examples:
            >>> transform = BaseTransform()
            >>> semantic_mask = np.zeros((100, 100), dtype=np.uint8)
            >>> transformed_mask = transform.apply_semantic(semantic_mask)
        """
        pass

    def __call__(self, labels):
        """
        Apply all label transformations to an image, instances, and semantic masks.

        This method orchestrates the application of various transformations defined in the BaseTransform class
        to the input labels. It sequentially calls the apply_image and apply_instances methods to process the
        image and object instances, respectively.

        Args:
            labels (dict): A dictionary containing image data and annotations. Expected keys include 'img' for
                the image data, and 'instances' for object instances.

        Returns:
            (dict): The input labels dictionary with transformed image and instances.

        Examples:
            >>> transform = BaseTransform()
            >>> labels = {"img": np.random.rand(640, 640, 3), "instances": []}
            >>> transformed_labels = transform(labels)
        """
        self.apply_image(labels)
        self.apply_instances(labels)
        self.apply_semantic(labels)


class Compose:
    """
    A class for composing multiple image transformations.

    Attributes:
        transforms (List[Callable]): A list of transformation functions to be applied sequentially.

    Methods:
        __call__: Apply a series of transformations to input data.
        append: Append a new transform to the existing list of transforms.
        insert: Insert a new transform at a specified index in the list of transforms.
        __getitem__: Retrieve a specific transform or a set of transforms using indexing.
        __setitem__: Set a specific transform or a set of transforms using indexing.
        tolist: Convert the list of transforms to a standard Python list.

    Examples:
        >>> transforms = [RandomFlip(), RandomPerspective(30)]
        >>> compose = Compose(transforms)
        >>> transformed_data = compose(data)
        >>> compose.append(CenterCrop((224, 224)))
        >>> compose.insert(0, RandomFlip())
    """

    def __init__(self, transforms):
        """
        Initialize the Compose object with a list of transforms.

        Args:
            transforms (List[Callable]): A list of callable transform objects to be applied sequentially.

        Examples:
            >>> from ultralytics.data.augment import Compose, RandomHSV, RandomFlip
            >>> transforms = [RandomHSV(), RandomFlip()]
            >>> compose = Compose(transforms)
        """
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """
        Apply a series of transformations to input data.

        This method sequentially applies each transformation in the Compose object's transforms to the input data.

        Args:
            data (Any): The input data to be transformed. This can be of any type, depending on the
                transformations in the list.

        Returns:
            (Any): The transformed data after applying all transformations in sequence.

        Examples:
            >>> transforms = [Transform1(), Transform2(), Transform3()]
            >>> compose = Compose(transforms)
            >>> transformed_data = compose(input_data)
        """
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """
        Append a new transform to the existing list of transforms.

        Args:
            transform (BaseTransform): The transformation to be added to the composition.

        Examples:
            >>> compose = Compose([RandomFlip(), RandomPerspective()])
            >>> compose.append(RandomHSV())
        """
        self.transforms.append(transform)

    def insert(self, index, transform):
        """
        Insert a new transform at a specified index in the existing list of transforms.

        Args:
            index (int): The index at which to insert the new transform.
            transform (BaseTransform): The transform object to be inserted.

        Examples:
            >>> compose = Compose([Transform1(), Transform2()])
            >>> compose.insert(1, Transform3())
            >>> len(compose.transforms)
            3
        """
        self.transforms.insert(index, transform)

    def __getitem__(self, index: Union[list, int]) -> "Compose":
        """
        Retrieve a specific transform or a set of transforms using indexing.

        Args:
            index (int | List[int]): Index or list of indices of the transforms to retrieve.

        Returns:
            (Compose): A new Compose object containing the selected transform(s).

        Raises:
            AssertionError: If the index is not of type int or list.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), RandomHSV(0.5, 0.5, 0.5)]
            >>> compose = Compose(transforms)
            >>> single_transform = compose[1]  # Returns a Compose object with only RandomPerspective
            >>> multiple_transforms = compose[0:2]  # Returns a Compose object with RandomFlip and RandomPerspective
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        return Compose([self.transforms[i] for i in index]) if isinstance(index, list) else self.transforms[index]

    def __setitem__(self, index: Union[list, int], value: Union[list, int]) -> None:
        """
        Set one or more transforms in the composition using indexing.

        Args:
            index (int | List[int]): Index or list of indices to set transforms at.
            value (Any | List[Any]): Transform or list of transforms to set at the specified index(es).

        Raises:
            AssertionError: If index type is invalid, value type doesn't match index type, or index is out of range.

        Examples:
            >>> compose = Compose([Transform1(), Transform2(), Transform3()])
            >>> compose[1] = NewTransform()  # Replace second transform
            >>> compose[0:2] = [NewTransform1(), NewTransform2()]  # Replace first two transforms
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(value, list), (
                f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
            )
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"list index {i} out of range {len(self.transforms)}."
            self.transforms[i] = v

    def tolist(self):
        """
        Convert the list of transforms to a standard Python list.

        Returns:
            (list): A list containing all the transform objects in the Compose instance.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), CenterCrop()]
            >>> compose = Compose(transforms)
            >>> transform_list = compose.tolist()
            >>> print(len(transform_list))
            3
        """
        return self.transforms

    def __repr__(self):
        """
        Return a string representation of the Compose object.

        Returns:
            (str): A string representation of the Compose object, including the list of transforms.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(degrees=10, translate=0.1, scale=0.1)]
            >>> compose = Compose(transforms)
            >>> print(compose)
            Compose([
                RandomFlip(),
                RandomPerspective(degrees=10, translate=0.1, scale=0.1)
            ])
        """
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"


class BaseMixTransform:
    """
    Base class for mix transformations like Cutmix, MixUp and Mosaic.

    This class provides a foundation for implementing mix transformations on datasets. It handles the
    probability-based application of transforms and manages the mixing of multiple images and labels.

    Attributes:
        dataset (Any): The dataset object containing images and labels.
        pre_transform (Callable | None): Optional transform to apply before mixing.
        p (float): Probability of applying the mix transformation.

    Methods:
        __call__: Apply the mix transformation to the input labels.
        _mix_transform: Abstract method to be implemented by subclasses for specific mix operations.
        get_indexes: Abstract method to get indexes of images to be mixed.
        _update_label_text: Update label text for mixed images.

    Examples:
        >>> class CustomMixTransform(BaseMixTransform):
        ...     def _mix_transform(self, labels):
        ...         # Implement custom mix logic here
        ...         return labels
        ...
        ...     def get_indexes(self):
        ...         return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        >>> dataset = YourDataset()
        >>> transform = CustomMixTransform(dataset, p=0.5)
        >>> mixed_labels = transform(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """
        Initialize the BaseMixTransform object for mix transformations like CutMix, MixUp and Mosaic.

        This class serves as a base for implementing mix transformations in image processing pipelines.

        Args:
            dataset (Any): The dataset object containing images and labels for mixing.
            pre_transform (Callable | None): Optional transform to apply before mixing.
            p (float): Probability of applying the mix transformation. Should be in the range [0.0, 1.0].

        Examples:
            >>> dataset = YOLODataset("path/to/data")
            >>> pre_transform = Compose([RandomFlip(), RandomPerspective()])
            >>> mix_transform = BaseMixTransform(dataset, pre_transform, p=0.5)
        """
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply pre-processing transforms and cutmix/mixup/mosaic transforms to labels data.

        This method determines whether to apply the mix transform based on a probability factor. If applied, it
        selects additional images, applies pre-transforms if specified, and then performs the mix transform.

        Args:
            labels (Dict[str, Any]): A dictionary containing label data for an image.

        Returns:
            (Dict[str, Any]): The transformed labels dictionary, which may include mixed data from other images.

        Examples:
            >>> transform = BaseMixTransform(dataset, pre_transform=None, p=0.5)
            >>> result = transform({"image": img, "bboxes": boxes, "cls": classes})
        """
        if random.uniform(0, 1) > self.p:
            return labels

        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic, CutMix or MixUp
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # Update cls and texts
        labels = self._update_label_text(labels)
        # Mosaic, CutMix or MixUp
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _mix_transform(self, labels: Dict[str, Any]):
        """
        Apply CutMix, MixUp or Mosaic augmentation to the label dictionary.

        This method should be implemented by subclasses to perform specific mix transformations like CutMix, MixUp or
        Mosaic. It modifies the input label dictionary in-place with the augmented data.

        Args:
            labels (Dict[str, Any]): A dictionary containing image and label data. Expected to have a 'mix_labels' key
                with a list of additional image and label data for mixing.

        Returns:
            (Dict[str, Any]): The modified labels dictionary with augmented data after applying the mix transform.

        Examples:
            >>> transform = BaseMixTransform(dataset)
            >>> labels = {"image": img, "bboxes": boxes, "mix_labels": [{"image": img2, "bboxes": boxes2}]}
            >>> augmented_labels = transform._mix_transform(labels)
        """
        raise NotImplementedError

    def get_indexes(self):
        """
        Get a list of shuffled indexes for mosaic augmentation.

        Returns:
            (List[int]): A list of shuffled indexes from the dataset.

        Examples:
            >>> transform = BaseMixTransform(dataset)
            >>> indexes = transform.get_indexes()
            >>> print(indexes)  # [3, 18, 7, 2]
        """
        return random.randint(0, len(self.dataset) - 1)

    @staticmethod
    def _update_label_text(labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update label text and class IDs for mixed labels in image augmentation.

        This method processes the 'texts' and 'cls' fields of the input labels dictionary and any mixed labels,
        creating a unified set of text labels and updating class IDs accordingly.

        Args:
            labels (Dict[str, Any]): A dictionary containing label information, including 'texts' and 'cls' fields,
                and optionally a 'mix_labels' field with additional label dictionaries.

        Returns:
            (Dict[str, Any]): The updated labels dictionary with unified text labels and updated class IDs.

        Examples:
            >>> labels = {
            ...     "texts": [["cat"], ["dog"]],
            ...     "cls": torch.tensor([[0], [1]]),
            ...     "mix_labels": [{"texts": [["bird"], ["fish"]], "cls": torch.tensor([[0], [1]])}],
            ... }
            >>> updated_labels = self._update_label_text(labels)
            >>> print(updated_labels["texts"])
            [['cat'], ['dog'], ['bird'], ['fish']]
            >>> print(updated_labels["cls"])
            tensor([[0],
                    [1]])
            >>> print(updated_labels["mix_labels"][0]["cls"])
            tensor([[2],
                    [3]])
        """
        if "texts" not in labels:
            return labels

        mix_texts = sum([labels["texts"]] + [x["texts"] for x in labels["mix_labels"]], [])
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for i, text in enumerate(mix_texts)}

        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts
        return labels


class Mosaic(BaseMixTransform):
    """
    Mosaic augmentation for image datasets.

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
    The augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
        p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
        n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
        border (Tuple[int, int]): Border size for width and height.

    Methods:
        get_indexes: Return a list of random indexes from the dataset.
        _mix_transform: Apply mixup transformation to the input image and labels.
        _mosaic3: Create a 1x3 image mosaic.
        _mosaic4: Create a 2x2 image mosaic.
        _mosaic9: Create a 3x3 image mosaic.
        _update_labels: Update labels with padding.
        _cat_labels: Concatenate labels and clips mosaic border instances.

    Examples:
        >>> from ultralytics.data.augment import Mosaic
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        >>> augmented_labels = mosaic_aug(original_labels)
    """

    def __init__(self, dataset, imgsz: int = 640, p: float = 1.0, n: int = 4):
        """
        Initialize the Mosaic augmentation object.

        This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
        The augmentation is applied to a dataset with a given probability.

        Args:
            dataset (Any): The dataset on which the mosaic augmentation is applied.
            imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
            p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
            n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).

        Examples:
            >>> from ultralytics.data.augment import Mosaic
            >>> dataset = YourDataset(...)
            >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        """
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert n in {4, 9}, "grid must be equal to 4 or 9."
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n
        self.buffer_enabled = self.dataset.cache != "ram"

    def get_indexes(self):
        """
        Return a list of random indexes from the dataset for mosaic augmentation.

        This method selects random image indexes either from a buffer or from the entire dataset, depending on
        the 'buffer' parameter. It is used to choose images for creating mosaic augmentations.

        Returns:
            (List[int]): A list of random image indexes. The length of the list is n-1, where n is the number
                of images used in the mosaic (either 3 or 8, depending on whether n is 4 or 9).

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> indexes = mosaic.get_indexes()
            >>> print(len(indexes))  # Output: 3
        """
        if self.buffer_enabled:  # select images from buffer
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply mosaic augmentation to the input image and labels.

        This method combines multiple images (3, 4, or 9) into a single mosaic image based on the 'n' attribute.
        It ensures that rectangular annotations are not present and that there are other images available for
        mosaic augmentation.

        Args:
            labels (Dict[str, Any]): A dictionary containing image data and annotations. Expected keys include:
                - 'rect_shape': Should be None as rect and mosaic are mutually exclusive.
                - 'mix_labels': A list of dictionaries containing data for other images to be used in the mosaic.

        Returns:
            (Dict[str, Any]): A dictionary containing the mosaic-augmented image and updated annotations.

        Raises:
            AssertionError: If 'rect_shape' is not None or if 'mix_labels' is empty.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> augmented_data = mosaic._mix_transform(labels)
        """
        assert labels.get("rect_shape", None) is None, "rect and mosaic are mutually exclusive."
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment."
        return (
            self._mosaic3(labels) if self.n == 3 else self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)
        )  # This code is modified for mosaic3 method.

    def _mosaic3(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a 1x3 image mosaic by combining three images.

        This method arranges three images in a horizontal layout, with the main image in the center and two
        additional images on either side. It's part of the Mosaic augmentation technique used in object detection.

        Args:
            labels (Dict[str, Any]): A dictionary containing image and label information for the main (center) image.
                Must include 'img' key with the image array, and 'mix_labels' key with a list of two
                dictionaries containing information for the side images.

        Returns:
            (Dict[str, Any]): A dictionary with the mosaic image and updated labels. Keys include:
                - 'img' (np.ndarray): The mosaic image array with shape (H, W, C).
                - Other keys from the input labels, updated to reflect the new image dimensions.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=3)
            >>> labels = {
            ...     "img": np.random.rand(480, 640, 3),
            ...     "mix_labels": [{"img": np.random.rand(480, 640, 3)} for _ in range(2)],
            ... }
            >>> result = mosaic._mosaic3(labels)
            >>> print(result["img"].shape)
            (640, 640, 3)
        """
        mosaic_labels = []
        s = self.imgsz
        for i in range(3):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img3
            if i == 0:  # center
                img3 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 3 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 2:  # left
                c = s - w, s + h0 - h, s, s + h0

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coordinates

            img3[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  # img3[ymin:ymax, xmin:xmax]
            # hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img3[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return final_labels

    def _mosaic4(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a 2x2 image mosaic from four input images.

        This method combines four images into a single mosaic image by placing them in a 2x2 grid. It also
        updates the corresponding labels for each image in the mosaic.

        Args:
            labels (Dict[str, Any]): A dictionary containing image data and labels for the base image (index 0) and three
                additional images (indices 1-3) in the 'mix_labels' key.

        Returns:
            (Dict[str, Any]): A dictionary containing the mosaic image and updated labels. The 'img' key contains the mosaic
                image as a numpy array, and other keys contain the combined and adjusted labels for all four images.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> labels = {
            ...     "img": np.random.rand(480, 640, 3),
            ...     "mix_labels": [{"img": np.random.rand(480, 640, 3)} for _ in range(3)],
            ... }
            >>> result = mosaic._mosaic4(labels)
            >>> assert result["img"].shape == (1280, 1280, 3)
        """
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels

    def _mosaic9(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a 3x3 image mosaic from the input image and eight additional images.

        This method combines nine images into a single mosaic image. The input image is placed at the center,
        and eight additional images from the dataset are placed around it in a 3x3 grid pattern.

        Args:
            labels (Dict[str, Any]): A dictionary containing the input image and its associated labels. It should have
                the following keys:
                - 'img' (np.ndarray): The input image.
                - 'resized_shape' (Tuple[int, int]): The shape of the resized image (height, width).
                - 'mix_labels' (List[Dict]): A list of dictionaries containing information for the additional
                  eight images, each with the same structure as the input labels.

        Returns:
            (Dict[str, Any]): A dictionary containing the mosaic image and updated labels. It includes the following keys:
                - 'img' (np.ndarray): The final mosaic image.
                - Other keys from the input labels, updated to reflect the new mosaic arrangement.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=9)
            >>> input_labels = dataset[0]
            >>> mosaic_result = mosaic._mosaic9(input_labels)
            >>> mosaic_image = mosaic_result["img"]
        """
        mosaic_labels = []
        s = self.imgsz
        hp, wp = -1, -1  # height, width previous
        for i in range(9):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coordinates

            # Image
            img9[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return final_labels

    @staticmethod
    def _update_labels(labels, padw: int, padh: int) -> Dict[str, Any]:
        """
        Update label coordinates with padding values.

        This method adjusts the bounding box coordinates of object instances in the labels by adding padding
        values. It also denormalizes the coordinates if they were previously normalized.

        Args:
            labels (Dict[str, Any]): A dictionary containing image and instance information.
            padw (int): Padding width to be added to the x-coordinates.
            padh (int): Padding height to be added to the y-coordinates.

        Returns:
            (dict): Updated labels dictionary with adjusted instance coordinates.

        Examples:
            >>> labels = {"img": np.zeros((100, 100, 3)), "instances": Instances(...)}
            >>> padw, padh = 50, 50
            >>> updated_labels = Mosaic._update_labels(labels, padw, padh)
        """
        nh, nw = labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Concatenate and process labels for mosaic augmentation.

        This method combines labels from multiple images used in mosaic augmentation, clips instances to the
        mosaic border, and removes zero-area boxes.

        Args:
            mosaic_labels (List[Dict[str, Any]]): A list of label dictionaries for each image in the mosaic.

        Returns:
            (Dict[str, Any]): A dictionary containing concatenated and processed labels for the mosaic image, including:
                - im_file (str): File path of the first image in the mosaic.
                - ori_shape (Tuple[int, int]): Original shape of the first image.
                - resized_shape (Tuple[int, int]): Shape of the mosaic image (imgsz * 2, imgsz * 2).
                - cls (np.ndarray): Concatenated class labels.
                - instances (Instances): Concatenated instance annotations.
                - mosaic_border (Tuple[int, int]): Mosaic border size.
                - texts (List[str], optional): Text labels if present in the original labels.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640)
            >>> mosaic_labels = [{"cls": np.array([0, 1]), "instances": Instances(...)} for _ in range(4)]
            >>> result = mosaic._cat_labels(mosaic_labels)
            >>> print(result.keys())
            dict_keys(['im_file', 'ori_shape', 'resized_shape', 'cls', 'instances', 'mosaic_border'])
        """
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        # Final labels
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels


class MixUp(BaseMixTransform):
    """
    Apply MixUp augmentation to image datasets.

    This class implements the MixUp augmentation technique as described in the paper [mixup: Beyond Empirical Risk
    Minimization](https://arxiv.org/abs/1710.09412). MixUp combines two images and their labels using a random weight.

    Attributes:
        dataset (Any): The dataset to which MixUp augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before MixUp.
        p (float): Probability of applying MixUp augmentation.

    Methods:
        _mix_transform: Apply MixUp augmentation to the input labels.

    Examples:
        >>> from ultralytics.data.augment import MixUp
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> mixup = MixUp(dataset, p=0.5)
        >>> augmented_labels = mixup(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p: float = 0.0) -> None:
        """
        Initialize the MixUp augmentation object.

        MixUp is an image augmentation technique that combines two images by taking a weighted sum of their pixel
        values and labels. This implementation is designed for use with the Ultralytics YOLO framework.

        Args:
            dataset (Any): The dataset to which MixUp augmentation will be applied.
            pre_transform (Callable | None): Optional transform to apply to images before MixUp.
            p (float): Probability of applying MixUp augmentation to an image. Must be in the range [0, 1].

        Examples:
            >>> from ultralytics.data.dataset import YOLODataset
            >>> dataset = YOLODataset("path/to/data.yaml")
            >>> mixup = MixUp(dataset, pre_transform=None, p=0.5)
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def _mix_transform(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply MixUp augmentation to the input labels.

        This method implements the MixUp augmentation technique as described in the paper
        "mixup: Beyond Empirical Risk Minimization" (https://arxiv.org/abs/1710.09412).

        Args:
            labels (Dict[str, Any]): A dictionary containing the original image and label information.

        Returns:
            (Dict[str, Any]): A dictionary containing the mixed-up image and combined label information.

        Examples:
            >>> mixer = MixUp(dataset)
            >>> mixed_labels = mixer._mix_transform(labels)
        """
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels


class CutMix(BaseMixTransform):
    """
    Apply CutMix augmentation to image datasets as described in the paper https://arxiv.org/abs/1905.04899.

    CutMix combines two images by replacing a random rectangular region of one image with the corresponding region from another image,
    and adjusts the labels proportionally to the area of the mixed region.

    Attributes:
        dataset (Any): The dataset to which CutMix augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before CutMix.
        p (float): Probability of applying CutMix augmentation.
        beta (float): Beta distribution parameter for sampling the mixing ratio.
        num_areas (int): Number of areas to try to cut and mix.

    Methods:
        _mix_transform: Apply CutMix augmentation to the input labels.
        _rand_bbox: Generate random bounding box coordinates for the cut region.

    Examples:
        >>> from ultralytics.data.augment import CutMix
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> cutmix = CutMix(dataset, p=0.5)
        >>> augmented_labels = cutmix(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p: float = 0.0, beta: float = 1.0, num_areas: int = 3) -> None:
        """
        Initialize the CutMix augmentation object.

        Args:
            dataset (Any): The dataset to which CutMix augmentation will be applied.
            pre_transform (Callable | None): Optional transform to apply before CutMix.
            p (float): Probability of applying CutMix augmentation.
            beta (float): Beta distribution parameter for sampling the mixing ratio.
            num_areas (int): Number of areas to try to cut and mix.
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        self.beta = beta
        self.num_areas = num_areas

    def _rand_bbox(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Generate random bounding box coordinates for the cut region.

        Args:
            width (int): Width of the image.
            height (int): Height of the image.

        Returns:
            (Tuple[int]): (x1, y1, x2, y2) coordinates of the bounding box.
        """
        # Sample mixing ratio from Beta distribution
        lam = np.random.beta(self.beta, self.beta)

        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        # Random center
        cx = np.random.randint(width)
        cy = np.random.randint(height)

        # Bounding box coordinates
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)

        return x1, y1, x2, y2

    def _mix_transform(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply CutMix augmentation to the input labels.

        Args:
            labels (Dict[str, Any]): A dictionary containing the original image and label information.

        Returns:
            (Dict[str, Any]): A dictionary containing the mixed image and adjusted labels.

        Examples:
            >>> cutter = CutMix(dataset)
            >>> mixed_labels = cutter._mix_transform(labels)
        """
        # Get a random second image
        h, w = labels["img"].shape[:2]

        cut_areas = np.asarray([self._rand_bbox(w, h) for _ in range(self.num_areas)], dtype=np.float32)
        ioa1 = bbox_ioa(cut_areas, labels["instances"].bboxes)  # (self.num_areas, num_boxes)
        idx = np.nonzero(ioa1.sum(axis=1) <= 0)[0]
        if len(idx) == 0:
            return labels

        labels2 = labels.pop("mix_labels")[0]
        area = cut_areas[np.random.choice(idx)]  # randomly select one
        ioa2 = bbox_ioa(area[None], labels2["instances"].bboxes).squeeze(0)
        indexes2 = np.nonzero(ioa2 >= (0.01 if len(labels["instances"].segments) else 0.1))[0]
        if len(indexes2) == 0:
            return labels

        instances2 = labels2["instances"][indexes2]
        instances2.convert_bbox("xyxy")
        instances2.denormalize(w, h)

        # Apply CutMix
        x1, y1, x2, y2 = area.astype(np.int32)
        labels["img"][y1:y2, x1:x2] = labels2["img"][y1:y2, x1:x2]

        # Restrain instances2 to the random bounding border
        instances2.add_padding(-x1, -y1)
        instances2.clip(x2 - x1, y2 - y1)
        instances2.add_padding(x1, y1)

        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"][indexes2]], axis=0)
        labels["instances"] = Instances.concatenate([labels["instances"], instances2], axis=0)
        return labels


class RandomPerspective:
    """
    Implement random perspective and affine transformations on images and corresponding annotations.

    This class applies random rotations, translations, scaling, shearing, and perspective transformations
    to images and their associated bounding boxes, segments, and keypoints. It can be used as part of an
    augmentation pipeline for object detection and instance segmentation tasks.

    Attributes:
        degrees (float): Maximum absolute degree range for random rotations.
        translate (float): Maximum translation as a fraction of the image size.
        scale (float): Scaling factor range, e.g., scale=0.1 means 0.9-1.1.
        shear (float): Maximum shear angle in degrees.
        perspective (float): Perspective distortion factor.
        border (Tuple[int, int]): Mosaic border size as (x, y).
        pre_transform (Callable | None): Optional transform to apply before the random perspective.

    Methods:
        affine_transform: Apply affine transformations to the input image.
        apply_bboxes: Transform bounding boxes using the affine matrix.
        apply_segments: Transform segments and generate new bounding boxes.
        apply_keypoints: Transform keypoints using the affine matrix.
        __call__: Apply the random perspective transformation to images and annotations.
        box_candidates: Filter transformed bounding boxes based on size and aspect ratio.

    Examples:
        >>> transform = RandomPerspective(degrees=10, translate=0.1, scale=0.1, shear=10)
        >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> labels = {"img": image, "cls": np.array([0, 1]), "instances": Instances(...)}
        >>> result = transform(labels)
        >>> transformed_image = result["img"]
        >>> transformed_instances = result["instances"]
    """

    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        border: Tuple[int, int] = (0, 0),
        pre_transform=None,
    ):
        """
        Initialize RandomPerspective object with transformation parameters.

        This class implements random perspective and affine transformations on images and corresponding bounding boxes,
        segments, and keypoints. Transformations include rotation, translation, scaling, and shearing.

        Args:
            degrees (float): Degree range for random rotations.
            translate (float): Fraction of total width and height for random translation.
            scale (float): Scaling factor interval, e.g., a scale factor of 0.5 allows a resize between 50%-150%.
            shear (float): Shear intensity (angle in degrees).
            perspective (float): Perspective distortion factor.
            border (Tuple[int, int]): Tuple specifying mosaic border (top/bottom, left/right).
            pre_transform (Callable | None): Function/transform to apply to the image before starting the random
                transformation.

        Examples:
            >>> transform = RandomPerspective(degrees=10.0, translate=0.1, scale=0.5, shear=5.0)
            >>> result = transform(labels)  # Apply random perspective to labels
        """
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border  # mosaic border
        self.pre_transform = pre_transform

    def affine_transform(self, img: np.ndarray, border: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Apply a sequence of affine transformations centered around the image center.

        This function performs a series of geometric transformations on the input image, including
        translation, perspective change, rotation, scaling, and shearing. The transformations are
        applied in a specific order to maintain consistency.

        Args:
            img (np.ndarray): Input image to be transformed.
            border (Tuple[int, int]): Border dimensions for the transformed image.

        Returns:
            img (np.ndarray): Transformed image.
            M (np.ndarray): 3x3 transformation matrix.
            s (float): Scale factor applied during the transformation.

        Examples:
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> border = (10, 10)
            >>> transformed_img, matrix, scale = affine_transform(img, border)
        """
        # Center
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
            if img.ndim == 2:
                img = img[..., None]
        return img, M, s

    def apply_bboxes(self, bboxes: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Apply affine transformation to bounding boxes.

        This function applies an affine transformation to a set of bounding boxes using the provided
        transformation matrix.

        Args:
            bboxes (np.ndarray): Bounding boxes in xyxy format with shape (N, 4), where N is the number
                of bounding boxes.
            M (np.ndarray): Affine transformation matrix with shape (3, 3).

        Returns:
            (np.ndarray): Transformed bounding boxes in xyxy format with shape (N, 4).

        Examples:
            >>> bboxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])
            >>> M = torch.eye(3)
            >>> transformed_bboxes = apply_bboxes(bboxes, M)
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def apply_segments(self, segments: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply affine transformations to segments and generate new bounding boxes.

        This function applies affine transformations to input segments and generates new bounding boxes based on
        the transformed segments. It clips the transformed segments to fit within the new bounding boxes.

        Args:
            segments (np.ndarray): Input segments with shape (N, M, 2), where N is the number of segments and M is the
                number of points in each segment.
            M (np.ndarray): Affine transformation matrix with shape (3, 3).

        Returns:
            bboxes (np.ndarray): New bounding boxes with shape (N, 4) in xyxy format.
            segments (np.ndarray): Transformed and clipped segments with shape (N, M, 2).

        Examples:
            >>> segments = np.random.rand(10, 500, 2)  # 10 segments with 500 points each
            >>> M = np.eye(3)  # Identity transformation matrix
            >>> new_bboxes, new_segments = apply_segments(segments, M)
        """
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0)
        segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        return bboxes, segments

    def apply_keypoints(self, keypoints: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Apply affine transformation to keypoints.

        This method transforms the input keypoints using the provided affine transformation matrix. It handles
        perspective rescaling if necessary and updates the visibility of keypoints that fall outside the image
        boundaries after transformation.

        Args:
            keypoints (np.ndarray): Array of keypoints with shape (N, 17, 3), where N is the number of instances,
                17 is the number of keypoints per instance, and 3 represents (x, y, visibility).
            M (np.ndarray): 3x3 affine transformation matrix.

        Returns:
            (np.ndarray): Transformed keypoints array with the same shape as input (N, 17, 3).

        Examples:
            >>> random_perspective = RandomPerspective()
            >>> keypoints = np.random.rand(5, 17, 3)  # 5 instances, 17 keypoints each
            >>> M = np.eye(3)  # Identity transformation
            >>> transformed_keypoints = random_perspective.apply_keypoints(keypoints, M)
        """
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > self.size[0]) | (xy[:, 1] > self.size[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply random perspective and affine transformations to an image and its associated labels.

        This method performs a series of transformations including rotation, translation, scaling, shearing,
        and perspective distortion on the input image and adjusts the corresponding bounding boxes, segments,
        and keypoints accordingly.

        Args:
            labels (Dict[str, Any]): A dictionary containing image data and annotations.
                Must include:
                    'img' (np.ndarray): The input image.
                    'cls' (np.ndarray): Class labels.
                    'instances' (Instances): Object instances with bounding boxes, segments, and keypoints.
                May include:
                    'mosaic_border' (Tuple[int, int]): Border size for mosaic augmentation.

        Returns:
            (Dict[str, Any]): Transformed labels dictionary containing:
                - 'img' (np.ndarray): The transformed image.
                - 'cls' (np.ndarray): Updated class labels.
                - 'instances' (Instances): Updated object instances.
                - 'resized_shape' (Tuple[int, int]): New image shape after transformation.

        Examples:
            >>> transform = RandomPerspective()
            >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> labels = {
            ...     "img": image,
            ...     "cls": np.array([0, 1, 2]),
            ...     "instances": Instances(bboxes=np.array([[10, 10, 50, 50], [100, 100, 150, 150]])),
            ... }
            >>> result = transform(labels)
            >>> assert result["img"].shape[:2] == result["resized_shape"]
        """
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # do not need ratio pad

        img = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")
        # Make sure the coord formats are right
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # Scale for func:`box_candidates`
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # Update bboxes if there are segments.
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        # Clip
        new_instances.clip(*self.size)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    @staticmethod
    def box_candidates(
        box1: np.ndarray,
        box2: np.ndarray,
        wh_thr: int = 2,
        ar_thr: int = 100,
        area_thr: float = 0.1,
        eps: float = 1e-16,
    ) -> np.ndarray:
        """
        Compute candidate boxes for further processing based on size and aspect ratio criteria.

        This method compares boxes before and after augmentation to determine if they meet specified
        thresholds for width, height, aspect ratio, and area. It's used to filter out boxes that have
        been overly distorted or reduced by the augmentation process.

        Args:
            box1 (np.ndarray): Original boxes before augmentation, shape (4, N) where n is the
                number of boxes. Format is [x1, y1, x2, y2] in absolute coordinates.
            box2 (np.ndarray): Augmented boxes after transformation, shape (4, N). Format is
                [x1, y1, x2, y2] in absolute coordinates.
            wh_thr (int): Width and height threshold in pixels. Boxes smaller than this in either
                dimension are rejected.
            ar_thr (int): Aspect ratio threshold. Boxes with an aspect ratio greater than this
                value are rejected.
            area_thr (float): Area ratio threshold. Boxes with an area ratio (new/old) less than
                this value are rejected.
            eps (float): Small epsilon value to prevent division by zero.

        Returns:
            (np.ndarray): Boolean array of shape (n) indicating which boxes are candidates.
                True values correspond to boxes that meet all criteria.

        Examples:
            >>> random_perspective = RandomPerspective()
            >>> box1 = np.array([[0, 0, 100, 100], [0, 0, 50, 50]]).T
            >>> box2 = np.array([[10, 10, 90, 90], [5, 5, 45, 45]]).T
            >>> candidates = random_perspective.box_candidates(box1, box2)
            >>> print(candidates)
            [True True]
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


class RandomHSV:
    """
    Randomly adjust the Hue, Saturation, and Value (HSV) channels of an image.

    This class applies random HSV augmentation to images within predefined limits set by hgain, sgain, and vgain.

    Attributes:
        hgain (float): Maximum variation for hue. Range is typically [0, 1].
        sgain (float): Maximum variation for saturation. Range is typically [0, 1].
        vgain (float): Maximum variation for value. Range is typically [0, 1].

    Methods:
        __call__: Apply random HSV augmentation to an image.

    Examples:
        >>> import numpy as np
        >>> from ultralytics.data.augment import RandomHSV
        >>> augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> labels = {"img": image}
        >>> augmenter(labels)
        >>> augmented_image = augmented_labels["img"]
    """

    def __init__(self, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5) -> None:
        """
        Initialize the RandomHSV object for random HSV (Hue, Saturation, Value) augmentation.

        This class applies random adjustments to the HSV channels of an image within specified limits.

        Args:
            hgain (float): Maximum variation for hue. Should be in the range [0, 1].
            sgain (float): Maximum variation for saturation. Should be in the range [0, 1].
            vgain (float): Maximum variation for value. Should be in the range [0, 1].

        Examples:
            >>> hsv_aug = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> hsv_aug(image)
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply random HSV augmentation to an image within predefined limits.

        This method modifies the input image by randomly adjusting its Hue, Saturation, and Value (HSV) channels.
        The adjustments are made within the limits set by hgain, sgain, and vgain during initialization.

        Args:
            labels (Dict[str, Any]): A dictionary containing image data and metadata. Must include an 'img' key with
                the image as a numpy array.

        Returns:
            (Dict[str, Any]): A dictionary containing the mixed image and adjusted labels.

        Examples:
            >>> hsv_augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> labels = {"img": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)}
            >>> labels = hsv_augmenter(labels)
            >>> augmented_img = labels["img"]
        """
        img = labels["img"]
        if img.shape[-1] != 3:  # only apply to RGB images
            return labels
        if self.hgain or self.sgain or self.vgain:
            dtype = img.dtype  # uint8

            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]  # random gains
            x = np.arange(0, 256, dtype=r.dtype)
            # lut_hue = ((x * (r[0] + 1)) % 180).astype(dtype)   # original hue implementation from ultralytics<=8.3.78
            lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
            lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
            lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
            lut_sat[0] = 0  # prevent pure white changing color, introduced in 8.3.79

            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return labels


class RandomFlip:
    """
    Apply a random horizontal or vertical flip to an image with a given probability.

    This class performs random image flipping and updates corresponding instance annotations such as
    bounding boxes and keypoints.

    Attributes:
        p (float): Probability of applying the flip. Must be between 0 and 1.
        direction (str): Direction of flip, either 'horizontal' or 'vertical'.
        flip_idx (array-like): Index mapping for flipping keypoints, if applicable.

    Methods:
        __call__: Apply the random flip transformation to an image and its annotations.

    Examples:
        >>> transform = RandomFlip(p=0.5, direction="horizontal")
        >>> result = transform({"img": image, "instances": instances})
        >>> flipped_image = result["img"]
        >>> flipped_instances = result["instances"]
    """

    def __init__(self, p: float = 0.5, direction: str = "horizontal", flip_idx: List[int] = None) -> None:
        """
        Initialize the RandomFlip class with probability and direction.

        This class applies a random horizontal or vertical flip to an image with a given probability.
        It also updates any instances (bounding boxes, keypoints, etc.) accordingly.

        Args:
            p (float): The probability of applying the flip. Must be between 0 and 1.
            direction (str): The direction to apply the flip. Must be 'horizontal' or 'vertical'.
            flip_idx (List[int] | None): Index mapping for flipping keypoints, if any.

        Raises:
            AssertionError: If direction is not 'horizontal' or 'vertical', or if p is not between 0 and 1.

        Examples:
            >>> flip = RandomFlip(p=0.5, direction="horizontal")
            >>> flip_with_idx = RandomFlip(p=0.7, direction="vertical", flip_idx=[1, 0, 3, 2, 5, 4])
        """
        assert direction in {"horizontal", "vertical"}, f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply random flip to an image and update any instances like bounding boxes or keypoints accordingly.

        This method randomly flips the input image either horizontally or vertically based on the initialized
        probability and direction. It also updates the corresponding instances (bounding boxes, keypoints) to
        match the flipped image.

        Args:
            labels (Dict[str, Any]): A dictionary containing the following keys:
                'img' (np.ndarray): The image to be flipped.
                'instances' (ultralytics.utils.instance.Instances): An object containing bounding boxes and
                    optionally keypoints.

        Returns:
            (Dict[str, Any]): The same dictionary with the flipped image and updated instances:
                'img' (np.ndarray): The flipped image.
                'instances' (ultralytics.utils.instance.Instances): Updated instances matching the flipped image.

        Examples:
            >>> labels = {"img": np.random.rand(640, 640, 3), "instances": Instances(...)}
            >>> random_flip = RandomFlip(p=0.5, direction="horizontal")
            >>> flipped_labels = random_flip(labels)
        """
        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        # WARNING: two separate if and calls to random.random() intentional for reproducibility with older versions
        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        labels["img"] = np.ascontiguousarray(img)
        labels["instances"] = instances
        return labels


class LetterBox:
    """
    Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates
    corresponding labels and bounding boxes.

    Attributes:
        new_shape (tuple): Target shape (height, width) for resizing.
        auto (bool): Whether to use minimum rectangle.
        scale_fill (bool): Whether to stretch the image to new_shape.
        scaleup (bool): Whether to allow scaling up. If False, only scale down.
        stride (int): Stride for rounding padding.
        center (bool): Whether to center the image or align to top-left.

    Methods:
        __call__: Resize and pad image, update labels and bounding boxes.

    Examples:
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result["img"]
        >>> updated_instances = result["instances"]
    """

    def __init__(
        self,
        new_shape: Tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
    ):
        """
        Initialize LetterBox object for resizing and padding images.

        This class is designed to resize and pad images for object detection, instance segmentation, and pose estimation
        tasks. It supports various resizing modes including auto-sizing, scale-fill, and letterboxing.

        Args:
            new_shape (Tuple[int, int]): Target size (height, width) for the resized image.
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly.
            scale_fill (bool): If True, stretch the image to new_shape without padding.
            scaleup (bool): If True, allow scaling up. If False, only scale down.
            center (bool): If True, center the placed image. If False, place image in top-left corner.
            stride (int): Stride of the model (e.g., 32 for YOLOv5).

        Attributes:
            new_shape (Tuple[int, int]): Target size for the resized image.
            auto (bool): Flag for using minimum rectangle resizing.
            scale_fill (bool): Flag for stretching image without padding.
            scaleup (bool): Flag for allowing upscaling.
            stride (int): Stride value for ensuring image size is divisible by stride.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, stride=32)
            >>> resized_img = letterbox(original_img)
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels: Dict[str, Any] = None, image: np.ndarray = None) -> Union[Dict[str, Any], np.ndarray]:
        """
        Resize and pad an image for object detection, instance segmentation, or pose estimation tasks.

        This method applies letterboxing to the input image, which involves resizing the image while maintaining its
        aspect ratio and adding padding to fit the new shape. It also updates any associated labels accordingly.

        Args:
            labels (Dict[str, Any] | None): A dictionary containing image data and associated labels, or empty dict if None.
            image (np.ndarray | None): The input image as a numpy array. If None, the image is taken from 'labels'.

        Returns:
            (Dict[str, Any] | nd.ndarray): If 'labels' is provided, returns an updated dictionary with the resized and padded image,
                updated labels, and additional metadata. If 'labels' is empty, returns the resized
                and padded image.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> result = letterbox(labels={"img": np.zeros((480, 640, 3)), "instances": Instances(...)})
            >>> resized_img = result["img"]
            >>> updated_instances = result["instances"]
        """
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            if img.ndim == 2:
                img = img[..., None]

        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        h, w, c = img.shape
        if c == 3:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        else:  # multispectral
            pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=114, dtype=img.dtype)
            pad_img[top : top + h, left : left + w] = img
            img = pad_img

        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, left, top)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    @staticmethod
    def _update_labels(labels: Dict[str, Any], ratio: Tuple[float, float], padw: float, padh: float) -> Dict[str, Any]:
        """
        Update labels after applying letterboxing to an image.

        This method modifies the bounding box coordinates of instances in the labels
        to account for resizing and padding applied during letterboxing.

        Args:
            labels (Dict[str, Any]): A dictionary containing image labels and instances.
            ratio (Tuple[float, float]): Scaling ratios (width, height) applied to the image.
            padw (float): Padding width added to the image.
            padh (float): Padding height added to the image.

        Returns:
            (Dict[str, Any]): Updated labels dictionary with modified instance coordinates.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> labels = {"instances": Instances(...)}
            >>> ratio = (0.5, 0.5)
            >>> padw, padh = 10, 20
            >>> updated_labels = letterbox._update_labels(labels, ratio, padw, padh)
        """
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class CopyPaste(BaseMixTransform):
    """
    CopyPaste class for applying Copy-Paste augmentation to image datasets.

    This class implements the Copy-Paste augmentation technique as described in the paper "Simple Copy-Paste is a Strong
    Data Augmentation Method for Instance Segmentation" (https://arxiv.org/abs/2012.07177). It combines objects from
    different images to create new training samples.

    Attributes:
        dataset (Any): The dataset to which Copy-Paste augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before Copy-Paste.
        p (float): Probability of applying Copy-Paste augmentation.

    Methods:
        _mix_transform: Apply Copy-Paste augmentation to the input labels.
        __call__: Apply the Copy-Paste transformation to images and annotations.

    Examples:
        >>> from ultralytics.data.augment import CopyPaste
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> copypaste = CopyPaste(dataset, p=0.5)
        >>> augmented_labels = copypaste(original_labels)
    """

    def __init__(self, dataset=None, pre_transform=None, p: float = 0.5, mode: str = "flip") -> None:
        """Initialize CopyPaste object with dataset, pre_transform, and probability of applying MixUp."""
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        assert mode in {"flip", "mixup"}, f"Expected `mode` to be `flip` or `mixup`, but got {mode}."
        self.mode = mode

    def _mix_transform(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Copy-Paste augmentation to combine objects from another image into the current image."""
        labels2 = labels["mix_labels"][0]
        return self._transform(labels, labels2)

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Copy-Paste augmentation to an image and its labels."""
        if len(labels["instances"].segments) == 0 or self.p == 0:
            return labels
        if self.mode == "flip":
            return self._transform(labels)

        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic or MixUp
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # Update cls and texts
        labels = self._update_label_text(labels)
        # Mosaic or MixUp
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _transform(self, labels1: Dict[str, Any], labels2: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Apply Copy-Paste augmentation to combine objects from another image into the current image."""
        im = labels1["img"]
        if "mosaic_border" not in labels1:
            im = im.copy()  # avoid modifying original non-mosaic image
        cls = labels1["cls"]
        h, w = im.shape[:2]
        instances = labels1.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)

        im_new = np.zeros(im.shape, np.uint8)
        instances2 = labels2.pop("instances", None)
        if instances2 is None:
            instances2 = deepcopy(instances)
            instances2.fliplr(w)
        ioa = bbox_ioa(instances2.bboxes, instances.bboxes)  # intersection over area, (N, M)
        indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
        n = len(indexes)
        sorted_idx = np.argsort(ioa.max(1)[indexes])
        indexes = indexes[sorted_idx]
        for j in indexes[: round(self.p * n)]:
            cls = np.concatenate((cls, labels2.get("cls", cls)[[j]]), axis=0)
            instances = Instances.concatenate((instances, instances2[[j]]), axis=0)
            cv2.drawContours(im_new, instances2.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

        result = labels2.get("img", cv2.flip(im, 1))  # augment segments
        if result.ndim == 2:  # cv2.flip would eliminate the last dimension for grayscale images
            result = result[..., None]
        i = im_new.astype(bool)
        im[i] = result[i]

        labels1["img"] = im
        labels1["cls"] = cls
        labels1["instances"] = instances
        return labels1


class Albumentations:
    """
    Albumentations transformations for image augmentation.

    This class applies various image transformations using the Albumentations library. It includes operations such as
    Blur, Median Blur, conversion to grayscale, Contrast Limited Adaptive Histogram Equalization (CLAHE), random changes
    in brightness and contrast, RandomGamma, and image quality reduction through compression.

    Attributes:
        p (float): Probability of applying the transformations.
        transform (albumentations.Compose): Composed Albumentations transforms.
        contains_spatial (bool): Indicates if the transforms include spatial operations.

    Methods:
        __call__: Apply the Albumentations transformations to the input labels.

    Examples:
        >>> transform = Albumentations(p=0.5)
        >>> augmented_labels = transform(labels)

    Notes:
        - The Albumentations package must be installed to use this class.
        - If the package is not installed or an error occurs during initialization, the transform will be set to None.
        - Spatial transforms are handled differently and require special processing for bounding boxes.
    """

    def __init__(self, p: float = 1.0) -> None:
        """
        Initialize the Albumentations transform object for YOLO bbox formatted parameters.

        This class applies various image augmentations using the Albumentations library, including Blur, Median Blur,
        conversion to grayscale, Contrast Limited Adaptive Histogram Equalization, random changes of brightness and
        contrast, RandomGamma, and image quality reduction through compression.

        Args:
            p (float): Probability of applying the augmentations. Must be between 0 and 1.

        Attributes:
            p (float): Probability of applying the augmentations.
            transform (albumentations.Compose): Composed Albumentations transforms.
            contains_spatial (bool): Indicates if the transforms include spatial transformations.

        Raises:
            ImportError: If the Albumentations package is not installed.
            Exception: For any other errors during initialization.

        Examples:
            >>> transform = Albumentations(p=0.5)
            >>> augmented = transform(image=image, bboxes=bboxes, class_labels=classes)
            >>> augmented_image = augmented["image"]
            >>> augmented_bboxes = augmented["bboxes"]

        Notes:
            - Requires Albumentations version 1.0.3 or higher.
            - Spatial transforms are handled differently to ensure bbox compatibility.
            - Some transforms are applied with very low probability (0.01) by default.
        """
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")

        try:
            import os

            os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # suppress Albumentations upgrade message
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            # List of possible spatial transforms
            spatial_transforms = {
                "Affine",
                "BBoxSafeRandomCrop",
                "CenterCrop",
                "CoarseDropout",
                "Crop",
                "CropAndPad",
                "CropNonEmptyMaskIfExists",
                "D4",
                "ElasticTransform",
                "Flip",
                "GridDistortion",
                "GridDropout",
                "HorizontalFlip",
                "Lambda",
                "LongestMaxSize",
                "MaskDropout",
                "MixUp",
                "Morphological",
                "NoOp",
                "OpticalDistortion",
                "PadIfNeeded",
                "Perspective",
                "PiecewiseAffine",
                "PixelDropout",
                "RandomCrop",
                "RandomCropFromBorders",
                "RandomGridShuffle",
                "RandomResizedCrop",
                "RandomRotate90",
                "RandomScale",
                "RandomSizedBBoxSafeCrop",
                "RandomSizedCrop",
                "Resize",
                "Rotate",
                "SafeRotate",
                "ShiftScaleRotate",
                "SmallestMaxSize",
                "Transpose",
                "VerticalFlip",
                "XYMasking",
            }  # from https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms

            # Transforms
            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_range=(75, 100), p=0.0),
            ]

            # Compose transforms
            self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
            self.transform = (
                A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
                if self.contains_spatial
                else A.Compose(T)
            )
            if hasattr(self.transform, "set_random_seed"):
                # Required for deterministic transforms in albumentations>=1.4.21
                self.transform.set_random_seed(torch.initial_seed())
            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass