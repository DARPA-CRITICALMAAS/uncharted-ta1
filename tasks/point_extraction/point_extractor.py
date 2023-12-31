from tasks.point_extraction.entities import MapTile, MapTiles, MapPointLabel, MapImage
from tasks.point_extraction.pytorch.mobilenet_rcnn import MobileNetRCNN
from tasks.point_extraction.pytorch.utils import PointInferenceDataset

from tasks.common.s3_data_cache import S3DataCache
from tasks.common.task import Task, TaskInput, TaskResult

import os
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Dict

import parmap
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results


class MobileNetPointDetector(Task):
    """
    Model for detecting points in images. Predicts the location of points using a Pytorch model.
    Predicts directionality features of each point, if necessary, using a separate model.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _VERSION = 1
    _PYTORCH_MODEL = MobileNetRCNN

    LABEL_MAPPING = {
        "mine_pt": 1,
        "stike_and_dip": 2,
        "strikedip": 3,
        "strike_dip": 4,
        "inclined_bedding": 5,
        "overturned_bedding": 6,
        "gravel_pit_pt": 7,
    }

    def __init__(self, task_id: str, path: str) -> None:
        self.model = self._PYTORCH_MODEL.from_pretrained(path)
        self.model.eval()
        self.model.to(self.device)
        super().__init__(task_id)

    @staticmethod
    def dataloader_factory(images: List[MapTile]) -> DataLoader:
        dataset = PointInferenceDataset(tiles=images)
        return DataLoader(
            dataset=dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            collate_fn=default_collate,
        )

    @staticmethod
    def model_factory(model_name: str):
        raise NotImplementedError

    def reformat_output(self, output: Dict) -> List[MapPointLabel]:
        """
        Reformats Pytorch model output to match the MapPointLabel schema.
        """

        formatted_data: List[MapPointLabel] = []

        id_to_name = {v: k for k, v in self.LABEL_MAPPING.items()}

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            formatted_data.append(
                MapPointLabel(
                    classifier_name=type(self).__name__,
                    classifier_version=self._VERSION,
                    class_id=label,
                    class_name=id_to_name[label],
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    score=float(score),
                )
            )
        return formatted_data

    def to(self, device: str) -> None:
        """
        Move underlying Pytorch model to specified device.
        """
        self.model.to(device)

    @classmethod
    def load(cls, task_id: str, path: str):
        return cls(task_id, path)

    @property
    def version(self):
        return self._VERSION

    @property
    def input_type(self):
        return List[MapTile]

    @property
    def output_type(self):
        return List[MapTile]

    def run(
        self,
        input: TaskInput,
    ) -> TaskResult:
        """
        Prediction utility for inference and evaluation.
        """
        map_tiles = MapTiles.model_validate(input.data["map_tiles"])
        dataloader = self.dataloader_factory(images=map_tiles.tiles)

        predictions: List[MapTile] = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Running {type(self).__name__}"):
                images, metadata = batch
                outputs = self.model(images)
                for idx, image in enumerate(images):
                    predictions.append(
                        MapTile(
                            x_offset=int(metadata["x_offset"][idx].item()),
                            y_offset=int(metadata["y_offset"][idx].item()),
                            width=int(metadata["width"][idx].item()),
                            height=int(metadata["height"][idx].item()),
                            image=image,
                            map_path=metadata["map_path"][idx],
                            predictions=self.reformat_output(outputs[idx]),
                        )
                    )
        result_map_tiles = MapTiles(raster_id=map_tiles.raster_id, tiles=predictions)
        return TaskResult(
            task_id=self._task_id,
            output={"map_tiles": result_map_tiles.model_dump()},
        )


class PointDirectionPredictor(Task):
    _VERSION = 1
    SUPPORTED_CLASSES = ["strike_and_dip"]
    TEMPLATE_PATH = "tasks/point_extraction/templates/strike_dip_implied_transparent_black_north.jpg"
    template: np.ndarray = np.empty((0, 0))

    def __init__(self, task_id: str):
        super().__init__(task_id)

    def load_template(self):
        template = np.array(Image.open(self.TEMPLATE_PATH))
        self.template = self.global_thresholding(
            template, otsu=False, threshold_value=225
        )  # Blacken template.
        self.template = template

    def _check_size(self, label: MapPointLabel, template: np.ndarray) -> bool:
        """
        Used to palliate issues arising from tiling..
        Temporary fix which prevents propagating tiling errors to directionality pred.
        """
        x1, y1, x2, y2 = label.x1, label.y1, label.x2, label.y2
        template_size = template.shape[:2]
        label_size = (x2 - x1, y2 - y1)
        if label_size[0] < template_size[0] or label_size[1] < template_size[1]:
            return False
        return True

    @staticmethod
    def create_mask(template: np.ndarray, ignore_color=[255, 255, 255]):
        """
        Creates a binary mask where pixels equal to ignore_color are set to 0, and all others are set to 1.

        Parameters:
        - template (np.array): The template image.
        - ignore_color (list of int): The RGB color to ignore. Default is white, [255, 255, 255].

        Returns:
        np.array: The binary mask.
        """
        # Check if the image is grayscale
        if len(template.shape) == 2:
            # If it's grayscale, create a simple binary mask
            mask = np.ones_like(template, dtype=np.uint8) * 255
        else:
            # If it's not grayscale, create a mask ignoring a specific color
            mask = np.all(template != ignore_color, axis=-1).astype(np.uint8) * 255

        return mask

    @staticmethod
    def rotate_image(image: np.ndarray, angle, border_color=(255, 255, 255)):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos_theta = np.abs(M[0, 0])
        sin_theta = np.abs(M[0, 1])
        new_w = int(h * sin_theta + w * cos_theta)
        new_h = int(h * cos_theta + w * sin_theta)

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_color,
        )

        return rotated

    @staticmethod
    def crop_edges(img: np.ndarray, size=(70, 70)):
        h, w = img.shape[:2]
        h_crop, w_crop = size
        h_offset, w_offset = (h - h_crop) // 2, (w - w_crop) // 2
        return img[h_offset : h_offset + h_crop, w_offset : w_offset + w_crop]

    def match_template(self, base_image: np.ndarray, template: np.ndarray):
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        base_image = base_image.astype(np.float32)
        template = template.astype(np.float32)

        diag_length = int(
            np.ceil(np.sqrt(template.shape[0] ** 2 + template.shape[1] ** 2))
        )
        pad_y = max(diag_length - base_image.shape[0], base_image.shape[0])
        pad_x = max(diag_length - base_image.shape[1], base_image.shape[1])
        base_image = self.pad_image(
            base_image,
            pad_size=(base_image.shape[0] + pad_y, base_image.shape[1] + pad_x),
        )
        if len(base_image.shape) == 3:
            base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        base_image = base_image.astype(np.float32)
        mask = self.create_mask(template)
        result = cv2.matchTemplate(base_image, template, cv2.TM_CCOEFF, mask=mask)
        return result

    @staticmethod
    def global_thresholding(img: np.ndarray, threshold_value=50, otsu=False):
        """
        Applies global thresholding to an image and displays the original and thresholded image.

        Parameters:
        - img (np.array): The input image in NumPy array format.
        - threshold_value (int): The global threshold value.
        """
        # Handle images with alpha channels.
        if len(img.shape) > 2 and img.shape[2] == 4:
            # Separate the alpha channel
            bgr, alpha = img[..., :3], img[..., 3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        elif len(img.shape) > 2 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            alpha = None
        else:
            gray = img.copy()
            alpha = None

        thresh = cv2.THRESH_BINARY

        if otsu:
            thresh = cv2.THRESH_OTSU

        _, thresh_img = cv2.threshold(gray, threshold_value, 255, thresh)

        # If the image has an alpha channel, merge the thresholded image and the alpha channel
        if alpha is not None and img.shape[2] == 4:
            thresh_img = cv2.merge((thresh_img, thresh_img, thresh_img, alpha))

        return thresh_img

    def _fetch_template_match_candidates(
        self, labels: List[MapPointLabel], class_name="strike_and_dip"
    ):
        if class_name not in self.SUPPORTED_CLASSES:
            raise RuntimeError(
                f"Class {class_name} is not supported by this predictor."
            )
        return [(i, p) for i, p in enumerate(labels) if p.class_name == class_name]

    def _construct_base_candidates(self, labels: List[MapPointLabel], img: MapImage):
        return [
            np.array(img.image.crop((p.x1, p.y1, p.x2, p.y2)))
            for p in labels
            if self._check_size(p, template=self.template)
        ]

    def pad_image(self, img: np.ndarray, pad_size=(100, 100)):
        h, w = img.shape[:2]
        h_pad, w_pad = pad_size
        h_offset, w_offset = (h_pad - h) // 2, (w_pad - w) // 2

        if len(img.shape) == 3:
            num_channels = img.shape[2]
        else:
            num_channels = 1

        if num_channels == 3:
            padded_img = np.ones((h_pad, w_pad, 3), dtype=np.uint8) * 255
            padded_img[h_offset : h_offset + h, w_offset : w_offset + w, :] = img
        else:
            padded_img = np.ones((h_pad, w_pad), dtype=np.uint8) * 255
            padded_img[h_offset : h_offset + h, w_offset : w_offset + w] = img

        return padded_img

    def predict(self, base_image: np.ndarray):
        """
        Parameters:
        - template (np.array): The template image.
        - base_image (np.array): The base image.
        """
        if self.template.shape[0] == 0:
            self.load_template()

        base_image = self.global_thresholding(base_image)

        deg = []
        score = []
        best_score = -np.inf
        best_angle = 0

        for i in range(0, 360, 1):
            rotated_template = self.rotate_image(self.template[:, :], i)
            result = self.match_template(base_image, rotated_template)
            base_image_midpoint = np.array(base_image.shape[:2]) // 2
            result = result[base_image_midpoint[0] - 10 : base_image_midpoint[0] + 10]
            max_score = result.max()

            if max_score > best_score:
                best_score = max_score
                best_angle = i
            deg.append(i)
            score.append(max_score)

        return 360 - best_angle

    def run(self, input: TaskInput) -> TaskResult:
        """
        Run batch predictions over a MapImage object.

        This modifies the MapImage object predictions inplace. Unit test this.
        """

        if PointDirectionPredictor.template.shape[0] == 0:
            self.load_template()
        input_map_image = MapImage.model_validate(
            input.data["map_image"]
        )  # todo should just use the task input
        map_image = input_map_image.model_copy()
        if map_image.labels is None:
            raise RuntimeError("MapImage must have labels to run batch_predict.")
        match_candidates = self._fetch_template_match_candidates(
            map_image.labels
        )  # [[idx, label], ...] Use idx to add direction to labels.
        idxs = [idx for idx, _ in match_candidates]
        labels_with_direction = [label for _, label in match_candidates]
        base_images = self._construct_base_candidates(
            labels_with_direction, map_image
        )  # Crop map to regions of interest.
        direction_preds = parmap.map(self.predict, base_images, pm_pbar=True)
        for idx, direction in zip(idxs, direction_preds):
            map_image.labels[idx].directionality = direction
        return TaskResult(
            task_id=self._task_id, output={"map_image": map_image.model_dump()}
        )

    @property
    def input_type(self):
        return MapImage

    @property
    def output_type(self):
        return MapImage


class YOLOPointDetector(Task):
    """
    Wrapper for Ultralytics YOLOv8 model inference.
    """

    _VERSION = 1

    def __init__(
        self,
        task_id: str,
        model_data_path: str,
        cache_path: str,
        bsz: int = 15,
        device: str = "auto",
    ):
        local_data_path = self._prep_model_data(model_data_path, cache_path)
        self.model = YOLO(local_data_path)
        self.bsz = bsz
        self.device = device

        super().__init__(task_id)

    def _prep_model_data(self, model_data_path: str, data_cache_path: str) -> Path:
        """
        prepare local data cache and download model weights, if needed

        Args:
            model_data_path (str): The path to the folder containing the model weights
            data_cache_path (str): The path to the local data cache

        Returns:
            Path: The path to the local copy of the model weights file
        """

        local_model_data_path = None

        # check if path is a URL
        if model_data_path.startswith("s3://") or model_data_path.startswith("http"):
            # need to specify data cache path when fetching from S3
            if data_cache_path == "":
                raise ValueError(
                    "'data_cache_path' must be specified when fetching model data from S3"
                )

            s3_host = ""
            s3_path = ""
            s3_bucket = ""

            res = urlparse(model_data_path)
            s3_host = res.scheme + "://" + res.netloc
            s3_path = res.path.lstrip("/")
            s3_bucket = s3_path.split("/")[0]
            s3_path = s3_path.lstrip(s3_bucket)
            s3_path = s3_path.lstrip("/")

            # create local data cache, if doesn't exist, and connect to S3
            s3_data_cache = S3DataCache(
                data_cache_path,
                s3_host,
                s3_bucket,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "<UNSET>"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "<UNSET>"),
            )

            # check for model weights and config files in the folder
            s3_subfolder = s3_path[: s3_path.rfind("/")]
            for s3_key in s3_data_cache.list_bucket_contents(s3_subfolder):
                if s3_key.endswith(".pt"):
                    local_model_data_path = Path(
                        s3_data_cache.fetch_file_from_s3(s3_key, overwrite=False)
                    )
        else:
            # check for model weights and config files in the folder
            # iterate over files in folder
            for f in Path(model_data_path).iterdir():
                if f.is_file():
                    if f.suffix == ".pth":
                        local_model_data_path = f

        # check that we have all the files we need
        if not local_model_data_path or not local_model_data_path.is_file():
            raise ValueError(f"Model weights file not found at {model_data_path}")

        return local_model_data_path

    def process_output(self, predictions: Results) -> List[MapPointLabel]:
        pt_labels = []
        for pred in predictions:
            assert pred.boxes is not None
            assert isinstance(pred.boxes.data, torch.Tensor)
            assert isinstance(self.model.names, Dict)

            if len(pred.boxes.data) == 0:
                continue
            for box in pred.boxes.data.detach().cpu().tolist():
                x1, y1, x2, y2, score, class_id = box
                pt_labels.append(
                    MapPointLabel(
                        classifier_name="unchartNet_point_extractor",
                        classifier_version=self._VERSION,
                        class_id=int(class_id),
                        class_name=self.model.names[int(class_id)],
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        score=score,
                    )
                )
        return pt_labels

    def run(self, input: TaskInput) -> TaskResult:
        map_tiles = MapTiles.model_validate(input.data["map_tiles"])

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {self.device}")

        output: List[MapTile] = []
        for i in tqdm(range(0, len(map_tiles.tiles), self.bsz)):
            print(f"Processing batch {i} to {i + self.bsz}")
            batch = map_tiles.tiles[i : i + self.bsz]
            images = [tile.image for tile in batch]
            batch_preds = self.model.predict(images, imgsz=768, device=self.device)
            for tile, preds in zip(batch, batch_preds):
                tile.predictions = self.process_output(preds)
                output.append(tile)
        result_map_tiles = MapTiles(raster_id=map_tiles.raster_id, tiles=output)
        return TaskResult(
            task_id=self._task_id, output={"map_tiles": result_map_tiles.model_dump()}
        )

    @property
    def version(self):
        return self._VERSION

    @property
    def input_type(self):
        return List[MapTile]

    @property
    def output_type(self):
        return List[MapTile]
