import logging
from pathlib import Path
from typing import List

from schema.mappers.cdr import PointsMapper
from tasks.point_extraction.legend_analyzer import PointLegendAnalyzer
from tasks.point_extraction.point_extractor import YOLOPointDetector
from tasks.point_extraction.point_orientation_extractor import PointOrientationExtractor
from tasks.point_extraction.point_extractor_utils import convert_preds_to_bitmasks
from tasks.point_extraction.template_match_point_extractor import (
    TemplateMatchPointExtractor,
)
from tasks.point_extraction.tiling import Tiler, Untiler
from tasks.point_extraction.entities import (
    MapImage,
    LegendPointItems,
    LEGEND_ITEMS_OUTPUT_KEY,
)
from tasks.common.pipeline import (
    BaseModelOutput,
    Pipeline,
    PipelineResult,
    OutputCreator,
    Output,
    ImageDictOutput,
)
from tasks.segmentation.detectron_segmenter import DetectronSegmenter
from tasks.segmentation.denoise_segments import DenoiseSegments
from tasks.text_extraction.text_extractor import TileTextExtractor


logger = logging.getLogger(__name__)

import importlib.metadata

MODEL_NAME = "lara-point-extraction"  # should match name in pyproject.toml
MODEL_VERSION = importlib.metadata.version(MODEL_NAME)


class PointExtractionPipeline(Pipeline):
    """
    Pipeline for extracting map point symbols, orientation, and their associated orientation and magnitude values, if present

    Args:
        model_path: path to point symbol extraction model weights
        model_path_segmenter: path to segmenter model weights
        work_dir: cache directory
    """

    def __init__(
        self,
        model_path: str,
        model_path_segmenter: str,
        work_dir: str,
        verbose=False,
        include_cdr_output=True,
        include_bitmasks_output=False,
        gpu=True,
    ):
        # extract text from image, segmentation to only keep the map area,
        # tile, extract points, untile, predict direction
        logger.info("Initializing Point Extraction Pipeline")
        tasks = []
        tasks.append(
            TileTextExtractor(
                "tile_text", Path(work_dir).joinpath("text"), gamma_correction=0.5
            )
        )
        if model_path_segmenter:
            tasks.extend(
                [
                    DetectronSegmenter(
                        "segmenter",
                        model_path_segmenter,
                        str(Path(work_dir).joinpath("segmentation")),
                        gpu=gpu,
                    ),
                    DenoiseSegments("segment_denoising"),
                ]
            )
        else:
            logger.warning(
                "Not using image segmentation. 'model_path_segmenter' param not given"
            )
        tasks.extend(
            [
                PointLegendAnalyzer("legend_analyzer", ""),
                Tiler("tiling"),
                YOLOPointDetector(
                    "point_detection",
                    model_path,
                    str(Path(work_dir).joinpath("points")),
                    batch_size=20,
                ),
                Untiler("untiling"),
                PointOrientationExtractor("point_orientation_extraction"),
                TemplateMatchPointExtractor(
                    "template_match_point_extraction",
                    str(Path(work_dir).joinpath("template_match_points")),
                ),
            ]
        )

        outputs: List[OutputCreator] = [
            MapPointLabelOutput("map_point_label_output"),
        ]
        if include_cdr_output:
            outputs.append(CDROutput("map_point_label_cdr_output"))
        if include_bitmasks_output:
            outputs.append(BitmasksOutput("map_point_label_bitmasks_output"))

        super().__init__("point_extraction", "Point Extraction", outputs, tasks)
        self._verbose = verbose


class MapPointLabelOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates a MapPointLabel object from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            MapPointLabel: The map point label extraction object.
        """
        map_image = MapImage.model_validate(pipeline_result.data["map_image"])
        return BaseModelOutput(
            pipeline_result.pipeline_id,
            pipeline_result.pipeline_name,
            map_image,
        )


class CDROutput(OutputCreator):
    """
    Create CDR output objects for point extraction pipeline.
    """

    def __init__(self, id):
        """
        Initializes the output creator.

        Args:
            id (str): The ID of the output creator.
        """
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Validates the point extraction pipeline result and converts into the TA1 schema representation

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            Output: The output of the pipeline.
        """
        map_image = MapImage.model_validate(pipeline_result.data["map_image"])
        mapper = PointsMapper(MODEL_NAME, MODEL_VERSION)

        cdr_points = mapper.map_to_cdr(map_image)
        return BaseModelOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, cdr_points
        )


class BitmasksOutput(OutputCreator):
    """
    Create bitmasks output (in CMA-contest format) for point extraction pipeline.
    """

    def __init__(self, id):
        """
        Initializes the output creator.

        Args:
            id (str): The ID of the output creator.
        """
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Validates the point extraction pipeline result and converts to bitmasks

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            Output: The output of the pipeline.
        """
        map_image = MapImage.model_validate(pipeline_result.data["map_image"])
        legend_labels = []
        if LEGEND_ITEMS_OUTPUT_KEY in pipeline_result.data:
            legend_pt_items = LegendPointItems.model_validate(
                pipeline_result.data[LEGEND_ITEMS_OUTPUT_KEY]
            )
            legend_labels = [pt_type.name for pt_type in legend_pt_items.items]

        if pipeline_result.image is None:
            raise ValueError("Pipeline result image is None")
        (w, h) = pipeline_result.image.size
        bitmasks_dict = convert_preds_to_bitmasks(map_image, legend_labels, (w, h))

        return ImageDictOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, bitmasks_dict
        )
