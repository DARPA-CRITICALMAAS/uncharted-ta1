import logging
from pathlib import Path
from typing import List

from schema.mappers.cdr import PointsMapper
from tasks.point_extraction.legend_analyzer import PointLegendAnalyzer
from tasks.point_extraction.point_extractor import YOLOPointDetector
from tasks.point_extraction.point_orientation_extractor import PointOrientationExtractor
from tasks.point_extraction.template_match_point_extractor import (
    TemplateMatchPointExtractor,
)
from tasks.point_extraction.tiling import Tiler, Untiler
from tasks.point_extraction.entities import MapImage
from tasks.common.pipeline import (
    BaseModelOutput,
    Pipeline,
    PipelineResult,
    OutputCreator,
    Output,
)
from tasks.segmentation.detectron_segmenter import DetectronSegmenter
from tasks.text_extraction.text_extractor import TileTextExtractor
from tasks.segmentation.detectron_segmenter import (
    DetectronSegmenter,
    SEGMENTATION_OUTPUT_KEY,
)


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
    ):
        # extract text from image, segmentation to only keep the map area,
        # tile, extract points, untile, predict direction
        logger.info("Initializing Point Extraction Pipeline")
        tasks = []
        tasks.append(
            TileTextExtractor(
                "tile_text",
                Path(work_dir).joinpath("text"),
            )
        )
        if model_path_segmenter:
            tasks.append(
                DetectronSegmenter(
                    "detectron_segmenter",
                    model_path_segmenter,
                    str(Path(work_dir).joinpath("segmentation")),
                )
            )
        else:
            logger.warning(
                "Not using image segmentation. 'model_path_segmenter' param not given"
            )
        tasks.append(PointLegendAnalyzer("legend_analyzer", ""))
        tasks.extend(
            [
                Tiler("tiling"),
                YOLOPointDetector(
                    "point_detection",
                    model_path,
                    str(Path(work_dir).joinpath("points")),
                    batch_size=20,
                ),
                Untiler("untiling"),
                PointOrientationExtractor("point_orientation_extraction"),
            ]
        )
        tasks.append(
            TemplateMatchPointExtractor(
                "template_match_point_extraction",
                str(Path(work_dir).joinpath("template_match_points")),
            ),
        )

        outputs: List[OutputCreator] = [
            MapPointLabelOutput("map_point_label_output"),
        ]
        if include_cdr_output:
            outputs.append(CDROutput("map_point_label_cdr_output"))

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
    OutputCreator for point extraction pipeline.
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
