import logging
from typing import List
from shapely import Polygon
from collections import defaultdict

from tasks.segmentation.entities import (
    SegmentationResult,
    MapSegmentation,
    SEGMENTATION_OUTPUT_KEY,
)
from tasks.segmentation.segmenter_utils import merge_overlapping_polygons
from tasks.common.task import Task, TaskInput, TaskResult


logger = logging.getLogger(__name__)


class DenoiseSegments(Task):
    """
    Task to de-noise segmenter results by filtering out low confidence results, and merging any overlapping segments for a given class
    """

    def __init__(
        self,
        task_id: str,
        denoise_classes: list = ["legend_points_lines", "legend_polygons"],
        denoise_conf_thres: float = 0.75,
    ):
        super().__init__(task_id)

        self.denoise_classes = denoise_classes
        self.denoise_conf_thres = denoise_conf_thres

    def run(self, task_input: TaskInput) -> TaskResult:
        """
        Run the de-noise segmentation task
        """

        if SEGMENTATION_OUTPUT_KEY not in task_input.data:
            logger.warning(
                "No segmentation results avialable. Skipping DenoiseSegments task."
            )
            result = self._create_result(task_input)
            return result

        segmentation = MapSegmentation.model_validate(
            task_input.data[SEGMENTATION_OUTPUT_KEY]
        )

        # group segments by class
        segments_dict = defaultdict(list)
        for seg in segmentation.segments:
            segments_dict[seg.class_label].append(seg)

        # denoise segmentation results...
        segments_out = []
        id_model = segmentation.segments[0].id_model

        for label, segments in segments_dict.items():

            if label not in self.denoise_classes or len(segments) == 1:
                segments_out.extend(segments)
                continue
            elif len(segments) == 0:
                continue

            # filter by confidence threshold and
            # merge any overlapping polygons...
            conf_thres = min(self.denoise_conf_thres, segments[0].confidence)
            segments = list(filter(lambda s: (s.confidence >= conf_thres), segments))

            polys = [Polygon(seg.poly_bounds) for seg in segments]
            polys_conf = [seg.confidence for seg in segments]

            merged_poly = merge_overlapping_polygons(polys)

            if len(merged_poly) < len(polys):
                logger.debug(
                    f"NOTE: Some overlapping segments have been merged for segmenter class: {label}"
                )
                # some polygons have been merged, so truncate confidence lists
                polys_conf = polys_conf[: len(merged_poly)]
                # TODO... ideally, we figure out which ones were merged
                polys = merged_poly

                segs_merged: List[SegmentationResult] = []
                for poly, poly_conf in zip(polys, polys_conf):
                    seg_result = SegmentationResult(
                        poly_bounds=[(x, y) for x, y in poly.exterior.coords],
                        bbox=[
                            poly.bounds[0],
                            poly.bounds[1],
                            poly.bounds[0] + poly.bounds[2],
                            poly.bounds[1] + poly.bounds[3],
                        ],
                        area=poly.area,
                        confidence=poly_conf,
                        class_label=label,
                        id_model=id_model,
                    )
                    segs_merged.append(seg_result)
                segments = segs_merged

            segments_out.extend(segments)

        # save (overwrite) de-noised segments, and save task result
        segmentation.segments = segments_out
        result = self._create_result(task_input)
        result.add_output(SEGMENTATION_OUTPUT_KEY, segmentation.model_dump())
        return result
