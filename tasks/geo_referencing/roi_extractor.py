import logging
from shapely.geometry import Polygon
from shapely import MultiPolygon, concave_hull
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.segmentation.entities import (
    MapSegmentation,
    SEGMENTATION_OUTPUT_KEY,
    SEGMENT_MAP_CLASS,
)
from tasks.segmentation.segmenter_utils import get_segment_bounds
from tasks.geo_referencing.entities import MapROI, ROI_MAP_OUTPUT_KEY
from typing import Tuple

logger = logging.getLogger(__name__)

BUFFER_PERCENT = 0.04
BUFFER_MIN = 100
NEGATIVE_BUFFER_FACTOR = 2.0


class ROIExtractor(Task):
    """
    Use the map segmentation result, and buffer (inwards and outwards) to create a ring ROI
    """

    def run(self, input: TaskInput) -> TaskResult:

        result = self._create_result(input)

        # read segmentation output
        poly_map = []
        if SEGMENTATION_OUTPUT_KEY in input.data:
            segmentation = MapSegmentation.model_validate(
                input.data[SEGMENTATION_OUTPUT_KEY]
            )
            poly_map = get_segment_bounds(segmentation, SEGMENT_MAP_CLASS)

        if not poly_map:
            # no map area found
            logger.warning("No map ROI found")
            return result

        # ---- buffer map ROI inwards and outwards by a percentage of the map size
        # to create a "ring" ROI
        # (use the 1st, highest ranked map segment)
        poly_map = poly_map[0]
        (minx, miny, maxx, maxy) = poly_map.bounds
        buffer_pixels = BUFFER_PERCENT * max(maxx - minx, maxy - miny)
        buffer_pixels = int(max(buffer_pixels, BUFFER_MIN))
        logger.info(f"Using ROI buffer of {buffer_pixels} pixels")

        # apply inner and outer buffer
        poly_outer = poly_map.buffer(buffer_pixels, join_style="mitre")
        poly_inner = poly_map.buffer(
            -NEGATIVE_BUFFER_FACTOR * buffer_pixels, join_style="mitre"
        )
        # check that inner buffering results in a single polygon
        if isinstance(poly_inner, MultiPolygon):
            logger.warning("ROI buffering resulted in a multipolygon; merging into one")
            # polygons may not overlap, so merge into one single 'parent' polygon using concave hull
            poly_inner = concave_hull(poly_inner, ratio=1)
            if not isinstance(poly_inner, Polygon):
                # unexpected, skipp inner buffering
                poly_inner = poly_map

        # limit outer buffer coords within image bounds
        w = input.image.width
        h = input.image.height
        outer_coords = list(
            map(
                lambda x: self._limit_polygon((x[0], x[1]), (0, 0), (w, h)),
                list(poly_outer.exterior.coords),
            )
        )

        map_roi_result = MapROI(
            map_bounds=[(float(x), float(y)) for x, y in poly_map.exterior.coords],
            buffer_outer=outer_coords,
            buffer_inner=[(float(x), float(y)) for x, y in poly_inner.exterior.coords],
        )

        result.add_output(ROI_MAP_OUTPUT_KEY, map_roi_result.model_dump())
        return result

    def _limit_polygon(
        self,
        coord: Tuple[float, float],
        lower_limit: Tuple[float, float],
        upper_limit: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        limit polygon xy coords within a range
        """

        return (
            min(max(lower_limit[0], coord[0]), upper_limit[0]),
            min(max(lower_limit[1], coord[1]), upper_limit[1]),
        )
