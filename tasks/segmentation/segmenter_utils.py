import logging
import math
from tasks.segmentation.entities import MapSegmentation
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def rank_segments(segmentation: MapSegmentation, class_labels: List):
    """
    Post-process the segmentation result, by ranking segments per class
    """
    segments_out = []
    # loop through segmenter class labels, and de-noise / rank segments per class
    for label in class_labels:
        segments = list(
            filter(lambda s: (s.class_label == label), segmentation.segments)
        )
        if len(segments) == 0:
            continue
        # rank the segments by heuristic of confidence x sqrt(area)
        segments.sort(key=lambda s: (s.confidence * math.sqrt(s.area)), reverse=True)
        segments_out.extend(segments)

    # save ranked segments
    segmentation.segments = segments_out


def get_segment_bounds(
    segmentation: MapSegmentation, segment_class: str, max_results: int = 0
) -> List[Polygon]:
    """
    Parse segmentation result and return the polygon bounds for the desired segmentation class, if present
    Assumes the segments per class have already been ranked using confidence * sqrt(area) as in 'segmenter_postprocess'

    segment_class -- class to fetch
    max_results -- max segments to return
                if =0, all matching segments are returned
    """

    # filter segments for desired class
    segments = list(
        filter(lambda s: (s.class_label == segment_class), segmentation.segments)
    )
    if not segments:
        logger.warning(f"No {segment_class} segment found")
        return []
    if max_results > 0:
        segments = segments[:max_results]
    return [Polygon(s.poly_bounds) for s in segments]


def merge_overlapping_polygons(polys: List[Polygon]) -> List[Polygon]:
    """
    Merge overlapping shapely polygons into single polygon objects
    If polygons are not overlapping, the original polygon list is returned
    """
    if not polys:
        return polys

    merged_polys = unary_union(polys)
    # convert merged geometry back to list of polygons
    if merged_polys.geom_type == "MultiPolygon":
        merged_polys = list(merged_polys.geoms)
    elif merged_polys.geom_type == "Polygon":
        merged_polys = [merged_polys]
    else:  # merged_poly.is_empty:
        # unary_union didn't work (unexpected, so return the original list
        merged_polys = polys

    return merged_polys


def segments_to_mask(
    segmentation: MapSegmentation,
    width_height: Tuple[int, int],
    roi_classes=["map", "legend_points_lines"],
    buffer_pixel=150,
    buffer_percent=0.03,
) -> np.ndarray:
    """
    Convert segmentation results into a binary mask, using "roi_classes" segments as the mask foreground
    A mask "buffer" is used to dilate segment regions prior to mask creation
    """
    if not segmentation:
        logger.warning(
            "No segmentation results available. Skipping creating of binary mask."
        )
        return np.array([])

    w, h = width_height
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    buffer_size = min(buffer_pixel, max(h, w) * buffer_percent)
    polys = []
    # get all foreground segments
    for seg in segmentation.segments:
        if seg.class_label in roi_classes:
            p = Polygon(seg.poly_bounds)
            p_buffered = p.buffer(buffer_size)  # join_style="mitre")
            polys.append(p_buffered)
    if not polys:
        logger.warning("No ROI segments available. Skipping creating of binary mask.")
        return np.array([])
    # handle overlapping ROI polygons, if present
    polys = merge_overlapping_polygons(polys)
    poly_arrays = [
        np.array([(int(x), int(y)) for x, y in p.exterior.coords]) for p in polys
    ]
    # convert segment polygons to a binary mask
    cv2.fillPoly(binary_mask, pts=poly_arrays, color=255)  # type: ignore

    return binary_mask
