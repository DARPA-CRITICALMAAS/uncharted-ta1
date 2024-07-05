import logging
from enum import Enum
from typing import List
from collections import defaultdict
from shapely import Polygon, distance

from tasks.point_extraction.entities import LegendPointItem, LegendPointItems
from tasks.point_extraction.label_map import LABEL_MAPPING
from schema.cdr_schemas.cdr_responses.legend_items import LegendItemResponse
from tasks.segmentation.entities import MapSegmentation


logger = logging.getLogger(__name__)

SEGMENT_PT_LEGEND_CLASS = (
    "legend_points_lines"  # class label for points legend area segmentation
)


# Legend item annotations "system" or provenance labels
class LEGEND_ANNOTATION_PROVENANCE(str, Enum):
    GROUND_TRUTH = "ground_truth"
    LABELME = "labelme"  # aka STEPUP
    POLYMER = "polymer"  # Jatware's annotation system

    def __str__(self):
        return self.value


def parse_legend_annotations(
    legend_anns: list,
    raster_id: str,
    system_filter=[
        LEGEND_ANNOTATION_PROVENANCE.POLYMER,
        LEGEND_ANNOTATION_PROVENANCE.LABELME,
    ],
    check_validated=False,
) -> LegendPointItems:
    """
    parse legend annotations JSON data (CDR LegendItemResponse json format)
    and convert to LegendPointItem objects
    """

    # parse legend annotations and group by system label
    legend_item_resps = defaultdict(list)
    count_leg_items = 0
    for leg_ann in legend_anns:
        try:
            leg_resp = LegendItemResponse(**leg_ann)
            if leg_resp.system in system_filter or (
                check_validated and leg_resp.validated
            ):
                # only keep legend item responses from desired systems
                # or with validated=True
                legend_item_resps[leg_resp.system].append(leg_resp)
                count_leg_items += 1
        except Exception as e:
            # legend_pt_items = LegendPointItems(items=[], provenance="")
            logger.error(
                f"EXCEPTION parsing legend annotations json for raster {raster_id}: {repr(e)}"
            )
    logger.info(f"Successfully loaded {count_leg_items} LegendItemResponse objects")

    # try to parse non-labelme annotations first
    legend_point_items = []
    system_label = ""
    for system, leg_anns in legend_item_resps.items():
        if system == LEGEND_ANNOTATION_PROVENANCE.LABELME:
            continue
        system_label = system
        legend_point_items.extend(legend_ann_to_legend_items(leg_anns, raster_id))
    if legend_point_items:
        return LegendPointItems(items=legend_point_items, provenance=system_label)
    else:
        # try to parse label annotations 2nd (since labelme anns have noisy data for point/line features)
        for system, leg_anns in legend_item_resps.items():
            if not system == LEGEND_ANNOTATION_PROVENANCE.LABELME:
                continue
            legend_point_items.extend(legend_ann_to_legend_items(leg_anns, raster_id))
        if legend_point_items:
            return LegendPointItems(
                items=legend_point_items,
                provenance=LEGEND_ANNOTATION_PROVENANCE.LABELME,
            )
    return LegendPointItems(items=[], provenance="")


def parse_legend_point_hints(legend_hints: dict, raster_id: str) -> LegendPointItems:
    """
    parse legend hints JSON data (from the CMA contest)
    and convert to LegendPointItem objects

    legend_hints -- input hints dict
    """

    legend_point_items = []
    for shape in legend_hints["shapes"]:
        label = shape["label"]
        if not label.endswith("_pt") and not label.endswith("_point"):
            continue  # not a point symbol, skip

        # contour coords for the legend item's thumbnail swatch
        xy_pts = shape.get("points", [])
        if xy_pts:
            x_min = xy_pts[0][0]
            x_max = xy_pts[0][0]
            y_min = xy_pts[0][1]
            y_max = xy_pts[0][1]
            for x, y in xy_pts:
                x_min = int(min(x, x_min))
                x_max = int(max(x, x_max))
                y_min = int(min(y, y_min))
                y_max = int(max(y, y_max))
        else:
            x_min = 0
            x_max = 0
            y_min = 0
            y_max = 0
        xy_pts = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]
        class_name = find_legend_keyword_match(label, raster_id)
        legend_point_items.append(
            LegendPointItem(
                name=label,
                class_name=class_name,
                legend_bbox=[x_min, y_min, x_max, y_max],
                legend_contour=xy_pts,
                confidence=1.0,
                system=LEGEND_ANNOTATION_PROVENANCE.GROUND_TRUTH,
                validated=True,
            )
        )
    return LegendPointItems(
        items=legend_point_items, provenance=LEGEND_ANNOTATION_PROVENANCE.GROUND_TRUTH
    )


def find_legend_keyword_match(legend_item_name: str, raster_id: str) -> str:
    """
    Use keyword matching to map legend item label to point extractor ontology class names
    """
    leg_label_norm = raster_id + "_" + legend_item_name.strip().lower()
    matches = []
    for symbol_class, suffixs in LABEL_MAPPING.items():
        for s in suffixs:
            if s in leg_label_norm:
                # match found
                matches.append((s, symbol_class))
    if matches:
        # sort to get longest suffix match
        matches.sort(key=lambda a: len(a[0]), reverse=True)
        symbol_class = matches[0][1]
        logger.info(
            f"Legend label: {legend_item_name} matches point class: {symbol_class}"
        )
        return symbol_class

    logger.info(f"No point class match found for legend label: {legend_item_name}")
    return ""


def legend_ann_to_legend_items(
    legend_anns: List[LegendItemResponse], raster_id: str
) -> List[LegendPointItems]:
    """
    convert LegendItemResponse (CDR schema format)
    to internal LegendPointItem objects
    """
    legend_point_items = []
    prev_label = ""
    for leg_ann in legend_anns:
        label = leg_ann.label if leg_ann.label else leg_ann.abbreviation
        if (
            leg_ann.system == LEGEND_ANNOTATION_PROVENANCE.LABELME
            and prev_label
            and prev_label == label
        ):
            # special base to handle labelme (STEPUP) annotations...
            # skip the 2nd labelme annotation in each pair
            # (this 2nd entry is just the bbox for the legend item text; TODO -- could extract and include this text too?)
            continue
        class_name = find_legend_keyword_match(label, raster_id)
        xy_pts = (
            leg_ann.px_geojson.coordinates[0]
            if leg_ann.px_geojson
            else [
                [leg_ann.px_bbox[0], leg_ann.px_bbox[1]],
                [leg_ann.px_bbox[2], leg_ann.px_bbox[1]],
                [leg_ann.px_bbox[2], leg_ann.px_bbox[3]],
                [leg_ann.px_bbox[0], leg_ann.px_bbox[3]],
            ]
        )

        legend_point_items.append(
            LegendPointItem(
                name=label,
                class_name=class_name,
                abbreviation=leg_ann.abbreviation,
                description=leg_ann.description,
                legend_bbox=leg_ann.px_bbox,
                legend_contour=xy_pts,
                system=leg_ann.system,
                confidence=leg_ann.confidence,
                validated=leg_ann.validated,
            )
        )
        prev_label = label

    return legend_point_items


def filter_labelme_annotations(
    leg_point_items: LegendPointItems,
    segmentation: MapSegmentation,
    width_thres=120,
    shape_thres=2.0,
):
    """
    labelme (aka STEPUP) legend annotations are noisy, with all items for polygons, points and lines grouped together.
    These are filtered using segmentation info and shape heuristics to estimate which items, if any, correspond to point features
    """

    segs_point_legend = list(
        filter(
            lambda s: (s.class_label == SEGMENT_PT_LEGEND_CLASS),
            segmentation.segments,
        )
    )
    if not segs_point_legend:
        logger.warning(
            "No Points-Legend segment found. Disregarding labelme legend annotations as noisy."
        )
        leg_point_items.items = []
        return

    filtered_leg_items = []
    for seg in segs_point_legend:
        p_seg = Polygon(seg.poly_bounds)
        for leg in leg_point_items.items:
            p_leg = Polygon(leg.legend_contour)
            if not p_seg.intersects(p_leg.centroid):
                # this legend swatch is not within the points legend area; disregard
                continue
            # legend swatch intersects the points legend area
            # check other properties to determine if swatch is line vs point symbol
            w = leg.legend_bbox[2] - leg.legend_bbox[0]
            h = leg.legend_bbox[3] - leg.legend_bbox[1]
            if leg.class_name:
                # legend item label is in points ontology
                filtered_leg_items.append(leg)
            elif w < width_thres and w < shape_thres * h:
                # legend item swatch bbox is close to square
                filtered_leg_items.append(leg)
    leg_point_items.items = filtered_leg_items
