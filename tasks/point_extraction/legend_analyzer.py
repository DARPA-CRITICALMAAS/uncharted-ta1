import json, logging, os
from collections import defaultdict
import httpx
from shapely import Polygon, distance
from typing import Optional
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.point_extraction.entities import (
    LegendPointItems,
    LegendPointItem,
    LEGEND_ITEMS_OUTPUT_KEY,
    LEGEND_PT_LABELS_OUTPUT_KEY,
    PointLabels,
)
from tasks.point_extraction.legend_item_utils import (
    filter_labelme_annotations,
    handle_duplicate_labels,
    parse_legend_annotations,
    LEGEND_ANNOTATION_PROVENANCE,
)
from tasks.segmentation.entities import MapSegmentation, SEGMENTATION_OUTPUT_KEY

logger = logging.getLogger(__name__)

CDR_API_TOKEN = os.environ.get("CDR_API_TOKEN", "")
CDR_HOST = "https://api.cdr.land"
CDR_LEGEND_SYSTEM_VERSION_DEFAULT = "polymer__0.0.1"


class LegendPreprocessor(Task):
    """
    Pre-processing of Point Symbol Legend Items
    """

    def __init__(
        self,
        task_id: str,
        cache_path: str,
        fetch_legend_items: bool = False,
    ):

        self.fetch_legend_items = fetch_legend_items
        super().__init__(task_id, cache_path)

    def run(self, task_input: TaskInput) -> TaskResult:
        """
        run point symbol legend analysis
        """

        legend_pt_items = None
        if LEGEND_ITEMS_OUTPUT_KEY in task_input.data:
            # legend items for point symbols already exist
            legend_pt_items = LegendPointItems.model_validate(
                task_input.data[LEGEND_ITEMS_OUTPUT_KEY]
            )
        elif LEGEND_ITEMS_OUTPUT_KEY in task_input.request:
            # legend items for point symbols already exist as a request param
            # (ie, loaded from a JSON hints or annotations file)
            # convert to a TaskResult...
            legend_pt_items = LegendPointItems.model_validate(
                task_input.request[LEGEND_ITEMS_OUTPUT_KEY]
            )

        if self.fetch_legend_items:
            # try to fetch legend annotations for COG id from the CDR (via REST)
            cdr_legend_items = self.fetch_cdr_legend_items(task_input.raster_id)
            if cdr_legend_items and cdr_legend_items.items:
                # legend items sucessfully fetched from the CDR
                # overwriting any pre-loaded legend items, if present
                legend_pt_items = cdr_legend_items
                logger.info(
                    f"Sucessfully fetched {len(legend_pt_items.items)} legend annotations from the CDR"
                )

        if legend_pt_items:
            if legend_pt_items.provenance == LEGEND_ANNOTATION_PROVENANCE.LABELME:
                # "labelme" legend items...
                # use segmentation results to filter noisy "labelme" legend annotations
                # (needed because all labelme annotations are set to type "polygon" regardless of feature type: polygons, lines or points)
                if SEGMENTATION_OUTPUT_KEY in task_input.data:
                    segmentation = MapSegmentation.model_validate(
                        task_input.data[SEGMENTATION_OUTPUT_KEY]
                    )

                    filter_labelme_annotations(legend_pt_items, segmentation)
                    logger.info(
                        f"Number of legend point annotations after filtering: {len(legend_pt_items.items)}"
                    )
                else:
                    logger.warning(
                        "No segmentation results available. Disregarding labelme legend annotations as noisy."
                    )
                    legend_pt_items.items = []

            handle_duplicate_labels(legend_pt_items)

            return TaskResult(
                task_id=self._task_id,
                output={LEGEND_ITEMS_OUTPUT_KEY: legend_pt_items.model_dump()},
            )

        return self._create_result(task_input)

    def fetch_cdr_legend_items(
        self,
        raster_id: str,
        system_version: str = CDR_LEGEND_SYSTEM_VERSION_DEFAULT,
        check_validated: bool = True,
    ) -> Optional[LegendPointItems]:
        """
        fetch legend annotations from the CDR for a given COG id
        """

        if not CDR_API_TOKEN:
            logger.warning("Unable to fetch legend items; CDR_API_TOKEN not set")
            return None

        try:
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {CDR_API_TOKEN}",
            }
            client = httpx.Client(follow_redirects=True)
            url = f"{CDR_HOST}/v1/features/{raster_id}/legend_items"
            if system_version:
                url += f"?system_version={system_version}"

            r = client.get(url, headers=headers)
            if r.status_code != 200:
                logger.warning(
                    f"Unable to fetch legend items for raster {raster_id}; cdr response code {r.status_code}"
                )
                return None
            legend_anns = json.loads(r.content)
            legend_pt_items = parse_legend_annotations(
                legend_anns, raster_id, check_validated=check_validated
            )
            if legend_pt_items.items:
                return legend_pt_items

        except Exception as e:
            logger.warning(
                f"Exception fetching legend items from the CDR for raster {raster_id}: {repr(e)}"
            )
        return None


class LegendPostprocessor(Task):
    """
    Post-processing of Point Symbol Legend Items
    """

    def __init__(
        self,
        task_id: str,
        cache_path: str,
    ):

        super().__init__(task_id, cache_path)

    def run(self, task_input: TaskInput) -> TaskResult:
        """
        Run point symbol legend post-processing

        The goal is to convert point symbol predictions (from a map's legend area)
        to legend item annotation objects (or join with existing)
        """

        if not LEGEND_PT_LABELS_OUTPUT_KEY in task_input.data:
            logger.info(
                "Point predictions not available for the legend area. Skipping legend post-processing."
            )
            return self._create_result(task_input)

        # --- load legend area ML point predictions
        legend_pt_preds = PointLabels.model_validate(
            task_input.data[LEGEND_PT_LABELS_OUTPUT_KEY]
        )
        if not legend_pt_preds.labels:
            return self._create_result(task_input)

        # --- load legend item annotations, if available
        legend_pt_items = LegendPointItems(items=[])
        if LEGEND_ITEMS_OUTPUT_KEY in task_input.data:
            legend_pt_items = LegendPointItems.model_validate(
                task_input.data[LEGEND_ITEMS_OUTPUT_KEY]
            )
        join_with_existing = len(legend_pt_items.items) > 0

        # group legend pt predictions by class name
        pred_groups = defaultdict(list)
        for label in legend_pt_preds.labels:
            pred_groups[label.class_name].append(label)

        # loop over groups
        for class_name, preds in pred_groups.items():
            if len(preds) > 1:
                # more than 1 legend swatch extracted for this class name
                # so choose the highest conf one, and discard the others as noisy
                logger.info(
                    f"{len(preds)} predictions found for point type {class_name}. Choosing the best one."
                )
                # sort by model confidence score
                preds.sort(key=lambda s: s.score, reverse=True)

            bbox = [preds[0].x1, preds[0].y1, preds[0].x2, preds[0].y2]
            xy_pts = [
                [preds[0].x1, preds[0].y1],
                [preds[0].x2, preds[0].y1],
                [preds[0].x2, preds[0].y2],
                [preds[0].x1, preds[0].y2],
            ]
            confidence = preds[0].score

            if join_with_existing:
                # check which legend ann swatch most overlaps with others,
                # if multiple overlap, choose the closest one
                # ... if this one is already in ontology choose that one
                p_pred = Polygon(xy_pts)
                leg_matches = list(
                    filter(
                        lambda leg: (leg.class_name == class_name),
                        legend_pt_items.items,
                    )
                )
                if leg_matches:
                    # legend swatch matches found
                    p_leg = Polygon(leg_matches[0].legend_contour)
                    smallest_dim = min(
                        [
                            p_leg.bounds[2] - p_leg.bounds[0],
                            p_leg.bounds[3] - p_leg.bounds[1],
                            p_pred.bounds[2] - p_pred.bounds[0],
                            p_pred.bounds[3] - p_pred.bounds[1],
                        ]
                    )
                    dist_norm = distance(p_leg.centroid, p_pred.centroid) / max(
                        smallest_dim, 1.0
                    )
                    logger.info(
                        f"Joining legend swatch prediction with existing legend item annotation for class {class_name}; normalized distance = {dist_norm:.3f}"
                    )
                else:
                    # get legend item annotations without any class label,
                    # and match the one with highest overlap
                    leg_unmatched = list(
                        filter(
                            lambda leg: (not leg.class_name),
                            legend_pt_items.items,
                        )
                    )
                    i_min = -1
                    dist_min = 9999999.0
                    for i, leg in enumerate(leg_unmatched):
                        p_leg = Polygon(leg.legend_contour)
                        smallest_dim = min(
                            [
                                p_leg.bounds[2] - p_leg.bounds[0],
                                p_leg.bounds[3] - p_leg.bounds[1],
                                p_pred.bounds[2] - p_pred.bounds[0],
                                p_pred.bounds[3] - p_pred.bounds[1],
                            ]
                        )
                        dist_norm = distance(p_leg.centroid, p_pred.centroid) / max(
                            smallest_dim, 1.0
                        )
                        if dist_norm < dist_min:
                            dist_min = dist_norm
                            i_min = i
                    if dist_min < 1.0:
                        # match found
                        logger.info(
                            f"Joining legend swatch prediction with existing legend item annotation for class {class_name}; normalized distance = {dist_min:.3f}"
                        )
                        leg_unmatched[i_min].class_name = class_name
                    else:
                        # add a new legend point item based on ML legend analysis
                        # TODO could skip, if yolo confidence is low?
                        logger.info(
                            f"Adding new legend item for point class {class_name}"
                        )
                        legend_pt_items.items.append(
                            LegendPointItem(
                                name=class_name,
                                class_name=class_name,
                                legend_bbox=bbox,
                                legend_contour=xy_pts,
                                confidence=confidence,
                            )
                        )
            else:
                # add a new legend point item based on ML legend analysis
                # TODO could skip, if yolo confidence is low?
                logger.info(f"Adding new legend item for point class {class_name}")
                legend_pt_items.items.append(
                    LegendPointItem(
                        name=class_name,
                        class_name=class_name,
                        legend_bbox=bbox,
                        legend_contour=xy_pts,
                        confidence=confidence,
                    )
                )

        logger.info(
            f"Number of Point Legend Items after post-processing: {len(legend_pt_items.items)}"
        )

        return TaskResult(
            task_id=self._task_id,
            output={LEGEND_ITEMS_OUTPUT_KEY: legend_pt_items.model_dump()},
        )
