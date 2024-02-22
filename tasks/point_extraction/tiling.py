from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import List
import logging
from shapely.geometry import Polygon

from common.task import Task, TaskInput, TaskResult

from tasks.point_extraction.entities import MapTile, MapTiles, MapImage, MapPointLabel
from tasks.segmentation.entities import MapSegmentation, SEGMENTATION_OUTPUT_KEY

SEGMENT_MAP_CLASS = "map"  # class label for map area segmentation
TILE_OVERLAP_DEFAULT = (  # default tliing overlap = point bbox + 10%
    int(1.1 * 90),
    int(1.1 * 90),
)

logger = logging.getLogger(__name__)


class Tiler(Task):
    """
    Decomposes a full image into smaller tiles

    NOTE: for point extractor inference, for best results it is recommended
    to use the same size tiles that were used during model training
    e.g., 1024x1024
    """

    # TODO: handle case where image already has labels attached to it.
    def __init__(
        self,
        task_id="",
        tile_size: tuple = (1024, 1024),
        overlap: tuple = TILE_OVERLAP_DEFAULT,
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        super().__init__(task_id)

    def run(
        self,
        task_input: TaskInput,
    ) -> TaskResult:
        image_array = np.array(task_input.image)
        x_min = 0
        y_min = 0
        y_max, x_max, _ = image_array.shape

        # use image segmentation to restrict point extraction to map area only
        if SEGMENTATION_OUTPUT_KEY in task_input.data:
            segments = MapSegmentation.model_validate(
                task_input.data[SEGMENTATION_OUTPUT_KEY]
            ).segments
            # filter segments for class "map"
            segments = list(
                filter(lambda s: (s.class_label == SEGMENT_MAP_CLASS), segments)
            )
            if not segments:
                logger.warning("No map area segment found. Tiling whole image")
                poly_xy = None
            elif len(segments) > 1:
                logger.warning(
                    f"{len(segments)} map segments found. Using segment with highest confidence for tiling"
                )
                # TODO: or could use largest map segment?
                segments.sort(key=lambda s: s.confidence, reverse=True)
                poly_xy = segments[0].poly_bounds
            else:
                poly_xy = segments[0].poly_bounds

            if poly_xy:
                # restrict tiling to use *only* the bounding rectangle of map area
                # TODO: ideally should use map polygon area as a binary mask
                p_map = Polygon(poly_xy)
                (x_min, y_min, x_max, y_max) = [int(b) for b in p_map.bounds]

        step_x = int(self.tile_size[0] - self.overlap[0])
        step_y = int(self.tile_size[1] - self.overlap[1])

        tiles: List[MapTile] = []

        for y in range(y_min, y_max, step_y):
            for x in range(x_min, x_max, step_x):
                width = min(self.tile_size[0], x_max - x)
                height = min(self.tile_size[1], y_max - y)

                tile_array = image_array[y : y + height, x : x + width]

                if (
                    tile_array.shape[0] < self.tile_size[1]
                    or tile_array.shape[1] < self.tile_size[0]
                ):
                    padded_tile = np.zeros(
                        (self.tile_size[1], self.tile_size[0], 3),
                        dtype=tile_array.dtype,
                    )

                    padded_tile[:height, :width] = tile_array
                    tile_array = padded_tile

                maptile = MapTile(
                    x_offset=x,
                    y_offset=y,
                    width=self.tile_size[0],
                    height=self.tile_size[1],
                    map_bounds=(x_min, y_min, x_max, y_max),
                    image=Image.fromarray(tile_array),
                    map_path="",
                )
                tiles.append(maptile)
        map_tiles = MapTiles(raster_id=task_input.raster_id, tiles=tiles)
        return TaskResult(
            task_id=self._task_id, output={"map_tiles": map_tiles.model_dump()}
        )

    @property
    def input_type(self):
        return MapImage

    @property
    def output_type(self):
        return List[MapTile]


class Untiler(Task):
    def __init__(self, task_id="", overlap: tuple = TILE_OVERLAP_DEFAULT):
        # NOTE: Untiler recommended to use the same tile overlap as corresponding Tiler class instance
        self.check_overlap_predictions: bool = overlap[0] > 0 or overlap[1] > 0
        super().__init__(task_id)

    """
    Used to reconstruct the original image from the tiles and map back the bounding boxes and labels.
    Note that new images aren't actually constructed here, we are just mapping predictions from tiles onto the original map.
    """

    def run(self, input: TaskInput) -> TaskResult:
        """
        Reconstructs the original image from the tiles and maps back the bounding boxes and labels.
        tile_predictions: List of MapPointLabel objects. Generated by the model. TILES MUST BE FROM ONLY ONE MAP.
        returns: List of MapPointLabel objects. These can be mapped directly onto the original map.
        """

        map_tiles = MapTiles.model_validate(input.get_data("map_tiles"))
        tiles = map_tiles.tiles

        assert all(
            i.predictions is not None for i in tiles
        ), "Tiles must have predictions attached to them."
        all_predictions = []
        map_path = tiles[0].map_path
        for tile in tiles:

            x_offset = tile.x_offset  # xmin of tile, absolute value in original map
            y_offset = tile.y_offset  # ymin of tile, absolute value in original map

            for pred in tqdm(
                tile.predictions,
                desc="Reconstructing original map with predictions on tiles",
            ):

                x1 = pred.x1
                x2 = pred.x2
                y1 = pred.y1
                y2 = pred.y2
                score = pred.score
                label_name = pred.class_name

                # filter noisy predictions due to tile overlap
                if self.check_overlap_predictions and self._is_prediction_redundant(
                    (pred.x1, pred.y1, pred.x2, pred.y2),
                    tile.map_bounds,
                    (tile.x_offset, tile.y_offset),
                    (tile.width, tile.height),
                ):
                    continue

                global_prediction = MapPointLabel(
                    classifier_name=pred.classifier_name,
                    classifier_version=pred.classifier_version,
                    class_id=pred.class_id,
                    class_name=label_name,
                    x1=x1
                    + x_offset,  # Add offset of tile to project onto original map.
                    y1=y1 + y_offset,
                    x2=x2 + x_offset,
                    y2=y2 + y_offset,
                    score=score,
                    direction=pred.direction,
                    dip=pred.dip,
                )

                all_predictions.append(global_prediction)
        map_image = MapImage(path=map_path, labels=all_predictions)
        return TaskResult(task_id=self._task_id, output={"map_image": map_image})

    def _is_prediction_redundant(
        self,
        pred_bbox: tuple,
        map_bbox,
        tile_offset: tuple,
        tile_wh: tuple,
        shape_thres=2,
    ) -> bool:
        """
        Check if a point symbol prediction is redundant, due to overlapping tiles
        """
        (x1, y1, x2, y2) = pred_bbox
        (map_xmin, map_ymin, map_xmax, map_ymax) = map_bbox
        tile_w, tile_h = tile_wh
        x_offset, y_offset = tile_offset

        # TODO - instead of checking at tile edge could check if bbox edge is in overlap region
        if (abs((x2 - x1) - (y2 - y1)) > shape_thres) and (
            x1 <= 1 or y1 <= 1 or x2 >= tile_w - 1 or y2 >= tile_h - 1
        ):
            # pred bbox is at a tile edge and NOT square,
            # check if bbox edges correspond to global image bounds
            if (
                x1 + x_offset > map_xmin
                and x2 + x_offset < map_xmax
                and y1 + y_offset > map_ymin
                and y2 + y_offset < map_ymax
            ):
                # point bbox not at map edges, assume this is a redundant prediction (due to tile overlap) and skip
                return True

        return False

    @property
    def input_type(self):
        return List[MapTile]

    @property
    def output_type(self):
        return MapImage
