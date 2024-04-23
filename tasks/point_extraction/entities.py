from __future__ import annotations
import logging
import numpy as np
from PIL import Image
from pydantic import BaseModel, validator, Field
import torch
from typing import Optional, List, Union, Any

logger = logging.getLogger(__name__)
## Data Objects

LEGEND_ITEMS_OUTPUT_KEY = "legend_point_items"


class MapPointLabel(BaseModel):
    """
    Represents a label on a map image.
    Class ID should correspond to the ID encoded in the underlying model.
    """

    classifier_name: str
    classifier_version: str
    class_id: int
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    direction: Optional[float] = None  # [deg] orientation of point symbol
    dip: Optional[float] = None  # [deg] dip angle associated with symbol
    legend_name: str
    legend_bbox: List[Union[float, int]]


class MapImage(BaseModel):
    """
    Represents a map image with point symbol prediction results
    """

    path: str
    raster_id: str
    labels: Optional[List[MapPointLabel]] = None
    map_bounds: Optional[List[int]] = (
        None  # [x1, y1, h, w] location of map. TODO: Accept polygonal seg mask.
    )
    point_legend_bounds: Optional[List[int]] = (
        None  # [x1, y1, h, w] location of point legend.
    )
    polygon_legend_bounds: Optional[List[int]] = (
        None  # [x1, y1, h, w] location of polygon legend.
    )

    _cached_image = None

    @property
    def size(self):
        # Make use of the PIL Image.open lazy image loading to avoid loading the image prematurely.
        return self.image.size

    @property
    def image(self):
        if self._cached_image:
            img = self._cached_image
        else:
            img = Image.open(self.path)
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError("Image cannot have 0 height or width")
            self._cached_image = img
        # TODO: Use polygonal segmask stored in self.map_bounds to filter the image and crop out the non-map regions.
        return img


class MapTile(BaseModel):
    """
    Represents a tile of a map image in (x, y, width, height) format.
    x and y are coordinates on the original map image.

    Image tensors are assumed to be in Torchvision format (C, H, W). These are automatically converted to PIL Images.
    """

    x_offset: int  # x offset of the tile in the original image.
    y_offset: int  # y offset of the tile in the original image.
    width: int
    height: int
    map_bounds: tuple  # map global bounds (x_min, y_min, x_max, y_max)
    image: Any  # torch.Tensor or PIL.Image
    map_path: str  # Path to the original map image.
    predictions: Optional[List[MapPointLabel]] = None

    @validator("image", pre=True, always=True)
    def validate_image(cls, value):
        if value is None:
            return value
        if isinstance(value, torch.Tensor):
            value = value.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            value = value.numpy()
            if value.dtype == np.float32:
                value = (value * 255).astype(np.uint8)
            value = Image.fromarray(value)
        if not isinstance(value, Image.Image):
            raise TypeError(f"Expected PIL or torch.Tensor Image, got {type(value)}")
        if value.size[0] == 0 or value.size[1] == 0:
            raise ValueError("Image cannot have 0 height or width")

        return value

    class Config:
        arbitrary_types_allowed = True

    def img_to_torchvision_tensor(self):
        return (
            torch.tensor(self.image).float().permute(2, 0, 1)
        )  # Convert from (H, W, C) to (C, H, W)


class MapTiles(BaseModel):
    raster_id: str
    tiles: List[MapTile]

    def format_for_caching(self) -> MapTiles:
        """
        Reformat point extraction tiles prior to caching
        - tile image raster is discarded
        """

        tiles_cache = []
        for t in self.tiles:
            t_cache = MapTile(
                x_offset=t.x_offset,
                y_offset=t.y_offset,
                width=t.width,
                height=t.height,
                map_bounds=t.map_bounds,
                image=None,
                map_path=t.map_path,
                predictions=t.predictions,
            )
            tiles_cache.append(t_cache)

        return MapTiles(raster_id=self.raster_id, tiles=tiles_cache)

    def join_with_cached_predictions(self, cached_preds: MapTiles) -> bool:
        """
        Append cached point predictions to MapTiles
        """
        try:
            # re-format cached predictions with key as (x_offset, y_offset)
            cached_dict = {}
            for p in cached_preds.tiles:
                cached_dict[(p.x_offset, p.y_offset)] = p
            for t in self.tiles:
                key = (t.x_offset, t.y_offset)
                if key not in cached_dict:
                    # cached predictions not found for this tile!
                    return False
                t_cached = cached_dict[key]
                t.predictions = t_cached.predictions
            return True
        except Exception as e:
            print(f"Exception in join_with_cached_predictions: {str(e)}")
            return False


class LegendPointItem(BaseModel):
    """
    Class for internally storing info about legend items for point symbols
    """

    # TODO -- could be modified to use CDR PointLegendAndFeaturesResult class in the future

    name: str = Field(description="Label of the map unit in the legend")
    description: str = Field(
        default="", description="Description of the map unit in the legend"
    )
    legend_bbox: List[Union[float, int]] = Field(
        default_factory=list,
        description="""The rough 2 point bounding box of the map units label.
                    Format is expected to be [x1,y1,x2,y2] where the top left
                    is the origin (0,0).""",
    )
    legend_contour: List[List[Union[float, int]]] = Field(
        default_factory=list,
        description="""The more precise polygon bounding box of the map units
                    label. Format is expected to be [x,y] coordinate pairs
                    where the top left is the origin (0,0).""",
    )


class LegendPointItems(BaseModel):
    """
    Class for a collection of LegendPointItem objects
    """

    items: List[LegendPointItem]
    provenance: str = Field(
        default="", description="where did these legend items come from"
    )

    @staticmethod
    def parse_legend_point_hints(legend_hints: dict) -> LegendPointItems:
        """
        parse legend hints JSON data (from the CMA contest)
        and convert to LegendPointItem objects

        legend_hints -- input hints dict
        only_keep_points -- if True, will discard any hints about line or polygon features
        """

        legend_point_items = []
        for shape in legend_hints["shapes"]:
            label = shape["label"]
            if not label.endswith("_pt") and not label.endswith("_point"):
                continue  # not a point symbol, skip

            # contour coords for the legend item's thumbnail swatch
            xy_pts = shape["points"]
            x_min = xy_pts[0][0]
            x_max = xy_pts[0][0]
            y_min = xy_pts[0][1]
            y_max = xy_pts[0][1]
            for x, y in xy_pts:
                x_min = int(min(x, x_min))
                x_max = int(max(x, x_max))
                y_min = int(min(y, y_min))
                y_max = int(max(y, y_max))
            legend_point_items.append(
                LegendPointItem(
                    name=label,
                    legend_bbox=[x_min, y_min, x_max, y_max],
                    legend_contour=xy_pts,
                )
            )
        return LegendPointItems(items=legend_point_items, provenance="ground_truth")
