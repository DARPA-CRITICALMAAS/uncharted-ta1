import logging
import uuid
import numpy as np
from copy import deepcopy
from sklearn.cluster import DBSCAN
from shapely import Polygon, box

from tasks.geo_referencing.entities import (
    Coordinate,
    SOURCE_STATE_PLANE,
    SOURCE_UTM,
    SOURCE_LAT_LON,
)
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.geo_projection import PolyRegression
from tasks.geo_referencing.util import ocr_to_coordinates
from tasks.geo_referencing.entities import MapROI, ROI_MAP_OUTPUT_KEY

from typing import Dict, Tuple, List

logger = logging.getLogger("coordinates_filter")

NAIVE_FILTER_MINIMUM = 10


class FilterCoordinates(Task):

    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        # get coordinates so far
        lon_pts = input.get_data("lons")
        lat_pts = input.get_data("lats")

        # filter the coordinates to retain only those that are deemed valid
        lon_pts_filtered, lat_pts_filtered = self._filter(input, lon_pts, lat_pts)

        logger.info(
            f"Num coordinates after filtering: {len(lat_pts_filtered)} latitude and {len(lon_pts_filtered)}"
        )

        # update the coordinates list
        return self._create_result(input, lon_pts_filtered, lat_pts_filtered)

    def _create_result(
        self,
        input: TaskInput,
        lons: Dict[Tuple[float, float], Coordinate],
        lats: Dict[Tuple[float, float], Coordinate],
    ) -> TaskResult:
        result = super()._create_result(input)

        result.output["lons"] = lons
        result.output["lats"] = lats

        return result

    def _filter(
        self,
        input: TaskInput,
        lon_coords: Dict[Tuple[float, float], Coordinate],
        lat_coords: Dict[Tuple[float, float], Coordinate],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        return lon_coords, lat_coords


class FilterAxisCoordinates(Task):

    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        # get coordinates so far
        lon_pts = input.get_data("lons")
        lat_pts = input.get_data("lats")

        # filter the coordinates to retain only those that are deemed valid
        lon_pts_filtered = lon_pts
        if len(lon_pts) > 0:
            lon_pts_filtered = self._filter(input, lon_pts)
        lat_pts_filtered = lat_pts
        if len(lat_pts) > 0:
            lat_pts_filtered = self._filter(input, lat_pts)

        logger.info(
            f"Num coordinates after axis filtering: {len(lat_pts_filtered)} latitude and {len(lon_pts_filtered)}"
        )

        # update the coordinates list
        return self._create_result(input, lon_pts_filtered, lat_pts_filtered)

    def _create_result(
        self,
        input: TaskInput,
        lons: Dict[Tuple[float, float], Coordinate],
        lats: Dict[Tuple[float, float], Coordinate],
    ) -> TaskResult:
        result = super()._create_result(input)

        result.output["lons"] = lons
        result.output["lats"] = lats

        return result

    def _filter(
        self, input: TaskInput, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        return coords


class OutlierFilter(FilterAxisCoordinates):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _filter(
        self, input: TaskInput, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        if len(coords) < 3:
            logger.debug(
                "Skipping outlier filtering since there are fewer than 3 coordinates"
            )
            return coords

        logger.info(f"Running outlier filter for {len(coords)} coords")
        updated_coords = coords
        test_length = 0
        while len(updated_coords) != test_length:
            test_length = len(updated_coords)
            updated_coords = self._filter_regression(input, updated_coords)
            logger.debug(
                f"outlier filter updated length {len(updated_coords)} compared to test length {test_length}"
            )
        return updated_coords

    def _filter_regression(
        self, input: TaskInput, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        # use leave one out approach using linear regression model

        # reduce coordinate to (degree, constant dimension) where the constant dimension for lat is y and lon is x
        coords_representation = []
        for _, c in coords.items():
            coords_representation.append(c)

        # get the regression quality when holding out each coordinate one at a time
        reduced = [
            self._reduce(coords_representation, i)
            for i in range(len(coords_representation))
        ]

        # identify potential outliers via the model quality (outliers should show a dip in the error)
        results = {}
        test = sum(reduced) / len(coords_representation)
        for i in range(len(coords_representation)):
            # arbitrary test to flag outliers
            # having a floor to the test prevents removing datapoints when the error is low due to points lining up correctly
            if test > 0.1 and reduced[i] < 0.5 * test:
                continue
            else:
                key, _ = coords_representation[i].to_deg_result()
                results[key] = coords_representation[i]
        return results

    def _reduce(self, coords: list[Coordinate], index: int) -> float:
        # remove the point for which to calculate the model quality
        coords_work = coords.copy()
        coords_work.pop(index)

        # build linear regression model using the remaining points
        regression = PolyRegression(1)
        pixels = []
        degrees = []
        for c in coords_work:
            pixels.append(c.get_pixel_alignment())
            degrees.append(c.get_parsed_degree())

        # do polynomial regression for axis
        regression.fit_polynomial_regression(pixels, degrees)
        predictions = regression.predict_pts(pixels)

        # calculate error
        # TODO: FOR NOW DO SIMPLE SUM
        return sum([abs(degrees[i] - predictions[i]) for i in range(len(predictions))])


class DistinctDegreeOutlierFilter(FilterAxisCoordinates):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _filter(
        self, input: TaskInput, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        # a more strict outlier filter focusing on coordinates with the same parsed degree
        distincts = {}
        to_process = False
        for _, c in coords.items():
            degree = c.get_parsed_degree()
            if degree not in distincts:
                distincts[degree] = []
            distincts[degree].append(c)
            if len(distincts) >= 3:
                to_process = True

        if not to_process:
            logger.debug(
                "Skipping distinct degree filtering since there are not enough duplicates"
            )
            return coords

        # cycle through all distinct groups
        remaining_coords = {}
        for k, g in distincts.items():
            to_keep = []
            # if less than 3, simply add to output since filter will not be possible
            if len(g) < 3:
                to_keep = g
            else:
                logger.debug(f"attempting to filter by distinct degree value for {k}")
                # identical degree values should have one of x or y be fairly similar
                # if one is misaligned then it is probably from some other context
                x, y = input.image.size
                size_relevant = (y if g[0].is_lat() else x) / 20
                for c in g:
                    c_d = c.get_constant_dimension()
                    for c_i in g:
                        if c_i.get_pixel_alignment() != c.get_pixel_alignment():
                            c_i_d = c_i.get_constant_dimension()
                            if abs(c_i_d - c_d) < size_relevant and c_i not in to_keep:
                                to_keep.append(c_i)
                if len(to_keep) == 0:
                    to_keep = g
                    logger.debug(f"kept all coordinates parsed as {k}")
                else:
                    logger.debug(f"kept only a subset of coordinates parsed as {k}")
            for c in to_keep:
                key = c.to_deg_result()[0]
                remaining_coords[key] = c

        return remaining_coords


class HighQualityCoordinateFilter(FilterAxisCoordinates):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _filter(
        self, input: TaskInput, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        # if two points have the same degree parsed and are roughly aligned
        # then throw out everything else that is within the same general area with a different degree
        distincts = {}
        coordinates = []
        to_process = False
        for _, c in coords.items():
            degree = c.get_parsed_degree()
            if degree not in distincts:
                distincts[degree] = []
            distincts[degree].append(c)
            if len(distincts) >= 2:
                to_process = True
            coordinates.append(c)

        if not to_process:
            logger.debug(
                "skipping high quality coordinate filtering since there are not enough duplicates"
            )
            return coords

        remove_coords = {}
        remaining_coords = {}
        for degree, g in distincts.items():
            if len(g) >= 2:
                # check to make sure at least one other coordinate in the group falls within the expected range
                x, y = input.image.size
                size_relevant = (y if g[0].is_lat() else x) / 20
                can_filter, pixel_range = self._can_filter(g, size_relevant)
                if can_filter:
                    _, rejected = self._filter_range(
                        coordinates,
                        degree,
                        (pixel_range - size_relevant, pixel_range + size_relevant),
                    )
                    for c in rejected:
                        logger.debug(
                            f"removing {c.get_parsed_degree()} since it falls within the pixel range of high confidence points"
                        )
                        remove_coords[c.get_pixel_alignment()] = c

            for c in coordinates:
                key = c.to_deg_result()[0]
                if c.get_pixel_alignment() not in remove_coords:
                    remaining_coords[key] = c
        return remaining_coords

    def _can_filter(
        self, coordinates: List[Coordinate], pixel_range: float
    ) -> Tuple[bool, float]:
        for c in coordinates:
            c_d = c.get_constant_dimension()
            for c_i in coordinates:
                if c_i.get_pixel_alignment() != c.get_pixel_alignment():
                    c_i_d = c_i.get_constant_dimension()
                    if abs(c_i_d - c_d) < pixel_range:
                        return True, c_d
        return False, -1

    def _filter_range(
        self,
        coordinates: List[Coordinate],
        degree: float,
        pixel_range: Tuple[float, float],
    ) -> Tuple[List[Coordinate], List[Coordinate]]:
        # split coordinates found within the range to either be kept (degree matches) or removed (degree does not match)
        to_keep = []
        rejected = []
        for c in coordinates:
            pixel = c.get_constant_dimension()
            if pixel_range[0] <= pixel <= pixel_range[1]:
                if c.get_parsed_degree() == degree:
                    to_keep.append(c)
                else:
                    rejected.append(c)
        return to_keep, rejected


class UTMStatePlaneFilter(FilterCoordinates):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _filter(
        self,
        input: TaskInput,
        lon_coords: Dict[Tuple[float, float], Coordinate],
        lat_coords: Dict[Tuple[float, float], Coordinate],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:

        # get the count and confidence of state plane and utm coordinates
        lon_count_sp, lon_conf_sp = self._get_score(lon_coords, SOURCE_STATE_PLANE)
        lon_count_utm, lon_conf_utm = self._get_score(lon_coords, SOURCE_UTM)
        lat_count_sp, lat_conf_sp = self._get_score(lat_coords, SOURCE_STATE_PLANE)
        lat_count_utm, lat_conf_utm = self._get_score(lat_coords, SOURCE_UTM)

        # if no utm or no state plane coordinates exist then nothing to filter
        if lon_count_sp + lat_count_sp == 0:
            return lon_coords, lat_coords
        if lon_count_utm + lat_count_utm == 0:
            return lon_coords, lat_coords

        # if one has coordinates in both directions while the other doesnt then keep that one
        source_filter = ""
        if (
            min(lon_count_utm, lat_count_utm) > 0
            and min(lon_count_sp, lat_count_sp) == 0
        ):
            logger.debug("removing state plane coordinates since one axis has none")
            source_filter = SOURCE_STATE_PLANE
        elif (
            min(lon_count_utm, lat_count_utm) == 0
            and min(lon_count_sp, lat_count_sp) > 0
        ):
            logger.debug("removing utm coordinates since one axis has none")
            source_filter = SOURCE_UTM

        # if still unsure then retain the one with the highest confidence
        # by this point both utm and state plane have coordinates in one or two directions
        if source_filter == "":
            source_filter = SOURCE_UTM
            if max(lon_conf_utm, lat_conf_utm) > max(lon_conf_sp, lat_conf_sp):
                logger.debug(
                    "removing state plane coordinates since utm coordinates have higher confidence"
                )
                source_filter = SOURCE_STATE_PLANE
            else:
                logger.debug(
                    "removing utm coordinates since state plane coordinates have higher confidence"
                )

        logger.debug(f"filtering {source_filter} latitude and longitude coordinates")

        return self._filter_source(source_filter, lon_coords), self._filter_source(
            source_filter, lat_coords
        )

    def _get_score(
        self, coords: Dict[Tuple[float, float], Coordinate], source: str
    ) -> Tuple[int, float]:
        conf = -1
        count = 0
        for _, c in coords.items():
            src = c.get_source()
            if src == source:
                count = count + 1
                if conf < c.get_confidence():
                    conf = c.get_confidence()
        return (count, conf)

    def _filter_source(
        self, source: str, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        coords_filtered = {}
        for k, c in coords.items():
            if not c.get_source() == source:
                coords_filtered[k] = c
        return coords_filtered


class NaiveFilter(FilterAxisCoordinates):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _filter(
        self, input: TaskInput, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        logger.debug(f"naive filter running against {len(coords)} coords")
        updated_coords = self._filter_coarse(input, coords)
        return updated_coords

    def _filter_coarse(
        self, input: TaskInput, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        # check range of coordinates to determine if filtering is required
        degs = []
        for _, c in coords.items():
            degs.append(c.get_parsed_degree())

        if max(degs) - min(degs) < NAIVE_FILTER_MINIMUM:
            return coords

        # cluster degrees
        data = np.array([[d] for d in degs])
        db = DBSCAN(eps=2.5, min_samples=2).fit(data)
        labels = db.labels_

        clusters = []
        max_cluster = []
        for i, l in enumerate(labels):
            if l == -1:
                continue
            while len(clusters) <= l:
                clusters.append([])
            clusters[l].append(degs[i])
            if len(clusters[l]) > len(max_cluster):
                max_cluster = clusters[l]

        # no clustering so unable to filter anything reliably
        if len(max_cluster) == 0:
            return coords

        filtered_coords = {}
        for k, v in coords.items():
            if v.get_parsed_degree() in max_cluster:
                filtered_coords[k] = v
        return filtered_coords


class ROIFilter(FilterCoordinates):
    """
    Coordinate filtering based on Map region-of-interest (ROI)
    """

    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _filter(
        self,
        input: TaskInput,
        lon_coords: Dict[Tuple[float, float], Coordinate],
        lat_coords: Dict[Tuple[float, float], Coordinate],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:

        if not ROI_MAP_OUTPUT_KEY in input.data:
            logger.warning("No ROI info available; skipping ROIFilter")
            return (lon_coords, lat_coords)

        map_roi = MapROI.model_validate(input.data[ROI_MAP_OUTPUT_KEY])

        # use the ROI buffering result to create a "ring" polygon shape around the map's boundaries
        try:
            roi_poly = Polygon(shell=map_roi.buffer_outer, holes=[map_roi.buffer_inner])
        except Exception as ex:
            logger.warning(
                "Exception using inner and outer ROI buffering; just using outer buffering"
            )
            roi_poly = Polygon(shell=map_roi.buffer_outer)

        lon_inputs = deepcopy(lon_coords)  # TODO: is deepcopy necessary?
        lat_inputs = deepcopy(lat_coords)
        lon_counts_initial, lat_counts_initial = self._get_distinct_degrees(
            lon_inputs, lat_inputs
        )

        # --- do ROI filtering
        lons, lats = self._filter_roi(lon_inputs, lat_inputs, roi_poly)
        lon_counts, lat_counts = self._get_distinct_degrees(lons, lats)

        # --- adjust filtering based on distance to roi if insufficient points
        if lon_counts < 2 and lon_counts < lon_counts_initial:
            logger.debug(
                f"only {lon_counts} lon coords after roi filtering so re-adding coordinates"
            )
            lons = self._adjust_filter(lons, lon_coords, roi_poly)

        if lat_counts < 2 and lat_counts < lat_counts_initial:
            logger.debug(
                f"only {lat_counts} lat coords after roi filtering so re-adding coordinates"
            )
            lats = self._adjust_filter(lats, lat_coords, roi_poly)

        # ---
        # TODO: this should be re-factored/merged with other similar geo-ref tasks
        lons, lats = self._validate_lonlat_extractions(lons, lats, input.image.size)

        return lons, lats

    def _adjust_filter(
        self,
        coords: Dict[Tuple[float, float], Coordinate],
        coords_all: Dict[Tuple[float, float], Coordinate],
        roi_poly: Polygon,
    ) -> Dict[Tuple[float, float], Coordinate]:
        """
        Adjust ROI filtering based on the rectangular ROI if insufficient points
        """
        distinct_degs = set(map(lambda x: x[1].get_parsed_degree(), coords.items()))
        # create bbox of the ROI polygon
        roi_bbox = box(*roi_poly.bounds)

        coords_to_add = {}
        for (deg, i), coord in coords_all.items():
            if (deg, i) in coords:
                # this coord is already in valid set
                continue
            coord_poly = Polygon([(pt.x, pt.y) for pt in coord.get_bounds()])
            if coord_poly.intersects(roi_bbox):
                # save as coord to keep
                coords_to_add[(deg, i)] = coord

        # sort by confidence
        coords_to_add = sorted(
            coords_to_add.items(), key=lambda x: x[1].get_confidence(), reverse=True
        )
        # add more coordinate results
        for (deg, i), coord in coords_to_add:
            if len(distinct_degs) >= 2:
                break
            # TODO: flag this coord as being outside the map's ROI
            coord._confidence *= 0.5  # re-add coord with reduced confidence
            coords[(deg, i)] = coord
            distinct_degs.add(deg)
            logger.debug(f"re-adding coordinate: {deg} ({coord.get_pixel_alignment()})")

        return coords

    def _get_distinct_degrees(
        self,
        lon_coords: Dict[Tuple[float, float], Coordinate],
        lat_coords: Dict[Tuple[float, float], Coordinate],
    ) -> Tuple[int, int]:
        """
        Get the number of unique degree values for extracted lat and lon values
        """
        lats_distinct = set(map(lambda x: x[1].get_parsed_degree(), lat_coords.items()))
        lons_distinct = set(map(lambda x: x[1].get_parsed_degree(), lon_coords.items()))
        return len(lons_distinct), len(lats_distinct)

    def _validate_lonlat_extractions(
        self,
        lon_results: Dict[Tuple[float, float], Coordinate],
        lat_results: Dict[Tuple[float, float], Coordinate],
        im_size: Tuple[float, float],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        """
        Add an inferred anchor lat/lon pt. if needed
        """
        # TODO: this should be re-factored/merged with other similar geo-ref tasks

        num_lat_pts = len(lat_results)
        num_lon_pts = len(lon_results)
        logger.debug(
            f"point count after exclusion lat,lon: {num_lat_pts},{num_lon_pts}"
        )

        # check number of unique lat and lon values
        num_lat_pts = len(set([x[0] for x in lat_results]))
        num_lon_pts = len(set([x[0] for x in lon_results]))
        logger.debug(f"distinct after roi lat,lon: {num_lat_pts},{num_lon_pts}")

        if num_lon_pts >= 2 and num_lat_pts == 1:
            # estimate additional lat pt (based on lon pxl resolution)
            lst = [
                (k[0], k[1], v.get_pixel_alignment()[1]) for k, v in lon_results.items()
            ]
            max_pt = max(lst, key=lambda p: p[1])
            min_pt = min(lst, key=lambda p: p[1])
            pxl_range = max_pt[1] - min_pt[1]
            deg_range = max_pt[0] - min_pt[0]
            if deg_range != 0 and pxl_range != 0:
                deg_per_pxl = abs(
                    deg_range / pxl_range
                )  # TODO could use geodesic dist here?
                lat_pt = list(lat_results.items())[0]
                # new_y = im_size[1]-1
                new_y = 0 if lat_pt[0][1] > im_size[1] / 2 else im_size[1] - 1
                new_lat = -deg_per_pxl * (new_y - lat_pt[0][1]) + lat_pt[0][0]
                coord = Coordinate(
                    "lat keypoint",
                    "",
                    new_lat,
                    SOURCE_LAT_LON,
                    True,
                    pixel_alignment=(lat_pt[1].to_deg_result()[1], new_y),
                    confidence=0.6,
                )
                lat_results[(new_lat, new_y)] = coord

        elif num_lat_pts >= 2 and num_lon_pts == 1:
            # estimate additional lon pt (based on lat pxl resolution)
            lst = [
                (k[0], k[1], v.get_pixel_alignment()[0]) for k, v in lat_results.items()
            ]
            max_pt = max(lst, key=lambda p: p[1])
            min_pt = min(lst, key=lambda p: p[1])
            pxl_range = max_pt[1] - min_pt[1]
            deg_range = max_pt[0] - min_pt[0]
            if deg_range != 0 and pxl_range != 0:
                deg_per_pxl = abs(
                    deg_range / pxl_range
                )  # TODO could use geodesic dist here?
                lon_pt = list(lon_results.items())[0]
                # new_x = im_size[0]-1
                new_x = 0 if lon_pt[0][1] > im_size[0] / 2 else im_size[0] - 1
                new_lon = -deg_per_pxl * (new_x - lon_pt[0][1]) + lon_pt[0][0]
                coord = Coordinate(
                    "lon keypoint",
                    "",
                    new_lon,
                    SOURCE_LAT_LON,
                    False,
                    pixel_alignment=(new_x, lon_pt[1].to_deg_result()[1]),
                    confidence=0.6,
                )
                lon_results[(new_lon, new_x)] = coord

        return (lon_results, lat_results)

    def _filter_roi(
        self,
        lon_coords: Dict[Tuple[float, float], Coordinate],
        lat_coords: Dict[Tuple[float, float], Coordinate],
        roi_poly: Polygon,
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        """
        Filter extracted coordinates based on the map ROI
        """

        if not roi_poly:
            return (lon_coords, lat_coords)

        lon_out = {}
        lat_out = {}
        for (deg, y), coord in list(lat_coords.items()):
            coord_poly = Polygon([(pt.x, pt.y) for pt in coord.get_bounds()])
            if coord_poly.intersects(roi_poly):
                # keep this latitude pt
                lat_out[(deg, y)] = coord
            else:
                logger.debug(
                    f"removing out-of-bounds latitude point: {deg} ({coord.get_pixel_alignment()})"
                )
        for (deg, x), coord in list(lon_coords.items()):
            coord_poly = Polygon([(pt.x, pt.y) for pt in coord.get_bounds()])
            if coord_poly.intersects(roi_poly):
                # keep this longitude pt
                lon_out[(deg, x)] = coord
            else:
                logger.debug(
                    f"removing out-of-bounds longitude point: {deg} ({coord.get_pixel_alignment()})"
                )

        return (lon_out, lat_out)
