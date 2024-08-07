import argparse
import logging
import os

from geopy.distance import distance as geo_distance
from PIL.Image import Image as PILIMAGE
from PIL import Image

from pipelines.geo_referencing.factory import create_geo_referencing_pipelines
from pipelines.geo_referencing.output import CSVWriter, JSONWriter
from tasks.common.io import ImageFileInputIterator
from tasks.common.pipeline import PipelineInput
from tasks.geo_referencing.georeference import QueryPoint
from util.coordinate import absolute_minmax
from util.json import read_json_file
from util import logging as logging_util

from typing import List, Optional, Tuple

FOV_RANGE_KM = (
    700  # [km] max range of a image's field-of-view (around the clue coord pt)
)
LON_MINMAX = [-66.0, -180.0]  # fallback geo-fence (ALL of USA + Alaska)
LAT_MINMAX = [24.0, 73.0]

IMG_FILE_EXT = "tif"
CLUE_FILEN_SUFFIX = "_clue"

Image.MAX_IMAGE_PIXELS = 400000000
GEOCODE_CACHE = "temp/geocode/"

logger: Optional[logging.Logger] = None


def main():
    global logger
    logger = logging.getLogger("georeferencing_pipeline")
    logging_util.config_logger(logger)

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--clue_dir", type=str, default="")
    parser.add_argument("--query_dir", type=str, default="")
    parser.add_argument("--points_dir", type=str, default="")
    parser.add_argument("--extract_metadata", type=bool, default=False)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--state_plane_lookup_filename",
        type=str,
        default="./data/state_plane_reference.csv",
    )
    parser.add_argument(
        "--state_plane_zone_filename",
        type=str,
        default="./data/USA_State_Plane_Zones_NAD27.geojson",
    )
    parser.add_argument(
        "--state_code_filename",
        type=str,
        default="./data/state_codes.csv",
    )
    parser.add_argument(
        "--country_code_filename",
        type=str,
        default="./data/country_codes.csv",
    )
    parser.add_argument(
        "--ocr_gamma_correction",
        type=float,
        default=0.5,
    )
    p = parser.parse_args()

    # setup an input stream
    input = ImageFileInputIterator(p.input)

    run_pipelines(p, input)


def create_input(
    raster_id: str, image: PILIMAGE, points_path: str, query_path: str, clue_path: str
) -> PipelineInput:
    input = PipelineInput()
    input.image = image
    input.raster_id = raster_id

    lon_minmax, lat_minmax, lon_sign_factor, clue_point = get_params(clue_path)
    input.params["lon_minmax"] = lon_minmax
    input.params["lat_minmax"] = lat_minmax
    input.params["lon_sign_factor"] = lon_sign_factor
    input.params["clue_point"] = clue_point

    query_pts = query_points_from_points(raster_id, points_path)
    if not query_pts:
        query_pts = parse_query_file(query_path, input.image.size)
    input.params["query_pts"] = query_pts

    return input


def run_pipelines(parsed, input_data: ImageFileInputIterator):
    assert logger is not None

    # get the pipelines
    pipelines = create_geo_referencing_pipelines(
        parsed.extract_metadata,
        parsed.output,
        parsed.workdir,
        parsed.model,
        parsed.state_plane_lookup_filename,
        parsed.state_plane_zone_filename,
        parsed.state_code_filename,
        parsed.country_code_filename,
        parsed.ocr_gamma_correction,
    )

    # get file paths
    clue_dir = parsed.clue_dir
    query_dir = parsed.query_dir
    points_dir = parsed.points_dir

    results = {}
    results_summary = {}
    results_levers = {}
    results_gcps = {}
    results_integration = {}
    writer_csv = CSVWriter()
    writer_json = JSONWriter()
    for p in pipelines:
        results[p.id] = []
        results_summary[p.id] = []
        results_levers[p.id] = []
        results_gcps[p.id] = []
        results_integration[p.id] = []

    for raster_id, image in input_data:
        logger.info(f"processing {raster_id}")

        clue_path = os.path.join(clue_dir, raster_id + CLUE_FILEN_SUFFIX + ".csv")
        query_path = os.path.join(query_dir, raster_id + ".csv")
        points_path = os.path.join(points_dir, f"pipeline_output_{raster_id}.json")

        input = create_input(raster_id, image, points_path, query_path, clue_path)

        for pipeline in pipelines:
            logger.info(f"running pipeline {pipeline.id}")
            output = pipeline.run(input)
            results[pipeline.id].append(output["geo"])
            results_summary[pipeline.id].append(output["summary"])
            results_levers[pipeline.id].append(output["levers"])
            results_gcps[pipeline.id].append(output["gcps"])
            results_integration[pipeline.id].append(output["schema"])
            schema_output_path = os.path.join(
                parsed.output, "maps", f"{pipeline.id}", f"{raster_id}.json"
            )
            writer_json.output([output["schema"]], {"path": schema_output_path})  # type: ignore
            logger.info(f"done pipeline {pipeline.id}\n\n")

        for p in pipelines:
            writer_csv.output(
                results[p.id], {"path": os.path.join(parsed.output, f"test-{p.id}.csv")}
            )
            writer_csv.output(
                results_summary[p.id],
                {"path": os.path.join(parsed.output, f"test_summary-{p.id}.csv")},
            )
            writer_json.output(
                results_levers[p.id],
                {"path": os.path.join(parsed.output, f"test_levers-{p.id}.json")},
            )
            writer_json.output(
                results_gcps[p.id],
                {"path": os.path.join(parsed.output, f"test_gcps-{p.id}.json")},
            )
            writer_json.output(
                results_integration[p.id],
                {"path": os.path.join(parsed.output, f"test_schema-{p.id}.json")},
            )

    for p in pipelines:
        writer_csv.output(
            results[p.id], {"path": os.path.join(parsed.output, f"test-{p.id}.csv")}
        )
        writer_csv.output(
            results_summary[p.id],
            {"path": os.path.join(parsed.output, f"test_summary-{p.id}.csv")},
        )
        writer_json.output(
            results_levers[p.id],
            {"path": os.path.join(parsed.output, f"test_levers-{p.id}.json")},
        )
        writer_json.output(
            results_gcps[p.id],
            {"path": os.path.join(parsed.output, f"test_gcps-{p.id}.json")},
        )
        writer_json.output(
            results_integration[p.id],
            {"path": os.path.join(parsed.output, f"test_schema-{p.id}.json")},
        )


def get_geofence(
    csv_clue_file: str,
    fov_range_km: float,
    lon_limits: List[float] = [-66.0, -180.0],
    lat_limits: List[float] = [24.0, 73.0],
) -> Tuple[List[float], List[float], float, Optional[Tuple[float, float]]]:
    # parse clue CSV file
    assert logger is not None
    (clue_lon, clue_lat, clue_ok) = parse_clue_file(csv_clue_file)
    clue_point = None
    if clue_ok:
        # if False:
        logger.info("using lon/lat clue {}, {}".format(clue_lon, clue_lat))
        dist_km = (
            fov_range_km / 2.0
        )  # distance from clue pt in all directions (N,E,S,W)
        fov_pt_north = geo_distance(kilometers=dist_km).destination(
            (clue_lat, clue_lon), bearing=0
        )
        fov_pt_east = geo_distance(kilometers=dist_km).destination(
            (clue_lat, clue_lon), bearing=90
        )
        fov_degrange_lon = abs(fov_pt_east[1] - clue_lon)
        fov_degrange_lat = abs(fov_pt_north[0] - clue_lat)
        lon_minmax = [clue_lon - fov_degrange_lon, clue_lon + fov_degrange_lon]
        lat_minmax = [clue_lat - fov_degrange_lat, clue_lat + fov_degrange_lat]
        clue_point = (clue_lon, clue_lat)

    else:
        # if no lat/lon clue, fall-back to full geo-fence of USA + Alaska
        logger.info("no lon/lat clue found so using full geo-fence for USA + Alaska")
        lon_minmax = lon_limits
        lat_minmax = lat_limits

    lon_sign_factor = 1.0

    return (
        lon_minmax,
        lat_minmax,
        lon_sign_factor,
        clue_point,
    )


def parse_query_file(
    csv_query_file: str, image_size: Optional[Tuple[float, float]] = None
) -> List[QueryPoint]:
    """
    Expected schema is of the form:
    raster_ID,row,col,NAD83_x,NAD83_y
    GEO_0004,8250,12796,-105.72065081057087,43.40255034572461
    ...
    Note: NAD83* columns may not be present
    row (y) and col (x) = pixel coordinates to query
    NAD83* = (if present) are ground truth answers (lon and lat) for the query x,y pt
    """

    first_line = True
    x_idx = 2
    y_idx = 1
    lon_idx = 3
    lat_idx = 4
    query_pts = []
    try:
        with open(csv_query_file) as f_in:
            for line in f_in:
                if line.startswith("raster_") or first_line:
                    first_line = False
                    continue  # header line, skip

                rec = line.split(",")
                if len(rec) < 3:
                    continue
                raster_id = rec[0]
                x = int(rec[x_idx])
                y = int(rec[y_idx])
                if image_size is not None:
                    # sanity check that query points are not > image dimensions!
                    if x > image_size[0] or y > image_size[1]:
                        err_msg = (
                            "Query point {}, {} is outside image dimensions".format(
                                x, y
                            )
                        )
                        raise IOError(err_msg)
                lonlat_gt = None
                if len(rec) >= 5:
                    lon = float(rec[lon_idx])
                    lat = float(rec[lat_idx])
                    if lon != 0 and lat != 0:
                        lonlat_gt = (lon, lat)
                query_pts.append(QueryPoint(raster_id, (x, y), lonlat_gt))

    except Exception as e:
        print("EXCEPTION parsing query file: {}".format(csv_query_file))
        print(e)

    # print('Num query points parsed: {}'.format(len(query_pts)))

    return query_pts


def query_points_from_points(
    raster_id: str, points_file: str
) -> Optional[List[QueryPoint]]:
    return None
    if not os.path.isfile(points_file):
        return None

    query_points = []
    points_raw = read_json_file(points_file)
    for pt in points_raw["labels"]:
        x = int((pt["x1"] + pt["x2"]) / 2)
        y = int((pt["y1"] + pt["y2"]) / 2)
        query_points.append(
            QueryPoint(raster_id, (x, y), None, properties={"label": pt["class_name"]})
        )

    return query_points


def get_params(
    clue_path: str,
) -> Tuple[List[float], List[float], float, Optional[Tuple[float, float]]]:
    return get_geofence(
        clue_path,
        fov_range_km=FOV_RANGE_KM,
        lon_limits=LON_MINMAX,
        lat_limits=LAT_MINMAX,
    )


def parse_clue_file(csv_clue_file: str) -> Tuple[float, float, bool]:
    """
    Expected schema is of the form:
    raster_ID,NAD83_x,NAD83_y
    GEO_0004,-105.72065081057087,43.40255034572461

    Or possibly
    raster_ID,row,col,NAD83_x,NAD83_y
    GEO_0004,,,-105.72065081057087,43.40255034572461
    """

    first_line = True
    got_clue = False
    lon = 0.0
    lat = 0.0
    if not os.path.isfile(csv_clue_file):
        assert logger is not None
        logger.info(f"clue file not found when looking for {csv_clue_file}")
        return (lon, lat, got_clue)

    try:
        with open(csv_clue_file) as f_in:
            for line in f_in:
                if line.startswith("raster_") or first_line:
                    first_line = False
                    continue  # header line, skip

                if got_clue:
                    break

                rec = line.split(",")
                if len(rec) < 3:
                    continue
                if len(rec) < 5:
                    lon = float(rec[1])
                    lat = float(rec[2])
                else:
                    lon = round(float(rec[3]), 1)  # round to 1 decimal place
                    lat = round(float(rec[4]), 1)
                got_clue = True
    except Exception as e:
        print("EXCEPTION parsing clue file!")
        print(e)

    return (lon, lat, got_clue)


if __name__ == "__main__":
    main()
