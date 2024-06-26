import argparse
import atexit
import datetime
from pathlib import Path
import re
from time import sleep
import httpx
import json
import logging

import coloredlogs
import ngrok
import os
import pika
from pydantic import BaseModel
import rasterio as rio
import rasterio.transform as riot
import threading

from flask import Flask, request, Response
from pika.adapters.blocking_connection import BlockingChannel as Channel
from pika import BlockingConnection, spec
from pika.exceptions import AMQPChannelError, AMQPConnectionError
from PIL import Image
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject

from tasks.common.io import ImageFileInputIterator, download_file
from tasks.common.queue import (
    GEO_REFERENCE_REQUEST_QUEUE,
    METADATA_REQUEST_QUEUE,
    POINTS_REQUEST_QUEUE,
    SEGMENTATION_REQUEST_QUEUE,
    OutputType,
    Request,
    RequestResult,
)

from schema.mappers.cdr import get_mapper
from schema.cdr_schemas.events import Event, MapEventPayload
from schema.cdr_schemas.feature_results import FeatureResults
from schema.cdr_schemas.georeference import GeoreferenceResults, GroundControlPoint
from schema.cdr_schemas.metadata import CogMetaData
from tasks.geo_referencing.coordinates_extractor import RE_DEG
from tasks.geo_referencing.entities import GeoreferenceResult as LARAGeoreferenceResult
from tasks.metadata_extraction.entities import MetadataExtraction as LARAMetadata
from tasks.point_extraction.entities import MapImage as LARAPoints
from tasks.segmentation.entities import MapSegmentation as LARASegmentation

from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cdr")

app = Flask(__name__)

request_channel: Optional[Channel] = None

CDR_API_TOKEN = os.environ["CDR_API_TOKEN"]
CDR_HOST = "https://api.cdr.land"
CDR_SYSTEM_NAME = "uncharted"
CDR_SYSTEM_VERSION = "0.0.3"
CDR_CALLBACK_SECRET = "maps rock"
APP_PORT = 5001
CDR_EVENT_LOG = "events.log"

LARA_RESULT_QUEUE_NAME = "lara_result_queue"


class JSONLog:
    def __init__(self, file: str):
        self._file = file

    def log(self, log_type: str, data: Dict[str, Any]):
        # append the data as json, treating the file as a json lines file
        log_data = {
            "timestamp": f"{datetime.datetime.now()}",
            "log_type": log_type,
            "data": data,
        }
        with open(self._file, "a") as log_file:
            log_file.write(f"{json.dumps(log_data)}\n")


class Settings:
    cdr_api_token: str
    cdr_host: str
    workdir: str
    imagedir: str
    output: str
    system_name: str
    system_version: str
    callback_secret: str
    callback_url: str
    registration_id: str
    rabbitmq_host: str
    json_log: JSONLog


settings: Settings


class LaraRequestPublisher:
    def __init__(self, request_queues: List[str], host="localhost") -> None:
        self._request_connection: Optional[BlockingConnection] = None
        self._request_channel: Optional[Channel] = None
        self._host = host
        self._request_queues = request_queues

    def _create_channel(self) -> Channel:
        """
        Creates a blocking connection and channel on the given host and declares the given queue.

        Args:
            host: The host to connect to.
            queue: The queue to declare.

        Returns:
            The created channel.
        """
        logger.info(f"creating channel on host {self._host}")
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                self._host,
                heartbeat=900,
                blocked_connection_timeout=600,
            )
        )
        channel = connection.channel()
        for queue in self._request_queues:
            channel.queue_declare(queue=queue)
        return channel

    def publish_lara_request(self, req: Request, request_queue: str):
        """
        Publishes a LARA request to a specified queue.

        Args:
            req (Request): The LARA request object to be published.
            request_channel (Channel): The channel used for publishing the request.
            request_queue (str): The name of the queue to publish the request to.
        """
        logger.info(f"sending request {req.id} for image {req.image_id} to lara queue")
        if self._request_connection is not None and self._request_channel is not None:
            self._request_connection.add_callback_threadsafe(
                lambda: self._request_channel.basic_publish(  #   type: ignore
                    exchange="",
                    routing_key=request_queue,
                    body=json.dumps(req.model_dump()),
                )
            )
            logger.info(f"request {req.id} published to {request_queue}")
        else:
            logger.error("request connection / channel not initialized")

    def _run_request_queue(self):
        """
        Main loop to service the request queue. process_data_events is set to block for a maximum
        of 1 second before returning to ensure that heartbeats etc. are processed.
        """
        self._request_connection: Optional[BlockingConnection] = None
        while True:
            try:
                if (
                    self._request_connection is None
                    or self._request_connection.is_closed
                ):
                    logger.info(
                        f"connecting to request queue {','.join(self._request_queues)}"
                    )
                    self._request_channel = self._create_channel()
                    self._request_connection = self._request_channel.connection

                if self._request_connection is not None:
                    self._request_connection.process_data_events(time_limit=1)
                else:
                    logger.error("request connection not initialized")
            except (AMQPChannelError, AMQPConnectionError):
                logger.warn("request connection closed, reconnecting")
                if (
                    self._request_connection is not None
                    and self._request_connection.is_open
                ):
                    self._request_connection.close()
                sleep(5)

    def start_lara_request_queue(self):
        """
        Starts the LARA request queue by running the `run_request_queue` function in a separate thread.

        Args:
            host (str): The host address to pass to the `run_request_queue` function.

        Returns:
            None
        """
        threading.Thread(
            target=self._run_request_queue,
        ).start()


def prefetch_image(working_dir: Path, image_id: str, image_url: str) -> None:
    """
    Prefetches the image from the CDR for use by the pipelines.
    """
    # check working dir for the image
    filename = working_dir / f"{image_id}.tif"

    if not os.path.exists(filename):
        # download image
        image_data = download_file(image_url)

        # write it to working dir, creating the directory if necessary
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as file:
            file.write(image_data)


def project_image(
    source_image_path: str, target_image_path: str, geo_transform: Affine, crs: str
):
    with rio.open(source_image_path) as raw:
        bounds = riot.array_bounds(raw.height, raw.width, geo_transform)
        pro_transform, pro_width, pro_height = calculate_default_transform(
            crs, crs, raw.width, raw.height, *tuple(bounds)
        )
        pro_kwargs = raw.profile.copy()
        pro_kwargs.update(
            {
                "driver": "COG",
                "crs": {"init": crs},
                "transform": pro_transform,
                "width": pro_width,
                "height": pro_height,
            }
        )
        _raw_data = raw.read()
        with rio.open(target_image_path, "w", **pro_kwargs) as pro:
            for i in range(raw.count):
                _ = reproject(
                    source=_raw_data[i],
                    destination=rio.band(pro, i + 1),
                    src_transform=geo_transform,
                    src_crs=crs,
                    dst_transform=pro_transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear,
                    num_threads=8,
                    warp_mem_limit=256,
                )


def cps_to_transform(
    gcps: List[GroundControlPoint], height: int, to_crs: str
) -> Affine:
    cps = [
        {
            "row": float(gcp.px_geom.rows_from_top),
            "col": float(gcp.px_geom.columns_from_left),
            "x": float(gcp.map_geom.longitude),  #   type: ignore
            "y": float(gcp.map_geom.latitude),  #   type: ignore
            "crs": gcp.crs,
        }
        for gcp in gcps
    ]
    cps_p = []
    for cp in cps:
        proj = Transformer.from_crs(cp["crs"], to_crs, always_xy=True)
        x_p, y_p = proj.transform(xx=cp["x"], yy=cp["y"])
        cps_p.append(
            riot.GroundControlPoint(row=cp["row"], col=cp["col"], x=x_p, y=y_p)
        )

    return riot.from_gcps(cps_p)


def project_georeference(
    source_image_path: str,
    target_image_path: str,
    target_crs: str,
    gcps: List[GroundControlPoint],
):
    # open the image
    img = Image.open(source_image_path)
    _, height = img.size

    # create the transform
    geo_transform = cps_to_transform(gcps, height=height, to_crs=target_crs)

    # use the transform to project the image
    project_image(source_image_path, target_image_path, geo_transform, target_crs)


@app.route("/process_event", methods=["POST"])
def process_cdr_event():
    logger.info("event callback started")
    evt = request.get_json(force=True)
    settings.json_log.log("event", evt)
    logger.info(f"event data received {evt['event']}")
    lara_reqs: Dict[str, Request] = {}

    try:
        # handle event directly or create lara request
        match evt["event"]:
            case "ping":
                logger.info("received ping event")
            case "map.process":
                logger.info("Received map event")
                map_event = MapEventPayload.model_validate(evt["payload"])
                lara_reqs[GEO_REFERENCE_REQUEST_QUEUE] = Request(
                    id=evt["id"],
                    task="georeference",
                    image_id=map_event.cog_id,
                    image_url=map_event.cog_url,
                    output_format="cdr",
                )
                lara_reqs[POINTS_REQUEST_QUEUE] = Request(
                    id=evt["id"],
                    task="points",
                    image_id=map_event.cog_id,
                    image_url=map_event.cog_url,
                    output_format="cdr",
                )
                lara_reqs[SEGMENTATION_REQUEST_QUEUE] = Request(
                    id=evt["id"],
                    task="segments",
                    image_id=map_event.cog_id,
                    image_url=map_event.cog_url,
                    output_format="cdr",
                )
                lara_reqs[METADATA_REQUEST_QUEUE] = Request(
                    id=evt["id"],
                    task="metadata",
                    image_id=map_event.cog_id,
                    image_url=map_event.cog_url,
                    output_format="cdr",
                )
            case _:
                logger.info(f"received unsupported {evt['event']} event")

    except Exception:
        logger.error(f"exception processing {evt['event']} event")
        raise

    if len(lara_reqs) == 0:
        # assume ping or ignored event type
        return Response({"ok": "success"}, status=200, mimetype="application/json")

    # Pre-fetch the image from th CDR for use by the pipelines.  The pipelines have an
    # imagedir arg that should be configured to point at this location.
    prefetch_image(Path(settings.imagedir), map_event.cog_id, map_event.cog_url)
    # queue event in background since it may be blocking on the queue
    # assert request_channel is not None
    for queue_name, lara_req in lara_reqs.items():
        request_publisher.publish_lara_request(lara_req, queue_name)

    return Response({"ok": "success"}, status=200, mimetype="application/json")


def process_image(image_id: str, request_publisher: LaraRequestPublisher):
    logger.info(f"processing image with id {image_id}")

    image_url = f"https://s3.amazonaws.com/public.cdr.land/cogs/{image_id}.cog.tif"

    # build the request
    lara_reqs: Dict[str, Request] = {}
    lara_reqs[GEO_REFERENCE_REQUEST_QUEUE] = Request(
        id="mock-georeference",
        task="georeference",
        image_id=image_id,
        image_url=image_url,
        output_format="cdr",
    )
    lara_reqs[POINTS_REQUEST_QUEUE] = Request(
        id="mock-points",
        task="points",
        image_id=image_id,
        image_url=image_url,
        output_format="cdr",
    )
    lara_reqs[SEGMENTATION_REQUEST_QUEUE] = Request(
        id="mock-segments",
        task="segments",
        image_id=image_id,
        image_url=image_url,
        output_format="cdr",
    )
    lara_reqs[METADATA_REQUEST_QUEUE] = Request(
        id="mock-metadata",
        task="metadata",
        image_id=image_id,
        image_url=image_url,
        output_format="cdr",
    )

    # Pre-fetch the image from th CDR for use by the pipelines.  The pipelines have an
    # imagedir arg that should be configured to point at this location.
    prefetch_image(Path(settings.imagedir), image_id, image_url)

    # push the request onto the queue
    for queue_name, request in lara_reqs.items():
        logger.info(
            f"publishing request for image {image_id} to {queue_name} task: {request.task}"
        )
        request_publisher.publish_lara_request(request, queue_name)


def write_cdr_result(image_id: str, output_type: OutputType, result: BaseModel):
    """
    Write the CDR result to a JSON file.

    Args:
        image_id (str): The ID of the image.
        output_type (OutputType): The type of output.
        result (BaseModel): The result to be written.

    Returns:
        None
    """
    if settings.output:
        output_file = os.path.join(
            settings.output,
            f"{image_id}_{output_type.name.lower()}.json",
        )
        os.makedirs(
            settings.output, exist_ok=True
        )  # Create the output directory if it doesn't exist
        with open(output_file, "a") as f:
            logger.info(f"writing result to {output_file}")
            f.write(json.dumps(result.model_dump()))
            f.write("\n")
        return


def push_georeferencing(result: RequestResult):
    # reproject image to file on disk for pushing to CDR
    georef_result_raw = json.loads(result.output)

    # validate the result by building the model classes
    cdr_result: Optional[GeoreferenceResults] = None
    files_ = []
    try:
        lara_result = LARAGeoreferenceResult.model_validate(georef_result_raw)
        mapper = get_mapper(lara_result, settings.system_name, settings.system_version)
        cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
        assert cdr_result is not None
        assert cdr_result.georeference_results is not None
        assert cdr_result.georeference_results[0] is not None
        assert cdr_result.georeference_results[0].projections is not None
        projection = cdr_result.georeference_results[0].projections[0]
        gcps = cdr_result.gcps
        output_file_name = projection.file_name
        output_file_name_full = os.path.join(settings.workdir, output_file_name)
        assert gcps is not None

        logger.info(
            f"projecting image {result.image_path} to {output_file_name_full} using crs {projection.crs}"
        )
        project_georeference(
            result.image_path, output_file_name_full, projection.crs, gcps
        )

        files_.append(("files", (output_file_name, open(output_file_name_full, "rb"))))
    except:
        logger.error(
            "bad georeferencing result received so creating an empty result to send to cdr"
        )

        # create an empty result to send to cdr
        cdr_result = GeoreferenceResults(
            cog_id=result.request.image_id,
            georeference_results=[],
            gcps=[],
            system=settings.system_name,
            system_version=settings.system_version,
        )

    assert cdr_result is not None
    try:
        # write the result to disk if output is set
        if settings.output:
            write_cdr_result(result.request.image_id, result.output_type, cdr_result)
            return

        # push the result to CDR
        logger.info(f"pushing result for request {result.request.id} to CDR")
        headers = {"Authorization": f"Bearer {settings.cdr_api_token}"}
        client = httpx.Client(follow_redirects=True)
        resp = client.post(
            f"{settings.cdr_host}/v1/maps/publish/georef",
            data={"georef_result": json.dumps(cdr_result.model_dump())},
            files=files_,
            headers=headers,
            timeout=None,
        )
        logger.info(
            f"result for request {result.request.id} sent to CDR with response {resp.status_code}: {resp.content}"
        )
    except:
        logger.info("error when attempting to submit georeferencing results")


def push_features(result: RequestResult, model: FeatureResults):
    """
    Pushes the features result to the CDR
    """
    if settings.output:
        write_cdr_result(result.request.image_id, result.output_type, model)
        return

    logger.info(f"pushing features result for request {result.request.id} to CDR")
    headers = {
        "Authorization": f"Bearer {settings.cdr_api_token}",
        "Content-Type": "application/json",
    }
    client = httpx.Client(follow_redirects=True)
    resp = client.post(
        f"{settings.cdr_host}/v1/maps/publish/features",
        data=model.model_dump_json(),  #   type: ignore
        headers=headers,
        timeout=None,
    )
    logger.info(
        f"result for request {result.request.id} sent to CDR with response {resp.status_code}: {resp.content}"
    )


def push_segmentation(result: RequestResult):
    """
    Pushes the segmentation result to the CDR
    """
    segmentation_raw_result = json.loads(result.output)

    # validate the result by building the model classes
    cdr_result: Optional[FeatureResults] = None
    try:
        lara_result = LARASegmentation.model_validate(segmentation_raw_result)
        mapper = get_mapper(lara_result, settings.system_name, settings.system_version)
        cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
    except:
        logger.error(
            "bad segmentation result received so unable to send results to cdr"
        )
        return

    assert cdr_result is not None
    push_features(result, cdr_result)


def push_points(result: RequestResult):
    points_raw_result = json.loads(result.output)

    # validate the result by building the model classes
    cdr_result: Optional[FeatureResults] = None
    try:
        lara_result = LARAPoints.model_validate(points_raw_result)
        mapper = get_mapper(lara_result, settings.system_name, settings.system_version)
        cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
    except:
        logger.error("bad points result received so unable to send results to cdr")
        return

    assert cdr_result is not None
    push_features(result, cdr_result)


def push_metadata(result: RequestResult):
    """
    Pushes the metadata result to the CDR
    """
    metadata_result_raw = json.loads(result.output)

    # validate the result by building the model classes
    cdr_result: Optional[CogMetaData] = None
    try:
        lara_result = LARAMetadata.model_validate(metadata_result_raw)
        mapper = get_mapper(lara_result, settings.system_name, settings.system_version)
        cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
    except:
        logger.error("bad metadata result received so unable to send results to cdr")
        return

    assert cdr_result is not None

    # wrap metadata into feature result
    final_result = FeatureResults(
        cog_id=result.request.image_id,
        cog_metadata_extractions=[cdr_result],
        system=cdr_result.system,
        system_version=cdr_result.system_version,
    )

    push_features(result, final_result)


def process_lara_result(
    channel: Channel,
    method: spec.Basic.Deliver,
    properties: spec.BasicProperties,
    body: bytes,
):
    try:
        logger.info("received data from result channel")
        # parse the result
        body_decoded = json.loads(body.decode())
        result = RequestResult.model_validate(body_decoded)
        logger.info(
            f"processing result for request {result.request.id} of type {result.output_type}"
        )

        # reproject image to file on disk for pushing to CDR
        match result.output_type:
            case OutputType.GEOREFERENCING:
                logger.info("georeferencing results received")
                push_georeferencing(result)
            case OutputType.METADATA:
                logger.info("metadata results received")
                push_metadata(result)
            case OutputType.SEGMENTATION:
                logger.info("segmentation results received")
                push_segmentation(result)
            case OutputType.POINTS:
                logger.info("points results received")
                push_points(result)
            case _:
                logger.info("unsupported output type received from queue")
        settings.json_log.log(
            "result", {"type": result.output_type, "cog_id": result.request.image_id}
        )

    except Exception as e:
        logger.exception(f"Error processing result: {str(e)}")

    logger.info("result processing finished")


def create_channel(host: str, queue: str) -> Channel:
    """
    Creates a blocking connection and channel on the given host and declares the given queue.

    Args:
        host: The host to connect to.
        queue: The queue to declare.

    Returns:
        The created channel.
    """
    logger.info(f"creating channel on host {host}")
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host,
            heartbeat=900,
            blocked_connection_timeout=600,
        )
    )
    channel = connection.channel()
    channel.queue_declare(queue=queue)
    return channel


def _run_lara_result_queue(result_queue: str, host="localhost"):
    while True:
        result_channel: Optional[Channel] = None
        try:
            logger.info(
                f"starting the listener on the result queue ({host}:{result_queue})"
            )
            # setup the result queue
            result_channel = create_channel(host, result_queue)
            result_channel.basic_qos(prefetch_count=1)

            # start consuming the results - will timeout after 5 seconds of inactivity
            # allowing things like heartbeats to be processed
            while True:
                for method_frame, properties, body in result_channel.consume(
                    result_queue,
                    inactivity_timeout=5,
                    auto_ack=True,
                ):
                    if method_frame:
                        process_lara_result(
                            result_channel, method_frame, properties, body
                        )
        except (AMQPConnectionError, AMQPChannelError):
            logger.warning(f"result channel closed, reconnecting")
            # channel is closed - make sure the connection is closed to facilitate a
            # clean reconnect
            if result_channel and not result_channel.connection.is_closed:
                logger.info("closing result connection")
                result_channel.connection.close()
            sleep(5)


def start_lara_result_queue(result_queue: str, host="localhost"):
    threading.Thread(
        target=_run_lara_result_queue,
        args=(result_queue, host),
    ).start()


def register_cdr_system():
    logger.info(f"registering system {settings.system_name} with cdr")
    headers = {"Authorization": f"Bearer {settings.cdr_api_token}"}

    registration = {
        "name": settings.system_name,
        "version": settings.system_version,
        "callback_url": settings.callback_url,
        "webhook_secret": settings.callback_secret,
        # Leave blank if callback url has no auth requirement
        # "auth_header": "",
        # "auth_token": "",
        # Registers for ALL events
        "events": [],
    }

    client = httpx.Client(follow_redirects=True)

    r = client.post(
        f"{settings.cdr_host}/user/me/register", json=registration, headers=headers
    )

    # Log our registration_id such we can delete it when we close the program.
    response_raw = r.json()
    settings.registration_id = response_raw["id"]
    logger.info(f"system {settings.system_name} registered with cdr")


def get_cdr_registrations() -> List[Dict[str, Any]]:
    logger.info("getting list of existing registrations in CDR")

    # query the listing endpoint in CDR
    headers = {"Authorization": f"Bearer {settings.cdr_api_token}"}
    client = httpx.Client(follow_redirects=True)
    response = client.get(
        f"{settings.cdr_host}/user/me/registrations",
        headers=headers,
    )

    # parse json response
    return json.loads(response.content)


def cdr_unregister(registration_id: str):
    headers = {"Authorization": f"Bearer {settings.cdr_api_token}"}
    client = httpx.Client(follow_redirects=True)
    client.delete(
        f"{settings.cdr_host}/user/me/register/{registration_id}",
        headers=headers,
    )


def cdr_clean_up():
    logger.info(f"unregistering system {settings.registration_id} with cdr")
    # delete our registered system at CDR on program end
    cdr_unregister(settings.registration_id)
    logger.info(f"system {settings.registration_id} no longer registered with cdr")


def cdr_startup(host: str):
    # check if already registered and delete existing registrations for this name and token combination
    registrations = get_cdr_registrations()
    if len(registrations) > 0:
        for r in registrations:
            if r["name"] == settings.system_name:
                cdr_unregister(r["id"])

    # make it accessible from the outside
    settings.callback_url = f"{host}/process_event"

    register_cdr_system()

    # wire up the cleanup of the registration
    atexit.register(cdr_clean_up)


def start_app():
    # forward ngrok port
    logger.info("using ngrok to forward ports")
    listener = ngrok.forward(APP_PORT, authtoken_from_env=True)
    cdr_startup(listener.url())

    app.run(host="0.0.0.0", port=APP_PORT)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s %(name)s\t: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    coloredlogs.DEFAULT_FIELD_STYLES["levelname"] = {"color": "white"}
    coloredlogs.install(logger=logger)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("process", "host"), required=True)
    parser.add_argument("--system", type=str, default=CDR_SYSTEM_NAME)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--imagedir", type=str, required=True)
    parser.add_argument("--cog_id", type=str, required=False)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--cdr_event_log", type=str, default=CDR_EVENT_LOG)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    p = parser.parse_args()

    global settings
    settings = Settings()
    settings.cdr_api_token = CDR_API_TOKEN
    settings.cdr_host = CDR_HOST
    settings.workdir = p.workdir
    settings.imagedir = p.imagedir
    settings.output = p.output
    settings.system_name = p.system
    settings.system_version = CDR_SYSTEM_VERSION
    settings.callback_secret = CDR_CALLBACK_SECRET
    settings.rabbitmq_host = p.host
    settings.json_log = JSONLog(os.path.join(p.workdir, p.cdr_event_log))

    # check parameter consistency: either the mode is process and a cog id is supplied or the mode is host without a cog id
    if p.mode == "process":
        if (p.cog_id == "" or p.cog_id is None) and (p.input == "" or p.input is None):
            logger.info("process mode requires a cog id or an input file")
            exit(1)
    elif p.mode == "host" and (not p.cog_id == "" and p.cog_id is not None):
        logger.info("a cog id cannot be provided if host mode is selected")
        exit(1)
    logger.info(f"starting cdr in {p.mode} mode")

    # start the listener for the results
    start_lara_result_queue(LARA_RESULT_QUEUE_NAME, host=settings.rabbitmq_host)

    # declare a global request publisher since we need to access it from the
    # CDR event endpoint
    global request_publisher
    request_publisher = LaraRequestPublisher(
        [
            SEGMENTATION_REQUEST_QUEUE,
            POINTS_REQUEST_QUEUE,
            GEO_REFERENCE_REQUEST_QUEUE,
            METADATA_REQUEST_QUEUE,
        ],
        host=settings.rabbitmq_host,
    )
    request_publisher.start_lara_request_queue()

    # either start the flask app if host mode selected or run the image specified if in process mode
    if p.mode == "host":
        start_app()
    elif p.mode == "process":
        cdr_startup("https://mock.example")
        if p.input:
            # open the cog csv file and process each line
            with open(p.input, "r") as f:
                for line in f:
                    cog_id = line.strip()
                    process_image(cog_id, request_publisher)
        else:
            process_image(p.cog_id, request_publisher)


if __name__ == "__main__":
    main()
