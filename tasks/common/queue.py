import abc
import argparse
import json
import logging
import os
from pathlib import Path

import pika

from PIL.Image import Image as PILImage

from tasks.common.pipeline import (
    BaseModelOutput,
    Pipeline,
    PipelineInput,
)
from tasks.common.io import ImageFileInputIterator
from tasks.common.image import download_file

from pika.adapters.blocking_connection import BlockingChannel as Channel
from pika import spec

from pydantic import BaseModel

from typing import Tuple

logger = logging.getLogger("process_queue")


class Request(BaseModel):
    """
    A request to run an image through a pipeline.
    """

    id: str
    task: str
    image_id: str
    image_url: str
    output_format: str


class RequestResult(BaseModel):
    """
    The result of a pipeline request.
    """

    request: Request

    success: bool
    output: str
    image_path: str


class RequestQueue:
    """
    Input and output messages queues for process pipeline requests and publishing
    the results.

    Args:
        pipeline: The pipeline to use for processing requests.
        request_queue: The name of the request queue.
        result_queue: The name of the result queue.
        host: The host of the queue.
        heartbeat: The heartbeat interval.
        blocked_connection_timeout: The blocked connection timeout.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        request_queue: str,
        result_queue: str,
        workdir: Path,
        host="localhost",
        heartbeat=900,
        blocked_connection_timeout=600,
    ) -> None:
        """
        Initialize the request queue.
        """
        self._pipeline = pipeline
        self._host = host
        self._request_queue = request_queue
        self._result_queue = result_queue
        self._heartbeat = heartbeat
        self._blocked_connection_timeout = blocked_connection_timeout
        self._working_dir = workdir

        self.setup_queues()

    def setup_queues(self) -> None:
        """
        Setup the input and output queues.
        """

        logger.info("wiring up request queue to input and output queues")

        # setup input and output queue
        request_connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                "localhost",
                heartbeat=900,
                blocked_connection_timeout=600,
            )
        )
        self._input_channel = request_connection.channel()
        self._input_channel.queue_declare(queue="metadata_request")
        self._input_channel.basic_qos(prefetch_count=1)
        self._input_channel.basic_consume(
            queue="metadata_request",
            on_message_callback=self._process_queue_input,
            auto_ack=False,  # manually ack based on message validity
        )

        result_connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                self._host,
                heartbeat=self._heartbeat,
                blocked_connection_timeout=self._blocked_connection_timeout,
            )
        )
        self._output_channel = result_connection.channel()
        self._output_channel.queue_declare(queue=self._result_queue)

    def start_request_queue(self):
        """Start the request queue."""
        logger.info("starting request queue")
        self._input_channel.start_consuming()

    def _process_queue_input(
        self,
        channel: Channel,
        method: spec.Basic.Deliver,
        _: spec.BasicProperties,
        body: bytes,
    ) -> None:
        """
        Process a request from the input queue.

        Args:
            channel: The channel the request was received on.
            method: The method used to deliver the request.
            body: The body of the request.

        """

        logger.info("request received from input queue")

        try:
            body_decoded = json.loads(body.decode())
            # parse body as request
            request = Request.model_validate(body_decoded)
            channel.basic_ack(delivery_tag=method.delivery_tag)

            # process the request
            # result = self._process_request(request)

            # create the input
            image_path, image_it = self._get_image(
                self._working_dir, request.image_id, request.image_url
            )
            input = self._create_pipeline_input(request, next(image_it)[1])

            # run the pipeline
            outputs = self._pipeline.run(input)

            # create the response
            output_raw: BaseModelOutput = outputs["lara"]  # type: ignore
            result = self._create_output(request, image_path, output_raw)

            logger.info("writing request result to output queue")

            # run queue operations
            self._output_channel.basic_publish(
                exchange="",
                routing_key=self._result_queue,
                body=json.dumps(result.model_dump()),
            )
            logger.info("result written to output queue")
        except Exception as e:
            logger.exception(e)
            channel.basic_reject(requeue=False, delivery_tag=method.delivery_tag)

    def _create_pipeline_input(
        self, request: Request, image: PILImage
    ) -> PipelineInput:
        """
        Create the pipeline input for the request.

        Args:
            request: The request.
            image: The image.

        Returns:
            The pipeline input.
        """
        input = PipelineInput()
        input.image = image
        input.raster_id = request.image_id

        return input

    def _create_output(
        self, request: Request, image_path: str, output: BaseModelOutput
    ) -> RequestResult:
        """
        Create the output for the request.

        Args:
            request: The request.
            image_path: The path to the image.
            output: The output of the pipeline.

        Returns:
            The request result.
        """
        return RequestResult(
            request=request,
            output=json.dumps(output.data.model_dump()),
            success=True,
            image_path=image_path,
        )

    def _get_image(
        self, working_dir: Path, image_id: str, image_url: str
    ) -> Tuple[str, ImageFileInputIterator]:
        """
        Get the image for the request.
        """
        # check working dir for the image
        disk_filename = str(working_dir / "images" / f"{image_id}.tif")

        if not os.path.isfile(disk_filename):
            # download image
            image_data = download_file(image_url)

            # write it to working dir
            with open(disk_filename, "wb") as file:
                file.write(image_data)

        # load image from disk
        return disk_filename, ImageFileInputIterator(disk_filename)
