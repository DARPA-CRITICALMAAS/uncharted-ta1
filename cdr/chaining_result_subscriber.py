import json
import logging

import pika.spec as spec
from pika.adapters.blocking_connection import BlockingChannel as Channel

from cdr.request_publisher import LaraRequestPublisher
from tasks.common.queue import RequestResult
from tasks.common.result_subscriber import LaraResultSubscriber

logger = logging.getLogger("chaining_result_subscriber")


class ChainingResultSubscriber(LaraResultSubscriber):
    # sequence of pipelines execution
    DEFAULT_PIPELINE_SEQUENCE = [
        LaraResultSubscriber.SEGMENTATION_PIPELINE,
        LaraResultSubscriber.METADATA_PIPELINE,
        LaraResultSubscriber.POINTS_PIPELINE,
        LaraResultSubscriber.GEOREFERENCE_PIPELINE,
    ]

    def __init__(
        self,
        request_publisher: LaraRequestPublisher,
        result_queue: str,
        cdr_host: str,
        cdr_token: str,
        output: str,
        workdir: str,
        imagedir: str,
        host="localhost",
        port=5672,
        vhost="/",
        uid="",
        pwd="",
        pipeline_sequence=DEFAULT_PIPELINE_SEQUENCE,
    ):

        super().__init__(
            result_queue,
            cdr_host,
            cdr_token,
            output,
            workdir,
            imagedir,
            host=host,
            port=port,
            vhost=vhost,
            uid=uid,
            pwd=pwd,
        )
        self._request_publisher = request_publisher
        self._pipeline_sequence = (
            pipeline_sequence
            if len(pipeline_sequence) > 0
            else self.DEFAULT_PIPELINE_SEQUENCE
        )

    def _process_lara_result(
        self,
        channel: Channel,
        method: spec.Basic.Deliver,
        properties: spec.BasicProperties,
        body: bytes,
    ):
        """
        Process the received LARA result.  In a serial execution model this will also
        trigger the next pipeline in the sequence.

        Args:
            channel (Channel): The channel object.
            method (spec.Basic.Deliver): The method object.
            properties (spec.BasicProperties): The properties object.
            body (bytes): The body of the message.

        Returns:
            None

        Raises:
            Exception: If there is an error processing the result.
        """

        try:
            logger.info("received data from result channel")
            # parse the result
            body_decoded = json.loads(body.decode())
            result = RequestResult.model_validate(body_decoded)
            logger.info(
                f"processing result for request {result.request.id} of type {result.output_type}"
            )

            # When a publisher has been supplied we run the next pipeline in the
            # sequence
            output_pipeline = self.PIPELINE_OUTPUTS[result.output_type]
            next = self._pipeline_sequence.index(output_pipeline) + 1
            next_pipeline = (
                self._pipeline_sequence[next]
                if next < len(self._pipeline_sequence)
                else self.NULL_PIPELINE
            )

            # if there is no next pipeline in the sequence then we are done
            if next_pipeline == self.NULL_PIPELINE:
                return

            request = self.next_request(
                next_pipeline,
                result.request.image_id,
                result.request.image_url,
            )
            logger.info(f"sending next request in sequence: {request.task}")
            self._request_publisher.publish_lara_request(
                request, self.PIPELINE_QUEUES[next_pipeline]
            )

        except Exception as e:
            logger.exception(f"Error processing lara result: {e}")

        logger.info("result processing finished")
