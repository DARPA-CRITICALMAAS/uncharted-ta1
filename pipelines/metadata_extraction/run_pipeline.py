import argparse
from pathlib import Path
import logging
import os

from PIL.Image import Image as PILImage
from numpy import isin

from tasks.common.pipeline import (
    EmptyOutput,
    PipelineInput,
    BaseModelOutput,
    ImageOutput,
)
from pipelines.metadata_extraction.metadata_extraction_pipeline import (
    MetadataExtractorPipeline,
)
from tasks.common.io import (
    ImageFileInputIterator,
    ImageFileWriter,
    JSONFileWriter,
    validate_s3_config,
)
from tasks.metadata_extraction.metadata_extraction import (
    DEFAULT_GPT_MODEL,
    DEFAULT_OPENAI_API_VERSION,
    LLM_PROVIDER,
)
from util import logging as logging_util


def main():
    logger = logging.getLogger("metadata_pipeline")

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--cdr_schema", action="store_true")
    parser.add_argument("--debug_images", action="store_true")
    parser.add_argument("--llm", type=str, default=DEFAULT_GPT_MODEL)
    parser.add_argument(
        "--llm_api_version", type=str, default=DEFAULT_OPENAI_API_VERSION
    )
    parser.add_argument(
        "--llm_provider",
        type=LLM_PROVIDER,
        choices=list(LLM_PROVIDER),
        default=LLM_PROVIDER.OPENAI,
    )
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    p = parser.parse_args()

    logging_util.config_logger(logger, p.log_level)

    logger.info(f"Args: {p}")

    # validate any s3 path args up front
    validate_s3_config(str(p.input), p.workdir, "", p.output)

    # setup an input stream
    input = ImageFileInputIterator(str(p.input))

    # setup output writers
    file_writer = JSONFileWriter()
    image_writer = ImageFileWriter()

    # create the pipeline
    pipeline = MetadataExtractorPipeline(
        p.workdir,
        p.model,
        p.debug_images,
        p.cdr_schema,
        p.llm,
        p.llm_api_version,
        p.llm_provider,
        not p.no_gpu,
    )

    # run the extraction pipeline
    for doc_id, image in input:
        image_input = PipelineInput(image=image, raster_id=doc_id)
        try:
            results = pipeline.run(image_input)
        except Exception as e:
            logger.exception(e)
            continue

        # write the results out to the file system or s3 bucket
        for output_type, output_data in results.items():
            if isinstance(output_data, BaseModelOutput):
                if output_type == "metadata_extraction_output":
                    path = os.path.join(p.output, f"{doc_id}_metadata_extraction.json")
                    file_writer.process(path, output_data.data)
                elif output_type == "metadata_cdr_output" and p.cdr_schema:
                    path = os.path.join(
                        p.output, f"{doc_id}_metadata_extraction_cdr.json"
                    )
                    file_writer.process(path, output_data.data)
            elif isinstance(output_data, ImageOutput):
                # write out the image
                path = os.path.join(p.output, f"{doc_id}_metadata_extraction.png")
                assert isinstance(output_data.data, PILImage)
                image_writer.process(path, output_data.data)
            elif isinstance(output_data, EmptyOutput):
                logger.info(f"Empty {output_type} output for {doc_id}")
            else:
                logger.warning(f"Unknown output type: {type(output_data)}")
                continue


if __name__ == "__main__":
    main()
