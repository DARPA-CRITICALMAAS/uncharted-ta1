import argparse
import json
import logging
import os
from PIL.Image import Image as PILImage
from tasks.common.pipeline import (
    ImageOutput,
    EmptyOutput,
    PipelineInput,
    BaseModelOutput,
    ImageDictOutput,
)
from pipelines.point_extraction.point_extraction_pipeline import PointExtractionPipeline
from tasks.common.io import (
    ImageFileInputIterator,
    JSONFileWriter,
    ImageFileWriter,
    validate_s3_config,
)
from tasks.point_extraction.legend_item_utils import (
    parse_legend_annotations,
    parse_legend_point_hints,
)
from tasks.point_extraction.entities import (
    LEGEND_ITEMS_OUTPUT_KEY,
)
from util import logging as logging_util


def main():

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workdir", type=str, default="tmp/lara/workdir")
    parser.add_argument("--model_point_extractor", type=str, required=True)
    parser.add_argument("--model_segmenter", type=str, default=None)
    parser.add_argument("--cdr_schema", action="store_true")  # False by default
    parser.add_argument("--bitmasks", action="store_true")  # False by default
    parser.add_argument("--fetch_legend_items", action="store_true")
    parser.add_argument("--legend_items_dir", type=str, default="")
    parser.add_argument("--legend_hints_dir", type=str, default="")
    parser.add_argument("--debug_images", action="store_true")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--log_level", default="INFO")
    p = parser.parse_args()

    logger = logging.getLogger("point_extraction_pipeline")
    logging_util.config_logger(logger, p.log_level)

    # validate any s3 path args up front
    validate_s3_config(p.input, p.workdir, "", p.output)

    # setup an input stream
    input = ImageFileInputIterator(p.input)

    # setup an output writer
    file_writer = JSONFileWriter()
    image_writer = ImageFileWriter()

    # create the pipeline
    pipeline = PointExtractionPipeline(
        p.model_point_extractor,
        p.model_segmenter,
        p.workdir,
        fetch_legend_items=p.fetch_legend_items,
        include_cdr_output=p.cdr_schema,
        include_bitmasks_output=p.bitmasks,
        gpu=not p.no_gpu,
        batch_size=p.batch_size,
        debug_images=p.debug_images,
    )

    # run the extraction pipeline
    for doc_id, image in input:

        # --- TEMP code needed to run with contest dir-based data
        if (
            doc_id.endswith("_pt")
            or doc_id.endswith("_poly")
            or doc_id.endswith("_line")
            or doc_id.endswith("_point")
        ):
            logger.info(f"Skipping {doc_id}")
            continue
        # ---
        logger.info(f"Processing {doc_id}")
        image_input = PipelineInput(image=image, raster_id=doc_id)

        if p.legend_items_dir:
            # load JSON legend annotations file, if present, parse and add to PipelineInput
            # expected format is LegendItemResponse CDR pydantic objects
            try:
                # check for legend annotations for this image
                with open(
                    os.path.join(p.legend_items_dir, doc_id + ".json"), "r"
                ) as fp:
                    legend_anns = json.load(fp)
                    legend_pt_items = parse_legend_annotations(legend_anns, doc_id)
                    # add legend item annotations as a pipeline input param
                    image_input.params[LEGEND_ITEMS_OUTPUT_KEY] = legend_pt_items
                    logger.info(
                        f"Number of legend point items loaded for this map: {len(legend_pt_items.items)}"
                    )

            except Exception as e:
                logger.error("EXCEPTION loading legend items json: " + repr(e))

        elif p.legend_hints_dir:
            # load JSON legend hints file, if present, parse and add to PipelineInput
            try:
                # check for legend hints for this image (JSON CMA contest data)
                with open(
                    os.path.join(p.legend_hints_dir, doc_id + ".json"), "r"
                ) as fp:
                    legend_hints = json.load(fp)
                    legend_pt_items = parse_legend_point_hints(legend_hints, doc_id)
                    # add legend item hints as a pipeline input param
                    image_input.params[LEGEND_ITEMS_OUTPUT_KEY] = legend_pt_items
                    logger.info(
                        f"Number of legend point items loaded for this map: {len(legend_pt_items.items)}"
                    )

            except Exception as e:
                logger.error("EXCEPTION loading legend hints json: " + repr(e))

        if p.bitmasks:
            bitmasks_out_dir = os.path.join(p.output, "bitmasks")
            os.makedirs(bitmasks_out_dir, exist_ok=True)
            if not p.legend_hints_dir and not p.legend_items_dir:
                logger.warning(
                    'Points pipeline is configured to create CMA contest bitmasks without using legend annotations! Setting "legend_hints_dir" or "legend_items_dir" param is recommended.'
                )

        try:
            results = pipeline.run(image_input)
        except Exception as e:
            logger.exception(e)
            continue

        # write the results out to the file system or s3 bucket
        for output_type, output_data in results.items():
            if isinstance(output_data, BaseModelOutput):
                if output_type == "map_point_label_output":
                    path = os.path.join(p.output, f"{doc_id}_point_extraction.json")
                    file_writer.process(path, output_data.data)
                elif output_type == "map_point_label_cdr_output" and p.cdr_schema:
                    path = os.path.join(p.output, f"{doc_id}_point_extraction_cdr.json")
                    file_writer.process(path, output_data.data)
            elif isinstance(output_data, ImageDictOutput) and p.bitmasks:
                # write out the binary raster images
                for pt_label, pil_im in output_data.data.items():
                    raster_path = os.path.join(
                        bitmasks_out_dir, f"{doc_id}_{pt_label}.tif"
                    )
                    image_writer.process(raster_path, pil_im)
            elif isinstance(output_data, ImageOutput):
                # write out the image
                path = os.path.join(p.output, f"{doc_id}_point_extraction.png")
                assert isinstance(output_data.data, PILImage)
                image_writer.process(path, output_data.data)
            elif isinstance(output_data, EmptyOutput):
                logger.info(f"Empty {output_type} output for {doc_id}")
            else:
                logger.warning(f"Unknown output data: {output_data}")


if __name__ == "__main__":
    main()
