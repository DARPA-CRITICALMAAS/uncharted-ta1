from flask import Flask, request, Response
import logging, json
import argparse
from hashlib import sha1
from io import BytesIO

from pipelines.point_extraction.point_extraction_pipeline import PointExtractionPipeline
from tasks.common.pipeline import PipelineInput, BaseModelOutput, BaseModelListOutput
from tasks.common import image_io


#
# Flask web app for Legend and Map Segmenter module
#

app = Flask(__name__)


@app.route("/api/process_image", methods=["POST"])
def process_image():
    """
    Perform point extraction on a map image.
    request.data is expected to contain binary image file buffer
    """

    # Adapted from code samples here: https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594
    try:
        # open the image from the supplied byte stream
        bytes_io = BytesIO(request.data)
        image = image_io.load_pil_image_stream(bytes_io)

        # use the hash as the doc id since we don't have a filename
        doc_id = sha1(request.data).hexdigest()

        # run the image through the point extraction pipeline
        pipeline_input = PipelineInput(image=image, raster_id=doc_id)
        result = point_extraction_pipeline.run(pipeline_input)
        if len(result) == 0:
            msg = "No point extraction results"
            logging.warning(msg)
            return (msg, 500)

        ta1_schema = app.config.get("ta1_schema", False)
        # get ta1 schema output or internal output format
        point_extraction_result = (
            result["integration_output"]
            if ta1_schema
            else result["map_point_label_output"]
        )

        # convert result to a JSON string and return
        if isinstance(point_extraction_result, BaseModelOutput):
            result_json = json.dumps(point_extraction_result.data.model_dump())
            return Response(result_json, status=200, mimetype="application/json")
        elif isinstance(point_extraction_result, BaseModelListOutput):
            result_json = json.dumps(
                [d.model_dump() for d in point_extraction_result.data]
            )
            return Response(result_json, status=200, mimetype="application/json")
        else:
            msg = "No point extraction results"
            logging.warning(msg)
            return (msg, 500)

    except Exception as e:
        msg = f"Error with process_image: {repr(e)}"
        logging.error(msg)
        print(repr(e))
        return Response(msg, status=500)


@app.route("/healthcheck")
def health():
    """
    healthcheck
    """
    return ("healthy", 200)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s %(name)s\t: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("point extraction app")
    logger.info("*** Starting Point Extraction App ***")

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default="tmp/lara/workdir")
    parser.add_argument("--model_point_extractor", type=str, required=True)
    parser.add_argument("--model_segmenter", type=str, default=None)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument(
        "--ta1_schema",
        type=bool,
        default=False,
        help="Output results as TA1 json schema format",
    )
    p = parser.parse_args()

    # init point extraction pipeline
    point_extraction_pipeline = PointExtractionPipeline(
        p.model_point_extractor, p.model_segmenter, p.workdir
    )

    #### start flask server
    app.config["ta1_schema"] = p.ta1_schema
    if p.debug:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    else:
        app.run(host="0.0.0.0", port=5000)
