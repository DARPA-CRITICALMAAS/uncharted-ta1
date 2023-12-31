from flask import Flask, request, Response
import logging, json
from PIL import Image
import argparse
from hashlib import sha1
from io import BytesIO

from pipelines.segmentation.segmentation_pipeline import SegmentationPipeline
from tasks.common.pipeline import PipelineInput, BaseModelOutput


#
# Flask web app for Legend and Map Segmenter module
#

app = Flask(__name__)


@app.route("/api/process_image", methods=["POST"])
def process_image():
    """
    Perform legend and map segmentation on an image
    request.data is expected to contain binary image file buffer
    """

    # Adapted from code samples here: https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594
    try:
        # open the image from the supplied byte stream
        bytes_io = BytesIO(request.data)
        image = Image.open(bytes_io)

        # use the hash as the doc id since we don't have a filename
        doc_id = sha1(request.data).hexdigest()

        # run the image through the metadata extraction pipeline
        pipeline_input = PipelineInput(image=image, raster_id=doc_id)
        result = segmentation_pipeline.run(pipeline_input)
        if len(result) == 0:
            msg = "No segmentation results"
            logging.warning(msg)
            return (msg, 500)

        segmentation_result = result["map_segmentation_output"]
        if isinstance(segmentation_result, BaseModelOutput):
            # convert result to a JSON array
            result_json = json.dumps(segmentation_result.data.model_dump())
            return Response(result_json, status=200, mimetype="application/json")
        else:
            msg = "No map segmentation results"
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
    logger = logging.getLogger("segmenter app")
    logger.info("*** Starting Legend and Map Segmenter App ***")

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--min_confidence", type=float, default=0.25)
    parser.add_argument("--debug", type=float, default=False)
    p = parser.parse_args()

    # init segmenter
    segmentation_pipeline = SegmentationPipeline(p.model, p.workdir, p.min_confidence)

    #### start flask server
    if p.debug:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    else:
        app.run(host="0.0.0.0", port=5000)

    # TEMP Use this for debug mode
