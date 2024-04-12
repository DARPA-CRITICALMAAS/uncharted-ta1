
## LARA Image Segmentation Pipeline


This pipeline performs segmentation to isolate the map and legend regions on an image

Segmentation is done using a fine-tuned version of the `LayoutLMv3` model:
https://github.com/microsoft/unilm/tree/master/layoutlmv3

See more info on pipeline tasks here: [../../tasks/README.md](../../tasks/README.md)

### Segmentation categories (classes)

The model currently supports 4 segmentation classes:
* Map
* Legend (polygons)
* Legend (points and lines)
* Cross section

### Installation

* python 3.10 or higher is required
* Installation of Detectron2 requires `torch` already be present in the environment, so it must be installed manually.

To install from the current directory:
```
# manually install torch - this is necessary due to issues with detectron2 dependencies
# (see https://github.com/facebookresearch/detectron2/issues/4472)
pip install torch==2.0.1

# install the task library
cd ../../tasks
pip install -e .[segmentation]

# install the segmentation pipeline
cd ../pipelines/segmenation
pip install -e .[segmentation]
```

### Overview ###

* Pipeline is defined in `segmentation_pipeline.py` and is suitable for integration into other systems
* Model weights can be input from S3 or local drive
* Input is a image (ie binary image file buffer)
* Ouput is the set of map polygons capturing the map region, legend areas and cross sections materialized as a:
  * `SegmentationResults` JSON object
  * `FeatureResults` JSON object (the latter being part of the CMA TA1 CDR schema)

### Command Line Execution ###
`run_pipeline.py` provides a command line wrapper around the segmentation pipeline, and allows for a directory map images to be processed serially.

To run from the repository root directory:
```
export AWS_ACCESS_KEY_ID=<KEY ID>
export AWS_SECRET_ACCESS_KEY=<SECRET KEY>

python3 -m pipelines.segmentation.run_pipeline \
    --input /image/input/dir \
    --output /model/output/dir \
    --workdir /model/working/dir \
    --model https://s3/compatible/endpoint/layoutlmv3_20230
```

Note that when the `model` parameter can point to a folder in the local file system, or to a resource on an S3-compatible endpoint. The folder/resource must contain the following files:
* `config.yaml`
* `config.json`
* `model_final.pth`

In the S3 case, the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables must be set accordingly.  The model weights an configuration files will be fetched from the S3 endpoint and cached.

### REST Service ###
`run_server.py` provides the pipeline as a REST service with the following endpoints:
* ```POST:  /api/process_image``` - Sends an image (as binary file buffer) to the segmenter pipeline for analysis. Results are JSON string.
* ```GET /healthcheck``` - Healthcheck endpoint

To start the server:
```
export AWS_ACCESS_KEY_ID=<KEY ID>
export AWS_SECRET_ACCESS_KEY=<SECRET KEY>

python3 -m pipelines.segmentation.run_server \
    --workdir /model/workingdir \
    --model https://s3/compatible/endpoint/layoutlmv3_20230
```

### Dockerized deployment
The `deploy/build.sh` script can be used to build the server above into a Docker image.  Once built, the server can be deployed as a container:

```
cd deploy

export AWS_ACCESS_KEY_ID=<KEY ID>
export AWS_SECRET_ACCESS_KEY=<SECRET KEY>

./run.sh /model/workingdir https://s3/compatible/endpoint/layoutlmv3_20230
```

