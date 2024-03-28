#!/bin/bash

# args: $1 - path to local directory to mount as /workdir in docker container
docker network create lara
docker run \
    --rm \
    --name point_extraction \
    -v $1:/workdir \
    --net lara \
    -p 5000:5000 \
    docker.uncharted.software/point_extraction:latest \
        --workdir /workdir \
        --model_point_extractor pipelines/point_extraction_weights/lara_yolo_20240320_best.pt \
        --model_segmenter pipelines/segmentation_weights/layoutlmv3_xsection_20231201

