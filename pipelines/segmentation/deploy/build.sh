#!/bin/bash

# copy the files to the build directory
mkdir -p pipelines/segmentation
cp ../*.py pipelines/segmentation
cp ../pyproject.toml pipelines/segmentation

cp -r ../../../schema .
cp -r ../../../util .
cp -r ../../../tasks .

mkdir -p pipelines/segmentation_weights
if [ -z "$1" ]
then
    echo "ERROR - No segment model weights dir supplied"
    segment_model=""
    exit 1
else
    segment_model=$1
    echo "Segment model weights dir: $segment_model"
    cp -r $segment_model pipelines/segmentation_weights
fi

# run the build
docker buildx build --platform linux/amd64,linux/arm64 -t uncharted/lara-segmentation:latest . --push

# cleanup the temp files
rm -rf pipelines
rm -rf tasks
rm -rf schema
rm -rf util
