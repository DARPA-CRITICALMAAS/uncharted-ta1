#!/bin/bash

# copy the files to the build directory
mkdir -p pipelines/point_extraction
cp ../*.py pipelines/point_extraction
cp ../pyproject.toml pipelines/point_extraction

cp -r ../../../schema .
cp -r ../../../tasks .
cp -r ../../../util .

# get the point model and segment model weights passed in as an argument
# we can leave it empty if not supplied
mkdir -p pipelines/point_extraction_weights
if [ -z "$1" ]
then
    echo "ERROR - No point model weights file supplied"
    point_model=""
    exit 1
else
    point_model=$1
    echo "Point model weights file: $point_model"
    cp $point_model pipelines/point_extraction_weights/points.pt
fi

mkdir -p pipelines/segmentation_weights
if [ -z "$2" ]
then
    echo "ERROR - No segment model weights dir supplied"
    segment_model=""
    exit 1
else
    segment_model=$2
    echo "Segment model weights dir: $segment_model"
    cp -r $segment_model/* pipelines/segmentation_weights
fi


# run the build with the platform argument if provided, otherwise build for the host architecture
platform=${3:-}
if [[ -n "$platform" ]]; then
    echo "Platform: $platform"
    docker buildx build --platform "$platform" -t uncharted/lara-point-extract:latest . --load
else
    docker build -t uncharted/lara-point-extract:latest .
fi

# cleanup the temp files
rm -rf pipelines
rm -rf tasks
rm -rf schema
rm -rf util

