#!/bin/bash

# args: $1 - path to local directory to mount as /workdir in docker container
# args: $2 - path to local directory to mount as /imagedir in docker container

docker network ls | grep -q 'lara' || docker network create lara
docker run \
    --rm \
    --name cdr \
    -e CDR_API_TOKEN=$CDR_API_TOKEN \
    -e NGROK_AUTHTOKEN=$NGROK_AUTHTOKEN \
    --net lara \
    -p 5000:5000 \
    uncharted/lara-cdr:latest --mode host
