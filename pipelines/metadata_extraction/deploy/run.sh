docker network create lara
docker run \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -v $GOOGLE_APPLICATION_CREDENTIALS:/credentials.json \
    -v $1:/workdir \
    -p 5000:5000 \
    docker.uncharted.software/metadata-extraction:latest
