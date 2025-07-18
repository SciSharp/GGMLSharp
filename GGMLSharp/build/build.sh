#!/bin/bash

set -e

IMAGE_NAME="ggml-multi-platform-builder"
OUTPUT_DIR="GGMLSharp/runtimes"

if [ ! -d "ggml" ]; then
    echo "Error: 'ggml' submodule directory not found."
    echo "Please run this script from the root of your project."
    exit 1
fi

GGML_COMMIT=$(git rev-parse HEAD:ggml)
echo "Using ggml commit: ${GGML_COMMIT}"

echo "Building the Docker image"
docker build -t ${IMAGE_NAME} --network host -f GGMLSharp/build/Dockerfile --build-arg GGML_COMMIT=${GGML_COMMIT} .

echo "Extracting the compiled runtimes"

CONTAINER_ID=$(docker create ${IMAGE_NAME})

if [ -d "${OUTPUT_DIR}" ]; then
    echo "Removing existing output directory: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

docker cp "${CONTAINER_ID}:/out/runtimes/." "${OUTPUT_DIR}/"
docker rm -v "${CONTAINER_ID}"

echo "Done"
