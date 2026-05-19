#!/bin/bash
set -e

BUILD_DIR="build"
BUILD_TYPE="Release"
CUDA="OFF"

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda) CUDA="ON"; shift ;;
        --debug) BUILD_TYPE="Debug"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring ternative.cpp (CUDA=$CUDA, BUILD_TYPE=$BUILD_TYPE)..."
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DTERNATIVE_CUDA="$CUDA" \
    -DCMAKE_CUDA_ARCHITECTURES="86"

echo "Building..."
cmake --build . --config "$BUILD_TYPE" --parallel "$(nproc)"

echo "Build complete. Binary: $BUILD_DIR/ternative"
