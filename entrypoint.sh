#!/bin/bash
set -e

USER_NAME=${LOCAL_USER_NAME:-developer}

# Export environment variables for CMake
export torch_DIR=/opt/libtorch/share/cmake/Torch
export CMAKE_PREFIX_PATH=$torch_DIR
export LD_LIBRARY_PATH=/usr/local/lib:/opt/libtorch/lib:$LD_LIBRARY_PATH

# If $PWD is a mounted folder, cd there; otherwise fallback
TARGET_DIR=${PWD:-$HOME/build}
cd "$TARGET_DIR"

exec "$@"
