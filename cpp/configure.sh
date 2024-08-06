#!/bin/bash

# Load intel libraries
source /opt/intel/oneapi/setvars.sh intel64

# Set compiler
# export CC=/usr/bin/clang
# export CXX=/usr/bin/clang++
export CXX=icpx

# Set variables
export SOURCE_DIR="/home/kratos/kratos-structured-solver-sandbox/cpp"
export BUILD_DIR="${SOURCE_DIR}/build"
#export FFTW

# Set basic configuration
export BUILD_TYPE="Debug"

# Set basic sycl flags to allow GCGPU
# export SYCL_FLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-unnamed-lambda"
export SYCL_FLAGS="-fsycl -fsycl-unnamed-lambda"

# Clean
clear
rm -rf "${BUILD_DIR}/${BUILD_TYPE}/cmake_install.cmake"
rm -rf "${BUILD_DIR}/${BUILD_TYPE}/CMakeCache.txt"
rm -rf "${BUILD_DIR}/${BUILD_TYPE}/CMakeFiles"

# Configure
# -G: Generator tool
# -H: directory where the code is
# -B: build directory (temporary files during compilation)
# -DCMAKE_INSTALL_PREFIX: installation directory (where the result of the compilation will appear)
# -DCMAKE_CXX_FLAGS: flags for the C++ compiler
cmake -H"${SOURCE_DIR}" -B"${BUILD_DIR}/${BUILD_TYPE}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX="${SOURCE_DIR}/bin" -DCMAKE_CXX_FLAGS="${SYCL_FLAGS}"

# Buid and install
cmake --build "${BUILD_DIR}/${BUILD_TYPE}" --target install -- -j$(nproc)
