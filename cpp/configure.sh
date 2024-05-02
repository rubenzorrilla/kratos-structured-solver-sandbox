#!/bin/bash

# Set compiler
export CC=/usr/bin/clang-17
export CXX=/usr/bin/clang++-17

# Set variables
export SOURCE_DIR="/home/rzorrilla/kratos-structured-solver-sandbox/cpp"
export BUILD_DIR="${SOURCE_DIR}/build"
#export FFTW

# Set basic configuration
export BUILD_TYPE="Release"

# Clean
clear
rm -rf "${BUILD_DIR}/${BUILD_TYPE}/cmake_install.cmake"
rm -rf "${BUILD_DIR}/${BUILD_TYPE}/CMakeCache.txt"
rm -rf "${BUILD_DIR}/${BUILD_TYPE}/CMakeFiles"

# Configure
# -H: directory where the code is
# -B: build directory (temporary files during compilation)
# -DCMAKE_INSTALL_PREFIX: installation directory (where the result of the compilation will appear)
cmake -H"${SOURCE_DIR}" -B"${BUILD_DIR}/${BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX="${SOURCE_DIR}/bin"

# Buid and install
cmake --build "${BUILD_DIR}/${BUILD_TYPE}" --target install -- -j$(nproc)
