#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${root_dir}/tmp/lowlevel-build"
output_dir="${root_dir}/lib"

cmake \
    -S "${script_dir}" \
    -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTACKPROCESSING_NATIVE_OUTPUT_DIR="${output_dir}"

cmake --build "${build_dir}" --config Release
