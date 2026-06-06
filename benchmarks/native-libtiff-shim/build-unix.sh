#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
output_dir="${1:-$repo_root/benchmarks/StackProcessing.Benchmarks/bin/Debug/net10.0}"
mkdir -p "$output_dir"

case "$(uname -s)" in
  Darwin)
    output="$output_dir/libsp_libtiff_shim.dylib"
    shared_flags=(-dynamiclib -install_name @rpath/libsp_libtiff_shim.dylib)
    ;;
  Linux)
    output="$output_dir/libsp_libtiff_shim.so"
    shared_flags=(-shared)
    ;;
  *)
    echo "Unsupported Unix platform: $(uname -s)" >&2
    exit 2
    ;;
esac

cc ${CFLAGS:-} -O3 -fPIC "${shared_flags[@]}" \
  "$script_dir/sp_libtiff_shim.c" \
  -o "$output" \
  $(pkg-config --cflags --libs libtiff-4)

echo "$output"
