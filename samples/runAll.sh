#!/bin/zsh

set -e

script_dir=${0:A:h}
cd "$script_dir"

parallel_jobs=1

usage() {
  cat <<EOF
Usage: ./runAll.sh [-j jobs]

Runs all sample projects. The default is sequential, which gives cleaner
per-sample timing measurements. Use -j with a value greater than 1 to run
multiple samples in parallel.

Options:
  -j, --jobs N       Run up to N samples at once.
  -p, --parallel    Run with one job per logical CPU.
  -h, --help        Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs)
      if [[ $# -lt 2 || ! "$2" == <-> || "$2" -lt 1 ]]; then
        echo "runAll.sh: -j/--jobs expects a positive integer" >&2
        exit 2
      fi
      parallel_jobs="$2"
      shift 2
      ;;
    -p|--parallel)
      parallel_jobs="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "runAll.sh: unknown option $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p tmp

dirs=(**/*.fsproj(:h))

run_sample() {
  local i="$1"
  local log="tmp/$i.out"
  mkdir -p "${log:h}"

  {
    echo "== $i =="
    pushd "$i" >/dev/null
    dotnet build
    /usr/bin/time env DYLD_LIBRARY_PATH="$(pwd)/lib" dotnet run --verbosity q -- -d 1
    popd >/dev/null
  } > "$log" 2>&1
}

for i in $dirs; do
  echo "$i"
  run_sample "$i" &

  while [[ "$(jobs -pr | wc -l | tr -d ' ')" -ge "$parallel_jobs" ]]; do
    sleep 0.2
  done
done

wait
