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

The solution is built once before sample execution. Sample runs use
dotnet run --no-build so parallel runs do not fight over shared build outputs.

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
pids=()
names=()
failures=0

cleanup() {
  trap - INT TERM HUP

  if (( ${#pids} > 0 )); then
    echo "Stopping ${#pids} running sample job(s)..." >&2

    for pid in $pids; do
      kill -TERM "$pid" 2>/dev/null || true
    done

    sleep 1

    for pid in $pids; do
      kill -KILL "$pid" 2>/dev/null || true
    done

    wait $pids 2>/dev/null || true
  fi

  exit 130
}

trap cleanup INT TERM HUP

run_sample() {
  local i="$1"
  local log="tmp/$i.out"

  (
    local child=""
    trap 'if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi; exit 130' INT TERM HUP

    echo "\n== Run $i =="
    cd "$i"

    /usr/bin/time env DYLD_LIBRARY_PATH="$(pwd)/lib" dotnet run --no-build --verbosity q -- -d 1 &
    child="$!"
    wait "$child"
  ) >> "$log" 2>&1
}

wait_oldest() {
  local pid="$pids[1]"
  local name="$names[1]"

  pids=(${pids[2,-1]})
  names=(${names[2,-1]})

  if wait "$pid"; then
    :
  else
    local exit_code="$?"
    failures=$((failures + 1))
    echo "$name failed with exit code $exit_code; see samples/tmp/$name.out" >&2
  fi
}

wait_for_slot() {
  while (( ${#pids} >= parallel_jobs )); do
    wait_oldest
  done
}

echo "build StackProcessing.sln"
if ! dotnet build ../StackProcessing.sln > tmp/build.out 2>&1; then
  echo "Build failed; see samples/tmp/build.out" >&2
  exit 1
fi

for i in $dirs; do
  wait_for_slot
  run_sample "$i" &
  pids+=("$!")
  names+=("$i")
  echo "run $i"
done

while (( ${#pids} > 0 )); do
  wait_oldest
done

if (( failures > 0 )); then
  exit 1
fi
