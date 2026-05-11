#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec dotnet run --project "$script_dir/RunAll/RunAll.fsproj" -- --samples-root "$script_dir" --gather-only "$@"
