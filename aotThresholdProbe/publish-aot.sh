#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"
rid="${1:-}"

if [[ -z "${rid}" ]]; then
    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64) rid="osx-arm64" ;;
        Darwin-x86_64) rid="osx-x64" ;;
        Linux-x86_64) rid="linux-x64" ;;
        Linux-aarch64) rid="linux-arm64" ;;
        *)
            echo "Could not infer .NET runtime identifier. Pass one explicitly, e.g. ./publish-aot.sh osx-arm64" >&2
            exit 2
            ;;
    esac
fi

dotnet publish "${script_dir}/aotThresholdProbe.fsproj" \
    -c Release \
    -r "${rid}" \
    -p:PublishTrimmed=true \
    -p:PublishAot=true \
    --self-contained true \
    -o "${root_dir}/tmp/publish-aot-hello-probe"
