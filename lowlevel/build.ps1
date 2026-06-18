param(
    [string]$VcpkgRoot = $env:VCPKG_ROOT,
    [string]$Triplet = "x64-windows"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$BuildDir = Join-Path $RootDir "tmp\lowlevel-build"
$OutputDir = Join-Path $RootDir "lib"

$ConfigureArgs = @(
    "-S", $ScriptDir,
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DSTACKPROCESSING_NATIVE_OUTPUT_DIR=$OutputDir"
)

if ($VcpkgRoot -and (Test-Path $VcpkgRoot)) {
    $Toolchain = Join-Path $VcpkgRoot "scripts\buildsystems\vcpkg.cmake"
    if (-not (Test-Path $Toolchain)) {
        throw "Could not find vcpkg toolchain file at '$Toolchain'."
    }

    $ConfigureArgs += @(
        "-DCMAKE_TOOLCHAIN_FILE=$Toolchain",
        "-DVCPKG_TARGET_TRIPLET=$Triplet"
    )
}

cmake @ConfigureArgs
cmake --build $BuildDir --config Release
