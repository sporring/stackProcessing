# Upstream Bugs To Consider Reporting

This is a running list of issues we have encountered in dependencies while developing StackProcessing. Items here should be verified with small reproductions before reporting upstream.

## Avalonia / NodeEditorAvalonia

- Restoring an object after its size has temporarily been set to zero does not restore its center position correctly.
  - Observed while debugging Studio node/pin layout.
  - The exact owner is still uncertain: it may be Avalonia layout behavior, NodeEditorAvalonia behavior, or an interaction between the two.

## ZarrNET

- ZarrNET appears to create debug log files as ordinary filesystem side effects.
  - We observed files named `log.txt` and a Windows-style path fragment `C:\Users\Public\biolog.txt` being created relative to the current directory or application base directory.
  - StackProcessing currently works around this in `StackIO.suppressZarrNetDebugLogging` by deleting those files and setting ZarrNET private static debug counters (`s_writeDebugCount`, `s_readDebugCount`, and local filesystem store `s_debugCount`) high enough to suppress further logging.
  - A minimal upstream report should check whether these logs are intentional, whether logging can be disabled through public API/configuration, and whether Windows absolute paths are being treated as relative filenames on non-Windows platforms.
  - A modest first pull request could simply remove these unconditional filesystem writes, or guard them behind an explicit debug option. The test case should create a small array, write it, read it back, and assert that no `log.txt` or `C:\Users\Public\biolog.txt` side files appear in the working directory or application directory on macOS/Linux.
  - This would be a good low-risk first contribution before proposing broader dtype support such as `float32` and complex arrays.

## MATLAB

- MATLAB R2024b Update 5 on macOS 26.5 crashed with a segmentation violation when running benchmark scripts through `matlab -nodisplay -nojvm -batch`.
  - Observed command shape:
    `matlab -nodisplay -nojvm -batch "addpath('.../benchmarks/matlab'); bench_stack('operation','median','pixelType','UInt8','input','tmp/benchmarks/input/UInt8_256x256x256','output','tmp/benchmarks/output/matlab/median_UInt8_256x256x256_radius-2_r01','radius','2')"`
  - The crash report showed `MATLAB Version 24.2.0.2863752 (R2024b) Update 5`, `Operating System Mac OS Version 26.5`, architecture `maca64`, and a failing thread named `GTP_4`.
  - The stack trace pointed into MATLAB internals, specifically `mwddux_matlab.dylib` at `ddux::matlab::LicenseLogger::initialize`, called from `libmwddux.dylib`, rather than into `bench_stack.m`, `medfilt3`, or our benchmark code.
  - The crash report was written as `/Users/jrh630/matlab_crash_dump.21431-1` on the development machine.
  - Removing `-nodisplay -nojvm` and invoking MATLAB as `matlab -batch ...` made the same benchmark path work in the local setup.
  - Before reporting upstream, prepare a minimal reproduction that runs a tiny `-batch` script with and without `-nojvm`, ideally without StackProcessing benchmark inputs, to confirm that the crash is tied to MATLAB startup flags rather than memory pressure from the median benchmark.

- MATLAB R2026a on the same macOS setup still fails immediately with the `-nojvm` benchmark command shape.
  - Observed on 2026-06-02 with `/Applications/MATLAB_R2026a.app/bin/matlab -nodisplay -nojvm -batch ...` during the first benchmark row, `copy UInt8 256x256x256`.
  - MATLAB printed:
    `Incompatible processor. This Qt build requires the following features: neon`
    followed by `Could not create on-disk crash report: failed opening file: Operation not permitted: unspecified iostream_category error` and `MATLAB is exiting because of fatal error`.
  - The benchmark wrapper recorded exit code `-9` before any successful R2026a measurement was produced.
  - The separate R2026a rerun CSV was discarded, leaving the existing MATLAB measurements in `benchmarks/results/raw.csv` unchanged.
  - This suggests that R2026a did not fix the local `-nojvm` brittleness. For the benchmark, keep using the existing non-`-nojvm` MATLAB measurements unless a smaller minimal reproduction shows a safe startup flag combination.

## Possible But Not Yet Classified

- Avalonia file dialog behavior has intermittently appeared delayed or non-selectable on macOS.
  - This may be application timing, StorageProvider usage, or platform dialog behavior rather than an upstream bug.
  - Keep this here only as a reminder until we have a minimal reproduction.
