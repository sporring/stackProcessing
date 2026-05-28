# Upstream Bugs To Consider Reporting

This is a running list of issues we have encountered in dependencies while developing StackProcessing. Items here should be verified with small reproductions before reporting upstream.

## Avalonia / NodeEditorAvalonia

- Restoring an object after its size has temporarily been set to zero does not restore its center position correctly.
  - Observed while debugging Studio node/pin layout.
  - The exact owner is still uncertain: it may be Avalonia layout behavior, NodeEditorAvalonia behavior, or an interaction between the two.

## SimpleITK

- `BinaryMorphologicalClosingImageFilter` appears not to expose the background value setter, unlike the other binary morphological operators.
  - This matters because erosion, dilation, and related binary morphology need explicit foreground/background semantics for 0-1 `uint8` images.
  - We should prepare a minimal comparison against erosion/dilation/opening before filing.

- Speckle noise generation appeared to hang when `stddev = 0`.
  - Needs a minimal reproduction that calls the SimpleITK speckle-noise path directly.
  - If confirmed, the expected behavior should probably be either identity output or a clear argument error.

- `DiscreteGaussianImageFilter` may have unexpected behavior in the streaming/boundary setup we need.
  - We saw suspicious smoothing results while using SimpleITK's discrete Gaussian path.
  - StackProcessing currently avoids that path and builds one explicit Gaussian kernel which is reused through the existing convolution/windowing code.
  - Before reporting, we should isolate a minimal comparison between SimpleITK discrete Gaussian filtering and explicit convolution with the same kernel and boundary assumptions.

## ITK / Homebrew

- The Homebrew ITK CMake package can make a small benchmark accidentally behave like a much larger ITK/VTK application unless the benchmark is explicit about both components and build type.
  - Initial benchmark configuration used broad `find_package(ITK REQUIRED)`, which pulled in Homebrew's full enabled ITK module set during configuration. The configure output loaded `ITKVtkGlue` and VTK package discovery, including warnings about Vulkan, Boost policy `CMP0167`, and freetype.
  - The resulting executable initially linked a broad ITK/VTK stack. Separately, `libitksys` and `libvtksys` contain a `ps -o rss= -p` path used for process/memory introspection; in the local sandbox this produced repeated `/bin/ps: Operation not permitted` messages.
  - A second issue was that `cmake --build ... --config Release` does not select Release optimization for Unix Makefiles unless `CMAKE_BUILD_TYPE=Release` was set at configure time. This likely left the benchmark in an unoptimized build despite looking like a Release build command.
  - The benchmark-side fix was to set a default `CMAKE_BUILD_TYPE=Release`, request only the ITK modules needed by the benchmark (`ITKIOTIFF`, thresholding, binary morphology, connected components, convolution, and smoothing), and explicitly attach `itk::TIFFImageIO` to each reader and writer rather than using ITK's format factory discovery for every TIFF slice.
  - After the fix, `otool -L benchmarks/cpp-itk/build/benchmark_itk` showed no VTK libraries linked, and the user observed a significant speedup on the cpp-itk benchmark path.
  - This may not be a reportable upstream bug in the strict sense: the main lesson may be documentation for our benchmark harness and for users of Homebrew ITK. A possible upstream request would be clearer guidance or less eager loading in `ITKConfig.cmake` when `COMPONENTS` are provided.

## ZarrNET

- ZarrNET appears to create debug log files as ordinary filesystem side effects.
  - We observed files named `log.txt` and a Windows-style path fragment `C:\Users\Public\biolog.txt` being created relative to the current directory or application base directory.
  - StackProcessing currently works around this in `StackIO.suppressZarrNetDebugLogging` by deleting those files and setting ZarrNET private static debug counters (`s_writeDebugCount`, `s_readDebugCount`, and local filesystem store `s_debugCount`) high enough to suppress further logging.
  - A minimal upstream report should check whether these logs are intentional, whether logging can be disabled through public API/configuration, and whether Windows absolute paths are being treated as relative filenames on non-Windows platforms.

## MATLAB

- MATLAB R2024b Update 5 on macOS 26.5 crashed with a segmentation violation when running benchmark scripts through `matlab -nodisplay -nojvm -batch`.
  - Observed command shape:
    `matlab -nodisplay -nojvm -batch "addpath('.../benchmarks/matlab'); bench_stack('operation','median','pixelType','UInt8','input','tmp/benchmarks/input/UInt8_256x256x256','output','tmp/benchmarks/output/matlab/median_UInt8_256x256x256_radius-2_r01','radius','2')"`
  - The crash report showed `MATLAB Version 24.2.0.2863752 (R2024b) Update 5`, `Operating System Mac OS Version 26.5`, architecture `maca64`, and a failing thread named `GTP_4`.
  - The stack trace pointed into MATLAB internals, specifically `mwddux_matlab.dylib` at `ddux::matlab::LicenseLogger::initialize`, called from `libmwddux.dylib`, rather than into `bench_stack.m`, `medfilt3`, or our benchmark code.
  - The crash report was written as `/Users/jrh630/matlab_crash_dump.21431-1` on the development machine.
  - Removing `-nodisplay -nojvm` and invoking MATLAB as `matlab -batch ...` made the same benchmark path work in the local setup.
  - Before reporting upstream, prepare a minimal reproduction that runs a tiny `-batch` script with and without `-nojvm`, ideally without StackProcessing benchmark inputs, to confirm that the crash is tied to MATLAB startup flags rather than memory pressure from the median benchmark.

## Possible But Not Yet Classified

- Avalonia file dialog behavior has intermittently appeared delayed or non-selectable on macOS.
  - This may be application timing, StorageProvider usage, or platform dialog behavior rather than an upstream bug.
  - Keep this here only as a reminder until we have a minimal reproduction.
