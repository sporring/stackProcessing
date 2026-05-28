# Comparative Benchmarks

This directory contains a report-facing benchmark harness for comparing StackProcessing with common image-processing environments on 3D TIFF-stack workflows.

The benchmark contract is deliberately simple:

```text
read TIFF slice stack -> assemble/process as a 3D volume -> write TIFF slice stack
```

Each backend should measure the same user-visible task, with the same input stack, and report wall-clock time plus peak resident memory through `tools/measure.py`. The raw results include both outer wall time, measured from launching `dotnet`, `python3`, `matlab`, or the C++ executable, and backend-reported internal time for the read-process-write work after process startup.

## Backends

- `StackProcessing.Benchmarks`: F# StackProcessing executable.
- `python-skimage-scipy`: Python/scikit-image/SciPy implementation.
- `cpp-itk`: C++/ITK implementation skeleton and CMake project.
- `matlab`: MATLAB script using Image Processing Toolbox, invoked as `matlab -nodisplay -nojvm -batch`.
- `python-dask-omezarr`: special-case chunk-native Dask/OME-Zarr implementation.

## Initial Operation Set

The first benchmark set avoids operations that demonstrate StackProcessing-specific mechanics directly. It focuses on common image-analysis tasks:

- `copy`: IO baseline.
- `threshold`: scalar map producing a binary `UInt8` mask with values `0` and `1`.
- `uniformConvolve`: 3D convolution with a uniform kernel.
- `median`: 3D median filter.
- `dilate`: binary 3D dilation producing a binary `UInt8` mask with values `0` and `1`.
- `connectedComponents`: 3D connected-component labelling.

The initial type set is:

- `UInt8`: values in the range `0..255`.
- `UInt16`: the same `0..255` values stored as `UInt16`.
- `Float32`: the same `0..255` values stored as `Float32`.

The baseline size sweep is:

- `256x256x256`
- `512x512x512`
- `1024x1024x1024`

For `median`, the benchmark uses radii `1`, `2`, and `3`, corresponding to `3x3x3`, `5x5x5`, and `7x7x7` neighbourhoods. For `dilate`, every backend first interprets the input as a binary mask using `input >= 128`, then applies spherical/ball structuring elements with radii `1`, `2`, and `3`, and writes a `UInt8` mask. For `uniformConvolve`, it uses uniform kernels of size `3`, `5`, and `7`.

The initial benchmark intentionally focuses on TIFF only. Format-specific behavior is large enough that TIFF should be understood before expanding to MHA, OME-Zarr, or HDF5/NeXus.

The fairness rule is that a case should read the same stack type, use the simplest native 3D operator available in that environment, and write the same output type across all backends. All generated inputs are deterministic ramps over the same value range `0..255`, regardless of storage type. `threshold` uses an inclusive lower threshold (`input >= 128`). `dilate` and `connectedComponents` use `input >= 128` to form their binary masks.

`uniformConvolve` is represented in StackProcessing by constructing a Float64 uniform kernel, casting typed benchmark inputs into the operation, and casting back to the requested output type. `copy`, `uniformConvolve`, and `median` write the same pixel type they read (`UInt8`, `UInt16`, or `Float32`). `threshold`, `dilate`, and `connectedComponents` write `UInt8` mask/label-style outputs for all backends. `connectedComponents` is included because it is an important dependency-sensitive operation. The semantic target for all TIFF-stack backends is 3D component labelling: StackProcessing may stream internally, while MATLAB, Python/scikit-image/SciPy, and C++/ITK read the full stack into a volume before processing.

The StackProcessing backend deliberately runs with the StackProcessing optimiser disabled. This keeps the benchmark focused on the current streaming implementation and avoids mixing benchmark results with experimental optimiser choices.

## Special-Case Chunk-Native Benchmark

OME-Zarr is an essential user-facing technology for microscopy and adjacent large-volume imaging communities, but it is not universally supported by the TIFF-stack tools in the baseline comparison. It therefore lives in a separate special-case matrix:

```text
read chunked OME-Zarr array -> process with Dask -> write chunked OME-Zarr array
```

This is intentionally not mixed into the TIFF-stack baseline. The special-case benchmarks answer a different user question: what happens if the workflow starts in a chunk-native ecosystem instead of a directory of TIFF slices?

The special cases are listed in `config/special-cases.csv`. The initial subset includes `copy`, `threshold`, uniform convolution, and small-radius 3D neighbourhood operations implemented with Dask overlap. These require `dask`, `zarr`, `numpy`, and `scipy`; spherical dilation uses `scikit-image` when available and otherwise falls back to an equivalent NumPy footprint.

## Suggested Workflow

For repeatable full runs, use the top-level benchmark driver. Install python packages once:

```bash
python3 -m venv .venv-benchmarks
source .venv-benchmarks/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-image tifffile
python -m pip install dask zarr
source .venv-benchmarks/bin/activate
```
Then run:
```bash
source .venv-benchmarks/bin/activate
bash benchmarks/run_all.sh --repeat 3
source .venv-benchmarks/bin/activate
```

This generates deterministic TIFF inputs, runs the default baseline backends, and writes:

```text
benchmarks/results/raw.csv
benchmarks/results/summary.csv
```

`raw.csv` contains `wallSeconds`, `internalSeconds`, and `peakRssKiB`. `summary.csv` reports median and mean wall time, internal time, startup overhead (`wallSeconds - internalSeconds`), and peak resident memory.

Use `--dry-run` to print the exact commands without executing them:

```bash
bash benchmarks/run_all.sh --repeat 3 --dry-run
```

Use `--backends` to run only selected implementations, and `--pixel-types` to restrict the case matrix while exploring:

```bash
bash benchmarks/run_all.sh \
  --repeat 3 \
  --backends python-skimage-scipy \
  --pixel-types UInt8
```

The pixel-type filter is comma-separated, for example `--pixel-types UInt8,Float32`.

Use `--shapes` to restrict the size matrix:

```bash
bash benchmarks/run_all.sh \
  --repeat 3 \
  --backends stackprocessing,python-skimage-scipy,cpp-itk,matlab \
  --pixel-types UInt8 \
  --shapes 256x256x256
```

The shape filter is also comma-separated, for example `--shapes 256x256x256,512x512x512`.

A broader report-facing run can include all baseline backends and the OME-Zarr special case:

```bash
bash benchmarks/run_all.sh \
  --repeat 3 \
  --backends stackprocessing,python-skimage-scipy,cpp-itk,matlab \
  --build-itk \
  --include-special
```

## C++/ITK Setup

The C++ backend uses CMake and a local ITK installation. On macOS with Homebrew, the usual setup is:

```bash
brew install cmake itk
cmake -S benchmarks/cpp-itk -B benchmarks/cpp-itk/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(brew --prefix itk)"
cmake --build benchmarks/cpp-itk/build --config Release -j
```

Then include the backend:

```bash
bash benchmarks/run_all.sh \
  --repeat 3 \
  --backends cpp-itk \
  --itk-exe benchmarks/cpp-itk/build/benchmark_itk
```

If ITK was installed from source or another package manager, pass the installation prefix through `CMAKE_PREFIX_PATH`, or set `ITK_DIR` to the directory containing `ITKConfig.cmake`.

The default backend set is intentionally modest:

```text
stackprocessing,python-skimage-scipy
```

This keeps a first run possible on ordinary developer machines. Add `cpp-itk`, `matlab`, and `--include-special` when those environments are installed.

If MATLAB is not named `matlab` on the current shell `PATH`, pass it explicitly:

```bash
bash benchmarks/run_all.sh --backends matlab --matlab-exe /path/to/matlab
```

Generate a deterministic input stack with StackProcessing:

```bash
dotnet run --project benchmarks/StackProcessing.Benchmarks/StackProcessing.Benchmarks.fsproj -- \
  generate --output tmp/benchmarks/input/uint8_512x512x64 \
  --shape 512x512x64 --pixel-type UInt8 --pattern ramp
```

Run one measured StackProcessing case:

```bash
python3 benchmarks/tools/measure.py \
  --output benchmarks/results/raw.csv \
  --backend stackprocessing \
  --operation median \
  --pixel-type UInt8 \
  --shape 512x512x64 \
  --parameter radius=3 \
  -- \
  dotnet run --project benchmarks/StackProcessing.Benchmarks/StackProcessing.Benchmarks.fsproj -- \
    run --operation median --pixel-type UInt8 \
    --input tmp/benchmarks/input/uint8_512x512x64 \
    --output tmp/benchmarks/output/stackprocessing/median_uint8_512x512x64_r3 \
    --radius 3
```

Aggregate results:

```bash
python3 benchmarks/tools/summarize_results.py \
  --input benchmarks/results/raw.csv \
  --output benchmarks/results/summary.csv
```

For the full case matrix, generate the needed inputs once per shape/type, then run a backend through `run_manifest.py`:

```bash
python3 benchmarks/tools/prepare_inputs.py

python3 benchmarks/tools/run_manifest.py \
  --backend stackprocessing \
  --repeat 3
```

Use `--dry-run` to print the exact commands without executing them.

Run the Dask/OME-Zarr special cases by converting the TIFF inputs to OME-Zarr stores, then using `config/special-cases.csv`:

```bash
python3 benchmarks/tools/tiff_stack_to_omezarr.py \
  --input tmp/benchmarks/input/UInt8_512x512x64 \
  --output tmp/benchmarks/input-omezarr/UInt8_512x512x64 \
  --shape 512x512x64 --pixel-type UInt8

python3 benchmarks/tools/run_manifest.py \
  --cases benchmarks/config/special-cases.csv \
  --backend python-dask-omezarr \
  --input-root tmp/benchmarks/input-omezarr \
  --repeat 3
```

The C++/ITK backend is built separately:

```bash
cmake -S benchmarks/cpp-itk -B benchmarks/cpp-itk/build
cmake --build benchmarks/cpp-itk/build --config Release
```

## Notes on Fairness

- All tools read the same TIFF stack and write a TIFF stack.
- The default operation semantics are 3D volume operations. Non-StackProcessing TIFF baselines read the full stack, process a 3D volume, and write the result as slices; StackProcessing is allowed to stream internally as long as the user-visible operation is the same 3D operation.
- Peak memory is measured at the process level by `tools/measure.py`.
- Startup costs are included by default because they matter for user-facing command-line workflows. For long-running batch studies, add separate warm-process measurements.
- MATLAB benchmarks assume Image Processing Toolbox.
- Python/scikit-image benchmarks assume `scikit-image`, `scipy`, and `numpy`.
- Python helper scripts use `tifffile` for TIFF stack generation and conversion.
- C++/ITK benchmarks assume a local ITK installation discoverable by CMake.
- Dask/OME-Zarr is reported as a special case because it uses a chunked array store instead of a TIFF stack.
