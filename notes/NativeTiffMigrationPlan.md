# Native TIFF Migration Plan

## Goal

Move the TIFF stack path in StackProcessing from `BitMiracle.LibTiff.NET` to the
native libtiff fast path demonstrated in the benchmarks, then remove
`BitMiracle.LibTiff.NET` from the production projects completely.

The benchmark result to promote is the raw uncompressed native-endian TIFF slice
path in `benchmarks/native-libtiff-shim` and the corresponding commands in
`benchmarks/StackProcessing.Benchmarks`.

## Current State

- `StackIO.fs` still opens TIFFs through `BitMiracle.LibTiff.Classic.Tiff`.
- `StackIO.fs` already contains a first native raw UInt8 read hook via
  `sp_libtiff_shim`, but it is not the complete TIFF implementation.
- `StackProcessing.Core.fsproj` still references `BitMiracle.LibTiff.NET`.
- The benchmark shim exposes the operations we need:
  - `sp_tiff_read_info`
  - `sp_tiff_read_raw_page`
  - `sp_tiff_read_raw_page_into`
  - `sp_tiff_write_raw_page`
  - `sp_tiff_write_raw_page_from`
  - scanline fallback helpers used by benchmarks
- The benchmark shim currently lives under `benchmarks/`, so it is not packaged
  as a normal StackProcessing runtime dependency.

## Intended Production Shape

Create one production native TIFF helper beside `spnth`, for example
`native/StackProcessing.NativeTiff`, and expose it to `StackIO.fs` through a
small F# module.

The normal DSL/Studio names should remain simple:

- `read`
- `readRandom`
- `readRange`
- `readThick`
- `write`
- `writeThick`
- `getImageInfo`

The implementation detail should be native libtiff. Users should not choose
between BitMiracle/native paths.

## Migration Steps

1. Move the benchmark shim into production native sources.

   - Copy `benchmarks/native-libtiff-shim/sp_libtiff_shim.c` into
     `native/StackProcessing.NativeTiff/`.
   - Add a CMake project that builds:
     - `libsp_tiff.so` on Linux
     - `libsp_tiff.dylib` on macOS
     - `sp_tiff.dll` on Windows
   - Use `pkg-config libtiff-4` on Linux/macOS.
   - Add a Windows path through vcpkg or explicit `TIFF_INCLUDE_DIR` and
     `TIFF_LIBRARY`.

2. Add production project copy rules.

   Add conditional copy items for the native TIFF library to the same projects
   that copy `spnth`:

   - `src/Chunk/Chunk.Core.fsproj` if the low-level declarations live there
   - `src/StackProcessing.Core/StackProcessing.Core.fsproj`
   - `src/StackProcessing/StackProcessing.fsproj`
   - benchmark project only if benchmarks call the production helper directly

3. Replace BitMiracle metadata reads.

   Replace all `Tiff.Open(..., "r")` metadata reads in `StackIO.fs` with
   native `sp_tiff_read_info`.

   `getImageInfo` and TIFF read planning need:

   - width
   - height
   - number of pages
   - bits per sample
   - sample format
   - samples per pixel
   - planar configuration
   - compression
   - tiled/stripped status
   - row bytes
   - raw page bytes
   - byte order/native-endian flag

4. Promote the raw-strip read fast path.

   For uncompressed native-endian scalar TIFF pages:

   - read directly into `Chunk.Bytes`
   - support an offset into a thick chunk
   - avoid temporary row buffers
   - avoid managed per-row calls
   - support `uint8`, `uint16`, and `float32` first

   The important `StackIO.fs` targets are:

   - `readChunkTiffSliceIntoOffset`
   - `readChunkTiffSliceByPlanIntoOffset`
   - `readChunkTiffSlice`
   - thick TIFF readers that currently populate one large `Chunk`.

5. Promote the raw-strip write fast path.

   For uncompressed native-endian scalar TIFF pages:

   - write one page from `Chunk.Bytes`
   - support an offset into a thick chunk
   - keep the filename/page-index bookkeeping from the current writer
   - write thin stack output from either thin chunks or thick chunks without
     copying into temporary thin chunks

   The important `StackIO.fs` targets are:

   - `writeChunkTiffSliceToOpenTiff`
   - `writeChunkTiffSliceFromOffset`
   - `writeChunkTiffSlice`
   - `writeChunkTiffFile`
   - thick TIFF writers.

6. Decide the fallback policy.

   To remove BitMiracle completely, unsupported TIFF layouts should fail with a
   clear error rather than silently falling back.

   Accept first:

   - uncompressed stripped TIFF
   - scalar contiguous pixels
   - native-endian data
   - `uint8`, `uint16`, and `float32`
   - single-page files and multi-page TIFF volumes

   Fail clearly for:

   - compressed TIFF
   - tiled TIFF
   - palette/color TIFF unless explicitly implemented
   - non-native endian data until byte swapping is implemented
   - unusual sample formats.

7. Remove BitMiracle.

   After all production TIFF paths are native:

   - remove `open BitMiracle.LibTiff.Classic` from `StackIO.fs`
   - remove `BitMiracle.LibTiff.NET` from `StackProcessing.Core.fsproj`
   - remove copied/generated BitMiracle references from run-project generation
     if any remain
   - update README and developer notes so TIFF support is described as native
     libtiff-backed.

8. Re-run the relevant checks.

   - `dotnet build src/StackProcessing.Core/StackProcessing.Core.fsproj -c Release --nologo`
   - `dotnet build StackProcessing.sln -c Release --nologo`
   - TIFF JSON samples through Studio/RunSamples
   - copy/threshold/convolve benchmark smoke runs for `uint8`, `uint16`, and
     `float32`.

## Benchmark Promotion Criteria

Before deleting BitMiracle, compare production `read -> write` against the
benchmark shim for:

- `uint8` 256^3, 512^3, 1024^3
- `uint16` 256^3, 512^3, 1024^3
- `float32` 256^3, 512^3, 1024^3

The production path should be within a small fixed overhead of the benchmark
shim. If it is not, inspect:

- extra Chunk allocation
- slice look-ahead
- avoidable copying between thick and thin chunks
- per-file directory cleanup accidentally included in timing
- antivirus/indexer interference during large file writes.

## FFTW Wrapper Status

StackProcessing no longer depends on the old managed FFTW wrapper. Current
production FFT calls go through the native `spnth` helper and FFTW C API
wrappers in `Chunk.NativeSp`.

The FFT benchmark comparison set should remain native `spnth` and C++/ITK only.
