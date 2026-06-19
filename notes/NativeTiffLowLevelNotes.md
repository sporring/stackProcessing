# Native TIFF Low-Level Notes

StackProcessing's user-facing TIFF surface is the Chunk DSL:

- `read`
- `readRandom`
- `readRange`
- `readThick`
- `readVolume`
- `readComplex64`
- `readComplex128`
- `readColor`
- `write`
- `writeThick`
- `getImageInfo`

The implementation goal is that these stages move TIFF pages directly into and
out of `Chunk<'T>` buffers with as little copying as possible. Users should not
select an implementation backend in graphs or DSL code.

## Reader Strategy

TIFF reads are planned once from the first slice/page in the stack or volume.
The resulting strategy is reused for all emitted chunks so StackIO does not
make a layout decision inside the stream loop. The current reader split is:

- `RawFastPath`
- `GeneralLibtiffPath`

`RawFastPath` is the benchmark-sensitive path:

- uncompressed stripped TIFF
- scalar contiguous pixels
- native-endian data
- accepted scalar storage types
- single-page slice stacks and multi-page TIFF volumes
- direct reads into `Chunk.Bytes`
- offset reads for thick chunks

`GeneralLibtiffPath` is the correctness/interoperability path:

- libtiff decoded reads
- compressed stripped TIFFs
- non-native-endian TIFFs
- scalar, complex, and RGB layouts that StackProcessing explicitly accepts
- the same `Chunk.Bytes` destination contract as the raw path

This two-path split is used by the TIFF read surface: scalar `read`,
`readRandom`, `readRange`, `readThick`, `readThickFiles`, `readVolume`,
complex `readComplex64`/`readComplex128` and thick variants, and RGB
`readColor`, `readColorRandom`, and `readColorRange`.

Tiled TIFF assembly, planar/sample rearrangement, palette/indexed colour,
sub-byte samples, non-RGB photometrics, and other broad TIFF normalisation
choices are deliberately postponed. They belong in `GeneralLibtiffPath`, but
the exact accepted output layouts still need explicit design.

## Writer Strategy

The default writer remains the raw native uncompressed path so existing graphs
and benchmarks keep the same performance contract. Option-bearing TIFF writers
select their write strategy before streaming:

- default options use raw native uncompressed output
- compression uses libtiff encoded output
- requested opposite-endian output uses a separate libtiff write entry point

The public DSL keeps `write` and `writeThick` as the fast defaults and exposes
TIFF-specific option writers for callers that need compression or byte-order
control.

## Native Helper Shape

Native TIFF calls should live with the other low-level C/C++ helpers under
`lowlevel/` and be built through the same CMake path. The public managed layer
should stay in `StackIO.fs`, where TIFF metadata and page IO become Chunk
sources and sinks.

The native side should expose a small C ABI for:

- reading TIFF image metadata into a format-neutral `ImageInfo`
- reading a page into a caller-owned byte buffer
- reading a page into a caller-owned byte buffer at an offset
- reading a decoded page into a caller-owned byte buffer at an offset
- writing a page from a caller-owned byte buffer
- writing a page from a caller-owned byte buffer at an offset
- writing an encoded page from a caller-owned byte buffer at an offset

## Chunk Integration

The important StackProcessing-side contracts are:

- readers produce owned `Chunk<'T>` values
- writers consume and release owned chunks
- thick readers may fill one larger `Chunk<'T>` and expose thin output slices
  only when the ownership rules remain explicit
- thick writers should be able to write pages from offsets without copying into
  temporary thin chunks
- timing measurements must clean output directories before the measured body
  starts and cleanup after the timer stops

The benchmark promotion criterion is simple: production `read -> write` should
stay close to the low-level raw page benchmark for `uint8`, `uint16`, and
`float32` at representative 256^3, 512^3, and 1024^3 stack sizes.

## Remaining Work

1. Finish moving accepted TIFF IO away from the managed TIFF package.

   Raw native and decoded/encoded native libtiff paths are now present, but
   `StackIO.fs` still contains managed TIFF fallback code and metadata helpers.
   Remove those production references once native coverage is complete enough
   for the accepted TIFF surface. This should include project references,
   generated graph project references, and `StackIO.fs` opens/usages.
   Benchmark-only experiments can keep their own dependencies if they are
   explicitly labelled as external comparison code.

2. Decide and implement tile assembly.

   `GeneralLibtiffPath` is the right place for tiled TIFFs, including
   compressed tiled TIFFs. The open design question is how tile assembly should
   map into StackProcessing's contiguous `Chunk.Bytes` layout without adding
   unnecessary copies or hot-loop branching.

3. Decide colour and unusual layout normalisation.

   RGB interleaved uint8 is accepted. Planar-separated RGB, palette/indexed
   images, sub-byte samples, alpha conventions, CMYK/YCbCr, and other TIFF
   photometrics should remain explicit unsupported cases until StackProcessing
   defines the output representation for each.

4. Expand write options deliberately.

   Compression and byte-order control exist for scalar TIFF output. Predictor,
   row-per-strip, tiled output, colour/complex option writers, and multipage
   chunk-file compression should be added only when their user-facing contract
   is clear.

5. Verify with smoke tests and benchmarks.

   Required checks:

   - build `StackProcessing.Core` and the full solution in Release
   - run TIFF JSON samples through Studio/RunSamples
   - run `read -> write`, threshold, and convolve smoke benchmarks for
     `uint8`, `uint16`, and `float32`
   - compare production copy timing against the low-level raw-page benchmark
   - check peak Chunk allocation and RSS for thin and thick read/write paths
