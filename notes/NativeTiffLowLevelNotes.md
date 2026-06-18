# Native TIFF Low-Level Notes

StackProcessing's user-facing TIFF surface is the Chunk DSL:

- `read`
- `readRandom`
- `readRange`
- `readThick`
- `write`
- `writeThick`
- `getImageInfo`

The implementation goal is that these stages move TIFF pages directly into and
out of `Chunk<'T>` buffers with as little copying as possible. Users should not
select an implementation backend in graphs or DSL code.

## Target Fast Path

The preferred TIFF fast path is:

- uncompressed stripped TIFF
- scalar contiguous pixels
- native-endian data
- `uint8`, `uint16`, and `float32` storage first
- single-page slice stacks and multi-page TIFF volumes
- direct reads into `Chunk.Bytes`
- offset reads and writes for thick chunks
- clear errors for layouts outside the supported fast path

Compressed, tiled, palette, non-native-endian, and unusual sample-format TIFFs
should fail with a specific message until they have explicit support.

## Native Helper Shape

Native TIFF calls should live with the other low-level C/C++ helpers under
`lowlevel/` and be built through the same CMake path. The public managed layer
should stay in `StackIO.fs`, where TIFF metadata and page IO become Chunk
sources and sinks.

The native side should expose a small C ABI for:

- reading TIFF image metadata into a format-neutral `ImageInfo`
- reading a page into a caller-owned byte buffer
- reading a page into a caller-owned byte buffer at an offset
- writing a page from a caller-owned byte buffer
- writing a page from a caller-owned byte buffer at an offset

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

1. Move the benchmark TIFF shim into the production low-level helper.

   The useful raw-page functions from `benchmarks/native-libtiff-shim` should
   become part of the `lowlevel/` CMake build, beside the current convolution,
   median, FFT, resampling, distance, and connected-component kernels.

2. Add TIFF metadata to the format-neutral `ImageInfo` path.

   TIFF stack planning needs width, height, page count, bits per sample, sample
   format, samples per pixel, planar configuration, compression, tiled/stripped
   layout, row bytes, raw page bytes, and byte-order status. The public API
   should remain `getImageInfo`.

3. Wire scalar TIFF reads through direct Chunk buffers.

   The first production targets are `uint8`, `uint16`, and `float32`.
   `read`, `readRandom`, `readRange`, and `readThick` should use the native
   raw-page path when the TIFF layout matches the fast-path contract. Reads
   into thick chunks must support byte offsets so internal grouped IO does not
   allocate temporary thin pages.

4. Wire scalar TIFF writes through direct Chunk buffers.

   `write` and `writeThick` should write from `Chunk.Bytes` and from offsets
   into thick chunks. Filename and page-index bookkeeping should stay in
   `StackIO.fs`, where the stack-level policy already lives.

5. Keep unsupported TIFF layouts explicit.

   Until support exists, compressed, tiled, palette, colour, non-native-endian,
   and unusual sample-format TIFFs should fail with clear messages. Silent
   fallback would hide performance and memory behaviour from the optimizer.

6. Remove production references to the managed TIFF package after the native
   path covers the accepted TIFF surface.

   This should include project references, generated graph project references,
   and `StackIO.fs` opens/usages. Benchmark-only experiments can keep their own
   dependencies if they are explicitly labelled as external comparison code.

7. Verify with smoke tests and benchmarks.

   Required checks:

   - build `StackProcessing.Core` and the full solution in Release
   - run TIFF JSON samples through Studio/RunSamples
   - run `read -> write`, threshold, and convolve smoke benchmarks for
     `uint8`, `uint16`, and `float32`
   - compare production copy timing against the low-level raw-page benchmark
   - check peak Chunk allocation and RSS for thin and thick read/write paths
