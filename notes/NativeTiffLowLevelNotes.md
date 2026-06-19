# BitMiracle Missing TIFF Decisions

This note tracks the TIFF cases that StackProcessing has deliberately not
normalised yet. The current production TIFF engine is BitMiracle's managed TIFF
package. StackProcessing owns image memory through `Chunk<'T>.Bytes`; BitMiracle
only reads from or writes to caller-owned byte arrays during a TIFF call.

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

The implementation goal is that these stages move accepted TIFF pages directly
into and out of `Chunk<'T>.Bytes` buffers with as little copying as possible.
Users should not select an implementation backend in graphs or DSL code.

## Current Reader Strategy

TIFF reads are planned once from the first slice/page in the stack or volume.
The resulting strategy is reused for all emitted chunks so StackIO does not
make a layout decision inside the stream loop. The current reader split is:

- `RawFastPath`
- `GeneralBitMiraclePath`

`RawFastPath` is the benchmark-sensitive path:

- uncompressed stripped TIFF
- scalar contiguous pixels
- native-endian data
- accepted scalar storage types
- single-page slice stacks and multi-page TIFF volumes
- direct BitMiracle `ReadRawStrip` reads into `Chunk.Bytes`
- offset reads for thick chunks

`GeneralBitMiraclePath` is the correctness/interoperability path for supported
stripped layouts:

- BitMiracle `ReadEncodedStrip` decoded reads
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
choices are deliberately postponed. They belong in future general-path work, but
the exact accepted output layouts still need explicit design.

Unsupported TIFF cases should fail explicitly for now. Tiling, non-RGB colour
representations, planar-separated data, sub-byte samples, and broader TIFF
normalisation must not sneak into the raw path or silently pick a lossy
representation.

## Current Writer Strategy

The default writer remains the raw uncompressed BitMiracle path so existing
graphs and benchmarks keep the fast raw performance contract. Option-bearing
TIFF writers select their write strategy before streaming:

- default options use `WriteRawStrip`
- compression uses `WriteEncodedStrip`
- requested byte order is selected in the BitMiracle open mode

The public DSL keeps `write` and `writeThick` as the fast defaults and exposes
TIFF-specific option writers for callers that need compression or byte-order
control.

## Memory Contract

The managed TIFF helper layer lives in `StackIO.fs`, where TIFF metadata and
page IO become Chunk sources and sinks. BitMiracle is used only as a TIFF file
reader/writer over caller-owned byte arrays:

- metadata inspection reads the first page/slice once to choose a strategy
- raw reads call `ReadRawStrip` into `Chunk.Bytes`
- general reads call `ReadEncodedStrip` into `Chunk.Bytes`
- raw writes call `WriteRawStrip` from `Chunk.Bytes`
- general writes call `WriteEncodedStrip` from `Chunk.Bytes`

BitMiracle does not own Chunk storage. StackProcessing's `Chunk.create`,
`incRef`, and `decRef` remain responsible for borrowing and returning the
underlying `ArrayPool<byte>` buffer.

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
stay close to the direct BitMiracle raw-strip benchmark for `uint8`, `uint16`,
and `float32` at representative 256^3, 512^3, and 1024^3 stack sizes.

## Missing TIFF Decisions

1. Decide and implement tile assembly.

   `GeneralBitMiraclePath` is the right place for tiled TIFFs, including
   compressed tiled TIFFs. The open design question is how tile assembly should
   map into StackProcessing's contiguous `Chunk.Bytes` layout without adding
   unnecessary copies or hot-loop branching.

2. Decide colour and unusual layout normalisation.

   RGB interleaved uint8 is accepted. Planar-separated RGB, palette/indexed
   images, sub-byte samples, alpha conventions, CMYK/YCbCr, and other TIFF
   photometrics should remain explicit unsupported cases until StackProcessing
   defines the output representation for each.

3. Decide which write options become public contracts.

   Compression and byte-order control exist for scalar TIFF output. Predictor,
   row-per-strip, tiled output, colour/complex option writers, and multipage
   chunk-file compression should be added only when their user-facing contract
   is clear.

4. Keep verification tied to user-visible DSL paths.

   Required checks:

   - build `StackProcessing.Core` and the full solution in Release
   - run TIFF JSON samples through Studio/RunSamples
   - run `read -> write`, threshold, and convolve smoke benchmarks for
     `uint8`, `uint16`, and `float32`
   - compare production copy timing against the direct BitMiracle raw-strip benchmark
   - check peak Chunk allocation and RSS for thin and thick read/write paths
