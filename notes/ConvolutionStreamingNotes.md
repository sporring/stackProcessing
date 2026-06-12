# Convolution Streaming Notes

This note records the current Chunk convolution design.

## Active Path

Convolution is Chunk-native. The preferred path for separable filters is:

```text
Chunk slice stream
  -> z-window/parallel collect where a halo is needed
  -> native convolveX / convolveY / convolveZ
  -> emitted Chunk slice stream
```

Single-axis native convolution uses `float32` kernels and supports:

- `uint8`
- `int8`
- `uint16`
- `int32`
- `float32`

Input and output keep the same pixel type. Integer outputs are converted back
with the same clamp/round policy used by the Chunk convolution wrappers.

## Separable Filters

Separable filters are stage compositions over the single-axis primitives:

- Box smoothing
- Gaussian smoothing
- finite differences
- Sobel-axis responses
- gradient vectors
- Hessian upper matrices
- Laplacian
- gradient magnitude
- Sobel magnitude
- structure tensor component smoothing

Box and Gaussian stages accept separate width/radius parameters per axis. This
keeps anisotropic volumes and pipeline experiments straightforward.

## Vector Components

Vector Chunk payloads store components in the chunk data layout. Structure
tensor and related derivative stages need to convolve each component without
unpacking every component into independent scalar chunks.

The active fast path is a native Float32 component-wise convolution helper. It
walks the byte-backed chunk storage directly, performs the requested axis pass
for each vector component, and returns a Float32 vector Chunk. This is the
right shape for smoothing the six structure-tensor outer-product components.

## Streaming Shape

Convolution stages should expose their halo needs through bounded windows, not
by materializing full volumes. A stage may read a bounded z-window, emit only
valid center slices, and release consumed chunks according to the window
resource rules.

For separable filters:

- X and Y passes are slice-local.
- Z passes require a bounded z-halo.
- Multi-pass filters compose ordinary stages so the plan graph and cost model
  can see the work.

## Performance Guidance

- Prefer native single-axis convolution for primitive numeric chunks.
- Prefer stage composition for separable filters; it keeps the implementation
  small and the graph/cost model visible.
- Avoid per-pixel generic callbacks in hot convolution loops.
- Use F# span loops for simple maps and reducers where native code would only
  add call overhead.
- Keep output chunks owned by the stage and release input chunks at the usual
  ownership boundary.

## FFT Boundary

FFT convolution is not the default path for local finite kernels. The current
FFT work is its own chunked complex64 pipeline. Local filters should stay on the
single-axis/native convolution path unless a benchmark shows the FFT route wins
for a specific large kernel and memory shape.
