# Chunk Backbone Stage Gaps

This note tracks which `Image`/`ImageFunctions` functionality now has a
`Chunk<>`-native path, and which stages still need a bridge or a native
implementation before Chunk can act as the regular StackProcessing backbone.

## Added or Present

- Structural buffer ownership: `Chunk.create`, `Chunk.incRef`, `Chunk.decRef`,
  typed spans, `map`, `mapi`, `iter`, `iteri`, `fold`, and `foldi`.
- Image bridges: `Chunk.ofImage`, `Chunk.toImage`, and `Chunk.toImageWith`.
  These deliberately copy at the boundary. They are compatibility bridges, not
  the desired hot path.
- Basic non-neighbourhood stages in `ChunkFunctions`:
  - `copy`
  - `map` and `map2`
  - scalar arithmetic: `addScalar`, `subScalar`, `mulScalar`, `divScalar`,
    `scalarAdd`, `scalarSub`, `scalarMul`, `scalarDiv`
  - pair arithmetic: `add`, `subtract`, `multiply`, `divide`
  - comparisons: `equal`, `notEqual`, `greater`, `greaterEqual`, `less`,
    `lessEqual`
  - mask logic: `maskAnd`, `maskOr`, `maskXor`, `maskNot`, `mask`
  - reducers: `sum`, `prod`, `minMax`, `getMinMax`
  - Float32 SIMD intensity helpers: `absFloat32`, `sqrtFloat32`,
    `squareFloat32`, `shiftScaleFloat32`, `clampFloat32`,
    `intensityWindowFloat32`, and `invertIntensityFloat32`
  - same-type intensity wrappers: `shiftScale`, `clamp`, `intensityWindow`,
    and `invertIntensity`; non-Float32 inputs explicitly detour through
    Float32 and cast back.
- Existing chunk-native analysis and neighbourhood stages:
  - `thresholdBinary`, `thresholdNative`, and `thresholdNativeParallelCollect`
  - `castToUInt8`, `castToFloat32`, `castFromFloat32`
  - sparse, dense, and left-edge histograms plus serial and parallel reducers
  - zonohedral binary dilation, erosion, opening, and closing
  - fixed-kernel convolution and window-parallel convolution
  - native single-axis convolution with `float32` kernels for `UInt8`, `Int8`,
    `UInt16`, `Int32`, and `Float32` chunks
  - separable convolution stages composed from native `convolveX`,
    `convolveY`, and `convolveZ`
  - separable Box and Gaussian filters, including separate per-axis radii and
    Gaussian sigma/radius parameters
  - finite-difference 1D kernels copied from `ImageFunctions` as `float32[]`
    and exposed as `finiteDiffNativeX/Y/ZParallelCollect`
  - separable Sobel-axis response stages (`sobelX/Y/ZNativeParallelCollect`)
  - UInt8 Perreault-Hebert dense median baseline with y-band workers
  - native nth-element median stages for `UInt8`, `UInt16`, `Int32`, and
    `Float32`, including `ParallelCollect` variants
  - connected-components SAUF stages for `UInt8` input with `UInt32` labels,
    including a `ParallelCollect` variant
  - XY FFT for `Float32` chunks to complex64-interleaved `Float32` chunks,
    including a `ParallelCollect` variant

## Still Needing Chunk Versions

- Stack-level facade functions that still expose only `Image<>` stages should
  gain Chunk-facing names or overloads once the API shape is clearer.
- Slab bridges: `ofSlab` and `toSlab` need a Chunk-native shape. The current
  `Slab<'T>` record contains an `Image<'T>`, so either the record should become
  storage-polymorphic or a parallel `ChunkSlab<'T>` should be introduced.
- Vector-valued image operations remain Image-only:
  `toVectorImage`, `appendVectorElement`, `vectorElement`,
  `mapVectorElements`, `vectorDot`, `vectorCross3D`, `vectorAngleTo`,
  structure-tensor helpers, and vector color conversion.
- Geometric and resampling operations remain Image/ITK paths:
  `euler2DTransform`, `euler2DRotate`, `resample2D`, affine resampling, and
  other coordinate-space operations.
- Full FFT workflows and complex-valued arithmetic remain Image/ITK paths.
  The Chunk path currently has native XY FFT for `Float32` chunks to
  complex64-interleaved `Float32` storage.
- Exact or ITK-backed neighbourhood filters still need either native Chunk
  versions or explicit bridge decisions:
  bilateral, gradient magnitude, Sobel magnitude, Laplacian, Gaussian
  derivatives, signed distance, label contour, and exact spherical morphology.
- Noise generators are not yet Chunk-native:
  normal, salt-and-pepper, shot, and speckle noise.
- Padding, crop, squeeze, concatenate, stack/unstack, and axis permutation need
  a Chunk policy. Some are structural enough to belong in `Chunk`; others may
  be better as `ChunkFunctions`.

## Design Notes

- `ChunkFunctions` should own algorithmic stages. `Chunk` should stay small and
  structural: ownership, span views, indexing, and boundary conversions.
- For integer same-type intensity operations, the current Chunk wrappers make
  widening explicit by using Float32 internally and casting back. This matches
  the direction we have been taking for benchmark fairness.
- Generic arithmetic helpers currently use tight span loops. Float32 intensity
  helpers use `System.Numerics.Vector<float32>` where the operation maps cleanly
  to vector lanes.
- The bridge functions copy by design. They should be used at ITK or legacy
  Image boundaries, not inside hot chunk pipelines.
