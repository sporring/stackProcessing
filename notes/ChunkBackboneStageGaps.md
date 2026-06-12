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
  - Normal, salt-and-pepper, and shot noise have simple Chunk-native stages.
  - Structural transforms: padding, crop, squeeze, concatenate along an axis,
    and axis permutation.
  - Native-backed 2D resampling and Euler 2D transform/rotation stages for
    Chunk slices.
  - Vector Chunk basis storage and operations: `toVectorImage`,
    `appendVectorElement`, `vectorElement`, `mapVectorElements`, `vectorDot`,
    `vectorMagnitude`, `vectorCross3D`, and `vectorAngleTo`. `Float32` vector
    chunks also have `mapVectorElementsFloat32`, `vectorDotFloat32`,
    `vectorMagnitudeFloat32`, and `vectorAngleToFloat32` for native derivative
    outputs.
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
  - derivative-family `Float32` Chunk stages built from separable Gaussian
    smoothing plus native finite differences:
    `gradientVectorNativeParallelCollect`,
    `gradientVectorNativeParallelCollectXYZ`,
    `hessianUpperNativeParallelCollect`,
    `hessianUpperNativeParallelCollectXYZ`,
    `laplacianNativeParallelCollect`, and
    `laplacianNativeParallelCollectXYZ`. Gradient output is a 3-component
    vector Chunk; Hessian output is the 6-component upper matrix
    `[Dxx; Dxy; Dxz; Dyy; Dyz; Dzz]`; Laplacian is scalar `Dxx + Dyy + Dzz`.
  - Convenience magnitude stages: `gradientMagnitudeNativeParallelCollect`,
    `gradientMagnitudeNativeParallelCollectXYZ`, and
    `sobelMagnitudeNativeParallelCollect`.
  - UInt8 Perreault-Hebert dense median baseline with y-band workers
  - native nth-element median stages for `UInt8`, `UInt16`, `Int32`, and
    `Float32`, including `ParallelCollect` variants
  - connected-components SAUF stages for `UInt8` input with `UInt32` labels,
    including a `ParallelCollect` variant
  - native exact signed distance band for `UInt8` mask chunks, emitted as
    `Float32` slices through `signedDistanceBandNativeParallelCollect`
  - XY FFT for `Float32` chunks to complex64-interleaved `Float32` chunks,
    including a `ParallelCollect` variant
  - StackProcessing facade and Studio compiler first-pass lowering for the
    regular TIFF stack path now use Chunk sources/sinks for:
    `readChunkSlices`, `writeChunkSlices`, `chunkZero`,
    `chunkCoordinateX/Y/Z`, Chunk padding/crop/permutation, finite
    differences, Gaussian/gradient/Sobel/Laplacian convenience stages,
    zonohedral binary morphology, signed distance band, cast, clamp,
    shift/scale, intensity window, threshold range, and normal/salt-pepper/shot
    noise stages.
  - Chunk stack source/stage conveniences: `readChunkSlicesRandom`,
    `readChunkSlicesRange`, `chunkRepeat`, and `chunkRepeatStage`.
  - Chunk Studio lowering for scalar image math, image-pair math,
    comparisons, mask logic, sum projection, and the streaming object
    workflow (`streamConnectedObjectsChunk`, `paintObjectsChunk`, and cropped
    painting).
    

## Still Needing Chunk Versions

- Stack-level facade functions that still expose only `Image<>` stages should
  gain Chunk-facing names or overloads once the API shape is clearer.
- Studio-flush smoke test currently exposes these Image-only islands when the
  default source/sink is switched to Chunk:
  - slab TIFF readers (`readSlab`) and a Chunk-native policy for formerly
    slab-shaped workflows
  - `polygonMask` and Euler-transform source creation
  - image stats, volume reducers, histogram reducers/quantiles, and chart/show
    stages
  - bias correction stages in `StackBias.fs` still lower as Image stages in
    Studio despite the Chunk work started there
  - serial registration/section stages still produce and consume Image streams
  - connected-components/relabel/update translation-table Studio paths still
    use Image stages even though Chunk SAUF labels exist
  - complex construction/arithmetic, full FFT workflows, and complex IO/write
    decisions still need Chunk policy beyond the current low-level XY FFT
  - list-backed vector-image Studio boxes (`toVectorImage`, element/range,
    color conversion, dot/cross/angle, PCA, structure tensor) need to lower to
    `VectorChunk` stages or be explicitly marked legacy
  - `speckleNoise`, bilateral filtering, grayscale morphology, label contour,
    change-label, marching cubes, and keypoints remain
    Image-backed in generated Studio graphs.
- Slab bridges: `ofSlab` and `toSlab` need a Chunk-native shape. The current
  `Slab<'T>` record contains an `Image<'T>`, so either the record should become
  storage-polymorphic or a parallel `ChunkSlab<'T>` should be introduced.
- Higher-level vector-valued operations still need Chunk stages or kernels:
  structure-tensor helpers and vector color conversion.
- Some geometric and resampling operations remain Image/ITK paths for now.
  Affine resampling has a first simple Chunk-slice stage, but it still needs
  optimization before it should be considered final.
- Full FFT workflows and complex-valued arithmetic remain Image/ITK paths.
  The Chunk path currently has native XY FFT for `Float32` chunks to
  complex64-interleaved `Float32` storage. Also, it needs speedup!
- Exact or ITK-backed neighbourhood filters still need either native Chunk
  versions or explicit bridge decisions:
  label contour and exact spherical morphology.
- Stack/unstack still need a Chunk policy. The simple structural single-Chunk
  transforms now live in `ChunkFunctions`.

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
- We are for the moment not porting recursive gaussian filter, speckle noise, bilateral filters
