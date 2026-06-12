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
  - Chunk Studio lowering for polygon masks, bias correction
    (`fitBiasModelChunk*`/`correctBiasChunk*`), vector dot/cross/angle,
    `VectorChunk` PCA, and marching cubes over Chunk slices.
  - More `VectorChunk` list-backed replacements: component range,
    append-element, element mapping, vector-to-color, and color-to-vector
    stages. These are native Chunk operations, but Studio still needs a clear
    sink/display policy for vector-valued outputs.
  - Complex64-interleaved Chunk construction and scalar operations:
    real/imaginary composition, polar composition, real, imaginary, modulus,
    argument, and conjugate. These share the same doubled-width `[re; im]`
    `Chunk<float32>` convention as the existing Chunk FFT/Zarr path.
  - Chunk keypoint stages for DoG/SIFT, LoG blob, Hessian, Harris3D, Forstner3D,
    and phase-congruency detectors. They use pure dense-volume Gaussian
    smoothing now; emitted Z is currently window-local because plain
    `Chunk<'T>` slices do not carry an absolute slice index.
  - A first Chunk-native serial-section subset: slice-to-slice translation
    estimation by SSD, serial bounding-box reduction, manifest extraction, and
    nearest-neighbour transform application. The old affine/SIFT serial
    registration behavior is not fully reproduced yet.
  - Euler-transform source creation now builds the polygon mask as a
    `Chunk<uint8>`, repeats it as Chunk slices, casts to the requested output
    type, and emits transformed slices through the native Chunk Euler 2D path.
  - Chunk image display and histogram chart lowering exist for Studio:
    `ShowImage` lowers through `chunkShow`/`showChunkWithLabels`, and
    `ImHistogram`/`ImHistogramData` use Chunk histogram reducers before the
    existing chart helpers.
    

## Still Needing Chunk Versions

- Stack-level facade functions that still expose only `Image<>` stages should
  gain Chunk-facing names or overloads once the API shape is clearer.
- Studio-flush smoke test currently exposes these Image-only islands when the
  default source/sink is switched to Chunk:
  - slab TIFF readers (`readSlab`) and a Chunk-native policy for formerly
    slab-shaped workflows
  - image stats, volume reducers, histogram quantiles/equalization, and the
    remaining scalar/vector chart or display policies
  - connected-components relabel/update translation-table Studio paths still
    use Image stages; the direct Chunk SAUF label stage exists, but the old
    labels-plus-object-count tuple shape is intentionally not recreated yet
  - full FFT workflows and complex IO/write decisions still need Chunk policy
    beyond the current low-level XY FFT and complex64-interleaved scalar helpers
  - structure tensor still needs a `VectorChunk` stage or should be explicitly
    marked legacy
  - speckle noise, bilateral filtering, grayscale morphology, label contour,
    and change-label are shelved and not being ported for now.
- Slab bridges: `ofSlab` and `toSlab` need a Chunk-native shape. The current
  `Slab<'T>` record contains an `Image<'T>`, so either the record should become
  storage-polymorphic or a parallel `ChunkSlab<'T>` should be introduced.
- Higher-level vector-valued operations still need Chunk stages or kernels:
  structure-tensor helpers.
- Some geometric and resampling operations remain Image/ITK paths for now.
  Affine resampling has a first simple Chunk-slice stage, but it still needs
  optimization before it should be considered final.
- Full FFT workflows remain Image/ITK paths. The Chunk path currently has
  native XY FFT for `Float32` chunks to complex64-interleaved `Float32` storage
  plus simple complex64-interleaved scalar helpers. Also, it needs speedup!
- Exact or ITK-backed neighbourhood filters still need either native Chunk
  versions or explicit bridge decisions:
  label contour and exact spherical morphology.
- Stack/unstack still need a Chunk policy. The simple structural single-Chunk
  transforms now live in `ChunkFunctions`.

## Sample And Graph Status

Preferred storage types for Chunk examples are `uint8`, `uint16`, `float32`,
and complex64-interleaved `float32` where the algorithm supports them. The
updated DSL samples intentionally use those types where it makes sense; a few
Chunk stages still produce `float` internally, for example bias correction.

Chunk-upgraded DSL samples now include:

- `samples/createByEuler2DTransform`
- `samples/showImage`
- `samples/polygonMask`
- `samples/noise`
- `samples/shotNoise`
- `samples/saltAndPepperNoise`
- `samples/addSaltAndPepperNoise`
- `samples/harris3DKeypoints`
- `samples/hessianKeypoints`
- `samples/forstner3DKeypoints`
- `samples/siftKeypoints`
- `samples/biasCorrection`
- `samples/serialTransform`
- `samples/objectsImage`
- `samples/objectsMarchingCubes`
- `samples/connectedComponents` for the hand-written DSL path. It now uses the
  Chunk SAUF relabel/stitch phase directly and writes relabelled `uint32`
  label slices without `WriteSlabSlices` or a temporary MHA stack.
- `samples/sumProjection`
- `samples/signedDistanceBand`
- `samples/laplacian`
- `samples/gradientMagnitude`
- `samples/sobelEdge`
- `samples/histogram`
- `samples/histogramShared`
- `samples/computeStats`
- `samples/normalize`
- `samples/quantileClamp`
- `samples/histogramEqualization`
- `samples/meshMeasurement`
- `samples/blackTopHat`
- `samples/whiteTopHat`
- `samples/morphologicalGradient`

Studio JSON graphs that use the corresponding boxes lower through the new
Chunk stages automatically; their box IDs are mostly unchanged because the
compiler now selects Chunk-backed stages for those boxes.

Sample JSON graphs that still are not supported without Image or Slabs:

- `samples/fft` because full FFT/inverse/shift workflows remain Image-backed.
- `samples/structureTensor` and `samples/pcaGradientDirection` JSON graphs
  because the handwritten DSL paths are Chunk-native now, but Studio vector
  boxes still need a clear Float32 lowering policy and true RGB/vector sink
  policy.
- `samples/volume` because the standalone sample still needs the same
  threshold-to-Chunk-volume cleanup now used by `samples/meshMeasurement`.
- `samples/resize`, `samples/randomRigidTransform`,
  `samples/affineKeypointRegistration`, and `samples/serialBiasCorrect`
  because those geometric/registration variants still use Image-only paths.
- `samples/binaryContour`, `samples/fillSmallHoles`, and
  `samples/removeSmallObjects` because those morphology/object cleanup paths
  remain Image-backed or intentionally deferred.

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
