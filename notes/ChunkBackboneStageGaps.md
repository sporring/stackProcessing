# Chunk Runtime Status

Chunk is the regular StackProcessing image payload. This note lists the active
Chunk surface, current sample/Studio coverage, and the few known technical
errors or shelved capabilities.

## Core Payload

- `Chunk.create`, `Chunk.incRef`, `Chunk.decRef`, typed spans, `map`, `mapi`,
  `iter`, `iteri`, `fold`, and `foldi`.
- `ChunkFunctions` owns image-processing algorithms. `Chunk` stays structural:
  ownership, spans, indexing helpers, shape, and boundary conversions.
- Preferred storage types are `uint8`, `uint16`, `float32`, and
  complex64-interleaved `float32` where the algorithm supports them.

## Scalar And Structural Operations

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
  `intensityStretchFloat32`, and `invertIntensityFloat32`
- same-type intensity wrappers: `shiftScale`, `clamp`, `intensityStretch`, and
  `invertIntensity`
- normal, salt-and-pepper, and Poisson noise
- padding, crop, squeeze, concatenate along an axis, and axis permutation
- native-backed 2D resampling and Euler 2D transform/rotation stages for Chunk
  slices

## Histograms, Stats, And Charts

- Dense, sparse, and left-edge histograms.
- Serial and parallel histogram reducers.
- Chunk stats and stats adders for min/max, sum, mean, variance, and pixel
  count.
- Quantiles and histogram equalization using Chunk histogram states.
- Histogram chart and histogram-data Studio lowering through Chunk reducers.
- `ShowImage` lowers through `show`/`showChunkWithLabels`.

## Convolution, Derivatives, And Vector Chunks

- Fixed-kernel convolution and window-parallel convolution.
- Native single-axis convolution with `float32` kernels for `UInt8`, `Int8`,
  `UInt16`, `Int32`, and `Float32` chunks.
- Separable convolution stages composed from native `convolveX`, `convolveY`,
  and `convolveZ`.
- Separable Box and Gaussian filters with separate per-axis radii and Gaussian
  sigma/radius parameters.
- Finite-difference 1D kernels exposed as `float32[]` and usable directly as
  convolution kernels.
- `finiteDiffX/Y/Z`
- `sobelX/Y/Z`
- `gradientVector` and XYZ variants
- `hessianUpper` and XYZ variants
- `laplacian` and XYZ variants
- `gradientMagnitude`, XYZ variants, and
  `sobelMagnitude`
- Vector Chunk basis storage and operations: `toVectorImage`,
  `appendVectorElement`, `vectorElement`, `mapVectorElements`, `vectorDot`,
  `vectorMagnitude`, `vectorCross3D`, and `vectorAngleTo`
- Float32 vector-specialized helpers for derivative outputs:
  `mapVectorElementsFloat32`, `vectorDotFloat32`, `vectorMagnitudeFloat32`,
  and `vectorAngleToFloat32`
- Structure tensor:
  `structureTensor` builds the smoothed gradient,
  6-component upper outer-product tensor, optional separable Gaussian tensor
  smoothing, and 12-component eigensystem vector Chunk
  `[eigenvalues; eigenvector0; eigenvector1; eigenvector2]`.
- PCA-gradient direction uses Chunk gradient/vector stages and Studio lowering.

## Morphology, Labels, And Distance

- Binary dilation, erosion, opening, and closing.
- Binary white top-hat, black top-hat, morphological gradient, and binary
  contour.
- UInt8 Perreault-Hebert dense median baseline with y-band workers.
- Median stages for `UInt8`, `UInt16`, `Int32`, and `Float32`.
- Connected-components SAUF stages for `UInt8` input with `UInt32` labels,
  including direct relabel/stitch output.
- Exact signed distance band for `UInt8` mask chunks, emitted as `Float32`
  slices through `signedDistanceBand`.
- Fill-small-holes and remove-small-objects have Chunk flood-fill style stages.

## FFT And Complex Chunks

- Complex64-interleaved Chunk construction and scalar operations:
  real/imaginary composition, polar composition, real, imaginary, modulus,
  argument, and conjugate.
- XY FFT for `Float32` chunks to complex64-interleaved `Float32` chunks.
- Inverse XY FFT sibling.
- 3D `fftshift` using temporary chunk files rather than full-volume
  materialization.

## IO, Sources, And Workflow Stages

- `read`, `write`
- `readRandom`, `readRange`
- Chunk-native scalar OME-Zarr range reads and slice writes for `uint8`,
  `uint16`, `float32`, and `float`
- `zero`, `coordinateX/Y/Z`, `repeat`, `repeatStage`
- scalar image math, image-pair math, comparisons, and mask logic
- sum projection and volume reducers
- streaming object workflow: `streamConnectedObjects`,
  `paintObjects`, and cropped painting
- polygon masks
- bias correction: `fitBiasModel*` and `correctBias*`
- vector dot/cross/angle stages
- `VectorChunk` PCA
- marching cubes over Chunk slices
- color-to-vector, vector-to-color, and RGB TIFF stack sink path
- file-info support at the stack boundary
- serial-section slice-to-slice translation estimation by SSD, serial
  bounding-box reduction, manifest extraction, nearest-neighbour transform
  application, affine resampling, and serial bias correction
- Euler-transform source creation through polygon masks, repeated Chunk
  slices, casts, and native Chunk Euler 2D transforms

## Keypoints

- DoG/SIFT
- LoG blob
- Hessian
- Harris3D
- Forstner3D
- phase congruency

The dense-volume smoothing used by keypoint stages is currently simple and
direct. Absolute-z handling is local to the surrounding stream metadata rather
than embedded in plain `Chunk<'T>`.

## Sample And Graph Status

Chunk-backed DSL samples include:

- `samples/createByEuler2DTransform`
- `samples/chunk`
- `samples/copy`
- `samples/serial`
- `samples/showImage`
- `samples/polygonMask`
- `samples/noise`
- `samples/poissonNoise`
- `samples/saltAndPepperNoise`
- `samples/addSaltAndPepperNoise`
- `samples/harris3DKeypoints`
- `samples/hessianKeypoints`
- `samples/forstner3DKeypoints`
- `samples/siftKeypoints`
- `samples/biasCorrection`
- `samples/serialTransform`
- `samples/objectsImage`
- `samples/objectsSizeHistogram`
- `samples/objectsMarchingCubes`
- `samples/connectedComponents`
- `samples/sumProjection`
- `samples/signedDistanceBand`
- `samples/laplacian`
- `samples/finiteDiff3D`
- `samples/gradientMagnitude`
- `samples/sobelEdge`
- `samples/histogram`
- `samples/histogramShared`
- `samples/computeStats`
- `samples/normalize`
- `samples/quantileClamp`
- `samples/histogramEqualization`
- `samples/multiplyConstant`
- `samples/multiplyMask`
- `samples/sharedStreamer`
- `samples/sharedImbalancedStreamer`
- `samples/meshMeasurement`
- `samples/blackTopHat`
- `samples/whiteTopHat`
- `samples/morphologicalGradient`
- `samples/structureTensor`
- `samples/pcaGradientDirection`
- `samples/volume`
- `samples/resize`
- `samples/binaryContour`
- `samples/randomRigidTransform`
- `samples/resampleAffine`
- `samples/affineKeypointRegistration`
- `samples/serialBiasCorrect`
- `samples/fillSmallHoles`
- `samples/removeSmallObjects`
- `samples/fft`

Studio JSON graphs using the corresponding boxes lower through Chunk stages.
The box IDs are mostly stable; the compiler selects Chunk-backed sources,
stages, reducers, and sinks.

## Active FFT Notes

- Slice-local `fft`/`invFftXY` stages remain available for explicit XY work.
- `fft3D`, `fft3DComplexXY`, `fft3DRealXY`, and `invFft3DRealXY` are the
  current Chunk-level separable 3D FFT stages. They reuse native FFTW plan
  caches over batches, use complex64-interleaved `float32` storage, and expose
  the real-XY/Hermitian-packed path needed by convolution-style round trips.
- Zarr-backed z-axis passes and subchunked spectral round trips are available
  for the larger-than-memory FFT workspace experiments. These are benchmarked
  separately from local-window image filters because the transform is a
  layout-changing global operation.

## Shelved

- Recursive Gaussian filtering.
- Speckle noise.
- Bilateral filtering.
- Grayscale morphology.
- Label contour.
- Exact spherical morphology.
- Change-label.
