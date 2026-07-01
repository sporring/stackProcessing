module StackProcessing

open System

type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>
type Profile = SlimPipeline.Profile
type ProfileTransition = SlimPipeline.ProfileTransition
type ResourceOps<'T> = SlimPipeline.ResourceOps<'T>
type StageTimeCoefficients = SlimPipeline.StageTimeCoefficients
type Window<'T> = SlimPipeline.Window<'T>

type ChunkIndex = StackCore.ChunkIndex
type ChunkLayout = StackCore.ChunkLayout
type Chunk<'T when 'T: equality> = StackCore.Chunk<'T>
type LocatedChunk<'T when 'T: equality> = StackCore.LocatedChunk<'T>
type EncodedLocatedChunk = StackCore.EncodedLocatedChunk
type VectorChunk<'T when 'T: equality> = StackCore.VectorChunk<'T>
type SpectralLayout = StackCore.SpectralLayout
type SpectralChunk = StackCore.SpectralChunk
type HistogramBinning = StackCore.HistogramBinning
type Histogram<'T when 'T: comparison> = StackCore.Histogram<'T>
type HistogramEstimate<'T when 'T: comparison> = ChunkFunctions.HistogramEstimate<'T>
type ImageStats = StackCore.ImageStats
type Point2D = StackCore.Point2D
type Polygon2D = StackCore.Polygon2D
module Chunk = StackCore.Chunk

let optimizerEnabled = StackCore.optimizerEnabled
let source = StackCore.source
let sourceWithOptimizer = StackCore.sourceWithOptimizer
let debug = StackCore.debug
let debugDefault = StackCore.debugDefault
let commandLineSource = StackCore.commandLineSource
let zip = StackCore.zip
let (>=>) = StackCore.(>=>)
let (-->) = StackCore.(-->)
let (>=>>) = StackCore.(>=>>)
let (>>=>) = StackCore.(>>=>)
let (>>=>>) = StackCore.(>>=>>)
let teeFst = StackCore.teeFst
let teeSnd = StackCore.teeSnd
let fork = StackCore.fork
let (-->>) = StackCore.(-->>)
let ignoreSingles = StackCore.ignoreSingles
let ignorePairs = StackCore.ignorePairs
let sink = StackCore.sink
let sinkList = StackCore.sinkList
let drain = StackCore.drain
let identity<'T> : Stage<'T, 'T> = StackCore.identityStage "identity"
let windowSkipTakeM outputStart outputCount = StackCore.windowSkipTakeM outputStart outputCount
let flattenList () = StackCore.flattenList ()
let failTypeMismatch<'T> = StackCore.failTypeMismatch<'T>
let tap<'T> (name: string) : Stage<'T, 'T> = SlimPipeline.Stage.tap name
let tapIt<'T> (toString: 'T -> string) : Stage<'T, 'T> = SlimPipeline.Stage.tapIt toString
let print<'T> () : Stage<'T, unit> =
    SlimPipeline.Stage.consumeWith "print" (fun _debug _index value -> printfn "%A" value) (fun _ -> 0UL)

let mutable private stackProcessingWorkers = 3

let getWorkers () = stackProcessingWorkers

let setWorkers workers =
    if workers < 1 then
        invalidArg "workers" $"StackProcessing worker count must be at least 1, got {workers}."
    stackProcessingWorkers <- workers

let showChartData = StackCharts.showChartData
let showChartDataWithLabels = StackCharts.showChartDataWithLabels
let showChart = StackCharts.showChart
let showChartWithLabels = StackCharts.showChartWithLabels
let showChartXY = StackCharts.showChartXY
let showChartXYWithLabels = StackCharts.showChartXYWithLabels
let showHistogram = StackCharts.showHistogram
let showHistogramWithLabels = StackCharts.showHistogramWithLabels
let showChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackCharts.showChunk<'T>
let showChunkWithLabels<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> colorMap title xAxis yAxis chunk =
    StackCharts.showChunkWithLabels<'T> colorMap title xAxis yAxis chunk

type FileInfo = StackIO.FileInfo
type ChunkInfo = StackIO.ChunkInfo
type ImageInfo = StackIO.ImageInfo
let getFileInfo = StackIO.getFileInfo
let getStackDepth = StackIO.getStackDepth
let getImageInfo = StackIO.getImageInfo
let getImageFileInfo = StackIO.getImageFileInfo
let volumeFilePath = StackIO.volumeFilePath
let getStackSize = StackIO.getStackSize
let getStackWidth = StackIO.getStackWidth
let getStackHeight = StackIO.getStackHeight
let getFilenames = StackIO.getFilenames
let getChunkInfo = StackIO.getChunkInfo
let getZarrInfo = StackIO.getZarrInfo
let getNexusInfo = StackIO.getNexusInfo
let getChunkFilename = StackIO.getChunkFilename
let deleteIfExists = StackIO.deleteIfExists

let read<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readChunkSlices<'T>
let readThick<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readChunkThick<'T>
let readRandom<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readChunkSlicesRandom<'T>
let sliceIndicesRandom = StackIO.sliceIndicesRandom
let sliceIndicesRange = StackIO.sliceIndicesRange
let readAtIndices<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readChunkSlicesAtIndices<'T>
let readRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readChunkSlicesRange<'T>
let readComplex = StackIO.readComplex64ChunkSlices
let readComplexThick = StackIO.readComplex64ChunkThick
let readTiffComplex = readComplex
let readTiffComplexThick = readComplexThick
let readVolume<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readChunkVolume<'T>
let readColor = StackIO.readColorChunkSlices
let readColorRandom = StackIO.readColorChunkSlicesRandom
let readColorRange = StackIO.readColorChunkSlicesRange
let write<'T when 'T: equality> = StackIO.writeChunkSlices<'T>
let writeTiffWithOptions<'T when 'T: equality> = StackIO.writeChunkSlicesWithOptions<'T>
let writeThick<'T when 'T: equality> = StackIO.writeChunkThick<'T>
let writeThickTiffWithOptions<'T when 'T: equality> = StackIO.writeChunkThickWithOptions<'T>
let writeComplex = StackIO.writeComplex64ChunkSlices
let writeComplexThick = StackIO.writeComplex64ChunkThick
let writeTiffComplex = writeComplex
let writeTiffComplexThick = writeComplexThick
let writeColor = StackIO.writeColorChunkSlices
let readZarrRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readZarrChunkSlicesRange<'T>
let readZarrThick<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readZarrChunkThickRange<'T>
let readZarrAlignedSlices<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readZarrChunkSlicesAlignedRange<'T>
let readZarrChunks<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readZarrLocatedChunks<'T>
let readZarrEncodedChunks = StackIO.readZarrEncodedChunks
let readZarrComplexHermitianRange = StackIO.readZarrSpectralComplex64InterleavedFloat32Range
let readZarrComplexHermitian path multiscaleIndex datasetIndex timepoint channel maxParallelChunks =
    readZarrComplexHermitianRange 0u 1 UInt32.MaxValue path multiscaleIndex datasetIndex timepoint channel maxParallelChunks
let readZarrComplexHermitianTiledRange = StackIO.readZarrSpectralComplex64InterleavedFloat32TiledRange
let readZarrComplexHermitianTiled path multiscaleIndex datasetIndex timepoint channel maxParallelChunks =
    readZarrComplexHermitianTiledRange 0u 1 UInt32.MaxValue path multiscaleIndex datasetIndex timepoint channel maxParallelChunks
let writeZarr<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> outputPath name depth chunkX chunkY chunkZ physicalSizeX physicalSizeY physicalSizeZ maxConcurrentWrites =
    StackIO.writeZarrChunkSlices<'T> outputPath name depth chunkX chunkY chunkZ physicalSizeX physicalSizeY physicalSizeZ maxConcurrentWrites
let writeZarrWithCompression<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.writeZarrChunkSlicesWithCompression<'T>
let writeZarrThick<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.writeZarrChunkThick<'T>
let writeZarrThickWithCompression<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.writeZarrChunkThickWithCompression<'T>
let writeZarrAlignedSlices<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.writeZarrChunkSlicesAligned<'T>
let writeZarrAlignedSlicesWithCompression<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.writeZarrChunkSlicesAlignedWithCompression<'T>
let writeZarrChunks<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.writeZarrLocatedChunks<'T>
let writeZarrChunksWithCompression<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.writeZarrLocatedChunksWithCompression<'T>
let writeZarrEncodedChunks = StackIO.writeZarrEncodedChunks
let writeZarrEncodedChunksWithCompression = StackIO.writeZarrEncodedChunksWithCompression
let writeZarrComplex = StackIO.writeZarrComplex64InterleavedFloat32
let writeZarrComplexWithCompression = StackIO.writeZarrComplex64InterleavedFloat32WithCompression
let writeZarrComplexHermitian = StackIO.writeZarrSpectralComplex64InterleavedFloat32
let writeZarrComplexHermitianWithCompression = StackIO.writeZarrSpectralComplex64InterleavedFloat32WithCompression
let writeZarrComplexHermitianTiled = StackIO.writeZarrSpectralComplex64InterleavedFloat32Tiled


type Position3D<'T> = StackPoints.Position3D<'T>
type CoordinatePoint = StackPoints.CoordinatePoint
type PointSet = StackPoints.PointSet
type VectorizedMatrix = StackCore.VectorizedMatrix
let toVector3 = StackPoints.toVector3
let ofVector3 = StackPoints.ofVector3
let toPointSet = StackPoints.toPointSet
let ofPointSet = StackPoints.ofPointSet
let coordinatePoint = StackPoints.coordinatePoint
let readPointSetFile = StackPoints.readPointSetFile
let writePointSetFile = StackPoints.writePointSetFile
let readPointSet = StackPoints.readPointSet
let writePointSet = StackPoints.writePointSet
let writeCSVPointSet = StackPoints.writeCSVPointSet
let vectorizeMatrix = StackPoints.vectorizeMatrix
let unvectorizeMatrix = StackPoints.unvectorizeMatrix
let pointPairDistances = StackPoints.pointPairDistances
let writeMatrix = StackPoints.writeMatrix
let writeCSVMatrix = StackPoints.writeCSVMatrix
let writeCSVHistogram<'T when 'T: comparison> = StackPoints.writeCSVHistogram<'T>
let selectGroupedValueOutput = StackPoints.selectGroupedValueOutput

type Affine = TinyLinAlg.Affine
type AffineRegistrationOptions = StackRegistration.AffineRegistrationOptions
type AffineRegistrationResult = StackRegistration.AffineRegistrationResult
type RansacResult<'Model, 'Item> = StackRansac.RansacResult<'Model, 'Item>
type PointMatch2D = StackRansac.PointMatch2D
let imageCenter = TinyLinAlg.imageCenter
let randomRigidTransformAround = TinyLinAlg.randomRigidTransformAround
let randomRigidTransform = TinyLinAlg.randomRigidTransform
let defaultAffineRegistrationOptions = StackRegistration.defaultAffineRegistrationOptions
let earthMoversDistance = StackRegistration.earthMoversDistance
let transformPointSet = StackRegistration.transformPointSet
let inverseAffine = StackRegistration.inverseAffine
let toHomogeneousMatrix = StackRegistration.toHomogeneousMatrix
let ofHomogeneousMatrix = StackRegistration.ofHomogeneousMatrix
let affineRegistration = StackRegistration.affineRegistration
let affineRegistrationMatrix = StackRegistration.affineRegistrationMatrix
let affineRegistrationInverseMatrix = StackRegistration.affineRegistrationInverseMatrix
let ransacFit = StackRansac.fit
let affine2DFromMatches = StackRansac.affine2DFromMatches
let affine2DRansac = StackRansac.affine2DRansac

type ImageSetCoordinateSystem = StackManifest.ImageSetCoordinateSystem
type ImageSetTransform = StackManifest.ImageSetTransform
type ImageSetGrid = StackManifest.ImageSetGrid
type ImageSetItem = StackManifest.ImageSetItem
type ImageSetMember = StackManifest.ImageSetItem
type ImageSetManifest = StackManifest.ImageSetManifest
let identityImageSetTransform = StackManifest.identityTransform
let imageSetTransformFromMatrix = StackManifest.transformFromMatrix
let imageSetTransformToMatrix = StackManifest.transformToMatrix
let imageSetTransformFromAffine = StackManifest.transformFromAffine
let imageSetTransformToAffine = StackManifest.transformToAffine
let createImageSetManifest = StackManifest.createManifest
let identityImageSetManifest = StackManifest.identityManifest
let imageSetGrid = StackManifest.imageSetGrid
let withImageSetGrid = StackManifest.withGrid
let imageSetGridIndexTransform = StackManifest.gridIndexTransform
let composeImageSetTransforms = StackManifest.composeTransforms
let updateMovingImageSetItemTransformFromRegistration = StackManifest.updateMovingItemTransformFromRegistration
let imageSetItem = StackManifest.spatialDataItem
let scalarImageSetItem = StackManifest.scalarImageItem
let gridImageSetItem = StackManifest.gridImageItem
let vectorImageSetItem = StackManifest.vectorImageItem
let pointSetManifestItem = StackManifest.pointSetItem
let triangleMeshManifestItem = StackManifest.triangleMeshItem
let matrixManifestItem = StackManifest.matrixItem
let imageSetMember = StackManifest.imageMember
let addImageSetItem = StackManifest.addItem
let addImageSetMember = StackManifest.addImage
let replaceImageSetItemTransform = StackManifest.replaceItemTransform
let replaceImageSetMemberTransform = StackManifest.replaceImageTransform
let writeImageSetManifest = StackManifest.writeManifest
let readImageSetManifest = StackManifest.readManifest

let zero<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkZero<'T>
let coordinateX<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkCoordinateX<'T>
let coordinateY<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkCoordinateY<'T>
let coordinateZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkCoordinateZ<'T>
let polygonMask = ChunkFunctions.chunkPolygonMask
let euler2DTransform<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType>
    rotation
    translation
    chunk =
    ChunkCore.ChunkFunctions.euler2DTransformNativeChunk<'T> rotation translation chunk
let mapi<'T, 'U when 'T: equality and 'U: equality
                  and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType
                  and 'U: (new: unit -> 'U) and 'U: struct and 'U :> ValueType>
    f =
    ChunkFunctions.mapi<'T, 'U> f
let repeat<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkRepeat<'T>
let repeatStage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkRepeatStage<'T>
let pad<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.pad<'T>
let crop<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.crop<'T>
let squeeze<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.squeeze<'T>
let concatenateAlong<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.concatenateAlong<'T>
let permuteAxes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> order =
    ChunkFunctions.permuteAxes<'T> order
let resize<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> outputWidth outputHeight outputDepth interpolationName =
    ChunkFunctions.chunkResize<'T> outputWidth outputHeight outputDepth interpolationName
let resample<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> factorX factorY factorZ interpolationName =
    ChunkFunctions.chunkResample<'T> factorX factorY factorZ interpolationName
let show<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.show<'T>
let signedDistanceBand bandRadius stride =
    ChunkFunctions.signedDistanceBandNativeParallelCollect stackProcessingWorkers bandRadius stride
let connectedComponents windowSize =
    ChunkFunctions.connectedComponentsSauf3DUInt8UInt32ParallelCollect stackProcessingWorkers windowSize
let toVectorImage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.toVectorImage<'T>
let vectorElement<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.vectorElement<'T>
let vectorRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.vectorRange<'T>
let appendVectorElement<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.appendVectorElement<'T>
let vectorMapElements = ChunkFunctions.vectorMapElements
let vectorCast<'S, 'T when 'S: equality and 'S: (new: unit -> 'S) and 'S: struct and 'S :> ValueType
                         and 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    ChunkFunctions.vectorCast<'S, 'T>
let colorToVector3 = ChunkFunctions.colorToVector3
let vectorDot = ChunkFunctions.vectorDot
let vectorMagnitude = ChunkFunctions.vectorMagnitude
let vectorCross3D = ChunkFunctions.vectorCross3D
let vectorAngleTo = ChunkFunctions.vectorAngleTo
let vectorDotFloat32 = ChunkFunctions.vectorDotFloat32
let vectorMagnitudeFloat32 = ChunkFunctions.vectorMagnitudeFloat32
let vectorMagnitudeSquaredFloat32 = ChunkFunctions.vectorMagnitudeSquaredFloat32
let vectorAngleToFloat32 = ChunkFunctions.vectorAngleToFloat32
let pca = ChunkFunctions.PCA
let covarianceMatrix = ChunkFunctions.covarianceMatrix
let symmetricMatrixEigenbasis = ChunkFunctions.symmetricMatrixEigenbasis
let symmetricMatrixEigenvaluesMatrix = ChunkFunctions.symmetricMatrixEigenvalues
let projectVectorBasis = ChunkFunctions.projectVectorBasisFloat32
let pcaFloat32 = covarianceMatrix
let structureTensor sigma radius rho rhoRadius =
    ChunkFunctions.structureTensorNativeParallelCollect stackProcessingWorkers sigma radius rho rhoRadius
let symmetricMatrixEigensystem () =
    ChunkFunctions.symmetricMatrixEigensystemFloat32 stackProcessingWorkers
let symmetricMatrixEigenvalues () =
    ChunkFunctions.symmetricMatrixEigenvaluesFloat32 stackProcessingWorkers
let symmetricMatrixEigenvector eigenIndex =
    ChunkFunctions.symmetricMatrixEigenvectorFloat32 stackProcessingWorkers eigenIndex
let selectGroupedVectorOutput = ChunkFunctions.selectGroupedVectorOutput
let toComplex = ChunkFunctions.toComplex64
let polarToComplex = ChunkFunctions.polarToComplex64
let real = ChunkFunctions.complex64Real
let imag = ChunkFunctions.complex64Imag
let modulus = ChunkFunctions.complex64Modulus
let length = modulus
let argument = ChunkFunctions.complex64Argument
let angle = argument
let conjugate = ChunkFunctions.complex64Conjugate
let fft = ChunkFunctions.fftXYFloat32ToComplex64Interleaved
let fftRealXY = ChunkFunctions.fftRealXYFloat32ToHermitianPackedComplex64Interleaved
let fft3D = ChunkFunctions.fft3DFloat32ToComplex64Interleaved
let fft3DComplexXY = ChunkFunctions.fft3DFloat32ToComplex64Interleaved
let fft3DRealXY = ChunkFunctions.fft3DRealXYFloat32ToComplex64Interleaved
let fftXYThenZPlanned = ChunkFunctions.fftXYThenZFloat32ToComplex64InterleavedPlanned
let invFft3DRealXY = ChunkFunctions.invFft3DRealXYComplex64InterleavedToFloat32
let invFft = ChunkFunctions.invFftXYComplex64InterleavedToFloat32
let invFftRealXY = ChunkFunctions.invFftXYHermitianPackedComplex64InterleavedToFloat32
let fftShift3D = ChunkFunctions.fftShift3DComplex64Interleaved
let imageHistogram<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () =
    ChunkFunctions.histogramReducer<'T> ()
let imageHistogramDense<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () =
    ChunkFunctions.histogramDenseReducer<'T> ()
let imageHistogramLeftEdges<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> leftEdges =
    ChunkFunctions.histogramLeftEdgesReducer<'T> leftEdges
let imageHistogramFixedBins<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> firstLeftEdge lastLeftEdge bins =
    ChunkFunctions.histogramFixedBinsReducer<'T> firstLeftEdge lastLeftEdge bins
let histogramEstimate<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> maxSlices input suffix method confidence targetError =
    StackIO.histogramEstimate<'T> maxSlices input suffix method confidence targetError
let histogramEstimateMap = ChunkFunctions.histogramEstimateMap
let histogramEqualization<'T, 'H when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (histogram: 'H) =
    ChunkFunctions.histogramEqualization<'T> (box histogram)
let quantiles quantileValues histogram =
    ChunkFunctions.quantiles quantileValues (box histogram)
let computeStats<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () = ChunkFunctions.computeStats<'T> ()
let objectVolume = ChunkFunctions.objectVolume
let sumProjection<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> projectionKind =
    ChunkFunctions.chunkSumProjection<'T> projectionKind

let momentsThresholdFromHistogram (histogram: Histogram<'T>) : float =
    let ordered =
        histogram.Counts
        |> Map.toList
        |> List.map (fun (value, count) -> Convert.ToDouble value, count)
        |> List.sortBy fst

    match ordered with
    | [] -> invalidArg "histogram" "Cannot estimate a moments threshold from an empty histogram."
    | [ value, _ ] -> value
    | _ ->
        let totalCount = ordered |> List.sumBy (snd >> float)
        let moment power =
            ordered
            |> List.sumBy (fun (value, count) -> (value ** power) * float count)
            |> fun value -> value / totalCount

        let m0 = 1.0
        let m1 = moment 1.0
        let m2 = moment 2.0
        let m3 = moment 3.0
        let cd = m0 * m2 - m1 * m1

        if abs cd < 1e-12 then
            m1
        else
            let c0 = (-m2 * m2 + m1 * m3) / cd
            let c1 = (-m3 + m2 * m1) / cd
            let discriminant = c1 * c1 - 4.0 * c0

            if discriminant < 0.0 then
                m1
            else
                let root = sqrt discriminant
                let z0 = 0.5 * (-c1 - root)
                let z1 = 0.5 * (-c1 + root)
                let denominator = z1 - z0

                if abs denominator < 1e-12 then
                    m1
                else
                    let p0 = (z1 - m1) / denominator |> max 0.0 |> min 1.0
                    let target = p0 * totalCount
                    let mutable cumulative = 0.0
                    let mutable threshold = fst (List.last ordered)
                    let mutable found = false

                    for index in 0 .. ordered.Length - 1 do
                        let value, count = ordered[index]
                        if not found then
                            cumulative <- cumulative + float count
                            if cumulative >= target then
                                threshold <-
                                    if index < ordered.Length - 1 then
                                        0.5 * (value + fst ordered[index + 1])
                                    else
                                        value
                                found <- true

                    threshold

let histogram2pairs<'T when 'T: comparison> : Stage<Histogram<'T>, ('T * uint64) list> =
    SlimPipeline.Stage.liftUnary "histogram2pairs" (fun histogram -> histogram.Counts |> Map.toList) id id

let histogramCounts<'T when 'T: comparison> : Stage<Histogram<'T>, Map<'T, uint64>> =
    SlimPipeline.Stage.liftUnary "histogramCounts" (fun histogram -> histogram.Counts) id id

let pairs2floats<'T, 'S> : Stage<('T * 'S) list, (float * float) list> =
    SlimPipeline.Stage.liftUnary
        "pairs2floats"
        (List.map (fun (x, y) -> Convert.ToDouble x, Convert.ToDouble y))
        id
        id

let plot (plt: float list -> float list -> unit) : Stage<(float * float) list, unit> =
    let consumer _debug _index points =
        let x, y = points |> List.unzip
        plt x y
    SlimPipeline.Stage.consumeWith "plot" consumer (fun _ -> 0UL)

let plotHistogramWithLabels title xAxis yAxis : Stage<Histogram<'T>, unit> =
    SlimPipeline.Stage.consumeWith
        "plotHistogramWithLabels"
        (fun _debug _index histogram -> showHistogramWithLabels title xAxis yAxis histogram)
        (fun _ -> 0UL)

let plotHistogram () : Stage<Histogram<'T>, unit> =
    SlimPipeline.Stage.consumeWith
        "plotHistogram"
        (fun _debug _index histogram -> showHistogram histogram)
        (fun _ -> 0UL)

let otsuThresholdFromHistogram (histogram: Histogram<'T>) : float =
    let ordered =
        histogram.Counts
        |> Map.toList
        |> List.choose (fun (value, count) ->
            if count = 0UL then
                None
            else
                Some(Convert.ToDouble value, count))
        |> List.sortBy fst

    match ordered with
    | [] -> invalidArg "histogram" "Cannot estimate an Otsu threshold from an empty histogram."
    | [ value, _ ] -> value
    | _ ->
        let totalCount = ordered |> List.sumBy (snd >> float)
        let totalMean = ordered |> List.sumBy (fun (value, count) -> value * float count)
        let mutable bestThreshold = fst ordered[0]
        let mutable bestVariance = Double.NegativeInfinity
        let mutable backgroundWeight = 0.0
        let mutable backgroundSum = 0.0

        for index in 0 .. ordered.Length - 2 do
            let value, count = ordered[index]
            backgroundWeight <- backgroundWeight + float count
            backgroundSum <- backgroundSum + value * float count
            let foregroundWeight = totalCount - backgroundWeight

            if backgroundWeight > 0.0 && foregroundWeight > 0.0 then
                let backgroundMean = backgroundSum / backgroundWeight
                let foregroundMean = (totalMean - backgroundSum) / foregroundWeight
                let variance = backgroundWeight * foregroundWeight * pown (backgroundMean - foregroundMean) 2

                if variance > bestVariance then
                    bestVariance <- variance
                    bestThreshold <- 0.5 * (value + fst ordered[index + 1])

        bestThreshold

let finiteDiffKernel1D = ChunkFunctions.finiteDiffKernel1D
let thresholdRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (lower: double) (upper: double) =
    ChunkFunctions.thresholdRange<'T> lower upper
let thresholdZarrChunksUInt8 = ChunkFunctions.thresholdLocatedNativeUInt8
let sqrt<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<float32> then
        unbox (box ChunkFunctions.sqrtFloat32)
    elif typeof<'T> = typeof<float> then
        unbox (box ChunkFunctions.sqrtFloat)
    else
        invalidArg "T" $"sqrt<'T> supports float32 and float chunks, got {typeof<'T>.Name}."
let abs<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<float32> then
        unbox (box ChunkFunctions.absFloat32)
    elif typeof<'T> = typeof<float> then
        unbox (box ChunkFunctions.absFloat)
    elif typeof<'T> = typeof<int8> then
        unbox (box (ChunkFunctions.map "chunkAbs.Int8" (fun (value: int8) -> Math.Abs value)))
    elif typeof<'T> = typeof<int16> then
        unbox (box (ChunkFunctions.map "chunkAbs.Int16" (fun (value: int16) -> Math.Abs value)))
    elif typeof<'T> = typeof<int32> then
        unbox (box (ChunkFunctions.map "chunkAbs.Int32" (fun (value: int32) -> Math.Abs value)))
    elif typeof<'T> = typeof<int64> then
        unbox (box (ChunkFunctions.map "chunkAbs.Int64" (fun (value: int64) -> Math.Abs value)))
    elif typeof<'T> = typeof<uint8>
      || typeof<'T> = typeof<uint16>
      || typeof<'T> = typeof<uint32>
      || typeof<'T> = typeof<uint64> then
        identity<Chunk<'T>>
    else
        invalidArg "T" $"abs<'T> supports real numeric chunks, got {typeof<'T>.Name}."
let log<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<float32> then
        unbox (box ChunkFunctions.logFloat32)
    elif typeof<'T> = typeof<float> then
        unbox (box ChunkFunctions.logFloat)
    else
        invalidArg "T" $"log<'T> supports float32 and float chunks, got {typeof<'T>.Name}."
let squareFloat32 = ChunkFunctions.squareFloat32
let clamp<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> lower upper =
    ChunkFunctions.clamp<'T> lower upper
let shiftScale<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.shiftScale<'T>
let intensityStretch = ChunkFunctions.intensityStretch
let cast<'S, 'T when 'S: equality and 'S: (new: unit -> 'S) and 'S: struct and 'S :> ValueType
                     and 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'S>, Chunk<'T>> =
    ChunkFunctions.castChunk<'S, 'T>

let private unsupportedScalarArithmetic<'T when 'T: equality> () : Stage<Chunk<'T>, Chunk<'T>> =
    invalidArg "T" $"Scalar image arithmetic supports real numeric chunks, got {typeof<'T>.Name}."

let private scalarStage<'T when 'T: equality> (stage: obj) : Stage<Chunk<'T>, Chunk<'T>> =
    unbox stage

let private addScalarFromDouble<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<uint8> then scalarStage<'T> (box (ChunkFunctions.addScalar (uint8 value) : Stage<Chunk<uint8>, Chunk<uint8>>))
    elif typeof<'T> = typeof<int8> then scalarStage<'T> (box (ChunkFunctions.addScalar (int8 value) : Stage<Chunk<int8>, Chunk<int8>>))
    elif typeof<'T> = typeof<uint16> then scalarStage<'T> (box (ChunkFunctions.addScalar (uint16 value) : Stage<Chunk<uint16>, Chunk<uint16>>))
    elif typeof<'T> = typeof<int16> then scalarStage<'T> (box (ChunkFunctions.addScalar (int16 value) : Stage<Chunk<int16>, Chunk<int16>>))
    elif typeof<'T> = typeof<uint32> then scalarStage<'T> (box (ChunkFunctions.addScalar (uint32 value) : Stage<Chunk<uint32>, Chunk<uint32>>))
    elif typeof<'T> = typeof<int32> then scalarStage<'T> (box (ChunkFunctions.addScalar (int32 value) : Stage<Chunk<int32>, Chunk<int32>>))
    elif typeof<'T> = typeof<uint64> then scalarStage<'T> (box (ChunkFunctions.addScalar (uint64 value) : Stage<Chunk<uint64>, Chunk<uint64>>))
    elif typeof<'T> = typeof<int64> then scalarStage<'T> (box (ChunkFunctions.addScalar (int64 value) : Stage<Chunk<int64>, Chunk<int64>>))
    elif typeof<'T> = typeof<float32> then scalarStage<'T> (box (ChunkFunctions.addScalar (float32 value) : Stage<Chunk<float32>, Chunk<float32>>))
    elif typeof<'T> = typeof<float> then scalarStage<'T> (box (ChunkFunctions.addScalar value : Stage<Chunk<float>, Chunk<float>>))
    else unsupportedScalarArithmetic<'T> ()

let private subScalarFromDouble<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<uint8> then scalarStage<'T> (box (ChunkFunctions.subScalar (uint8 value) : Stage<Chunk<uint8>, Chunk<uint8>>))
    elif typeof<'T> = typeof<int8> then scalarStage<'T> (box (ChunkFunctions.subScalar (int8 value) : Stage<Chunk<int8>, Chunk<int8>>))
    elif typeof<'T> = typeof<uint16> then scalarStage<'T> (box (ChunkFunctions.subScalar (uint16 value) : Stage<Chunk<uint16>, Chunk<uint16>>))
    elif typeof<'T> = typeof<int16> then scalarStage<'T> (box (ChunkFunctions.subScalar (int16 value) : Stage<Chunk<int16>, Chunk<int16>>))
    elif typeof<'T> = typeof<uint32> then scalarStage<'T> (box (ChunkFunctions.subScalar (uint32 value) : Stage<Chunk<uint32>, Chunk<uint32>>))
    elif typeof<'T> = typeof<int32> then scalarStage<'T> (box (ChunkFunctions.subScalar (int32 value) : Stage<Chunk<int32>, Chunk<int32>>))
    elif typeof<'T> = typeof<uint64> then scalarStage<'T> (box (ChunkFunctions.subScalar (uint64 value) : Stage<Chunk<uint64>, Chunk<uint64>>))
    elif typeof<'T> = typeof<int64> then scalarStage<'T> (box (ChunkFunctions.subScalar (int64 value) : Stage<Chunk<int64>, Chunk<int64>>))
    elif typeof<'T> = typeof<float32> then scalarStage<'T> (box (ChunkFunctions.subScalar (float32 value) : Stage<Chunk<float32>, Chunk<float32>>))
    elif typeof<'T> = typeof<float> then scalarStage<'T> (box (ChunkFunctions.subScalar value : Stage<Chunk<float>, Chunk<float>>))
    else unsupportedScalarArithmetic<'T> ()

let private mulScalarFromDouble<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<uint8> then scalarStage<'T> (box (ChunkFunctions.mulScalar (uint8 value) : Stage<Chunk<uint8>, Chunk<uint8>>))
    elif typeof<'T> = typeof<int8> then scalarStage<'T> (box (ChunkFunctions.mulScalar (int8 value) : Stage<Chunk<int8>, Chunk<int8>>))
    elif typeof<'T> = typeof<uint16> then scalarStage<'T> (box (ChunkFunctions.mulScalar (uint16 value) : Stage<Chunk<uint16>, Chunk<uint16>>))
    elif typeof<'T> = typeof<int16> then scalarStage<'T> (box (ChunkFunctions.mulScalar (int16 value) : Stage<Chunk<int16>, Chunk<int16>>))
    elif typeof<'T> = typeof<uint32> then scalarStage<'T> (box (ChunkFunctions.mulScalar (uint32 value) : Stage<Chunk<uint32>, Chunk<uint32>>))
    elif typeof<'T> = typeof<int32> then scalarStage<'T> (box (ChunkFunctions.mulScalar (int32 value) : Stage<Chunk<int32>, Chunk<int32>>))
    elif typeof<'T> = typeof<uint64> then scalarStage<'T> (box (ChunkFunctions.mulScalar (uint64 value) : Stage<Chunk<uint64>, Chunk<uint64>>))
    elif typeof<'T> = typeof<int64> then scalarStage<'T> (box (ChunkFunctions.mulScalar (int64 value) : Stage<Chunk<int64>, Chunk<int64>>))
    elif typeof<'T> = typeof<float32> then scalarStage<'T> (box (ChunkFunctions.mulScalar (float32 value) : Stage<Chunk<float32>, Chunk<float32>>))
    elif typeof<'T> = typeof<float> then scalarStage<'T> (box (ChunkFunctions.mulScalar value : Stage<Chunk<float>, Chunk<float>>))
    else unsupportedScalarArithmetic<'T> ()

let private divScalarFromDouble<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<uint8> then scalarStage<'T> (box (ChunkFunctions.divScalar (uint8 value) : Stage<Chunk<uint8>, Chunk<uint8>>))
    elif typeof<'T> = typeof<int8> then scalarStage<'T> (box (ChunkFunctions.divScalar (int8 value) : Stage<Chunk<int8>, Chunk<int8>>))
    elif typeof<'T> = typeof<uint16> then scalarStage<'T> (box (ChunkFunctions.divScalar (uint16 value) : Stage<Chunk<uint16>, Chunk<uint16>>))
    elif typeof<'T> = typeof<int16> then scalarStage<'T> (box (ChunkFunctions.divScalar (int16 value) : Stage<Chunk<int16>, Chunk<int16>>))
    elif typeof<'T> = typeof<uint32> then scalarStage<'T> (box (ChunkFunctions.divScalar (uint32 value) : Stage<Chunk<uint32>, Chunk<uint32>>))
    elif typeof<'T> = typeof<int32> then scalarStage<'T> (box (ChunkFunctions.divScalar (int32 value) : Stage<Chunk<int32>, Chunk<int32>>))
    elif typeof<'T> = typeof<uint64> then scalarStage<'T> (box (ChunkFunctions.divScalar (uint64 value) : Stage<Chunk<uint64>, Chunk<uint64>>))
    elif typeof<'T> = typeof<int64> then scalarStage<'T> (box (ChunkFunctions.divScalar (int64 value) : Stage<Chunk<int64>, Chunk<int64>>))
    elif typeof<'T> = typeof<float32> then scalarStage<'T> (box (ChunkFunctions.divScalar (float32 value) : Stage<Chunk<float32>, Chunk<float32>>))
    elif typeof<'T> = typeof<float> then scalarStage<'T> (box (ChunkFunctions.divScalar value : Stage<Chunk<float>, Chunk<float>>))
    else unsupportedScalarArithmetic<'T> ()

let private scalarSubFromDouble<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<uint8> then scalarStage<'T> (box (ChunkFunctions.scalarSub (uint8 value) : Stage<Chunk<uint8>, Chunk<uint8>>))
    elif typeof<'T> = typeof<int8> then scalarStage<'T> (box (ChunkFunctions.scalarSub (int8 value) : Stage<Chunk<int8>, Chunk<int8>>))
    elif typeof<'T> = typeof<uint16> then scalarStage<'T> (box (ChunkFunctions.scalarSub (uint16 value) : Stage<Chunk<uint16>, Chunk<uint16>>))
    elif typeof<'T> = typeof<int16> then scalarStage<'T> (box (ChunkFunctions.scalarSub (int16 value) : Stage<Chunk<int16>, Chunk<int16>>))
    elif typeof<'T> = typeof<uint32> then scalarStage<'T> (box (ChunkFunctions.scalarSub (uint32 value) : Stage<Chunk<uint32>, Chunk<uint32>>))
    elif typeof<'T> = typeof<int32> then scalarStage<'T> (box (ChunkFunctions.scalarSub (int32 value) : Stage<Chunk<int32>, Chunk<int32>>))
    elif typeof<'T> = typeof<uint64> then scalarStage<'T> (box (ChunkFunctions.scalarSub (uint64 value) : Stage<Chunk<uint64>, Chunk<uint64>>))
    elif typeof<'T> = typeof<int64> then scalarStage<'T> (box (ChunkFunctions.scalarSub (int64 value) : Stage<Chunk<int64>, Chunk<int64>>))
    elif typeof<'T> = typeof<float32> then scalarStage<'T> (box (ChunkFunctions.scalarSub (float32 value) : Stage<Chunk<float32>, Chunk<float32>>))
    elif typeof<'T> = typeof<float> then scalarStage<'T> (box (ChunkFunctions.scalarSub value : Stage<Chunk<float>, Chunk<float>>))
    else unsupportedScalarArithmetic<'T> ()

let private scalarDivFromDouble<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) : Stage<Chunk<'T>, Chunk<'T>> =
    if typeof<'T> = typeof<uint8> then scalarStage<'T> (box (ChunkFunctions.scalarDiv (uint8 value) : Stage<Chunk<uint8>, Chunk<uint8>>))
    elif typeof<'T> = typeof<int8> then scalarStage<'T> (box (ChunkFunctions.scalarDiv (int8 value) : Stage<Chunk<int8>, Chunk<int8>>))
    elif typeof<'T> = typeof<uint16> then scalarStage<'T> (box (ChunkFunctions.scalarDiv (uint16 value) : Stage<Chunk<uint16>, Chunk<uint16>>))
    elif typeof<'T> = typeof<int16> then scalarStage<'T> (box (ChunkFunctions.scalarDiv (int16 value) : Stage<Chunk<int16>, Chunk<int16>>))
    elif typeof<'T> = typeof<uint32> then scalarStage<'T> (box (ChunkFunctions.scalarDiv (uint32 value) : Stage<Chunk<uint32>, Chunk<uint32>>))
    elif typeof<'T> = typeof<int32> then scalarStage<'T> (box (ChunkFunctions.scalarDiv (int32 value) : Stage<Chunk<int32>, Chunk<int32>>))
    elif typeof<'T> = typeof<uint64> then scalarStage<'T> (box (ChunkFunctions.scalarDiv (uint64 value) : Stage<Chunk<uint64>, Chunk<uint64>>))
    elif typeof<'T> = typeof<int64> then scalarStage<'T> (box (ChunkFunctions.scalarDiv (int64 value) : Stage<Chunk<int64>, Chunk<int64>>))
    elif typeof<'T> = typeof<float32> then scalarStage<'T> (box (ChunkFunctions.scalarDiv (float32 value) : Stage<Chunk<float32>, Chunk<float32>>))
    elif typeof<'T> = typeof<float> then scalarStage<'T> (box (ChunkFunctions.scalarDiv value : Stage<Chunk<float>, Chunk<float>>))
    else unsupportedScalarArithmetic<'T> ()

let addScalar<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) : Stage<Chunk<'T>, Chunk<'T>> =
    addScalarFromDouble<'T> value
let subScalar<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) : Stage<Chunk<'T>, Chunk<'T>> =
    subScalarFromDouble<'T> value
let mulScalar<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) : Stage<Chunk<'T>, Chunk<'T>> =
    mulScalarFromDouble<'T> value
let divScalar<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) : Stage<Chunk<'T>, Chunk<'T>> =
    divScalarFromDouble<'T> value
let scalarAdd<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) : Stage<Chunk<'T>, Chunk<'T>> =
    addScalarFromDouble<'T> value
let scalarSub<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) : Stage<Chunk<'T>, Chunk<'T>> =
    scalarSubFromDouble<'T> value
let scalarMul<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) : Stage<Chunk<'T>, Chunk<'T>> =
    mulScalarFromDouble<'T> value
let scalarDiv<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) : Stage<Chunk<'T>, Chunk<'T>> =
    scalarDivFromDouble<'T> value
let inline addPair<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType and 'T: (static member ( + ) : 'T * 'T -> 'T)> = ChunkFunctions.add<'T>
let inline subPair<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType and 'T: (static member ( - ) : 'T * 'T -> 'T)> = ChunkFunctions.subtract<'T>
let inline mulPair<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType and 'T: (static member ( * ) : 'T * 'T -> 'T)> = ChunkFunctions.multiply<'T>
let inline divPair<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType and 'T: (static member ( / ) : 'T * 'T -> 'T)> = ChunkFunctions.divide<'T>
let inline maxOfPair<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.maximum<'T>
let inline minOfPair<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.minimum<'T>
let equal<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.equal<'T>
let notEqual<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.notEqual<'T>
let greater<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.greater<'T>
let greaterEqual<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.greaterEqual<'T>
let less<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.less<'T>
let lessEqual<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.lessEqual<'T>
let maskAnd = ChunkFunctions.maskAnd
let maskOr = ChunkFunctions.maskOr
let maskXor = ChunkFunctions.maskXor
let maskNot = ChunkFunctions.maskNot
let convolveFixedKernel<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel =
    ChunkFunctions.convolveFixedKernelNativeParallel<'T> kernel stackProcessingWorkers
let convolveX<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel =
    ChunkFunctions.convolveNativeXParallelCollect<'T> kernel stackProcessingWorkers
let convolveY<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel =
    ChunkFunctions.convolveNativeYParallelCollect<'T> kernel stackProcessingWorkers
let convolveZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> kernel =
    ChunkFunctions.convolveNativeZParallelCollect<'T> kernel stackProcessingWorkers
let finiteDiffX<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> order =
    ChunkFunctions.finiteDiffNativeXParallelCollect<'T> order stackProcessingWorkers
let finiteDiffY<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> order =
    ChunkFunctions.finiteDiffNativeYParallelCollect<'T> order stackProcessingWorkers
let finiteDiffZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> order =
    ChunkFunctions.finiteDiffNativeZParallelCollect<'T> order stackProcessingWorkers
let separableConvolve<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> xKernel yKernel zKernel =
    ChunkFunctions.separableConvolveNativeParallelCollect<'T> xKernel yKernel zKernel stackProcessingWorkers
let boxFilter<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> radiusX radiusY radiusZ =
    ChunkFunctions.boxFilterNativeParallelCollectXYZ<'T> radiusX radiusY radiusZ stackProcessingWorkers
let private defaultGaussianWindowSize sigma =
    if sigma <= 0.0 then
        invalidArg "sigma" $"Gaussian filter expects positive sigma, got {sigma}."
    let raw = int (Math.Ceiling(2.0 * sigma + 1.0))
    if raw % 2 = 0 then raw + 1 else raw

let private radiusFromOddWindowSize name (windowSize: uint) =
    if windowSize > uint Int32.MaxValue then
        invalidArg name $"Gaussian filter window size is too large: {windowSize}."
    let windowSize = int windowSize
    if windowSize <= 0 then
        invalidArg name $"Gaussian filter expects a positive odd window size, got {windowSize}."
    if windowSize % 2 = 0 then
        invalidArg name $"Gaussian filter expects an odd window size, got {windowSize}."
    windowSize / 2

let private gaussianWindowSizeOrDefault sigma windowSize =
    windowSize |> Option.defaultWith (fun () -> uint (defaultGaussianWindowSize sigma))

let gaussianFilter<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> sigma windowSize =
    let radius = windowSize |> gaussianWindowSizeOrDefault sigma |> radiusFromOddWindowSize "windowSize"
    ChunkFunctions.gaussianFilterNativeParallelCollect<'T> sigma radius stackProcessingWorkers
let gaussianFilterXYZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> sigmaX windowSizeX sigmaY windowSizeY sigmaZ windowSizeZ =
    let radiusX = windowSizeX |> gaussianWindowSizeOrDefault sigmaX |> radiusFromOddWindowSize "windowSizeX"
    let radiusY = windowSizeY |> gaussianWindowSizeOrDefault sigmaY |> radiusFromOddWindowSize "windowSizeY"
    let radiusZ = windowSizeZ |> gaussianWindowSizeOrDefault sigmaZ |> radiusFromOddWindowSize "windowSizeZ"
    ChunkFunctions.gaussianFilterNativeParallelCollectXYZ<'T> sigmaX radiusX sigmaY radiusY sigmaZ radiusZ stackProcessingWorkers
let gradientVector sigma radius =
    ChunkFunctions.gradientVectorNativeParallelCollect stackProcessingWorkers sigma radius
let gradientVectorXYZ sigmaX radiusX sigmaY radiusY sigmaZ radiusZ =
    ChunkFunctions.gradientVectorNativeParallelCollectXYZ stackProcessingWorkers sigmaX radiusX sigmaY radiusY sigmaZ radiusZ
let gradientMagnitudeSquared sigma windowSize =
    let radius = windowSize |> gaussianWindowSizeOrDefault sigma |> radiusFromOddWindowSize "windowSize"
    ChunkFunctions.gradientMagnitudeSquaredNativeParallelCollect sigma radius stackProcessingWorkers
let gradientMagnitudeSquaredXYZ sigmaX windowSizeX sigmaY windowSizeY sigmaZ windowSizeZ =
    let radiusX = windowSizeX |> gaussianWindowSizeOrDefault sigmaX |> radiusFromOddWindowSize "windowSizeX"
    let radiusY = windowSizeY |> gaussianWindowSizeOrDefault sigmaY |> radiusFromOddWindowSize "windowSizeY"
    let radiusZ = windowSizeZ |> gaussianWindowSizeOrDefault sigmaZ |> radiusFromOddWindowSize "windowSizeZ"
    ChunkFunctions.gradientMagnitudeSquaredNativeParallelCollectXYZ sigmaX radiusX sigmaY radiusY sigmaZ radiusZ stackProcessingWorkers
let hessianUpper sigma radius =
    ChunkFunctions.hessianUpperNativeParallelCollect sigma radius stackProcessingWorkers
let hessianUpperXYZ sigmaX radiusX sigmaY radiusY sigmaZ radiusZ =
    ChunkFunctions.hessianUpperNativeParallelCollectXYZ sigmaX radiusX sigmaY radiusY sigmaZ radiusZ stackProcessingWorkers
let laplacian sigma windowSize =
    let radius = windowSize |> gaussianWindowSizeOrDefault sigma |> radiusFromOddWindowSize "windowSize"
    ChunkFunctions.laplacianNativeParallelCollect sigma radius stackProcessingWorkers
let laplacianXYZ sigmaX windowSizeX sigmaY windowSizeY sigmaZ windowSizeZ =
    let radiusX = windowSizeX |> gaussianWindowSizeOrDefault sigmaX |> radiusFromOddWindowSize "windowSizeX"
    let radiusY = windowSizeY |> gaussianWindowSizeOrDefault sigmaY |> radiusFromOddWindowSize "windowSizeY"
    let radiusZ = windowSizeZ |> gaussianWindowSizeOrDefault sigmaZ |> radiusFromOddWindowSize "windowSizeZ"
    ChunkFunctions.laplacianNativeParallelCollectXYZ sigmaX radiusX sigmaY radiusY sigmaZ radiusZ stackProcessingWorkers
let sobelMagnitude () =
    ChunkFunctions.sobelMagnitudeNativeParallelCollect stackProcessingWorkers
let sobelX<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () =
    ChunkFunctions.sobelXNativeParallelCollect<'T> stackProcessingWorkers
let sobelY<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () =
    ChunkFunctions.sobelYNativeParallelCollect<'T> stackProcessingWorkers
let sobelZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () =
    ChunkFunctions.sobelZNativeParallelCollect<'T> stackProcessingWorkers
let medianUInt8 radius =
    ChunkFunctions.medianNativeNthElementUInt8ParallelCollect radius stackProcessingWorkers
let medianUInt16 radius =
    ChunkFunctions.medianNativeNthElementUInt16ParallelCollect radius stackProcessingWorkers
let medianInt32 radius =
    ChunkFunctions.medianNativeNthElementInt32ParallelCollect radius stackProcessingWorkers
let medianFloat32 radius =
    ChunkFunctions.medianNativeNthElementFloat32ParallelCollect radius stackProcessingWorkers
let addNormalNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> mean std =
    ChunkFunctions.addNormalNoise<'T> mean std
let addSaltAndPepperNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> probability pepper salt =
    ChunkFunctions.addSaltAndPepperNoise<'T> probability pepper salt
let addPoissonNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> lambda =
    ChunkFunctions.addPoissonNoise<'T> lambda
let binaryDilate = ChunkFunctions.binaryDilateZonohedral
let binaryErode = ChunkFunctions.binaryErodeZonohedral
let binaryOpening = ChunkFunctions.binaryOpeningZonohedral
let binaryClosing = ChunkFunctions.binaryClosingZonohedral
let binaryWhiteTopHat = ChunkFunctions.binaryWhiteTopHatZonohedral
let binaryWhiteTopHatWindowed = ChunkFunctions.binaryWhiteTopHatZonohedralParallel
let binaryBlackTopHat = ChunkFunctions.binaryBlackTopHatZonohedral
let binaryBlackTopHatWindowed = ChunkFunctions.binaryBlackTopHatZonohedralParallel
let binaryGradient = ChunkFunctions.binaryGradientZonohedral
let binaryGradientWindowed = ChunkFunctions.binaryGradientZonohedralParallel
let binaryContour = ChunkFunctions.binaryContourZonohedral
let binaryContourWindowed = ChunkFunctions.binaryContourZonohedralParallel

type ObjectConnectivity = StackObjects.ObjectConnectivity
type ObjectBounds = StackObjects.ObjectBounds
type StreamedObject = StackObjects.StreamedObject
type ObjectStream<'T> = StackObjects.ObjectStream<'T>
type ObjectMeasurements = StackObjects.ObjectMeasurements
type ObjectSizeStats = StackObjects.ObjectSizeStats
let streamConnectedObjects<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> connectivity =
    StackObjects.streamConnectedObjectsChunk<'T> connectivity
let objectSource = StackObjects.objectSource
let readObjects<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackObjects.readObjects<'T>
let paintObjects = StackObjects.paintObjectsChunk
let paintObjectsCropped<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackObjects.paintObjectsCroppedChunk<'T>
let writeObjects<'T> = StackObjects.writeObjects<'T>
let measureObjects<'T> = StackObjects.measureObjects<'T>
let objectSizes = StackObjects.objectSizes
let objectSizeStats = StackObjects.objectSizeStats
let histogram = StackObjects.histogram
let removeSmallObjects = StackObjects.removeSmallObjectsChunk
let fillSmallHoles = StackObjects.fillSmallHolesChunk

type Point3D = StackMesh.Point3D
type Triangle = StackMesh.Triangle
type TriangleSet = StackMesh.TriangleSet
let marchingCubes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> surfaceValue =
    StackMesh.marchingCubesChunk<'T> surfaceValue
let objectSurfaceArea = StackMesh.objectSurfaceArea
let writeMesh = StackMesh.writeMesh
let meshFilePath = StackMesh.meshFilePath

let dogKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.dogKeypointsChunk<'T>
let logBlobKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.logBlobKeypointsChunk<'T>
let hessianKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> sigma responseKind threshold stride =
    StackPoints.hessianKeypointsChunk<'T> sigma responseKind threshold stride
let harris3DKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> sigma rho k threshold stride =
    StackPoints.harris3DKeypointsChunk<'T> sigma rho k threshold stride
let forstner3DKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> sigma rho threshold stride =
    StackPoints.forstner3DKeypointsChunk<'T> sigma rho threshold stride
let phaseCongruencyKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.phaseCongruencyKeypointsChunk<'T>
let siftKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> sigma0 scaleFactor scaleLevels contrastThreshold stride =
    StackPoints.siftKeypointsChunk<'T> sigma0 scaleFactor scaleLevels contrastThreshold stride

type BiasPolynomialTerm = StackBias.BiasPolynomialTerm
type BiasPolynomialModel = StackBias.BiasPolynomialModel
let fitBiasModel<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> order =
    StackBias.fitBiasModelChunk<'T> order
let fitBiasModelMasked<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> order =
    StackBias.fitBiasModelChunkMasked<'T> order
let correctBias<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> model =
    StackBias.correctBiasChunk<'T> model
let correctBiasMasked<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> model =
    StackBias.correctBiasChunkMasked<'T> model
let serialPolynomialBiasCorrect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> order =
    StackBias.serialPolynomialBiasCorrectChunk<'T> order

type SerialSliceTransform = StackSerialSections.SerialSliceTransform
type SerialSliceManifest = StackSerialSections.SerialSliceManifest
type SerialVolumeGeometry = StackSerialSections.SerialVolumeGeometry
let serialIdentityManifest = StackSerialSections.serialIdentityManifest
let serialEstTrans<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> searchRadius method scale pixelFraction =
    StackSerialSections.serialEstTransChunk<'T> searchRadius method scale pixelFraction
let serialEstBoundingBox<'T when 'T: equality> : Stage<Chunk<'T> * SerialSliceManifest, SerialVolumeGeometry> =
    StackSerialSections.serialEstBoundingBoxChunk<'T>
let private serialBackgroundFromDouble<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (value: double) =
    if typeof<'T> = typeof<uint8> then unbox<'T> (box (uint8 value))
    elif typeof<'T> = typeof<int8> then unbox<'T> (box (int8 value))
    elif typeof<'T> = typeof<uint16> then unbox<'T> (box (uint16 value))
    elif typeof<'T> = typeof<int16> then unbox<'T> (box (int16 value))
    elif typeof<'T> = typeof<uint32> then unbox<'T> (box (uint32 value))
    elif typeof<'T> = typeof<int32> then unbox<'T> (box (int32 value))
    elif typeof<'T> = typeof<uint64> then unbox<'T> (box (uint64 value))
    elif typeof<'T> = typeof<int64> then unbox<'T> (box (int64 value))
    elif typeof<'T> = typeof<float32> then unbox<'T> (box (float32 value))
    elif typeof<'T> = typeof<float> then unbox<'T> (box value)
    else invalidArg "T" $"serialApplyTrans background supports real numeric chunks, got {typeof<'T>.Name}."
let serialApplyTrans<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (background: double) geometry =
    StackSerialSections.serialApplyTransChunk<'T> (serialBackgroundFromDouble<'T> background) geometry
let serialTrans<'T when 'T: equality> = StackSerialSections.serialTransChunk<'T>
let serialTransManifest<'T when 'T: equality> = StackSerialSections.serialTransManifestChunk<'T>
let serialApplyManifestInBoundingBox<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackSerialSections.serialApplyManifestInBoundingBoxChunk<'T>

type ImageGeom = StackAffineResampler.ImageGeom
let indexToPhysical = StackAffineResampler.indexToPhysical
let physicalToContIndex = StackAffineResampler.physicalToContIndex
let requiredChunksForSliceTrilinear = StackAffineResampler.requiredChunksForSliceTrilinear
let resampleAffineSlices<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackAffineResampler.resampleAffineChunkSlices<'T>
let resampleAffine<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackAffineResampler.resampleAffineChunk<'T>
