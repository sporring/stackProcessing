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
type VectorChunk<'T when 'T: equality> = StackCore.VectorChunk<'T>
type HistogramBinning = StackCore.HistogramBinning
type Histogram<'T when 'T: comparison> = StackCore.Histogram<'T>
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

let showChartData = StackCharts.showChartData
let showChartDataWithLabels = StackCharts.showChartDataWithLabels
let showChart = StackCharts.showChart
let showChartWithLabels = StackCharts.showChartWithLabels
let showChartXY = StackCharts.showChartXY
let showChartXYWithLabels = StackCharts.showChartXYWithLabels
let showChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackCharts.showChunk<'T>
let showChunkWithLabels<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackCharts.showChunkWithLabels<'T>

type FileInfo = StackIO.FileInfo
type ChunkInfo = StackIO.ChunkInfo
let getFileInfo = StackIO.getFileInfo
let getStackDepth = StackIO.getStackDepth
let getStackInfo = StackIO.getStackInfo
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

let readChunkSlices<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readChunkSlices<'T>
let readChunkSlicesRandom<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readChunkSlicesRandom<'T>
let readChunkSlicesRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readChunkSlicesRange<'T>
let readChunkVolume<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readChunkVolume<'T>
let readColorChunkSlices = StackIO.readColorChunkSlices
let readColorChunkSlicesRandom = StackIO.readColorChunkSlicesRandom
let readColorChunkSlicesRange = StackIO.readColorChunkSlicesRange
let writeChunkSlices<'T when 'T: equality> = StackIO.writeChunkSlices<'T>
let writeColorChunkSlices = StackIO.writeColorChunkSlices
let readZarrChunkSlicesRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readZarrChunkSlicesRange<'T>
let writeZarrChunkSlices<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.writeZarrChunkSlices<'T>
let writeZarrChunkSlicesWithCompression<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.writeZarrChunkSlicesWithCompression<'T>
let writeZarrComplex64InterleavedFloat32 = StackIO.writeZarrComplex64InterleavedFloat32
let writeZarrComplex64InterleavedFloat32WithCompression = StackIO.writeZarrComplex64InterleavedFloat32WithCompression
let fftZComplex64InterleavedZarrTiles = StackIO.fftZComplex64InterleavedZarrTiles

type Position3D<'T> = StackPoints.Position3D<'T>
type CoordinatePoint = StackPoints.CoordinatePoint
type PointSet = StackPoints.PointSet
type VectorizedMatrix = StackPoints.VectorizedMatrix
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
let affineToMatrix = StackRegistration.affineToMatrix
let matrixToAffine = StackRegistration.matrixToAffine
let affineRegistration = StackRegistration.affineRegistration
let affineRegistrationMatrices = StackRegistration.affineRegistrationMatrices
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

let chunkZero<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkZero<'T>
let chunkCoordinateX<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkCoordinateX<'T>
let chunkCoordinateY<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkCoordinateY<'T>
let chunkCoordinateZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkCoordinateZ<'T>
let chunkPolygonMask = ChunkFunctions.chunkPolygonMask
let chunkEuler2DTransformPath = ChunkFunctions.euler2DTransformPath
let chunkCreateByEuler2DTransformFromChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.createByEuler2DTransformFromChunk<'T>
let chunkRepeat<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkRepeat<'T>
let chunkRepeatStage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkRepeatStage<'T>
let chunkPad<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.pad<'T>
let chunkCrop<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.crop<'T>
let chunkSqueeze<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.squeeze<'T>
let chunkConcatenateAlong<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.concatenateAlong<'T>
let chunkPermuteAxes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.permuteAxes<'T>
let chunkResample2DNative<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.resample2DNative<'T>
let chunkEuler2DTransformNative<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.euler2DTransformNative<'T>
let chunkEuler2DRotateNative<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.euler2DRotateNative<'T>
let chunkResize<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkResize<'T>
let chunkResample<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkResample<'T>
let chunkShow<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.show<'T>

let chunkSignedDistanceBandNativeParallelCollect = ChunkFunctions.signedDistanceBandNativeParallelCollect
let chunkConnectedComponentsSauf3DUInt8UInt32 = ChunkFunctions.connectedComponentsSauf3DUInt8UInt32
let chunkConnectedComponentsSauf3DUInt8UInt32ParallelCollect = ChunkFunctions.connectedComponentsSauf3DUInt8UInt32ParallelCollect
let chunkConnectedComponentsSauf3DUInt8 = ChunkFunctions.connectedComponentsSauf3DUInt8

let chunkToVectorImage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.toVectorImage<'T>
let chunkVectorElement<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.vectorElement<'T>
let chunkVectorRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.vectorRange<'T>
let chunkAppendVectorElement<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.appendVectorElement<'T>
let chunkVectorMapElements = ChunkFunctions.vectorMapElements
let chunkVector3ToColor = ChunkFunctions.vector3ToColor
let chunkColorToVector3 = ChunkFunctions.colorToVector3
let chunkVectorDot = ChunkFunctions.vectorDot
let chunkVectorMagnitude = ChunkFunctions.vectorMagnitude
let chunkVectorCross3D = ChunkFunctions.vectorCross3D
let chunkVectorAngleTo = ChunkFunctions.vectorAngleTo
let chunkVectorDotFloat32 = ChunkFunctions.vectorDotFloat32
let chunkVectorMagnitudeFloat32 = ChunkFunctions.vectorMagnitudeFloat32
let chunkVector3ToColorFloat32 = ChunkFunctions.vector3ToColorFloat32
let chunkVectorAngleToFloat32 = ChunkFunctions.vectorAngleToFloat32
let chunkConvolveVectorComponentsFloat32NativeParallelCollect = ChunkFunctions.convolveVectorComponentsFloat32NativeParallelCollect
let chunkPCA = ChunkFunctions.PCA
let chunkPCAFloat32 = ChunkFunctions.PCAFloat32
let chunkStructureTensorNativeParallelCollect = ChunkFunctions.structureTensorNativeParallelCollect
let chunkSelectGroupedVectorOutput = ChunkFunctions.selectGroupedVectorOutput

let chunkToComplex64 = ChunkFunctions.toComplex64
let chunkPolarToComplex64 = ChunkFunctions.polarToComplex64
let chunkComplex64Real = ChunkFunctions.complex64Real
let chunkComplex64Imag = ChunkFunctions.complex64Imag
let chunkComplex64Modulus = ChunkFunctions.complex64Modulus
let chunkComplex64Argument = ChunkFunctions.complex64Argument
let chunkComplex64Conjugate = ChunkFunctions.complex64Conjugate
let chunkFftXYFloat32ToComplex64Interleaved = ChunkFunctions.fftXYFloat32ToComplex64Interleaved
let chunkFftXYFloat32ToComplex64InterleavedParallelCollect = ChunkFunctions.fftXYFloat32ToComplex64InterleavedParallelCollect
let chunkInvFftXYComplex64InterleavedToFloat32 = ChunkFunctions.invFftXYComplex64InterleavedToFloat32
let chunkInvFftXYComplex64InterleavedToFloat32ParallelCollect = ChunkFunctions.invFftXYComplex64InterleavedToFloat32ParallelCollect
let chunkFftShift3DComplex64Interleaved = ChunkFunctions.fftShift3DComplex64Interleaved

let chunkHistogram<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.histogramReducer<'T>
let chunkHistogramDense<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.histogramDenseReducer<'T>
let chunkHistogramLeftEdges<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.histogramLeftEdgesReducer<'T>
let chunkHistogramFixedBins<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.histogramFixedBinsReducer<'T>
let chunkHistogramEqualizationDense<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.histogramEqualizationDense<'T>
let chunkHistogramEqualizationLeftEdges<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.histogramEqualizationLeftEdges<'T>
let chunkHistogramEqualizationSparse<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.histogramEqualizationSparse<'T>
let chunkHistogramEqualization<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.histogramEqualization<'T>
let chunkQuantiles = ChunkFunctions.quantiles
let chunkComputeStats<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () =
    ChunkFunctions.computeStats<'T> ()
let chunkVolume = ChunkFunctions.volume
let chunkSumProjection<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkSumProjection<'T>

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

let chunkThresholdRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (lower: double) (upper: double) =
    ChunkFunctions.thresholdRange<'T> lower upper
let chunkAbsFloat32 = ChunkFunctions.absFloat32
let chunkSqrtFloat32 = ChunkFunctions.sqrtFloat32
let chunkSquareFloat32 = ChunkFunctions.squareFloat32
let chunkClamp<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.clamp<'T>
let chunkShiftScale<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.shiftScale<'T>
let chunkIntensityWindow<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.intensityWindow<'T>
let chunkCastToUInt8<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.castToUInt8<'T>
let chunkCastToFloat32<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.castToFloat32<'T>
let chunkCastFromFloat32<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.castFromFloat32<'T>
let chunkCast<'S, 'T when 'S: equality and 'S: (new: unit -> 'S) and 'S: struct and 'S :> ValueType
                         and 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'S>, Chunk<'T>> =
    if typeof<'S> = typeof<'T> then
        unbox (box (StackCore.identityStage "chunkCast.identity"))
    elif typeof<'T> = typeof<float32> then
        unbox (box (ChunkFunctions.castToFloat32<'S>))
    elif typeof<'T> = typeof<uint8> then
        unbox (box (ChunkFunctions.castToUInt8<'S>))
    elif typeof<'S> = typeof<float32> then
        unbox (box (ChunkFunctions.castFromFloat32<'T>))
    else
        ChunkFunctions.castToFloat32<'S> --> ChunkFunctions.castFromFloat32<'T>

let inline chunkImageAddScalar value = ChunkFunctions.addScalar value
let inline chunkImageSubScalar value = ChunkFunctions.subScalar value
let inline chunkImageMulScalar value = ChunkFunctions.mulScalar value
let inline chunkImageDivScalar value = ChunkFunctions.divScalar value
let inline chunkScalarAddImage value = ChunkFunctions.scalarAdd value
let inline chunkScalarSubImage value = ChunkFunctions.scalarSub value
let inline chunkScalarMulImage value = ChunkFunctions.scalarMul value
let inline chunkScalarDivImage value = ChunkFunctions.scalarDiv value
let inline chunkAddPair<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType and 'T: (static member ( + ) : 'T * 'T -> 'T)> = ChunkFunctions.add<'T>
let inline chunkSubPair<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType and 'T: (static member ( - ) : 'T * 'T -> 'T)> = ChunkFunctions.subtract<'T>
let inline chunkMulPair<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType and 'T: (static member ( * ) : 'T * 'T -> 'T)> = ChunkFunctions.multiply<'T>
let inline chunkDivPair<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType and 'T: (static member ( / ) : 'T * 'T -> 'T)> = ChunkFunctions.divide<'T>
let inline chunkMaxOfPair<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.maximum<'T>
let inline chunkMinOfPair<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.minimum<'T>
let chunkEqual<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.equal<'T>
let chunkNotEqual<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.notEqual<'T>
let chunkGreater<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.greater<'T>
let chunkGreaterEqual<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.greaterEqual<'T>
let chunkLess<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.less<'T>
let chunkLessEqual<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.lessEqual<'T>
let chunkMaskAnd = ChunkFunctions.maskAnd
let chunkMaskOr = ChunkFunctions.maskOr
let chunkMaskXor = ChunkFunctions.maskXor
let chunkMaskNot = ChunkFunctions.maskNot

let convolveNativeXParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.convolveNativeXParallelCollect<'T>
let convolveNativeYParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.convolveNativeYParallelCollect<'T>
let convolveNativeZParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.convolveNativeZParallelCollect<'T>
let finiteDiffKernel1D = ChunkFunctions.finiteDiffKernel1D
let finiteDiffNativeXParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.finiteDiffNativeXParallelCollect<'T>
let finiteDiffNativeYParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.finiteDiffNativeYParallelCollect<'T>
let finiteDiffNativeZParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.finiteDiffNativeZParallelCollect<'T>
let separableConvolveNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.separableConvolveNativeParallelCollect<'T>
let boxFilterNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.boxFilterNativeParallelCollect<'T>
let boxFilterNativeParallelCollectXYZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.boxFilterNativeParallelCollectXYZ<'T>
let gaussianFilterNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.gaussianFilterNativeParallelCollect<'T>
let gaussianFilterNativeParallelCollectXYZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.gaussianFilterNativeParallelCollectXYZ<'T>
let gradientVectorNativeParallelCollect = ChunkFunctions.gradientVectorNativeParallelCollect
let gradientVectorNativeParallelCollectXYZ = ChunkFunctions.gradientVectorNativeParallelCollectXYZ
let gradientMagnitudeNativeParallelCollect = ChunkFunctions.gradientMagnitudeNativeParallelCollect
let gradientMagnitudeNativeParallelCollectXYZ = ChunkFunctions.gradientMagnitudeNativeParallelCollectXYZ
let hessianUpperNativeParallelCollect = ChunkFunctions.hessianUpperNativeParallelCollect
let hessianUpperNativeParallelCollectXYZ = ChunkFunctions.hessianUpperNativeParallelCollectXYZ
let laplacianNativeParallelCollect = ChunkFunctions.laplacianNativeParallelCollect
let laplacianNativeParallelCollectXYZ = ChunkFunctions.laplacianNativeParallelCollectXYZ
let sobelMagnitudeNativeParallelCollect = ChunkFunctions.sobelMagnitudeNativeParallelCollect
let sobelXNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.sobelXNativeParallelCollect<'T>
let sobelYNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.sobelYNativeParallelCollect<'T>
let sobelZNativeParallelCollect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.sobelZNativeParallelCollect<'T>
let chunkMedianNativeNthElementUInt8 = ChunkFunctions.medianNativeNthElementUInt8
let chunkMedianNativeNthElementUInt8ParallelCollect = ChunkFunctions.medianNativeNthElementUInt8ParallelCollect
let chunkMedianNativeNthElementUInt16 = ChunkFunctions.medianNativeNthElementUInt16
let chunkMedianNativeNthElementUInt16ParallelCollect = ChunkFunctions.medianNativeNthElementUInt16ParallelCollect
let chunkMedianNativeNthElementInt32 = ChunkFunctions.medianNativeNthElementInt32
let chunkMedianNativeNthElementInt32ParallelCollect = ChunkFunctions.medianNativeNthElementInt32ParallelCollect
let chunkMedianNativeNthElementFloat32 = ChunkFunctions.medianNativeNthElementFloat32
let chunkMedianNativeNthElementFloat32ParallelCollect = ChunkFunctions.medianNativeNthElementFloat32ParallelCollect

let chunkAddNormalNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.addNormalNoise<'T>
let chunkAddSaltAndPepperNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.addSaltAndPepperNoise<'T>
let chunkAddShotNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.addShotNoise<'T>

let chunkBinaryDilateZonohedral = ChunkFunctions.binaryDilateZonohedral
let chunkBinaryErodeZonohedral = ChunkFunctions.binaryErodeZonohedral
let chunkBinaryOpeningZonohedral = ChunkFunctions.binaryOpeningZonohedral
let chunkBinaryClosingZonohedral = ChunkFunctions.binaryClosingZonohedral
let chunkBinaryWhiteTopHatZonohedral = ChunkFunctions.binaryWhiteTopHatZonohedral
let chunkBinaryWhiteTopHatZonohedralParallel = ChunkFunctions.binaryWhiteTopHatZonohedralParallel
let chunkBinaryBlackTopHatZonohedral = ChunkFunctions.binaryBlackTopHatZonohedral
let chunkBinaryBlackTopHatZonohedralParallel = ChunkFunctions.binaryBlackTopHatZonohedralParallel
let chunkBinaryGradientZonohedral = ChunkFunctions.binaryGradientZonohedral
let chunkBinaryGradientZonohedralParallel = ChunkFunctions.binaryGradientZonohedralParallel
let chunkBinaryContourZonohedral = ChunkFunctions.binaryContourZonohedral
let chunkBinaryContourZonohedralParallel = ChunkFunctions.binaryContourZonohedralParallel

type ObjectConnectivity = StackObjects.ObjectConnectivity
type ObjectBounds = StackObjects.ObjectBounds
type StreamedObject = StackObjects.StreamedObject
type ObjectMeasurements = StackObjects.ObjectMeasurements
type ObjectSizeStats = StackObjects.ObjectSizeStats
let streamConnectedObjectsChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackObjects.streamConnectedObjectsChunk<'T>
let paintObjectsChunk = StackObjects.paintObjectsChunk
let paintObjectsCroppedChunk = StackObjects.paintObjectsCroppedChunk
let measureObjects = StackObjects.measureObjects
let objectSizes = StackObjects.objectSizes
let objectSizeStats = StackObjects.objectSizeStats
let histogram = StackObjects.histogram
let chunkRemoveSmallObjects = StackObjects.removeSmallObjectsChunk
let chunkFillSmallHoles = StackObjects.fillSmallHolesChunk

type Point3D = StackMesh.Point3D
type Triangle = StackMesh.Triangle
type TriangleSet = StackMesh.TriangleSet
let marchingCubesChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackMesh.marchingCubesChunk<'T>
let surfaceArea = StackMesh.surfaceArea
let writeMesh = StackMesh.writeMesh
let meshFilePath = StackMesh.meshFilePath

let chunkDogKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.dogKeypointsChunk<'T>
let chunkLogBlobKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.logBlobKeypointsChunk<'T>
let chunkHessianKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.hessianKeypointsChunk<'T>
let chunkHarris3DKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.harris3DKeypointsChunk<'T>
let chunkForstner3DKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.forstner3DKeypointsChunk<'T>
let chunkPhaseCongruencyKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.phaseCongruencyKeypointsChunk<'T>
let chunkSiftKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.siftKeypointsChunk<'T>

type BiasPolynomialTerm = StackBias.BiasPolynomialTerm
type BiasPolynomialModel = StackBias.BiasPolynomialModel
let fitBiasModelChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackBias.fitBiasModelChunk<'T>
let fitBiasModelChunkMasked<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackBias.fitBiasModelChunkMasked<'T>
let correctBiasChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackBias.correctBiasChunk<'T>
let correctBiasChunkMasked<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackBias.correctBiasChunkMasked<'T>
let chunkSerialPolynomialBiasCorrect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackBias.serialPolynomialBiasCorrectChunk<'T>

type SerialSliceTransform = StackSerialSections.SerialSliceTransform
type SerialSliceManifest = StackSerialSections.SerialSliceManifest
type SerialVolumeGeometry = StackSerialSections.SerialVolumeGeometry
let serialIdentityManifest = StackSerialSections.serialIdentityManifest
let chunkSerialEstTrans<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackSerialSections.serialEstTransChunk<'T>
let chunkSerialEstBoundingBox<'T when 'T: equality> = StackSerialSections.serialEstBoundingBoxChunk<'T>
let chunkSerialApplyTrans<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackSerialSections.serialApplyTransChunk<'T>
let chunkSerialTransChunk<'T when 'T: equality> = StackSerialSections.serialTransChunk<'T>
let chunkSerialTransManifest<'T when 'T: equality> = StackSerialSections.serialTransManifestChunk<'T>
let chunkSerialApplyManifestInBoundingBox<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackSerialSections.serialApplyManifestInBoundingBoxChunk<'T>

type ImageGeom = StackAffineResampler.ImageGeom
let indexToPhysical = StackAffineResampler.indexToPhysical
let physicalToContIndex = StackAffineResampler.physicalToContIndex
let requiredChunksForSliceTrilinear = StackAffineResampler.requiredChunksForSliceTrilinear
let resampleAffineChunkSlices<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackAffineResampler.resampleAffineChunkSlices<'T>
let resampleAffineChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackAffineResampler.resampleAffineChunk<'T>
