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

let writeZarrComplex64InterleavedFloat32 = StackIO.writeZarrComplex64InterleavedFloat32
let writeZarrComplex64InterleavedFloat32WithCompression = StackIO.writeZarrComplex64InterleavedFloat32WithCompression
let fftZComplex64InterleavedZarrTiles = StackIO.fftZComplex64InterleavedZarrTiles

let read<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readChunkSlices<'T>
let readRandom<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readChunkSlicesRandom<'T>
let readRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readChunkSlicesRange<'T>
let readVolume<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readChunkVolume<'T>
let readColor = StackIO.readColorChunkSlices
let readColorRandom = StackIO.readColorChunkSlicesRandom
let readColorRange = StackIO.readColorChunkSlicesRange
let write<'T when 'T: equality> = StackIO.writeChunkSlices<'T>
let writeColor = StackIO.writeColorChunkSlices
let readZarrRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.readZarrChunkSlicesRange<'T>
let writeZarr<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.writeZarrChunkSlices<'T>
let writeZarrWithCompression<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackIO.writeZarrChunkSlicesWithCompression<'T>

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

let zero<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkZero<'T>
let coordinateX<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkCoordinateX<'T>
let coordinateY<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkCoordinateY<'T>
let coordinateZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkCoordinateZ<'T>
let polygonMask = ChunkFunctions.chunkPolygonMask
let euler2DTransformPath = ChunkFunctions.euler2DTransformPath
let createByEuler2DTransform<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.createByEuler2DTransformFromChunk<'T>
let repeat<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkRepeat<'T>
let repeatStage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkRepeatStage<'T>
let pad<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.pad<'T>
let crop<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.crop<'T>
let squeeze<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.squeeze<'T>
let concatenateAlong<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.concatenateAlong<'T>
let permuteAxes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.permuteAxes<'T>
let resize<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkResize<'T>
let resample<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkResample<'T>
let show<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.show<'T>
let signedDistanceBand = ChunkFunctions.signedDistanceBandNativeParallelCollect
let connectedComponents = ChunkFunctions.connectedComponentsSauf3DUInt8
let connectedComponentsUInt32 () = ChunkFunctions.connectedComponentsSauf3DUInt8UInt32 ()
let connectedComponentsUInt32Windowed = ChunkFunctions.connectedComponentsSauf3DUInt8UInt32ParallelCollect
let toVectorImage<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.toVectorImage<'T>
let vectorElement<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.vectorElement<'T>
let vectorRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.vectorRange<'T>
let appendVectorElement<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.appendVectorElement<'T>
let vectorMapElements = ChunkFunctions.vectorMapElements
let vector3ToColor = ChunkFunctions.vector3ToColor
let colorToVector3 = ChunkFunctions.colorToVector3
let vectorDot = ChunkFunctions.vectorDot
let vectorMagnitude = ChunkFunctions.vectorMagnitude
let vectorCross3D = ChunkFunctions.vectorCross3D
let vectorAngleTo = ChunkFunctions.vectorAngleTo
let vectorDotFloat32 = ChunkFunctions.vectorDotFloat32
let vectorMagnitudeFloat32 = ChunkFunctions.vectorMagnitudeFloat32
let vector3ToColorFloat32 = ChunkFunctions.vector3ToColorFloat32
let vectorAngleToFloat32 = ChunkFunctions.vectorAngleToFloat32
let pca = ChunkFunctions.PCA
let pcaFloat32 = ChunkFunctions.PCAFloat32
let structureTensor = ChunkFunctions.structureTensorNativeParallelCollect
let selectGroupedVectorOutput = ChunkFunctions.selectGroupedVectorOutput
let toComplex64 = ChunkFunctions.toComplex64
let polarToComplex64 = ChunkFunctions.polarToComplex64
let complex64Real = ChunkFunctions.complex64Real
let complex64Imag = ChunkFunctions.complex64Imag
let complex64Modulus = ChunkFunctions.complex64Modulus
let complex64Argument = ChunkFunctions.complex64Argument
let complex64Conjugate = ChunkFunctions.complex64Conjugate
let fft = ChunkFunctions.fftXYFloat32ToComplex64Interleaved
let invFft = ChunkFunctions.invFftXYComplex64InterleavedToFloat32
let fftShift3D = ChunkFunctions.fftShift3DComplex64Interleaved
let imageHistogram<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.histogramReducer<'T>
let imageHistogramDense<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.histogramDenseReducer<'T>
let imageHistogramLeftEdges<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.histogramLeftEdgesReducer<'T>
let imageHistogramFixedBins<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.histogramFixedBinsReducer<'T>
let histogramEqualization<'T when 'T: equality and 'T: comparison and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.histogramEqualization<'T>
let quantiles = ChunkFunctions.quantiles
let computeStats<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () = ChunkFunctions.computeStats<'T> ()
let volume = ChunkFunctions.volume
let sumProjection<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.chunkSumProjection<'T>

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

let finiteDiffKernel1D = ChunkFunctions.finiteDiffKernel1D
let thresholdRange<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (lower: double) (upper: double) =
    ChunkFunctions.thresholdRange<'T> lower upper
let sqrtFloat32 = ChunkFunctions.sqrtFloat32
let absFloat32 = ChunkFunctions.absFloat32
let squareFloat32 = ChunkFunctions.squareFloat32
let clamp<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.clamp<'T>
let shiftScale<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.shiftScale<'T>
let intensityWindow<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.intensityWindow<'T>
let cast<'S, 'T when 'S: equality and 'S: (new: unit -> 'S) and 'S: struct and 'S :> ValueType
                     and 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> : Stage<Chunk<'S>, Chunk<'T>> =
    if typeof<'S> = typeof<'T> then
        unbox (box (StackCore.identityStage "cast.identity"))
    elif typeof<'T> = typeof<float32> then
        unbox (box (ChunkFunctions.castToFloat32<'S>))
    elif typeof<'T> = typeof<uint8> then
        unbox (box (ChunkFunctions.castToUInt8<'S>))
    elif typeof<'S> = typeof<float32> then
        unbox (box (ChunkFunctions.castFromFloat32<'T>))
    else
        ChunkFunctions.castToFloat32<'S> --> ChunkFunctions.castFromFloat32<'T>
let inline addScalar value = ChunkFunctions.addScalar value
let inline subScalar value = ChunkFunctions.subScalar value
let inline mulScalar value = ChunkFunctions.mulScalar value
let inline divScalar value = ChunkFunctions.divScalar value
let inline scalarAdd value = ChunkFunctions.scalarAdd value
let inline scalarSub value = ChunkFunctions.scalarSub value
let inline scalarMul value = ChunkFunctions.scalarMul value
let inline scalarDiv value = ChunkFunctions.scalarDiv value
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
let convolveX<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.convolveNativeXParallelCollect<'T>
let convolveY<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.convolveNativeYParallelCollect<'T>
let convolveZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.convolveNativeZParallelCollect<'T>
let finiteDiffX<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.finiteDiffNativeXParallelCollect<'T>
let finiteDiffY<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.finiteDiffNativeYParallelCollect<'T>
let finiteDiffZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.finiteDiffNativeZParallelCollect<'T>
let separableConvolve<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.separableConvolveNativeParallelCollect<'T>
let boxFilter<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.boxFilterNativeParallelCollect<'T>
let boxFilterXYZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.boxFilterNativeParallelCollectXYZ<'T>
let gaussianFilter<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.gaussianFilterNativeParallelCollect<'T>
let gaussianFilterXYZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.gaussianFilterNativeParallelCollectXYZ<'T>
let gradientVector = ChunkFunctions.gradientVectorNativeParallelCollect
let gradientVectorXYZ = ChunkFunctions.gradientVectorNativeParallelCollectXYZ
let gradientMagnitude = ChunkFunctions.gradientMagnitudeNativeParallelCollect
let gradientMagnitudeXYZ = ChunkFunctions.gradientMagnitudeNativeParallelCollectXYZ
let hessianUpper = ChunkFunctions.hessianUpperNativeParallelCollect
let hessianUpperXYZ = ChunkFunctions.hessianUpperNativeParallelCollectXYZ
let laplacian = ChunkFunctions.laplacianNativeParallelCollect
let laplacianXYZ = ChunkFunctions.laplacianNativeParallelCollectXYZ
let sobelMagnitude = ChunkFunctions.sobelMagnitudeNativeParallelCollect
let sobelX<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.sobelXNativeParallelCollect<'T>
let sobelY<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.sobelYNativeParallelCollect<'T>
let sobelZ<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.sobelZNativeParallelCollect<'T>
let medianUInt8 = ChunkFunctions.medianNativeNthElementUInt8ParallelCollect
let medianUInt16 = ChunkFunctions.medianNativeNthElementUInt16ParallelCollect
let medianInt32 = ChunkFunctions.medianNativeNthElementInt32ParallelCollect
let medianFloat32 = ChunkFunctions.medianNativeNthElementFloat32ParallelCollect
let addNormalNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.addNormalNoise<'T>
let addSaltAndPepperNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.addSaltAndPepperNoise<'T>
let addShotNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = ChunkFunctions.addShotNoise<'T>
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
type ObjectMeasurements = StackObjects.ObjectMeasurements
type ObjectSizeStats = StackObjects.ObjectSizeStats
let streamConnectedObjects<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackObjects.streamConnectedObjectsChunk<'T>
let paintObjects = StackObjects.paintObjectsChunk
let paintObjectsCropped = StackObjects.paintObjectsCroppedChunk
let measureObjects = StackObjects.measureObjects
let objectSizes = StackObjects.objectSizes
let objectSizeStats = StackObjects.objectSizeStats
let histogram = StackObjects.histogram
let removeSmallObjects = StackObjects.removeSmallObjectsChunk
let fillSmallHoles = StackObjects.fillSmallHolesChunk

type Point3D = StackMesh.Point3D
type Triangle = StackMesh.Triangle
type TriangleSet = StackMesh.TriangleSet
let marchingCubes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackMesh.marchingCubesChunk<'T>
let surfaceArea = StackMesh.surfaceArea
let writeMesh = StackMesh.writeMesh
let meshFilePath = StackMesh.meshFilePath

let dogKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.dogKeypointsChunk<'T>
let logBlobKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.logBlobKeypointsChunk<'T>
let hessianKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.hessianKeypointsChunk<'T>
let harris3DKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.harris3DKeypointsChunk<'T>
let forstner3DKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.forstner3DKeypointsChunk<'T>
let phaseCongruencyKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.phaseCongruencyKeypointsChunk<'T>
let siftKeypoints<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackPoints.siftKeypointsChunk<'T>

type BiasPolynomialTerm = StackBias.BiasPolynomialTerm
type BiasPolynomialModel = StackBias.BiasPolynomialModel
let fitBiasModel<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackBias.fitBiasModelChunk<'T>
let fitBiasModelMasked<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackBias.fitBiasModelChunkMasked<'T>
let correctBias<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackBias.correctBiasChunk<'T>
let correctBiasMasked<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackBias.correctBiasChunkMasked<'T>
let serialPolynomialBiasCorrect<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackBias.serialPolynomialBiasCorrectChunk<'T>

type SerialSliceTransform = StackSerialSections.SerialSliceTransform
type SerialSliceManifest = StackSerialSections.SerialSliceManifest
type SerialVolumeGeometry = StackSerialSections.SerialVolumeGeometry
let serialIdentityManifest = StackSerialSections.serialIdentityManifest
let serialEstTrans<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackSerialSections.serialEstTransChunk<'T>
let serialEstBoundingBox<'T when 'T: equality> = StackSerialSections.serialEstBoundingBoxChunk<'T>
let serialApplyTrans<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackSerialSections.serialApplyTransChunk<'T>
let serialTrans<'T when 'T: equality> = StackSerialSections.serialTransChunk<'T>
let serialTransManifest<'T when 'T: equality> = StackSerialSections.serialTransManifestChunk<'T>
let serialApplyManifestInBoundingBox<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackSerialSections.serialApplyManifestInBoundingBoxChunk<'T>

type ImageGeom = StackAffineResampler.ImageGeom
let indexToPhysical = StackAffineResampler.indexToPhysical
let physicalToContIndex = StackAffineResampler.physicalToContIndex
let requiredChunksForSliceTrilinear = StackAffineResampler.requiredChunksForSliceTrilinear
let resampleAffineSlices<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackAffineResampler.resampleAffineChunkSlices<'T>
let resampleAffine<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> = StackAffineResampler.resampleAffineChunk<'T>
