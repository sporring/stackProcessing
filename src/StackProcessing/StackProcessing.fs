module StackProcessing

// open StackCore

// //////////////////// StackCore
type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>
type Profile = SlimPipeline.Profile
type ProfileTransition = SlimPipeline.ProfileTransition
type ResourceOps<'T> = SlimPipeline.ResourceOps<'T>
type StageTimeCoefficients = SlimPipeline.StageTimeCoefficients
type Window<'T> = SlimPipeline.Window<'T>
//type Slice<'S when 'S: equality> = Slice.Slice<'S>
type Image<'S when 'S: equality> = Image.Image<'S>
type ImageFacts = Image.ImageFacts

let source = StackCore.source
let debug = StackCore.debug
let commandLineSource = StackCore.commandLineSource
let zip = StackCore.zip
let (>=>) = StackCore.(>=>)
let (-->) = StackCore.(-->)
let (>=>>) = StackCore.(>=>>)
let (>>=>) = StackCore.(>>=>)
let (>>=>>) = StackCore.(>>=>>)
let teeFst = StackCore.teeFst
let teeSnd = StackCore.teeSnd
let ignoreSingles = StackCore.ignoreSingles
let ignorePairs = StackCore.ignorePairs
let zeroMaker = StackCore.zeroMaker
let sink = StackCore.sink
let sinkList = StackCore.sinkList
let drain = StackCore.drain
let drainList = StackCore.drainList
let drainLast = StackCore.drainLast
let tap = StackCore.tap
let tapIt = StackCore.tapIt

// //////////////////// StackIO
type FileInfo = ImageFunctions.FileInfo
type ChunkInfo = StackIO.ChunkInfo
type Position3D<'T> = StackPoints.Position3D<'T>
type CoordinatePoint = StackPoints.CoordinatePoint
type PointSetChunk = StackPoints.PointSetChunk
type Affine = TinyLinAlg.Affine
type AffineRegistrationOptions = StackRegistration.AffineRegistrationOptions
type AffineRegistrationResult = StackRegistration.AffineRegistrationResult
type ObjectConnectivity = StackObjects.ObjectConnectivity
type ObjectBounds = StackObjects.ObjectBounds
type StreamedObject = StackObjects.StreamedObject
type ObjectMeasurements = StackObjects.ObjectMeasurements
type ObjectSizeStats = StackObjects.ObjectSizeStats
type Point3D = StackMesh.Point3D
type Triangle = StackMesh.Triangle
type MeshChunk = StackMesh.MeshChunk

let getStackDepth = StackIO.getStackDepth
let getStackInfo = StackIO.getStackInfo
let getStackSize = StackIO.getStackSize
let getStackWidth = StackIO.getStackWidth
let getStackHeight = StackIO.getStackHeight
let getFilenames = StackIO.getFilenames
let readFiles<'T when 'T: equality> = StackIO.readFiles<'T>
let readFilesWithShape<'T when 'T: equality> = StackIO.readFilesWithShape<'T>
let readFilePairs<'T when 'T: equality> = StackIO.readFilePairs<'T>
let readFiltered<'T when 'T: equality> = StackIO.readFiltered<'T>
let read<'T when 'T: equality> = StackIO.read<'T>
let readRandom<'T when 'T: equality> = StackIO.readRandom<'T>
let readRange<'T when 'T: equality> = StackIO.readRange<'T>
let getChunkInfo = StackIO.getChunkInfo
let getZarrInfo = StackIO.getZarrInfo
let getNexusInfo = StackIO.getNexusInfo
let getChunkFilename = StackIO.getChunkFilename
let readSlabStacked<'T when 'T: equality> = StackIO.readSlabStacked<'T>
let readSlabAsWindows<'T when 'T: equality> = StackIO.readSlabAsWindows<'T>
let readSlab<'T when 'T: equality> = StackIO.readSlab<'T>
let readZarrSlabStacked<'T when 'T: equality> = StackIO.readZarrSlabStacked<'T>
let readZarrSlab<'T when 'T: equality> = StackIO.readZarrSlab<'T>
let readNexusSlabStacked<'T when 'T: equality> = StackIO.readNexusSlabStacked<'T>
let readNexusSlab<'T when 'T: equality> = StackIO.readNexusSlab<'T>
let readPointSet = StackPoints.readPointSet

let deleteIfExists = StackIO.deleteIfExists
let write = StackIO.write
let writeZarr = StackIO.writeZarr
let writeNexus = StackIO.writeNexus
let writeInSlabs = StackIO.writeInSlabs
let writePointSet = StackPoints.writePointSet
let writeMesh = StackMesh.writeMesh

// //////////////////// StackRegistration
let defaultAffineRegistrationOptions = StackRegistration.defaultAffineRegistrationOptions
let earthMoversDistance = StackRegistration.earthMoversDistance
let transformPointSet = StackRegistration.transformPointSet
let inverseAffine = StackRegistration.inverseAffine
let affineRegistration = StackRegistration.affineRegistration

// //////////////////// StackObjects
let streamConnectedObjects<'T when 'T: equality> = StackObjects.streamConnectedObjects<'T>
let removeSmallObjects = StackObjects.removeSmallObjects
let fillSmallHoles = StackObjects.fillSmallHoles
let paintObjects = StackObjects.paintObjects
let paintObjectsCropped = StackObjects.paintObjectsCropped
let measureObjects = StackObjects.measureObjects
let objectSizeStats = StackObjects.objectSizeStats
let objectSizeHistogram = StackObjects.objectSizeHistogram

// //////////////////// StackAffineResampler
let resampleAffineTrilinearSlices = StackAffineResampler.resampleAffineTrilinearSlices

// //////////////////// StackImageFunctions
type ImageStats = ImageFunctions.ImageStats

let cast<'S,'T when 'S: equality and 'T: equality> = StackImageFunctions.cast<'S,'T>
let add = StackImageFunctions.add
let addPair = StackImageFunctions.addPair
let inline scalarAddImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> = StackImageFunctions.scalarAddImage<^T>
let inline imageAddScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> = StackImageFunctions.imageAddScalar<^T>
let sub = StackImageFunctions.sub
let subPair = StackImageFunctions.subPair
let inline scalarSubImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> = StackImageFunctions.scalarSubImage<^T>
let inline imageSubScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> = StackImageFunctions.imageSubScalar<^T>
let mul = StackImageFunctions.mul
let mulPair = StackImageFunctions.mulPair
let inline scalarMulImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> = StackImageFunctions.scalarMulImage<^T>
let inline imageMulScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> = StackImageFunctions.imageMulScalar<^T>
let div = StackImageFunctions.div
let divPair = StackImageFunctions.divPair
let inline scalarDivImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> = StackImageFunctions.scalarDivImage<^T>
let inline imageDivScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> = StackImageFunctions.imageDivScalar<^T>
let maxOfPair = StackImageFunctions.maxOfPair
let minOfPair = StackImageFunctions.minOfPair
let getMinMax = StackImageFunctions.getMinMax
let failTypeMismatch<'T> = StackImageFunctions.failTypeMismatch<'T>
let abs<'T when 'T: equality>= StackImageFunctions.abs<'T>
let acos<'T when 'T: equality>= StackImageFunctions.acos<'T>
let asin<'T when 'T: equality>= StackImageFunctions.asin<'T>
let atan<'T when 'T: equality>= StackImageFunctions.atan<'T>
let cos<'T when 'T: equality>= StackImageFunctions.cos<'T>
let sin<'T when 'T: equality>= StackImageFunctions.sin<'T>
let tan<'T when 'T: equality>= StackImageFunctions.tan<'T>
let exp<'T when 'T: equality>= StackImageFunctions.exp<'T>
let log10<'T when 'T: equality>= StackImageFunctions.log10<'T>
let log<'T when 'T: equality>= StackImageFunctions.log<'T>
let round<'T when 'T: equality>= StackImageFunctions.round<'T>
let sqrt<'T when 'T: equality>= StackImageFunctions.sqrt<'T>
let sqrtWindowed<'T when 'T: equality> = StackImageFunctions.sqrtWindowed<'T>
let square<'T when 'T: equality>= StackImageFunctions.square<'T>
let clamp<'T when 'T: equality> lower upper : Stage<Image<'T>, Image<'T>> = StackImageFunctions.clamp<'T> lower upper
let shiftScale<'T when 'T: equality> shift scale : Stage<Image<'T>, Image<'T>> = StackImageFunctions.shiftScale<'T> shift scale
let intensityStretch<'T when 'T: equality> inputMinimum inputMaximum outputMinimum outputMaximum : Stage<Image<'T>, Image<'T>> = StackImageFunctions.intensityStretch<'T> inputMinimum inputMaximum outputMinimum outputMaximum
let median<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> = StackImageFunctions.median<'T> radius winSz
let bilateral<'T when 'T: equality> domainSigma rangeSigma winSz : Stage<Image<'T>, Image<'T>> = StackImageFunctions.bilateral<'T> domainSigma rangeSigma winSz
let gradientMagnitude<'T when 'T: equality> winSz : Stage<Image<'T>, Image<'T>> = StackImageFunctions.gradientMagnitude<'T> winSz
let sobelEdge<'T when 'T: equality> winSz : Stage<Image<'T>, Image<'T>> = StackImageFunctions.sobelEdge<'T> winSz
let laplacian<'T when 'T: equality> winSz : Stage<Image<'T>, Image<'T>> = StackImageFunctions.laplacian<'T> winSz
let grayscaleErode<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> = StackImageFunctions.grayscaleErode<'T> radius winSz
let grayscaleDilate<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> = StackImageFunctions.grayscaleDilate<'T> radius winSz
let grayscaleOpening<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> = StackImageFunctions.grayscaleOpening<'T> radius winSz
let grayscaleClosing<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> = StackImageFunctions.grayscaleClosing<'T> radius winSz
let whiteTopHat<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> = StackImageFunctions.whiteTopHat<'T> radius winSz
let blackTopHat<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> = StackImageFunctions.blackTopHat<'T> radius winSz
let morphologicalGradient<'T when 'T: equality> radius winSz : Stage<Image<'T>, Image<'T>> = StackImageFunctions.morphologicalGradient<'T> radius winSz
let binaryContour = StackImageFunctions.binaryContour
let binaryMedian = StackImageFunctions.binaryMedian
let equal<'T when 'T: equality> = StackImageFunctions.equal<'T>
let notEqual<'T when 'T: equality> = StackImageFunctions.notEqual<'T>
let greater<'T when 'T: equality> = StackImageFunctions.greater<'T>
let greaterEqual<'T when 'T: equality> = StackImageFunctions.greaterEqual<'T>
let less<'T when 'T: equality> = StackImageFunctions.less<'T>
let lessEqual<'T when 'T: equality> = StackImageFunctions.lessEqual<'T>
let andMask = StackImageFunctions.andMask
let orMask = StackImageFunctions.orMask
let xorMask = StackImageFunctions.xorMask
let notMask = StackImageFunctions.notMask
let labelContour<'T when 'T: equality> fullyConnected winSz : Stage<Image<'T>, Image<'T>> = StackImageFunctions.labelContour<'T> fullyConnected winSz
let changeLabel<'T when 'T: equality> fromLabel toLabel : Stage<Image<'T>, Image<'T>> = StackImageFunctions.changeLabel<'T> fromLabel toLabel
let marchingCubes<'T when 'T: equality> = StackMesh.marchingCubes<'T>
let dogKeypoints<'T when 'T: equality> = StackPoints.dogKeypoints<'T>
let resize<'T when 'T: equality> = StackImageFunctions.resize<'T>
let resample<'T when 'T: equality> = StackImageFunctions.resample<'T>
let histogram = StackImageFunctions.histogram
let sumProjection<'T when 'T: equality> = StackImageFunctions.sumProjection<'T>
let quantiles = StackImageFunctions.quantiles
let otsuThresholdFromHistogram = StackImageFunctions.otsuThresholdFromHistogram
let momentsThresholdFromHistogram = StackImageFunctions.momentsThresholdFromHistogram
let inline map2pairs< ^T, ^S when ^T: comparison and ^T: (static member op_Explicit: ^T -> float) and  ^S: (static member op_Explicit: ^S -> float) > = StackImageFunctions.map2pairs<'T,'S>
let inline pairs2floats< ^T, ^S when ^T: (static member op_Explicit: ^T -> float) and  ^S: (static member op_Explicit: ^S -> float) > = StackImageFunctions.pairs2floats<'T,'S>
let inline pairs2ints< ^T, ^S when ^T: (static member op_Explicit: ^T -> int) and  ^S: (static member op_Explicit: ^S -> int) > = StackImageFunctions.pairs2ints<'T,'S>
let computeStats = StackImageFunctions.computeStats
let discreteGaussian = StackImageFunctions.discreteGaussian
let convGauss = StackImageFunctions.convGauss
let createPadding<'T when 'T: equality> = StackImageFunctions.createPadding<'T>
let crop<'T when 'T: equality> = StackImageFunctions.crop<'T>
let convolve = StackImageFunctions.convolve
let conv = StackImageFunctions.conv
let finiteDiff = StackImageFunctions.finiteDiff
let erode = StackImageFunctions.erode
let dilate = StackImageFunctions.dilate
let opening = StackImageFunctions.opening
let closing = StackImageFunctions.closing
let connectedComponents = StackImageFunctions.connectedComponents
let relabelComponents = StackImageFunctions.relabelComponents
let signedDistanceBand = StackImageFunctions.signedDistanceBand
let threshold = StackImageFunctions.threshold
let addNormalNoise = StackImageFunctions.addNormalNoise
let show = StackImageFunctions.show
let plot = StackImageFunctions.plot
let print = StackImageFunctions.print
let zero<'T when 'T: equality>= StackImageFunctions.zero<'T>
let createByEuler2DTransform<'T when 'T: equality>= StackImageFunctions.createByEuler2DTransform<'T>
let empty = StackImageFunctions.empty
let writeSlabSlices = StackImageFunctions.writeSlabSlices
type ComponentStatistics = StackImageFunctions.ComponentStatistics
type ConnectedComponentTranslationTable = StackImageFunctions.ConnectedComponentTranslationTable
let makeConnectedComponentTranslationTable = StackImageFunctions.makeConnectedComponentTranslationTable
let updateConnectedComponents = StackImageFunctions.updateConnectedComponents
let permuteAxes = StackImageFunctions.permuteAxes
