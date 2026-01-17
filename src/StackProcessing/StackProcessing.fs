module StackProcessing

open StackCore

// //////////////////// StackCore
type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>
type Profile = SlimPipeline.Profile
type ProfileTransition = SlimPipeline.ProfileTransition
//type Slice<'S when 'S: equality> = Slice.Slice<'S>
type Image<'S when 'S: equality> = Image.Image<'S>

let getMem = StackCore.getMem
let incIfImage = StackCore.incIfImage
let incRef = StackCore.incRef
let decIfImage = StackCore.decIfImage
let decRef = StackCore.decRef
let releaseAfter = StackCore.releaseAfter
let releaseAfter2 = StackCore.releaseAfter2
let volFctToLstFctReleaseAfter = StackCore.volFctToLstFctReleaseAfter
let (>=>) = StackCore.(>=>)
let (-->) = StackCore.(-->)
let source = StackCore.source
let debug = StackCore.debug
let zip = StackCore.zip
let promoteStreamingToWindow = StackCore.promoteStreamingToWindow
let (>=>>) = StackCore.(>=>>)
let (>>=>) = StackCore.(>>=>)
let (>>=>>) = StackCore.(>>=>>)
let ignoreSingles = StackCore.ignoreSingles
let ignorePairs = StackCore.ignorePairs
let zeroMaker = StackCore.zeroMaker
let window = StackCore.window
let flatten = StackCore.flatten
let map = StackCore.map
let sinkOp = StackCore.sinkOp
let sink = StackCore.sink
let sinkList = StackCore.sinkList
let drain = StackCore.drain
let drainList = StackCore.drainList
let drainLast = StackCore.drainLast
let tap = StackCore.tap
let tapIt = StackCore.tapIt
let idStage<'T> = StackCore.idStage<'T>

// //////////////////// StackIO
type FileInfo = ImageFunctions.FileInfo
type ChunkInfo = StackIO.ChunkInfo

let getStackDepth = StackIO.getStackDepth
let getStackInfo = StackIO.getStackInfo
let getStackSize = StackIO.getStackSize
let getStackWidth = StackIO.getStackWidth
let getStackHeight = StackIO.getStackHeight
let getFilenames = StackIO.getFilenames
let readFiles<'T when 'T: equality> = StackIO.readFiles<'T>
let readFilePairs<'T when 'T: equality> = StackIO.readFilePairs<'T>
let readFiltered<'T when 'T: equality> = StackIO.readFiltered<'T>
let read<'T when 'T: equality> = StackIO.read<'T>
let readRandom<'T when 'T: equality> = StackIO.readRandom<'T>
let getChunkInfo = StackIO.getChunkInfo
let getChunkFilename = StackIO.getChunkFilename
let readChunksAsWindows<'T when 'T: equality> = StackIO.readChunksAsWindows<'T>
let readChunks<'T when 'T: equality> = StackIO.readChunks<'T>
let icompare = StackIO.icompare

let deleteIfExists = StackIO.deleteIfExists
let write = StackIO.write
let writeInChunks = StackIO.writeInChunks

// //////////////////// ChunkedAffineResampler
let resampleAffineTrilinearSlices = ChunkedAffineResampler.resampleAffineTrilinearSlices

// //////////////////// StackImageFunctions
type ImageStats = ImageFunctions.ImageStats

let liftUnary = StackImageFunctions.liftUnary
let liftUnaryReleaseAfter = StackImageFunctions.liftUnaryReleaseAfter
let getBytesPerComponent<'T> = StackImageFunctions.getBytesPerComponent<'T>
let cast<'S,'T when 'S: equality and 'T: equality> = StackImageFunctions.cast<'S,'T>
let liftRelease2 = StackImageFunctions.liftRelease2
let memNeeded<'T> = StackImageFunctions.memNeeded<'T>
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
let square<'T when 'T: equality>= StackImageFunctions.square<'T>
let imageHistogram = StackImageFunctions.imageHistogram
let imageHistogramFold = StackImageFunctions.imageHistogramFold
let histogram = StackImageFunctions.histogram
let inline map2pairs< ^T, ^S when ^T: comparison and ^T: (static member op_Explicit: ^T -> float) and  ^S: (static member op_Explicit: ^S -> float) > = StackImageFunctions.map2pairs<'T,'S>
let inline pairs2floats< ^T, ^S when ^T: (static member op_Explicit: ^T -> float) and  ^S: (static member op_Explicit: ^S -> float) > = StackImageFunctions.pairs2floats<'T,'S>
let inline pairs2ints< ^T, ^S when ^T: (static member op_Explicit: ^T -> int) and  ^S: (static member op_Explicit: ^S -> int) > = StackImageFunctions.pairs2ints<'T,'S>
let imageComputeStats = StackImageFunctions.imageComputeStats
let imageComputeStatsFold = StackImageFunctions.imageComputeStatsFold
let computeStats = StackImageFunctions.computeStats
let stackFUnstack = StackImageFunctions.stackFUnstack
let skipNTakeM = StackImageFunctions.skipNTakeM
let stackFUnstackTrim = StackImageFunctions.stackFUnstackTrim
let discreteGaussianOp = StackImageFunctions.discreteGaussianOp
let discreteGaussian = StackImageFunctions.discreteGaussian
let convGauss = StackImageFunctions.convGauss
let createPadding = StackImageFunctions.createPadding
let convolveOp = StackImageFunctions.convolveOp
let convolve = StackImageFunctions.convolve
let conv = StackImageFunctions.conv
let finiteDiff = StackImageFunctions.finiteDiff
let erode = StackImageFunctions.erode
let dilate = StackImageFunctions.dilate
let opening = StackImageFunctions.opening
let closing = StackImageFunctions.closing
let binaryFillHoles = StackImageFunctions.binaryFillHoles
let connectedComponents = StackImageFunctions.connectedComponents
let relabelComponents = StackImageFunctions.relabelComponents
let watershed = StackImageFunctions.watershed
let signedDistanceMap = StackImageFunctions.signedDistanceMap
let otsuThreshold = StackImageFunctions.otsuThreshold
let momentsThreshold = StackImageFunctions.momentsThreshold
let threshold = StackImageFunctions.threshold
let addNormalNoise = StackImageFunctions.addNormalNoise
let ImageConstantPad<'T when 'T: equality>= StackImageFunctions.ImageConstantPad<'T>
let show = StackImageFunctions.show
let plot = StackImageFunctions.plot
let print = StackImageFunctions.print
let srcStage = StackImageFunctions.srcStage
let srcPlan = StackImageFunctions.srcPlan
let zero<'T when 'T: equality>= StackImageFunctions.zero<'T>
let createByEuler2DTransform<'T when 'T: equality>= StackImageFunctions.createByEuler2DTransform<'T>
let empty = StackImageFunctions.empty
let getConnectedChunkNeighbours = StackImageFunctions.getConnectedChunkNeighbours
let makeAdjacencyGraph = StackImageFunctions.makeAdjacencyGraph
let makeTranslationTable = StackImageFunctions.makeTranslationTable
let trd = StackImageFunctions.trd
let updateConnectedComponents = StackImageFunctions.updateConnectedComponents
let permuteAxes = StackImageFunctions.permuteAxes