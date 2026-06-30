module ProbeProbing

open System
open System.Collections.Generic
open System.Diagnostics
open System.IO
open System.Runtime.InteropServices
open System.Globalization
open System.Text.Json
open System.Text.Json.Serialization
open SlimPipeline
open StackProcessing
open Studio.Graph

[<CLIMutable>]
type ProbeSourceJson =
    { name: string
      elementBytes: uint64
      hasLength: bool
      length: uint64
      shape: Dictionary<string, string> }

[<CLIMutable>]
type ProbeIterationJson =
    { iteration: int
      observedElements: uint64
      observedBytes: uint64
      rssBaselineBytes: uint64
      rssPeakBytes: uint64
      rssDeltaBytes: uint64
      rssPostGcBytes: uint64
      rssRetainedDeltaBytes: uint64
      elapsedMilliseconds: int64
      elapsedTotalMilliseconds: float
      elapsedTicks: int64
      throughputElementsPerSecond: float
      throughputBytesPerSecond: float }

[<CLIMutable>]
type ProbeTimeCostJson =
    { calibrationKey: string
      cpuCostUnits: float
      nativeCostUnits: float
      ioReadBytes: uint64
      ioWriteBytes: uint64
      ioReadOps: uint64
      ioWriteOps: uint64 }

[<CLIMutable>]
type ProbeResultJson =
    { name: string
      description: string
      parameters: Dictionary<string, string>
      observedElements: uint64
      observedBytes: uint64
      predictedImagePeakBytes: uint64
      predictedMemoryPeakBytes: uint64
      actualMemoryDeltaBytes: uint64
      predictedElapsedMilliseconds: float option
      actualElapsedMedianMilliseconds: float
      rssBaselineBytes: uint64
      rssPeakBytes: uint64
      rssDeltaBytes: uint64
      rssDeltaMinBytes: uint64
      rssDeltaMedianBytes: uint64
      rssDeltaMaxBytes: uint64
      rssRetainedDeltaMinBytes: uint64
      rssRetainedDeltaMedianBytes: uint64
      rssRetainedDeltaMaxBytes: uint64
      elapsedMilliseconds: int64
      elapsedTotalMilliseconds: float
      elapsedMinMilliseconds: float
      elapsedMedianMilliseconds: float
      elapsedMaxMilliseconds: float
      throughputElementsPerSecond: float
      throughputMinElementsPerSecond: float
      throughputMedianElementsPerSecond: float
      throughputMaxElementsPerSecond: float
      throughputBytesPerSecond: float
      throughputMinBytesPerSecond: float
      throughputMedianBytesPerSecond: float
      throughputMaxBytesPerSecond: float
      warmupCount: int
      repetitionCount: int
      source: ProbeSourceJson
      costPeakTime: ProbeTimeCostJson option
      costTimes: ProbeTimeCostJson array
      iterations: ProbeIterationJson array }

[<CLIMutable>]
type ProbeReportJson =
    { generatedUtc: string
      osDescription: string
      processArchitecture: string
      frameworkDescription: string
      configuration: string
      sampleIntervalMilliseconds: int
      warmupCount: int
      repetitionCount: int
      workingDirectory: string
      tempDirectory: string
      calibrations: Dictionary<string, StageTimeCoefficients>
      probes: ProbeResultJson array }

let private availableMemory = 2UL * 1024UL * 1024UL * 1024UL
let private sampleIntervalMs = 10
let private warmupCount = 1
let private repetitionCount = 5
let private canonicalDepth = 64u
let private canonicalRadius = 3u
let private canonicalWindowSize = 2u * canonicalRadius + 1u
let private canonicalSigma = 3.0
let private canonicalSigmaText = "3.0"
let private canonicalGaussianWindowSize = 13u
let private canonicalKernelSize = canonicalWindowSize

let private zero<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    StackProcessing.zero<'T>

let private read<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    StackProcessing.read<'T>

let private write<'T when 'T: equality> =
    StackProcessing.write<'T>

let private cast<'S, 'T when 'S: equality and 'S: (new: unit -> 'S) and 'S: struct and 'S :> ValueType
                          and 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    ChunkFunctions.castChunk<'S, 'T>

let private addNormalNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> mean std =
    StackProcessing.addNormalNoise<'T> mean std

let private addSaltAndPepperNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> probability pepper salt =
    StackProcessing.addSaltAndPepperNoise<'T> probability pepper salt

let private addPoissonNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> lambda =
    StackProcessing.addPoissonNoise<'T> lambda

let private threshold<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (lower: double) (upper: double) =
    thresholdRange<'T> lower upper

let private imageAddScalar value =
    addScalar value

let private imageMulScalar value =
    mulScalar value

let private intensityStretch =
    StackProcessing.intensityStretch

let private computeStats<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> () =
    StackProcessing.computeStats<'T> ()

let private sumProjection<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    StackProcessing.sumProjection<'T>

let private smoothWGauss<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> sigma _outputRegionMode _boundaryCondition windowSize =
    gaussianFilter<'T> sigma windowSize

let private gradientMagnitude<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> windowSize =
    let radius = windowSize |> Option.map (fun w -> int ((w - 1u) / 2u)) |> Option.defaultValue 3
    cast<'T, float32> --> StackProcessing.gradientMagnitude 1.0 radius

let private sobelEdge<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> _windowSize =
    cast<'T, float32> --> StackProcessing.sobelMagnitude ()

let private laplacian<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> windowSize =
    let radius = windowSize |> Option.map (fun w -> int ((w - 1u) / 2u)) |> Option.defaultValue 3
    cast<'T, float32> --> StackProcessing.laplacian 1.0 radius

let private sqrt<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    if typeof<'T> = typeof<float32> then
        unbox (box ChunkFunctions.sqrtFloat32)
    else
        cast<'T, float32> --> ChunkFunctions.sqrtFloat32 --> cast<float32, 'T>

let private sqrtWindowed<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (_windowSize: uint) =
    sqrt<'T>

let private erode radius =
    StackProcessing.binaryErode radius

let private dilate radius =
    StackProcessing.binaryDilate radius

let private opening radius =
    StackProcessing.binaryOpening radius

let private closing radius =
    StackProcessing.binaryClosing radius

let private blackTopHat<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (radius: uint) (windowSize: int option) =
    match windowSize with
    | Some w -> StackProcessing.binaryBlackTopHatWindowed radius w
    | None -> StackProcessing.binaryBlackTopHat radius

let private whiteTopHat<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (radius: uint) (windowSize: int option) =
    match windowSize with
    | Some w -> StackProcessing.binaryWhiteTopHatWindowed radius w
    | None -> StackProcessing.binaryWhiteTopHat radius

let private morphologicalGradient<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (radius: uint) (windowSize: int option) =
    match windowSize with
    | Some w -> StackProcessing.binaryGradientWindowed radius w
    | None -> StackProcessing.binaryGradient radius

let private binaryContour fullyConnected (windowSize: int option) =
    match windowSize with
    | Some w -> StackProcessing.binaryContourWindowed fullyConnected w
    | None -> StackProcessing.binaryContour fullyConnected

let private fillSmallHoles maximumVolume connectivity =
    StackProcessing.fillSmallHoles maximumVolume connectivity

let private resize<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    StackProcessing.resize<'T>

let private resample<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> =
    StackProcessing.resample<'T>

let private normalNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> width height depth mean std pl =
    pl
    |> zero<'T> width height depth
    >=> addNormalNoise mean std

let private saltAndPepperNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> width height depth probability pl =
    pl
    |> zero<'T> width height depth
    >=> addSaltAndPepperNoise probability None None

let private poissonNoise<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> width height depth lambda pl =
    pl
    |> zero<'T> width height depth
    >=> addPoissonNoise lambda

// Workflow-shape inspiration for realistic boilerplate probes:
// - Robert Haase et al., Bio-image Analysis Notebooks
// - NEUBIAS Bioimage Analysis Training Resources
// - BIAFLOWS curated microscopy workflow benchmarks
// These probes are hand-written StackProcessing equivalents for now; later we may scrape/import
// community workflows and map their steps onto this limited StackProcessing vocabulary.

type ImageSize =
    { Width: uint
      Height: uint
      Depth: uint }

let private xySizes =
    [ 64u ]

let private imageSize xy depth =
    { Width = xy
      Height = xy
      Depth = depth }

let private defaultDepths =
    [ canonicalDepth ]

let private singletonDepths =
    [ canonicalDepth ]

let private inputDepths =
    defaultDepths @ singletonDepths
    |> List.distinct
    |> List.sort

let private inputSizes =
    [ for xy in xySizes do
          for depth in inputDepths do
              imageSize xy depth ]

let private boilerplateXySizes =
    xySizes

let private boilerplateDepths =
    defaultDepths

let private boilerplateInputSizes =
    [ for xy in boilerplateXySizes do
          for depth in boilerplateDepths do
              imageSize xy depth ]

let private convolutionBreakdownSizes =
    xySizes
    |> List.map (fun side -> imageSize side canonicalDepth)

let private gaussianWindowSizes =
    [ canonicalGaussianWindowSize ]

let private unaryWindowSizes =
    [ canonicalWindowSize ]

let private stackUnstackWindowSizes =
    gaussianWindowSizes

let private forceFullGc () =
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()

let private sourceToJson (peek: SourcePeek option) =
    match peek with
    | Some source ->
        let shape = Dictionary<string, string>()
        source.Shape |> Map.iter (fun key value -> shape[key] <- value)
        { name = source.Name
          elementBytes = source.ElementBytes
          hasLength = source.Length.IsSome
          length = source.Length |> Option.defaultValue 0UL
          shape = shape }
    | None ->
        { name = ""
          elementBytes = 0UL
          hasLength = false
          length = 0UL
          shape = Dictionary<string, string>() }

let private observedElements (plan: Plan<unit, _>) =
    plan.length * (plan.nElemsPerSlice |> SingleOrPair.sum |> SingleOrPair.fst)

let private observedBytes (source: ProbeSourceJson) (fallbackElements: uint64) =
    if source.hasLength && source.elementBytes > 0UL then
        source.length * source.elementBytes
    else
        fallbackElements

let private median (values: uint64 array) =
    if values.Length = 0 then 0UL
    else
        let sorted = Array.sort values
        sorted[sorted.Length / 2]

let private medianFloat (values: float array) =
    if values.Length = 0 then 0.0
    else
        let sorted = Array.sort values
        sorted[sorted.Length / 2]

let private parameters pairs =
    let dict = Dictionary<string, string>()
    pairs |> List.iter (fun (key, value) -> dict[key] <- value)
    dict

let private csvEscape (value: string) =
    if value.Contains(',') || value.Contains('"') || value.Contains('\n') || value.Contains('\r') then
        "\"" + value.Replace("\"", "\"\"") + "\""
    else
        value

let private writeCsv path (rows: seq<string list>) =
    let directory = Path.GetDirectoryName(path: string)
    if not (String.IsNullOrWhiteSpace directory) then
        Directory.CreateDirectory directory |> ignore

    rows
    |> Seq.map (List.map csvEscape >> String.concat ",")
    |> fun lines -> File.WriteAllLines(path, lines)

let private invariant (value: float) =
    value.ToString("G17", CultureInfo.InvariantCulture)

let private invariantInt64 (value: int64) =
    value.ToString(CultureInfo.InvariantCulture)

let private invariantUInt64 (value: uint64) =
    value.ToString(CultureInfo.InvariantCulture)

let private tryParameter key (probe: ProbeResultJson) =
    if isNull probe.parameters then
        None
    else
        match probe.parameters.TryGetValue key with
        | true, value -> Some value
        | _ -> None

let private operationName probe =
    probe
    |> tryParameter "operation"
    |> Option.defaultValue probe.name

let private normalizedIdentifier (value: string) =
    value.Trim().ToLowerInvariant().Replace("-", "").Replace("_", "").Replace(".", "")

let private calibrationParameterKey (probe: ProbeResultJson) =
    let excluded =
        set [ "operation"
              "baselineOperation"
              "stage"
              "derivedFrom"
              "isNPlus2"
              "depthOffset" ]

    if isNull probe.parameters then
        ""
    else
        probe.parameters
        |> Seq.cast<KeyValuePair<string, string>>
        |> Seq.choose (fun kvp ->
            if excluded.Contains kvp.Key then
                None
            else
                Some $"{kvp.Key}={kvp.Value}")
        |> Seq.sort
        |> String.concat "|"

let private analysisParameterText (probe: ProbeResultJson) =
    if isNull probe.parameters then
        ""
    else
        probe.parameters
        |> Seq.cast<KeyValuePair<string, string>>
        |> Seq.sortBy (fun kvp -> kvp.Key)
        |> Seq.map (fun kvp -> $"{kvp.Key}={kvp.Value}")
        |> String.concat "|"

let private stageFeatureKey (probe: ProbeResultJson) =
    let stageOrOperation =
        probe
        |> tryParameter "stage"
        |> Option.defaultValue (operationName probe)

    match calibrationParameterKey probe with
    | "" -> stageOrOperation
    | parameters -> $"{stageOrOperation}:{parameters}"

let private probeParameter key (probe: ProbeResultJson) =
    probe |> tryParameter key |> Option.defaultValue ""

let private stageParameter key defaultValue (probe: ProbeResultJson) =
    probe |> tryParameter key |> Option.defaultValue defaultValue

let private sampleCompatibleProbeFeatures (probe: ProbeResultJson) =
    let depth = probeParameter "depth" probe
    let width = probeParameter "width" probe
    let height = probeParameter "height" probe
    let pixelType = probeParameter "pixelType" probe
    let capitalizedType =
        match pixelType.ToLowerInvariant() with
        | "uint8" -> "UInt8"
        | "float32" -> "Float32"
        | other -> other

    let readFeature =
        $"Read:type={capitalizedType}:format=Image stack:suffix=.tiff:thickDepth=1:multiscaleIndex=0:datasetIndex=0:timepoint=0:channel=0:maxParallelChunks=0:frameAxis=0"

    let readFeatureWithAxes =
        $"Read:type={capitalizedType}:format=Image stack:suffix=.tiff:thickDepth=1:multiscaleIndex=0:datasetIndex=0:timepoint=0:channel=0:maxParallelChunks=0:frameAxis=0:yAxis=1:xAxis=2"

    let writeFeature =
        "Write:format=Image stack:suffix=.tiff:depth=1:chunkX=64:chunkY=64:chunkZ=8:maxConcurrentWrites=0:frameAxis=0"

    let writeFeatureWithAxes =
        "Write:format=Image stack:suffix=.tiff:depth=1:chunkX=64:chunkY=64:chunkZ=8:maxConcurrentWrites=0:frameAxis=0:yAxis=1:xAxis=2"

    let zeroFeature =
        $"Zero:type={capitalizedType}:width={width}:height={height}:depth={depth}"
    let mean = stageParameter "mean" "128.0" probe
    let std = stageParameter "std" "50.0" probe
    let thresholdValue = stageParameter "threshold" "128.0" probe
    let windowSize = stageParameter "windowSize" "1" probe
    let sigma = stageParameter "sigma" canonicalSigmaText probe
    let kernelSize = stageParameter "kernelSize" "<empty>" probe

    seq {
        match tryParameter "features" probe with
        | Some features ->
            for feature in features.Split([| "||" |], StringSplitOptions.RemoveEmptyEntries) do
                yield feature
        | None -> ()

        match operationName probe with
        | "zero" ->
            yield zeroFeature
        | "zero-write" ->
            yield zeroFeature
            yield writeFeature
            yield writeFeatureWithAxes
        | "noise-write" ->
            yield zeroFeature
            yield $"AddNormalNoise:type={capitalizedType}:mean={mean}:std={std}"
            yield writeFeature
            yield writeFeatureWithAxes
        | "read-ignore" ->
            yield readFeature
            yield readFeatureWithAxes
        | "read-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield writeFeature
            yield writeFeatureWithAxes
        | "threshold-write" ->
            yield zeroFeature
            yield $"AddNormalNoise:type={capitalizedType}:mean={mean}:std={std}"
            yield $"Threshold:type={capitalizedType}:lower={thresholdValue}:upper=infinity"
            yield "ImageOpScalar:operation=*:type=UInt8:value=255"
            yield writeFeature
            yield writeFeatureWithAxes
        | "compute-stats" ->
            yield readFeature
            yield readFeatureWithAxes
            yield "ComputeStats"
        | "add-salt-and-pepper-write" ->
            yield zeroFeature
            yield $"AddSaltAndPepperNoise:type={capitalizedType}:probability=0.02"
            yield writeFeature
            yield writeFeatureWithAxes
        | "salt-and-pepper-write" ->
            yield $"SaltAndPepperNoise:type={capitalizedType}:width={width}:height={height}:depth={depth}:probability=0.02"
            yield writeFeature
            yield writeFeatureWithAxes
        | "poisson-noise-write" ->
            yield $"PoissonNoise:type={capitalizedType}:width={width}:height={height}:depth={depth}:lambda=2.0"
            yield "Cast:sourceType=Float32:targetType=UInt8"
            yield writeFeature
            yield writeFeatureWithAxes
        | "speckle-noise-write" ->
            yield $"SpeckleNoise:type={capitalizedType}:width={width}:height={height}:depth={depth}:std=0.5"
            yield "Cast:sourceType=Float32:targetType=UInt8"
            yield writeFeature
            yield writeFeatureWithAxes
        | "add-poisson-speckle-write" ->
            yield zeroFeature
            yield "AddPoissonNoise:type=Float32:lambda=2.0"
            yield "AddSpeckleNoise:type=Float32:std=0.5"
            yield "Cast:sourceType=Float32:targetType=UInt8"
            yield writeFeature
            yield writeFeatureWithAxes
        | "fft-roundtrip-write" ->
            yield $"NormalNoise:type={capitalizedType}:width={width}:height={height}:depth={depth}:mean=128.0:std=25.0"
            yield "FFT:type=Float32:chunkX=16:chunkY=16:chunkZ=8"
            yield "InvFFT:chunkX=16:chunkY=16:chunkZ=8"
            yield "Cast:sourceType=Float32:targetType=Float32"
            yield writeFeature
            yield writeFeatureWithAxes
        | "grayscale-erode-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield $"GrayscaleErode:type={capitalizedType}:radius=3:windowSize=7"
            yield writeFeature
            yield writeFeatureWithAxes
        | "grayscale-dilate-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield $"GrayscaleDilate:type={capitalizedType}:radius=3:windowSize=7"
            yield writeFeature
            yield writeFeatureWithAxes
        | "grayscale-opening-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield $"GrayscaleOpening:type={capitalizedType}:radius=3:windowSize=7"
            yield writeFeature
            yield writeFeatureWithAxes
        | "grayscale-closing-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield $"GrayscaleClosing:type={capitalizedType}:radius=3:windowSize=7"
            yield writeFeature
            yield writeFeatureWithAxes
        | "binary-median-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield "BinaryMedian:radius=3:windowSize=7"
            yield writeFeature
            yield writeFeatureWithAxes
        | "fill-small-holes-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield "FillSmallHoles:maximumVolume=128:connectivity=TwentySix"
            yield writeFeature
            yield writeFeatureWithAxes
        | "binary-contour-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield "BinaryContour:fullyConnected=false:windowSize=7"
            yield writeFeature
            yield writeFeatureWithAxes
        | "erode-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield "Erode:radius=3"
            yield writeFeature
            yield writeFeatureWithAxes
        | "dilate-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield "Dilate:radius=3"
            yield writeFeature
            yield writeFeatureWithAxes
        | "opening-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield "Opening:radius=3"
            yield writeFeature
            yield writeFeatureWithAxes
        | "closing-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield "Closing:radius=3"
            yield writeFeature
            yield writeFeatureWithAxes
        | "black-top-hat-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield $"BlackTopHat:type={capitalizedType}:radius=3:windowSize=7"
            yield writeFeature
            yield writeFeatureWithAxes
        | "white-top-hat-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield $"WhiteTopHat:type={capitalizedType}:radius=3:windowSize=7"
            yield writeFeature
            yield writeFeatureWithAxes
        | "morphological-gradient-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield $"MorphologicalGradient:type={capitalizedType}:radius=3:windowSize=7"
            yield writeFeature
            yield writeFeatureWithAxes
        | "filters-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield "SmoothWMedian:type=Float32:radius=3:windowSize=7"
            yield "SmoothWBilateral:type=Float32:domainSigma=1.5:rangeSigma=30.0:windowSize=7"
            yield "GradientMagnitude:type=Float32:windowSize=7"
            yield "SobelEdge:type=Float32:windowSize=7"
            yield "Laplacian:type=Float32:windowSize=7"
            yield "IntensityStretch:type=Float32:inputMinimum=0.0:inputMaximum=255.0:outputMinimum=0.0:outputMaximum=255.0"
            yield "Cast:sourceType=Float32:targetType=UInt8"
            yield writeFeature
            yield writeFeatureWithAxes
        | "resize-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield $"Resize:type={capitalizedType}:width=96:height=96:depth=96:interpolation=Linear"
            yield writeFeature
            yield writeFeatureWithAxes
        | "resize64-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield $"Resize:type={capitalizedType}:width=64:height=64:depth=64:interpolation=Linear"
            yield writeFeature
            yield writeFeatureWithAxes
        | "resample-write" ->
            yield $"NormalNoise:type={capitalizedType}:width={width}:height={height}:depth={depth}:mean=128.0:std=25.0"
            yield $"Resample:type={capitalizedType}:factorX=1.5:factorY=1.5:factorZ=1.5:interpolation=Linear"
            yield "Cast:sourceType=Float32:targetType=UInt8"
            yield writeFeature
            yield writeFeatureWithAxes
        | "sqrt" ->
            yield zeroFeature
            yield $"ImageOpScalar:operation=+:type={capitalizedType}:value=4.0"
            yield $"Sqrt:type={capitalizedType}"
        | "sqrt-write" ->
            yield zeroFeature
            yield $"ImageOpScalar:operation=+:type={capitalizedType}:value=4.0"
            yield $"Sqrt:type={capitalizedType}"
            yield "Cast:sourceType=Float32:targetType=UInt8"
            yield writeFeature
        | "sqrt-windowed"
        | "sqrt-windowed-write" ->
            yield zeroFeature
            yield $"ImageOpScalar:operation=+:type={capitalizedType}:value=4.0"
            yield $"SqrtWindowed:type={capitalizedType}:windowSize={windowSize}"
            if operationName probe = "sqrt-windowed-write" then
                yield "Cast:sourceType=Float32:targetType=UInt8"
                yield writeFeature
                yield writeFeatureWithAxes
        | "smoothWGauss-write" ->
            yield readFeature
            yield readFeatureWithAxes
            yield $"SmoothWGauss:type={capitalizedType}:sigma={sigma}:kernelSize={kernelSize}:boundary=<empty>:windowSize={windowSize}"
            yield "Cast:sourceType=Float32:targetType=UInt8"
            yield writeFeature
            yield writeFeatureWithAxes
        | _ -> ()
    }
    |> Seq.toList

let private parseCsvLine (line: string) =
    let values = ResizeArray<string>()
    let current = System.Text.StringBuilder()
    let mutable quoted = false
    let mutable i = 0

    while i < line.Length do
        match line[i] with
        | '"' when quoted && i + 1 < line.Length && line[i + 1] = '"' ->
            current.Append('"') |> ignore
            i <- i + 2
        | '"' ->
            quoted <- not quoted
            i <- i + 1
        | ',' when not quoted ->
            values.Add(current.ToString())
            current.Clear() |> ignore
            i <- i + 1
        | ch ->
            current.Append(ch) |> ignore
            i <- i + 1

    values.Add(current.ToString())
    values |> Seq.toList

let private loadAnalysisFeatureTokens supportThreshold path =
    if String.IsNullOrWhiteSpace path || not (File.Exists path) then
        Set.empty
    else
        let lines = File.ReadAllLines path

        match lines |> Array.tryHead with
        | None -> Set.empty
        | Some header ->
            let columns = parseCsvLine header
            let featureIndex = columns |> List.tryFindIndex ((=) "featureKey")

            match featureIndex with
            | None -> Set.empty
            | Some featureIndex ->
                let features =
                    lines
                    |> Seq.skip 1
                    |> Seq.choose (fun line ->
                        let columns = parseCsvLine line
                        if featureIndex < columns.Length then Some columns[featureIndex] else None)
                    |> Seq.countBy id
                    |> Seq.filter (fun (feature, count) ->
                        feature <> "intercept"
                        && match supportThreshold with
                           | Some threshold -> count <= threshold
                           | None -> true)

                features
                |> Seq.collect (fun (feature, _) ->
                    let normalized = normalizedIdentifier feature
                    let operation =
                        match feature.Split([| ':' |], 2) with
                        | [| first; _ |] -> normalizedIdentifier first
                        | _ -> normalized

                    [ normalized; operation ])
                |> Set.ofSeq

let private probeMatchesAnalysisTokens tokens (probe: ProbeResultJson) =
    if Set.isEmpty tokens then
        true
    else
        let candidates =
            seq {
                operationName probe
                stageFeatureKey probe
                match tryParameter "stage" probe with
                | Some stage -> stage
                | None -> ()

                for timeCost in probe.costTimes do
                    timeCost.calibrationKey

                for feature in sampleCompatibleProbeFeatures probe do
                    feature
            }
            |> Seq.map normalizedIdentifier
            |> Seq.toList

        tokens
        |> Set.exists (fun (token: string) ->
            candidates
            |> List.exists (fun (candidate: string) ->
                candidate.Contains(token, StringComparison.Ordinal)
                || token.Contains(candidate, StringComparison.Ordinal)))

let private filterProbesByAnalysisFeatures supportThreshold analysisFeaturesPath probes =
    match analysisFeaturesPath with
    | None -> probes
    | Some path ->
        let tokens = loadAnalysisFeatureTokens supportThreshold path
        let filtered = probes |> Array.filter (probeMatchesAnalysisTokens tokens)
        printfn "Analysis feature restriction kept %d of %d probes from %s." filtered.Length probes.Length path
        filtered

let private imageParameters size pixelType windowSize baseDepth =
    let p =
        parameters
            [ "pixelType", pixelType
              "width", string size.Width
              "height", string size.Height
              "depth", string size.Depth
              "depthBase", string baseDepth
              "depthOffset", string (size.Depth - baseDepth)
              "isNPlus2", string (size.Depth = baseDepth + 2u)
              "imagePixels", string (uint64 size.Width * uint64 size.Height)
              "imageVoxels", string (uint64 size.Width * uint64 size.Height * uint64 size.Depth)
              "windowSize", string windowSize ]
    p

let private defaultImageParameters size pixelType windowSize =
    imageParameters size pixelType windowSize 3u

let private singletonImageParameters size pixelType windowSize =
    imageParameters size pixelType windowSize 1u

let private encodedFeatures (features: string list) =
    String.concat "||" features

let private timeCostToJson (timeCost: StageTimeCostEstimate) =
    timeCost.CalibrationKey
    |> Option.map (fun key ->
        { calibrationKey = key
          cpuCostUnits = timeCost.CpuCostUnits
          nativeCostUnits = timeCost.NativeCostUnits
          ioReadBytes = timeCost.IoReadBytes
          ioWriteBytes = timeCost.IoWriteBytes
          ioReadOps = timeCost.IoReadOps
          ioWriteOps = timeCost.IoWriteOps })

let private scaleTimeCostEstimate (factor: uint64) (timeCost: StageTimeCostEstimate) =
    let factorFloat = float factor
    { timeCost with
        CpuCostUnits = timeCost.CpuCostUnits * factorFloat
        NativeCostUnits = timeCost.NativeCostUnits * factorFloat
        IoReadBytes = timeCost.IoReadBytes * factor
        IoWriteBytes = timeCost.IoWriteBytes * factor
        IoReadOps = timeCost.IoReadOps * factor
        IoWriteOps = timeCost.IoWriteOps * factor }

let private timeCostFromJson (timeCost: ProbeTimeCostJson) =
    { CpuCostUnits = timeCost.cpuCostUnits
      NativeCostUnits = timeCost.nativeCostUnits
      IoReadBytes = timeCost.ioReadBytes
      IoWriteBytes = timeCost.ioWriteBytes
      IoReadOps = timeCost.ioReadOps
      IoWriteOps = timeCost.ioWriteOps
      CalibrationKey = Some timeCost.calibrationKey
      Tags = [] }

let private coefficientsFromTimeCost elapsedMilliseconds (timeCost: StageTimeCostEstimate) =
    let elapsed = elapsedMilliseconds
    { CpuMillisecondsPerUnit =
        if timeCost.CpuCostUnits > 0.0 then elapsed / timeCost.CpuCostUnits else 0.0
      NativeMillisecondsPerUnit =
        if timeCost.NativeCostUnits > 0.0 then elapsed / timeCost.NativeCostUnits else 0.0
      IoReadMillisecondsPerByte =
        if timeCost.IoReadBytes > 0UL then elapsed / float timeCost.IoReadBytes else 0.0
      IoWriteMillisecondsPerByte =
        if timeCost.IoWriteBytes > 0UL then elapsed / float timeCost.IoWriteBytes else 0.0
      IoReadMillisecondsPerOp =
        if timeCost.IoReadBytes = 0UL && timeCost.IoReadOps > 0UL then elapsed / float timeCost.IoReadOps else 0.0
      IoWriteMillisecondsPerOp =
        if timeCost.IoWriteBytes = 0UL && timeCost.IoWriteOps > 0UL then elapsed / float timeCost.IoWriteOps else 0.0 }

let private runProbe name description probeParameters execute (planFactory: unit -> Plan<unit, _>) =
    printfn "Probing %s" name
    let metadataPlan = planFactory()
    let source = sourceToJson metadataPlan.sourcePeek
    let observedElements = observedElements metadataPlan
    let observedBytes = observedBytes source observedElements

    for _ in 1..warmupCount do
        forceFullGc()
        planFactory() |> execute
        forceFullGc()

    let iterations =
        [| for i in 1..repetitionCount do
               forceFullGc()
               let plan = planFactory()
               let stopwatch = Stopwatch.StartNew()
               let _, snapshot =
                   MemoryProbe.measure sampleIntervalMs (fun () ->
                       execute plan)
               stopwatch.Stop()
               let elapsedTotalMilliseconds = stopwatch.Elapsed.TotalMilliseconds
               let throughput =
                   if observedElements > 0UL && elapsedTotalMilliseconds > 0.0 then
                       float observedElements / (elapsedTotalMilliseconds / 1000.0)
                   else
                       0.0
               let byteThroughput =
                   if observedBytes > 0UL && elapsedTotalMilliseconds > 0.0 then
                       float observedBytes / (elapsedTotalMilliseconds / 1000.0)
                   else
                       0.0
               forceFullGc()
               let postGcRss = MemoryProbe.currentRssBytes()
               let retained =
                   if postGcRss > snapshot.Baseline then
                       postGcRss - snapshot.Baseline
                   else
                       0UL

               { iteration = i
                 observedElements = observedElements
                 observedBytes = observedBytes
                 rssBaselineBytes = snapshot.Baseline
                 rssPeakBytes = snapshot.Peak
                 rssDeltaBytes = snapshot.Delta
                 rssPostGcBytes = postGcRss
                 rssRetainedDeltaBytes = retained
                 elapsedMilliseconds = stopwatch.ElapsedMilliseconds
                 elapsedTotalMilliseconds = elapsedTotalMilliseconds
                 elapsedTicks = stopwatch.ElapsedTicks
                 throughputElementsPerSecond = throughput
                 throughputBytesPerSecond = byteThroughput } |]

    let medianDelta = iterations |> Array.map _.rssDeltaBytes |> median
    let medianRetained = iterations |> Array.map _.rssRetainedDeltaBytes |> median
    let medianElapsed = iterations |> Array.map _.elapsedTotalMilliseconds |> medianFloat
    let medianThroughput = iterations |> Array.map _.throughputElementsPerSecond |> medianFloat
    let medianByteThroughput = iterations |> Array.map _.throughputBytesPerSecond |> medianFloat
    let medianIteration =
        iterations
        |> Array.sortBy (fun iteration ->
            let deltaDistance =
                if iteration.rssDeltaBytes > medianDelta then
                    iteration.rssDeltaBytes - medianDelta
                else
                    medianDelta - iteration.rssDeltaBytes
            let retainedDistance =
                if iteration.rssRetainedDeltaBytes > medianRetained then
                    iteration.rssRetainedDeltaBytes - medianRetained
                else
                    medianRetained - iteration.rssRetainedDeltaBytes
            let elapsedDistance = Math.Abs(iteration.elapsedTotalMilliseconds - medianElapsed)
            float (deltaDistance + retainedDistance) + elapsedDistance)
        |> Array.head
    let totalTimeUnitScale = max 1UL metadataPlan.length
    let toTotalTimeUnits timeCost =
        timeCost
        |> scaleTimeCostEstimate totalTimeUnitScale
        |> timeCostToJson
    let costPeakTime = metadataPlan.costPeak |> Option.bind (fun cost -> toTotalTimeUnits cost.Time)
    let costTimes =
        metadataPlan.costObservations
        |> List.rev
        |> List.choose (fun cost -> toTotalTimeUnits cost.Time)
        |> List.toArray

    { name = name
      description = description
      parameters = probeParameters
      observedElements = observedElements
      observedBytes = observedBytes
      predictedImagePeakBytes = metadataPlan.memPeak
      predictedMemoryPeakBytes = metadataPlan.memPeak
      actualMemoryDeltaBytes = medianDelta
      predictedElapsedMilliseconds = None
      actualElapsedMedianMilliseconds = medianElapsed
      rssBaselineBytes = medianIteration.rssBaselineBytes
      rssPeakBytes = medianIteration.rssPeakBytes
      rssDeltaBytes = medianIteration.rssDeltaBytes
      rssDeltaMinBytes = iterations |> Array.map _.rssDeltaBytes |> Array.min
      rssDeltaMedianBytes = medianDelta
      rssDeltaMaxBytes = iterations |> Array.map _.rssDeltaBytes |> Array.max
      rssRetainedDeltaMinBytes = iterations |> Array.map _.rssRetainedDeltaBytes |> Array.min
      rssRetainedDeltaMedianBytes = medianRetained
      rssRetainedDeltaMaxBytes = iterations |> Array.map _.rssRetainedDeltaBytes |> Array.max
      elapsedMilliseconds = medianIteration.elapsedMilliseconds
      elapsedTotalMilliseconds = medianIteration.elapsedTotalMilliseconds
      elapsedMinMilliseconds = iterations |> Array.map _.elapsedTotalMilliseconds |> Array.min
      elapsedMedianMilliseconds = medianElapsed
      elapsedMaxMilliseconds = iterations |> Array.map _.elapsedTotalMilliseconds |> Array.max
      throughputElementsPerSecond = medianIteration.throughputElementsPerSecond
      throughputMinElementsPerSecond = iterations |> Array.map _.throughputElementsPerSecond |> Array.min
      throughputMedianElementsPerSecond = medianThroughput
      throughputMaxElementsPerSecond = iterations |> Array.map _.throughputElementsPerSecond |> Array.max
      throughputBytesPerSecond = medianIteration.throughputBytesPerSecond
      throughputMinBytesPerSecond = iterations |> Array.map _.throughputBytesPerSecond |> Array.min
      throughputMedianBytesPerSecond = medianByteThroughput
      throughputMaxBytesPerSecond = iterations |> Array.map _.throughputBytesPerSecond |> Array.max
      warmupCount = warmupCount
      repetitionCount = repetitionCount
      source = source
      costPeakTime = costPeakTime
      costTimes = costTimes
      iterations = iterations }

let private runSinkProbe name description probeParameters (planFactory: unit -> Plan<unit, _>) =
    let execute plan = sink plan
    runProbe name description probeParameters execute planFactory

let private runDrainProbe name description probeParameters (planFactory: unit -> Plan<unit, _>) =
    let execute plan = drain plan |> ignore
    runProbe name description probeParameters execute planFactory

let private tryStageTimeCost (probe: ProbeResultJson) =
    match tryParameter "stage" probe with
    | Some stage ->
        let normalizedStage = normalizedIdentifier stage
        probe.costTimes
        |> Array.tryFind (fun timeCost ->
            (normalizedIdentifier timeCost.calibrationKey).Contains(normalizedStage))
    | None ->
        match probe.costTimes with
        | [| timeCost |] -> Some timeCost
        | _ -> None

let private coefficientObservations (probes: ProbeResultJson array) =
    let probesByOperationAndShape =
        probes
        |> Array.groupBy (fun probe -> operationName probe, calibrationParameterKey probe)
        |> Array.map (fun (key, matches) -> key, matches[0])
        |> Map.ofArray

    seq {
        for probe in probes do
            match tryParameter "baselineOperation" probe, tryStageTimeCost probe with
            | Some baselineOperation, Some timeCost ->
                let baselineKey = baselineOperation, calibrationParameterKey probe

                match probesByOperationAndShape |> Map.tryFind baselineKey with
                | Some baseline ->
                    let elapsed =
                        max 0.0 (probe.elapsedMedianMilliseconds - baseline.elapsedMedianMilliseconds)

                    if elapsed > 0.0 then
                        yield timeCost.calibrationKey, coefficientsFromTimeCost elapsed (timeCostFromJson timeCost)
                | None -> ()
            | None, Some timeCost when probe.costTimes.Length = 1 ->
                if probe.elapsedMedianMilliseconds > 0.0 then
                    yield timeCost.calibrationKey, coefficientsFromTimeCost probe.elapsedMedianMilliseconds (timeCostFromJson timeCost)
            | _ -> ()
    }
    |> Seq.toList

let private averageCoefficients (items: StageTimeCoefficients list) =
    let count = float items.Length
    let average selector =
        items |> List.sumBy selector |> fun total -> total / count

    { CpuMillisecondsPerUnit = average _.CpuMillisecondsPerUnit
      NativeMillisecondsPerUnit = average _.NativeMillisecondsPerUnit
      IoReadMillisecondsPerByte = average _.IoReadMillisecondsPerByte
      IoWriteMillisecondsPerByte = average _.IoWriteMillisecondsPerByte
      IoReadMillisecondsPerOp = average _.IoReadMillisecondsPerOp
      IoWriteMillisecondsPerOp = average _.IoWriteMillisecondsPerOp }

let private buildCalibrations probes =
    let calibrations = Dictionary<string, StageTimeCoefficients>()

    probes
    |> coefficientObservations
    |> List.groupBy fst
    |> List.iter (fun (key, observations) ->
        calibrations[key] <-
            observations
            |> List.map snd
            |> averageCoefficients)

    calibrations

let private tryPredictElapsedMilliseconds (calibrations: Dictionary<string, StageTimeCoefficients>) (probe: ProbeResultJson) =
    let mutable any = false
    let mutable total = 0.0

    for timeCost in probe.costTimes do
        match calibrations.TryGetValue timeCost.calibrationKey with
        | true, coefficients ->
            any <- true
            total <- total + StageTimeCoefficients.estimateMilliseconds coefficients (timeCostFromJson timeCost)
        | _ -> ()

    if any then Some total else None

let private attachPredictions calibrations probes =
    probes
    |> Array.map (fun probe ->
        { probe with
            predictedElapsedMilliseconds = tryPredictElapsedMilliseconds calibrations probe
            predictedMemoryPeakBytes = probe.predictedImagePeakBytes
            actualMemoryDeltaBytes = probe.rssDeltaMedianBytes
            actualElapsedMedianMilliseconds = probe.elapsedMedianMilliseconds })

let private probingCsvPrefix reportPath =
    let directory = Path.GetDirectoryName(reportPath: string)
    let name = Path.GetFileNameWithoutExtension reportPath

    if String.IsNullOrWhiteSpace directory then
        name
    else
        Path.Combine(directory, name)

let private writeProbeAnalysisCsvs reportPath (calibrations: Dictionary<string, StageTimeCoefficients>) (probes: ProbeResultJson array) =
    let prefix = probingCsvPrefix reportPath
    let rowsPath = prefix + "-rows.csv"
    let featuresPath = prefix + "-features.csv"
    let vectorsPath = prefix + "-vectors.csv"
    let calibrationsPath = prefix + "-calibrations.csv"
    let predictionsPath = prefix + "-predictions.csv"

    writeCsv
        rowsPath
        (seq {
            yield
                [ "rowId"
                  "source"
                  "description"
                  "operation"
                  "stage"
                  "baselineOperation"
                  "parameterKey"
                  "parameters"
                  "warmupCount"
                  "repetitionCount" ]

            for probe in probes do
                yield
                    [ probe.name
                      "probing"
                      probe.description
                      operationName probe
                      tryParameter "stage" probe |> Option.defaultValue ""
                      tryParameter "baselineOperation" probe |> Option.defaultValue ""
                      calibrationParameterKey probe
                      analysisParameterText probe
                      string probe.warmupCount
                      string probe.repetitionCount ]
        })

    writeCsv
        featuresPath
        (seq {
            yield [ "rowId"; "featureKey"; "value"; "featureKind" ]

            for probe in probes do
                yield [ probe.name; $"operation:{operationName probe}"; "1"; "operation" ]
                yield [ probe.name; $"stage:{stageFeatureKey probe}"; "1"; "stage" ]

                for feature in sampleCompatibleProbeFeatures probe do
                    yield [ probe.name; feature; "1"; "sampleCompatible" ]

                for timeCost in probe.costTimes do
                    yield [ probe.name; $"time:{timeCost.calibrationKey}:present"; "1"; "timePresence" ]

                    if timeCost.cpuCostUnits <> 0.0 then
                        yield [ probe.name; $"time:{timeCost.calibrationKey}:cpuUnits"; invariant timeCost.cpuCostUnits; "timeUnits" ]

                    if timeCost.nativeCostUnits <> 0.0 then
                        yield [ probe.name; $"time:{timeCost.calibrationKey}:nativeUnits"; invariant timeCost.nativeCostUnits; "timeUnits" ]

                    if timeCost.ioReadBytes <> 0UL then
                        yield [ probe.name; $"time:{timeCost.calibrationKey}:ioReadBytes"; invariantUInt64 timeCost.ioReadBytes; "timeUnits" ]

                    if timeCost.ioWriteBytes <> 0UL then
                        yield [ probe.name; $"time:{timeCost.calibrationKey}:ioWriteBytes"; invariantUInt64 timeCost.ioWriteBytes; "timeUnits" ]

                    if timeCost.ioReadOps <> 0UL then
                        yield [ probe.name; $"time:{timeCost.calibrationKey}:ioReadOps"; invariantUInt64 timeCost.ioReadOps; "timeUnits" ]

                    if timeCost.ioWriteOps <> 0UL then
                        yield [ probe.name; $"time:{timeCost.calibrationKey}:ioWriteOps"; invariantUInt64 timeCost.ioWriteOps; "timeUnits" ]
        })

    writeCsv
        vectorsPath
        (seq {
            yield [ "rowId"; "measurement"; "value"; "source" ]

            for probe in probes do
                yield [ probe.name; "actualElapsedMedianMilliseconds"; invariant probe.elapsedMedianMilliseconds; "probing" ]
                yield [ probe.name; "actualElapsedMinMilliseconds"; invariant probe.elapsedMinMilliseconds; "probing" ]
                yield [ probe.name; "actualElapsedMaxMilliseconds"; invariant probe.elapsedMaxMilliseconds; "probing" ]
                yield [ probe.name; "rssDeltaMedianBytes"; invariantUInt64 probe.rssDeltaMedianBytes; "probing" ]
                yield [ probe.name; "rssDeltaMinBytes"; invariantUInt64 probe.rssDeltaMinBytes; "probing" ]
                yield [ probe.name; "rssDeltaMaxBytes"; invariantUInt64 probe.rssDeltaMaxBytes; "probing" ]
                yield [ probe.name; "rssRetainedDeltaMedianBytes"; invariantUInt64 probe.rssRetainedDeltaMedianBytes; "probing" ]
                yield [ probe.name; "predictedMemoryPeakBytes"; invariantUInt64 probe.predictedMemoryPeakBytes; "probing" ]
                yield [ probe.name; "observedElements"; invariantUInt64 probe.observedElements; "probing" ]
                yield [ probe.name; "observedBytes"; invariantUInt64 probe.observedBytes; "probing" ]
                yield [ probe.name; "throughputMedianElementsPerSecond"; invariant probe.throughputMedianElementsPerSecond; "probing" ]
                yield [ probe.name; "throughputMedianBytesPerSecond"; invariant probe.throughputMedianBytesPerSecond; "probing" ]

                match probe.predictedElapsedMilliseconds with
                | Some predicted -> yield [ probe.name; "predictedElapsedMilliseconds"; invariant predicted; "probing" ]
                | None -> ()
        })

    writeCsv
        calibrationsPath
        (seq {
            yield
                [ "calibrationKey"
                  "cpuMillisecondsPerUnit"
                  "nativeMillisecondsPerUnit"
                  "ioReadMillisecondsPerByte"
                  "ioWriteMillisecondsPerByte"
                  "ioReadMillisecondsPerOp"
                  "ioWriteMillisecondsPerOp" ]

            for kvp in calibrations |> Seq.cast<KeyValuePair<string, StageTimeCoefficients>> |> Seq.sortBy (fun kvp -> kvp.Key) do
                let coefficients = kvp.Value

                yield
                    [ kvp.Key
                      invariant coefficients.CpuMillisecondsPerUnit
                      invariant coefficients.NativeMillisecondsPerUnit
                      invariant coefficients.IoReadMillisecondsPerByte
                      invariant coefficients.IoWriteMillisecondsPerByte
                      invariant coefficients.IoReadMillisecondsPerOp
                      invariant coefficients.IoWriteMillisecondsPerOp ]
        })

    writeCsv
        predictionsPath
        (seq {
            yield [ "rowId"; "actualElapsedMedianMilliseconds"; "predictedElapsedMilliseconds"; "residualMilliseconds" ]

            for probe in probes do
                match probe.predictedElapsedMilliseconds with
                | Some predicted ->
                    yield
                        [ probe.name
                          invariant probe.elapsedMedianMilliseconds
                          invariant predicted
                          invariant (probe.elapsedMedianMilliseconds - predicted) ]
                | None -> ()
        })

    printfn "Wrote probing analysis CSVs to %s-*.csv" prefix

let private sizeName size =
    $"{size.Width}x{size.Height}x{size.Depth}"

let private savedParameter key value =
    { Key = key
      Value = value
      UseInput = false }

let private savedNode id functionId x y parameters =
    { Id = id
      FunctionId = functionId
      X = x
      Y = y
      Parameters = parameters |> List.map (fun (key, value) -> savedParameter key value) |> List.toArray }

let private savedEdge fromNode toNode =
    { FromNode = fromNode
      FromKind = "dataOutput"
      FromPort = 0
      ToNode = toNode
      ToKind = "dataInput"
      ToPort = 0 }

let private outputPathForProbe name =
    Path.Combine("outputs", name).Replace('\\', '/')

let private repositoryRoot () =
    let cwd = Directory.GetCurrentDirectory()

    if File.Exists(Path.Combine(cwd, "StackProcessing.sln")) then
        cwd
    else
        let rec walk (dir: DirectoryInfo) =
            if isNull dir then
                cwd
            elif File.Exists(Path.Combine(dir.FullName, "StackProcessing.sln")) then
                dir.FullName
            else
                walk dir.Parent

        walk (DirectoryInfo cwd)

let private sampleDataPath name =
    Path.Combine(repositoryRoot (), "samples", "data", name).Replace('\\', '/')

let private graphForLinearPipeline name nodes =
    let spacedNodes =
        nodes
        |> List.mapi (fun index (id, functionId, parameters) ->
            savedNode id functionId (100.0 + float index * 220.0) 120.0 parameters)

    { Version = 1
      Nodes = spacedNodes |> List.toArray
      Edges =
        spacedNodes
        |> List.pairwise
        |> List.map (fun (left, right) -> savedEdge left.Id right.Id)
        |> List.toArray }

type GraphTemplate =
    { Name: string
      Description: string
      Features: string list
      Graph: SavedGraph }

type BottomUpInputConfig =
    { Shape: ImageSize
      Depth: uint
      ShapeInput: string
      NoisyInput: string
      NoisyPixelType: string
      TypedInputs: Map<string, string>
      TypedFormatInputs: Map<string * string, string> }

let private sourceNode probe =
    let width = probeParameter "width" probe
    let height = probeParameter "height" probe
    let depth = probeParameter "depth" probe
    let pixelType = probeParameter "pixelType" probe
    let capitalizedType =
        match pixelType.ToLowerInvariant() with
        | "uint8" -> "UInt8"
        | "float32" -> "Float32"
        | other -> other

    "source",
    "Zero",
    [ "availableMemory", string availableMemory + "UL"
      "type", capitalizedType
      "width", width
      "height", height
      "depth", depth ]

let private writeNode name =
    "write",
    "Write",
    [ "format", "Image stack"
      "output", outputPathForProbe name
      "suffix", ".tiff" ]

let private writeNodeWithSuffix name suffix =
    "write",
    "Write",
    [ "format", "Image stack"
      "output", outputPathForProbe name
      "suffix", suffix ]

let private writeZarrNode name depth =
    "write",
    "Write",
    [ "format", "OME-Zarr"
      "output", outputPathForProbe name
      "name", "benchmark"
      "depth", string depth
      "chunkX", "256"
      "chunkY", "256"
      "chunkZ", "16"
      "physicalSizeX", "1.0"
      "physicalSizeY", "1.0"
      "physicalSizeZ", "1.0"
      "maxConcurrentWrites", "0" ]

let private ignoreNode =
    "ignore", "Ignore", []

let private zeroNode pixelType width height depth =
    "source",
    "Zero",
    [ "availableMemory", string availableMemory + "UL"
      "type", pixelType
      "width", string width
      "height", string height
      "depth", string depth ]

let private emptyNode =
    "empty",
    "Empty",
    [ "availableMemory", string availableMemory + "UL" ]

let private coordinateNode axis width height depth =
    "source",
    $"Coordinate{axis}",
    [ "availableMemory", string availableMemory + "UL"
      "type", "Float32"
      "width", string width
      "height", string height
      "depth", string depth ]

let private readNodeFromWithSuffix input pixelType suffix =
    "read",
    "Read",
    [ "availableMemory", string availableMemory + "UL"
      "type", pixelType
      "format", "Image stack"
      "input", input
      "suffix", suffix ]

let private readZarrNode input pixelType =
    "read",
    "Read",
    [ "availableMemory", string availableMemory + "UL"
      "type", pixelType
      "format", "OME-Zarr"
      "input", input
      "thickDepth", "16"
      "multiscaleIndex", "0"
      "datasetIndex", "0"
      "timepoint", "0"
      "channel", "0"
      "maxParallelChunks", "0" ]

let private readNodeFrom input pixelType =
    readNodeFromWithSuffix input pixelType ".tiff"

let private readNode pixelType =
    readNodeFrom (sampleDataPath "rotatingBoxes") pixelType

let private bottomUpIoPixelTypes =
    [ "UInt8"
      "UInt16"
      "Float32" ]

let private bottomUpTiffPixelTypes =
    Set.ofList bottomUpIoPixelTypes

let private bottomUpComplexPixelTypes =
    []

let private bottomUpMhaPixelTypes =
    Set.empty

let private bottomUpSupportedFormats pixelType =
    [ if bottomUpTiffPixelTypes.Contains pixelType then
          ".tiff"
      if bottomUpMhaPixelTypes.Contains pixelType then
          ".mha" ]

let private bottomUpIoSuffix pixelType =
    bottomUpSupportedFormats pixelType
    |> List.tryHead
    |> Option.defaultValue ".mha"

let private formatLabel (suffix: string) =
    suffix.TrimStart('.').ToLowerInvariant()

let private castNode sourceType targetType =
    "cast",
    "Cast",
    [ "sourceType", sourceType
      "targetType", targetType ]

let private thresholdNode pixelType lower =
    "threshold",
    "Threshold",
    [ "type", pixelType
      "lower", lower
      "upper", "infinity" ]

let private windowSlabRoundtripNode pixelType windowSize =
    "windowSlabRoundtrip",
    "WindowSlabRoundtrip",
    [ "type", pixelType
      "windowSize", string windowSize ]

let private windowedCastNode sourceType targetType windowSize =
    "windowedCast",
    "WindowedCast",
    [ "sourceType", sourceType
      "targetType", targetType
      "windowSize", string windowSize ]

let private windowedThresholdNode pixelType lower windowSize =
    "windowedThreshold",
    "WindowedThreshold",
    [ "type", pixelType
      "lower", lower
      "upper", "infinity"
      "windowSize", string windowSize ]

let private intensityStretchNode id pixelType inputMinimum inputMaximum =
    id,
    "IntensityStretch",
    [ "type", pixelType
      "inputMinimum", inputMinimum
      "inputMaximum", inputMaximum
      "outputMinimum", "0.0"
      "outputMaximum", "255.0" ]

let private graphTemplateMatchesAnalysisTokens tokens (template: GraphTemplate) =
    if Set.isEmpty tokens then
        true
    else
        let candidates =
            seq {
                template.Name
                template.Description
                for feature in template.Features do
                    feature
                    match feature.Split([| ':' |], 2) with
                    | [| operation; _ |] -> operation
                    | _ -> ()
            }
            |> Seq.map normalizedIdentifier
            |> Seq.toList

        tokens
        |> Set.exists (fun (token: string) ->
            candidates
            |> List.exists (fun candidate ->
                candidate.Contains(token, StringComparison.Ordinal)
                || token.Contains(candidate, StringComparison.Ordinal)))

let private bottomUpGraphTemplates config =
    let sizeText = string config.Shape.Width
    let heightText = string config.Shape.Height
    let depthText = string config.Depth
    let sizeSuffix = sizeName config.Shape
    let writeFeatureFor suffix = $"Write:format=Image stack:suffix={suffix}:depth=1:chunkX=64:chunkY=64:chunkZ=8:maxConcurrentWrites=0:frameAxis=0"
    let writeFeature = writeFeatureFor ".tiff"
    let writeZarrFeature = "Write:format=OME-Zarr:depth=1:chunkX=256:chunkY=256:chunkZ=16:maxConcurrentWrites=0"
    let ignoreFeature = "Ignore"

    let emptyTemplate name description =
        { Name = name
          Description = description
          Features = [ "intercept" ]
          Graph = graphForLinearPipeline name [ emptyNode ] }

    let templateWithSink name description sinkFeature nodes sink =
        { Name = name
          Description = description
          Features = "intercept" :: sinkFeature :: []
          Graph = graphForLinearPipeline name (nodes @ [ sink ]) }

    let ignoreTemplate name description nodes =
        templateWithSink name description ignoreFeature nodes ignoreNode

    let writeTemplate name description nodes =
        templateWithSink name description writeFeature nodes (writeNode name)

    let writeTemplateWithSuffix name description suffix nodes =
        templateWithSink name description (writeFeatureFor suffix) nodes (writeNodeWithSuffix name suffix)

    let writeZarrTemplate name description depth nodes =
        templateWithSink name description writeZarrFeature nodes (writeZarrNode name depth)

    let writeUInt8Template name description nodes outputType =
        if outputType = "UInt8" then
            writeTemplate name description nodes
        else
            writeTemplate name description (nodes @ [ castNode outputType "UInt8" ])

    let typedInput pixelType =
        config.TypedInputs
        |> Map.tryFind pixelType
        |> Option.defaultValue
            (match pixelType with
             | "UInt8" -> config.ShapeInput
             | _ -> config.NoisyInput)

    let typedInputWithSuffix pixelType suffix =
        config.TypedFormatInputs
        |> Map.tryFind (pixelType, suffix)
        |> Option.defaultWith (fun () -> typedInput pixelType)

    let zeroUInt8 = zeroNode "UInt8" config.Shape.Width config.Shape.Height config.Depth
    let readTyped pixelType =
        readNodeFromWithSuffix (typedInput pixelType) pixelType (bottomUpIoSuffix pixelType)

    let readUInt8 = readTyped "UInt8"
    let readUInt16 = readTyped "UInt16"
    let readInt32 = readTyped "Int32"
    let readFloat = readTyped "Float32"
    let readFloat32 = readTyped "Float32"

    let coreScalarTypes =
        [ "uint8", "UInt8", readUInt8, "128.0", "2"
          "uint16", "UInt16", readUInt16, "32768.0", "2"
          "int32", "Int32", readInt32, "128.0", "2"
          "float32", "Float32", readFloat32, "0.5", "1.25" ]

    let primaryFloatTypes =
        [ "float32", "Float32", readFloat32
          "float64", "Float32", readFloat ]

    let emptyLayer =
        [| emptyTemplate "bottomup-00-empty" "Empty graph for process startup/shutdown intercept." |]

    let sourceLayer =
        [| let mutable index = 1
           for pixelType in bottomUpIoPixelTypes do
               let typeKey = pixelType.ToLowerInvariant()
               let zero = zeroNode pixelType config.Shape.Width config.Shape.Height config.Depth

               yield
                   ignoreTemplate
                       (sprintf "bottomup-%02d-zero-%s-ignore-%s" index typeKey sizeSuffix)
                       $"Minimal {pixelType} zero traversal through ignore."
                       [ zero ]
               index <- index + 1

               for suffix in bottomUpSupportedFormats pixelType do
                   let formatKey = formatLabel suffix
                   let read = readNodeFromWithSuffix (typedInputWithSuffix pixelType suffix) pixelType suffix

                   yield
                       writeTemplateWithSuffix
                           (sprintf "bottomup-%02d-zero-%s-%s-write-%s" index typeKey formatKey sizeSuffix)
                           $"{pixelType} zero source written to {suffix} stack."
                           suffix
                           [ zero ]
                   index <- index + 1

                   yield
                       ignoreTemplate
                           (sprintf "bottomup-%02d-read-%s-%s-ignore-%s" index typeKey formatKey sizeSuffix)
                           $"{pixelType} {suffix} read consumed without writing."
                           [ read ]
                   index <- index + 1

                   yield
                       writeTemplateWithSuffix
                           (sprintf "bottomup-%02d-read-%s-%s-write-%s" index typeKey formatKey sizeSuffix)
                           $"{pixelType} {suffix} read then write."
                           suffix
                           [ read ]
                   index <- index + 1

               if pixelType = "UInt8" || pixelType = "UInt16" then
                   match config.TypedFormatInputs |> Map.tryFind (pixelType, ".zarr") with
                   | Some zarrInput ->
                       let formatKey = "zarr"
                       let read = readZarrNode zarrInput pixelType

                       yield
                           ignoreTemplate
                               (sprintf "bottomup-%02d-read-%s-%s-ignore-%s" index typeKey formatKey sizeSuffix)
                               $"{pixelType} OME-Zarr read consumed without writing."
                               [ read ]
                       index <- index + 1

                       yield
                           writeZarrTemplate
                               (sprintf "bottomup-%02d-read-%s-%s-write-%s" index typeKey formatKey sizeSuffix)
                               $"{pixelType} OME-Zarr read then write."
                               config.Depth
                               [ read ]
                       index <- index + 1
                   | None -> ()

           for complexPixelType in bottomUpComplexPixelTypes do
               match config.TypedFormatInputs |> Map.tryFind (complexPixelType, ".mha") with
               | Some complexInput ->
                   let typeKey = complexPixelType.ToLowerInvariant()
                   let suffix = ".mha"
                   let formatKey = formatLabel suffix
                   let read = readNodeFromWithSuffix complexInput complexPixelType suffix

                   yield
                       ignoreTemplate
                           (sprintf "bottomup-%02d-read-%s-%s-ignore-%s" index typeKey formatKey sizeSuffix)
                           $"{complexPixelType} .mha read consumed without writing."
                           [ read ]
                   index <- index + 1

                   yield
                       writeTemplateWithSuffix
                           (sprintf "bottomup-%02d-read-%s-%s-write-%s" index typeKey formatKey sizeSuffix)
                           $"{complexPixelType} .mha read then write."
                           suffix
                           [ read ]
                   index <- index + 1
               | None -> () |]

    let readCastLayer =
        let castPairs =
            [ "uint8-float32", "UInt8", "Float32"
              "uint8-float64", "UInt8", "Float32"
              "uint16-float32", "UInt16", "Float32"
              "uint16-float64", "UInt16", "Float32"
              "int32-float64", "Int32", "Float32"
              "float32-float64", "Float32", "Float32"
              "float64-float32", "Float32", "Float32" ]

        [| let mutable index = 1
           for caseName, diskType, targetType in castPairs do
               let targetSuffix = bottomUpIoSuffix targetType
               for diskSuffix in bottomUpSupportedFormats diskType do
                   let formatKey = formatLabel diskSuffix
                   let diskInput = typedInputWithSuffix diskType diskSuffix
                   let implicitRead = readNodeFromWithSuffix diskInput targetType diskSuffix
                   let explicitRead = readNodeFromWithSuffix diskInput diskType diskSuffix
                   let explicitNodes = [ explicitRead; castNode diskType targetType ]

                   yield
                       ignoreTemplate
                           (sprintf "bottomup-%02d-read-%s-%s-implicit-ignore-%s" index caseName formatKey sizeSuffix)
                           $"{diskType} {diskSuffix} on disk read directly as {targetType}, consumed without writing."
                           [ implicitRead ]
                   index <- index + 1

                   yield
                       ignoreTemplate
                           (sprintf "bottomup-%02d-read-%s-%s-explicit-cast-ignore-%s" index caseName formatKey sizeSuffix)
                           $"{diskType} {diskSuffix} on disk read as {diskType}, cast to {targetType}, consumed without writing."
                           explicitNodes
                   index <- index + 1

                   yield
                       writeTemplateWithSuffix
                           (sprintf "bottomup-%02d-read-%s-%s-implicit-write-%s" index caseName formatKey sizeSuffix)
                           $"{diskType} {diskSuffix} on disk read directly as {targetType}, then written."
                           targetSuffix
                           [ implicitRead ]
                   index <- index + 1

                   yield
                       writeTemplateWithSuffix
                           (sprintf "bottomup-%02d-read-%s-%s-explicit-cast-write-%s" index caseName formatKey sizeSuffix)
                           $"{diskType} {diskSuffix} on disk read as {diskType}, cast to {targetType}, then written."
                           targetSuffix
                           explicitNodes
                   index <- index + 1 |]

    let syntheticSourcesLayer =
        [| yield
               ignoreTemplate
               $"bottomup-10-normalNoise-float-ignore-{sizeSuffix}"
               "Float32 normalNoise source consumed without writing."
               [ "noise", "NormalNoise", [ "availableMemory", string availableMemory + "UL"; "type", "Float32"; "width", sizeText; "height", heightText; "depth", depthText; "mean", "128.0"; "std", "25.0" ] ]
           yield
               writeUInt8Template
               $"bottomup-11-normalNoise-float-write-{sizeSuffix}"
               "Float32 normalNoise source cast and written."
               [ "noise", "NormalNoise", [ "availableMemory", string availableMemory + "UL"; "type", "Float32"; "width", sizeText; "height", heightText; "depth", depthText; "mean", "128.0"; "std", "25.0" ] ]
               "Float32"
           yield
               ignoreTemplate
               $"bottomup-12-addNormalNoise-uint8-ignore-{sizeSuffix}"
               "UInt8 zero plus addNormalNoise consumed without writing."
               [ zeroUInt8; "noise", "AddNormalNoise", [ "type", "UInt8"; "mean", "128.0"; "std", "50.0" ] ]
           yield
               writeTemplate
               $"bottomup-13-addNormalNoise-uint8-write-{sizeSuffix}"
               "UInt8 zero plus addNormalNoise written."
               [ zeroUInt8; "noise", "AddNormalNoise", [ "type", "UInt8"; "mean", "128.0"; "std", "50.0" ] ]
           yield
               ignoreTemplate
               $"bottomup-14-saltAndPepper-source-ignore-{sizeSuffix}"
               "UInt8 saltAndPepperNoise source consumed without writing."
               [ "noise", "SaltAndPepperNoise", [ "availableMemory", string availableMemory + "UL"; "type", "UInt8"; "width", sizeText; "height", heightText; "depth", depthText; "probability", "0.02" ] ]
           yield
               writeTemplate
               $"bottomup-15-saltAndPepper-source-write-{sizeSuffix}"
               "UInt8 saltAndPepperNoise source written."
               [ "noise", "SaltAndPepperNoise", [ "availableMemory", string availableMemory + "UL"; "type", "UInt8"; "width", sizeText; "height", heightText; "depth", depthText; "probability", "0.02" ] ]
           yield
               ignoreTemplate
               $"bottomup-16-poissonNoise-source-ignore-{sizeSuffix}"
               "Float32 poissonNoise source consumed without writing."
               [ "noise", "PoissonNoise", [ "availableMemory", string availableMemory + "UL"; "type", "Float32"; "width", sizeText; "height", heightText; "depth", depthText; "lambda", "2.0" ] ]
           yield
               writeUInt8Template
               $"bottomup-17-poissonNoise-source-write-{sizeSuffix}"
               "Float32 poissonNoise source cast and written."
               [ "noise", "PoissonNoise", [ "availableMemory", string availableMemory + "UL"; "type", "Float32"; "width", sizeText; "height", heightText; "depth", depthText; "lambda", "2.0" ] ]
               "Float32"
           yield
               ignoreTemplate
               $"bottomup-18-speckleNoise-source-ignore-{sizeSuffix}"
               "Float32 speckleNoise source consumed without writing."
               [ "noise", "SpeckleNoise", [ "availableMemory", string availableMemory + "UL"; "type", "Float32"; "width", sizeText; "height", heightText; "depth", depthText; "std", "0.5" ] ]
           yield
               writeUInt8Template
               $"bottomup-19-speckleNoise-source-write-{sizeSuffix}"
               "Float32 speckleNoise source cast and written."
               [ "noise", "SpeckleNoise", [ "availableMemory", string availableMemory + "UL"; "type", "Float32"; "width", sizeText; "height", heightText; "depth", depthText; "std", "0.5" ] ]
               "Float32"

           for axis in [ "X"; "Y"; "Z" ] do
               let axisLower = axis.ToLowerInvariant()
               let node = coordinateNode axis config.Shape.Width config.Shape.Height config.Depth

               yield
                   ignoreTemplate
                       $"bottomup-20-coordinate{axisLower}-float-ignore-{sizeSuffix}"
                       $"Float32 coordinate{axis} source consumed without writing."
                       [ node ]

               yield
                   writeUInt8Template
                       $"bottomup-21-coordinate{axisLower}-float-write-{sizeSuffix}"
                       $"Float32 coordinate{axis} source cast and written."
                       [ node ]
                       "Float32" |]

    let simpleUnaryLayer =
        [| yield ignoreTemplate $"bottomup-20-cast-float32-uint8-ignore-{sizeSuffix}" "Float32 read cast to UInt8 and consumed." [ readFloat32; castNode "Float32" "UInt8" ]
           yield writeTemplate $"bottomup-21-cast-float32-uint8-write-{sizeSuffix}" "Float32 read cast to UInt8 and written." [ readFloat32; castNode "Float32" "UInt8" ]
           yield ignoreTemplate $"bottomup-22-cast-uint16-uint8-ignore-{sizeSuffix}" "UInt16 read cast to UInt8 and consumed." [ readUInt16; castNode "UInt16" "UInt8" ]
           yield writeTemplate $"bottomup-23-cast-uint16-uint8-write-{sizeSuffix}" "UInt16 read cast to UInt8 and written." [ readUInt16; castNode "UInt16" "UInt8" ]

           let mutable index = 24
           for typeKey, pixelType, read, threshold, scalar in coreScalarTypes do
               yield
                   ignoreTemplate
                       (sprintf "bottomup-%02d-threshold-%s-ignore-%s" index typeKey sizeSuffix)
                       $"{pixelType} read threshold consumed without writing."
                       [ read; thresholdNode pixelType threshold ]
               index <- index + 1

               yield
                   writeUInt8Template
                       (sprintf "bottomup-%02d-threshold-%s-write-%s" index typeKey sizeSuffix)
                       $"{pixelType} read threshold cast and written."
                       [ read; thresholdNode pixelType threshold ]
                       "UInt8"
               index <- index + 1

               yield
                   ignoreTemplate
                       (sprintf "bottomup-%02d-imageOpScalar-%s-ignore-%s" index typeKey sizeSuffix)
                       $"{pixelType} read multiplied by scalar and consumed."
                       [ read; "op", "ImageOpScalar", [ "operation", "*"; "type", pixelType; "value", scalar ] ]
               index <- index + 1

               yield
                   writeUInt8Template
                       (sprintf "bottomup-%02d-imageOpScalar-%s-write-%s" index typeKey sizeSuffix)
                       $"{pixelType} read multiplied by scalar, cast, and written."
                       [ read; "op", "ImageOpScalar", [ "operation", "*"; "type", pixelType; "value", scalar ] ]
                       pixelType
               index <- index + 1

           for typeKey, pixelType, read in primaryFloatTypes do
               yield
                   ignoreTemplate
                       (sprintf "bottomup-%02d-unarySqrt-%s-ignore-%s" index typeKey sizeSuffix)
                       $"{pixelType} read square-rooted and consumed."
                       [ read; "op", "UnaryImageFunction", [ "function", "sqrt" ] ]
               index <- index + 1

               yield
                   writeUInt8Template
                       (sprintf "bottomup-%02d-unarySqrt-%s-write-%s" index typeKey sizeSuffix)
                       $"{pixelType} read square-rooted, cast, and written."
                       [ read; "op", "UnaryImageFunction", [ "function", "sqrt" ] ]
                       pixelType
               index <- index + 1 |]

    let windowSlabLayer =
        let windowSizes =
            [ 3u; 5u; 9u ]
            |> List.filter (fun windowSize -> windowSize <= config.Shape.Depth)

        [| for windowSize in windowSizes do
               yield ignoreTemplate $"bottomup-20-windowSlab-roundtrip-float32-ignore-w{windowSize}-{sizeSuffix}" $"Float32 read through window-to-slab roundtrip with window {windowSize}, then consumed." [ readFloat32; windowSlabRoundtripNode "Float32" windowSize ]
               yield writeUInt8Template $"bottomup-21-windowSlab-roundtrip-float32-write-w{windowSize}-{sizeSuffix}" $"Float32 read through window-to-slab roundtrip with window {windowSize}, cast, and written." [ readFloat32; windowSlabRoundtripNode "Float32" windowSize ] "Float32"
               yield ignoreTemplate $"bottomup-22-windowSlab-cast-float32-uint8-ignore-w{windowSize}-{sizeSuffix}" $"Float32 read windowed through slab cast to UInt8 with window {windowSize}, then consumed." [ readFloat32; windowedCastNode "Float32" "UInt8" windowSize ]
               yield writeTemplate $"bottomup-23-windowSlab-cast-float32-uint8-write-w{windowSize}-{sizeSuffix}" $"Float32 read windowed through slab cast to UInt8 with window {windowSize}, then written." [ readFloat32; windowedCastNode "Float32" "UInt8" windowSize ]
               yield ignoreTemplate $"bottomup-24-windowSlab-threshold-uint8-ignore-w{windowSize}-{sizeSuffix}" $"UInt8 read windowed through slab threshold with window {windowSize}, then consumed." [ readUInt8; windowedThresholdNode "UInt8" "128.0" windowSize ]
               yield writeTemplate $"bottomup-25-windowSlab-threshold-uint8-write-w{windowSize}-{sizeSuffix}" $"UInt8 read windowed through slab threshold with window {windowSize}, then written." [ readUInt8; windowedThresholdNode "UInt8" "128.0" windowSize ]
               yield ignoreTemplate $"bottomup-26-windowSlab-threshold-uint16-ignore-w{windowSize}-{sizeSuffix}" $"UInt16 read windowed through slab threshold with window {windowSize}, then consumed." [ readUInt16; windowedThresholdNode "UInt16" "32768.0" windowSize ]
               yield writeUInt8Template $"bottomup-27-windowSlab-threshold-uint16-write-w{windowSize}-{sizeSuffix}" $"UInt16 read windowed through slab threshold with window {windowSize}, then cast and written." [ readUInt16; windowedThresholdNode "UInt16" "32768.0" windowSize ] "UInt8" |]

    let windowLayer =
        let floatFilters =
            [ "gauss", "SmoothWGauss", [ "sigma", "3.0"; "outputRegionMode", "None"; "boundaryCondition", "None"; "windowSize", "None" ]
              "median", "SmoothWMedian", [ "radius", "3"; "windowSize", "7" ]
              "gradient", "GradientMagnitude", [ "windowSize", "7" ]
              "laplacian", "Laplacian", [ "windowSize", "7" ]
              "sobel", "SobelEdge", [ "windowSize", "7" ] ]

        let binaryOps =
            [ "erode", "Erode", [ "radius", "3" ]
              "dilate", "Dilate", [ "radius", "3" ]
              "opening", "Opening", [ "radius", "3" ]
              "closing", "Closing", [ "radius", "3" ]
              "binaryMedian", "BinaryMedian", [ "radius", "3"; "windowSize", "7" ]
              "binaryContour", "BinaryContour", [ "fullyConnected", "false"; "windowSize", "7" ]
              "fillSmallHoles", "FillSmallHoles", [ "maximumVolume", "128"; "connectivity", "TwentySix" ] ]

        let grayscaleOps =
            [ "grayErode", "GrayscaleErode", [ "radius", "3"; "windowSize", "7" ]
              "grayDilate", "GrayscaleDilate", [ "radius", "3"; "windowSize", "7" ]
              "grayOpening", "GrayscaleOpening", [ "radius", "3"; "windowSize", "7" ]
              "grayClosing", "GrayscaleClosing", [ "radius", "3"; "windowSize", "7" ]
              "blackTopHat", "BlackTopHat", [ "radius", "3"; "windowSize", "7" ]
              "whiteTopHat", "WhiteTopHat", [ "radius", "3"; "windowSize", "7" ]
              "morphGradient", "MorphologicalGradient", [ "radius", "3"; "windowSize", "7" ] ]

        [| let mutable index = 30
           for id, functionId, parameters in floatFilters do
               for typeKey, pixelType, read in primaryFloatTypes do
                   let typedParameters = ("type", pixelType) :: parameters
                   let nodes = [ read; "op", functionId, typedParameters ]

                   yield
                       ignoreTemplate
                           (sprintf "bottomup-%02d-%s-%s-ignore-%s" index id typeKey sizeSuffix)
                           $"{pixelType} {functionId} consumed without writing."
                           nodes
                   index <- index + 1

                   yield
                       writeUInt8Template
                           (sprintf "bottomup-%02d-%s-%s-write-%s" index id typeKey sizeSuffix)
                           $"{pixelType} {functionId} cast and written."
                           nodes
                           pixelType
                   index <- index + 1

           for id, functionId, parameters in binaryOps do
               let nodes = [ readUInt8; thresholdNode "UInt8" "128.0"; "op", functionId, parameters ]
               yield ignoreTemplate (sprintf "bottomup-%02d-%s-uint8-ignore-%s" index id sizeSuffix) $"UInt8 threshold then {functionId} consumed." nodes
               index <- index + 1
               yield writeTemplate (sprintf "bottomup-%02d-%s-uint8-write-%s" index id sizeSuffix) $"UInt8 threshold then {functionId} written." nodes
               index <- index + 1

           for id, functionId, parameters in grayscaleOps do
               for typeKey, pixelType, read, _, _ in coreScalarTypes do
                   if pixelType <> "Float32" then
                       let nodes = [ read; "op", functionId, ("type", pixelType) :: parameters ]
                       yield ignoreTemplate (sprintf "bottomup-%02d-%s-%s-ignore-%s" index id typeKey sizeSuffix) $"{pixelType} {functionId} consumed without writing." nodes
                       index <- index + 1
                       yield writeUInt8Template (sprintf "bottomup-%02d-%s-%s-write-%s" index id typeKey sizeSuffix) $"{pixelType} {functionId} cast and written." nodes pixelType
                       index <- index + 1 |]

    let intensityAndAdditiveLayer =
        [| ignoreTemplate
               $"bottomup-40-addSaltAndPepper-uint8-ignore-{sizeSuffix}"
               "UInt8 read plus addSaltAndPepperNoise consumed without writing."
               [ readUInt8; "noise", "AddSaltAndPepperNoise", [ "type", "UInt8"; "probability", "0.02" ] ]
           writeTemplate
               $"bottomup-41-addSaltAndPepper-uint8-write-{sizeSuffix}"
               "UInt8 read plus addSaltAndPepperNoise written."
               [ readUInt8; "noise", "AddSaltAndPepperNoise", [ "type", "UInt8"; "probability", "0.02" ] ]
           ignoreTemplate
               $"bottomup-42-addPoissonNoise-float32-ignore-{sizeSuffix}"
               "Float32 read plus addPoissonNoise consumed without writing."
               [ readFloat32; "noise", "AddPoissonNoise", [ "type", "Float32"; "lambda", "2.0" ] ]
           writeUInt8Template
               $"bottomup-43-addPoissonNoise-float32-write-{sizeSuffix}"
               "Float32 read plus addPoissonNoise cast and written."
               [ readFloat32; "noise", "AddPoissonNoise", [ "type", "Float32"; "lambda", "2.0" ] ]
               "Float32"
           ignoreTemplate
               $"bottomup-44-addSpeckleNoise-float32-ignore-{sizeSuffix}"
               "Float32 read plus addSpeckleNoise consumed without writing."
               [ readFloat32; "noise", "AddSpeckleNoise", [ "type", "Float32"; "std", "0.5" ] ]
           writeUInt8Template
               $"bottomup-45-addSpeckleNoise-float32-write-{sizeSuffix}"
               "Float32 read plus addSpeckleNoise cast and written."
               [ readFloat32; "noise", "AddSpeckleNoise", [ "type", "Float32"; "std", "0.5" ] ]
               "Float32"
           ignoreTemplate
               $"bottomup-46-shiftScale-float32-ignore-{sizeSuffix}"
               "Float32 read shifted and scaled, then consumed."
               [ readFloat32; "intensity", "ShiftScale", [ "type", "Float32"; "shift", "10.0"; "scale", "0.5" ] ]
           writeUInt8Template
               $"bottomup-47-shiftScale-float32-write-{sizeSuffix}"
               "Float32 read shifted/scaled, cast, and written."
               [ readFloat32; "intensity", "ShiftScale", [ "type", "Float32"; "shift", "10.0"; "scale", "0.5" ] ]
               "Float32"
           ignoreTemplate
               $"bottomup-48-intensityStretch-float32-ignore-{sizeSuffix}"
               "Float32 read intensity-stretched, then consumed."
               [ readFloat32; intensityStretchNode "stretch" "Float32" "0.0" "255.0" ]
           writeUInt8Template
               $"bottomup-49-intensityStretch-float32-write-{sizeSuffix}"
               "Float32 read intensity-stretched, cast, and written."
               [ readFloat32; intensityStretchNode "stretch" "Float32" "0.0" "255.0" ]
               "Float32" |]

    let geometryAndProjectionLayer =
        [| ignoreTemplate
               $"bottomup-50-crop-uint8-ignore-{sizeSuffix}"
               "UInt8 read cropped on every side and consumed."
               [ readUInt8; "geometry", "Crop", [ "type", "UInt8"; "beforeX", "4"; "afterX", "4"; "beforeY", "4"; "afterY", "4"; "beforeZ", "4"; "afterZ", "4" ] ]
           writeTemplate
               $"bottomup-51-crop-uint8-write-{sizeSuffix}"
               "UInt8 read cropped on every side and written."
               [ readUInt8; "geometry", "Crop", [ "type", "UInt8"; "beforeX", "4"; "afterX", "4"; "beforeY", "4"; "afterY", "4"; "beforeZ", "4"; "afterZ", "4" ] ]
           ignoreTemplate
               $"bottomup-52-padding-uint8-ignore-{sizeSuffix}"
               "UInt8 read padded on every side and consumed."
               [ readUInt8; "geometry", "CreatePadding", [ "type", "UInt8"; "beforeX", "4"; "afterX", "4"; "beforeY", "4"; "afterY", "4"; "beforeZ", "4"; "afterZ", "4"; "value", "0.0" ] ]
           writeTemplate
               $"bottomup-53-padding-uint8-write-{sizeSuffix}"
               "UInt8 read padded on every side and written."
               [ readUInt8; "geometry", "CreatePadding", [ "type", "UInt8"; "beforeX", "4"; "afterX", "4"; "beforeY", "4"; "afterY", "4"; "beforeZ", "4"; "afterZ", "4"; "value", "0.0" ] ]
           ignoreTemplate
               $"bottomup-54-resize-uint8-ignore-{sizeSuffix}"
               "UInt8 read resized to 48^3 and consumed."
               [ readUInt8; "geometry", "Resize", [ "type", "UInt8"; "width", "48"; "height", "48"; "depth", "48"; "interpolation", "Linear" ] ]
           writeTemplate
               $"bottomup-55-resize-uint8-write-{sizeSuffix}"
               "UInt8 read resized to 48^3 and written."
               [ readUInt8; "geometry", "Resize", [ "type", "UInt8"; "width", "48"; "height", "48"; "depth", "48"; "interpolation", "Linear" ] ]
           ignoreTemplate
               $"bottomup-56-resample-float-ignore-{sizeSuffix}"
               "Float32 read resampled by 0.75 and consumed."
               [ readFloat32; "geometry", "Resample", [ "type", "Float32"; "factorX", "0.75"; "factorY", "0.75"; "factorZ", "0.75"; "interpolation", "Linear" ] ]
           writeUInt8Template
               $"bottomup-57-resample-float-write-{sizeSuffix}"
               "Float32 read resampled by 0.75, cast, and written."
               [ readFloat32; "geometry", "Resample", [ "type", "Float32"; "factorX", "0.75"; "factorY", "0.75"; "factorZ", "0.75"; "interpolation", "Linear" ] ]
               "Float32"
           ignoreTemplate
               $"bottomup-58-permuteAxes-uint8-ignore-{sizeSuffix}"
               "UInt8 read with x/y axes swapped and consumed."
               [ readUInt8; "geometry", "PermuteAxes", [ "axes", "(1,0,2)"; "tileSize", "64" ] ]
           writeTemplate
               $"bottomup-59-permuteAxes-uint8-write-{sizeSuffix}"
               "UInt8 read with x/y axes swapped and written."
               [ readUInt8; "geometry", "PermuteAxes", [ "axes", "(1,0,2)"; "tileSize", "64" ] ]
           ignoreTemplate
               $"bottomup-60-sumProjection-uint8-ignore-{sizeSuffix}"
               "UInt8 read reduced to a sum projection and consumed."
               [ readUInt8; "projection", "SumProjection", [ "type", "UInt8"; "function", "Identity" ] ]
           writeUInt8Template
               $"bottomup-61-sumProjection-uint8-write-{sizeSuffix}"
               "UInt8 read reduced to a sum projection, cast, and written."
               [ readUInt8; "projection", "SumProjection", [ "type", "UInt8"; "function", "Identity" ] ]
               "Float32" |]

    let fourierAndVectorLayer =
        [| ignoreTemplate
               $"bottomup-70-fft-float-ignore-{sizeSuffix}"
               "Float32 read transformed with FFT and consumed."
               [ readFloat; "fft", "FFT", [ "type", "Float32"; "chunkX", "64"; "chunkY", "64"; "chunkZ", "16" ] ]
           ignoreTemplate
               $"bottomup-71-fft-shift-ignore-{sizeSuffix}"
               "Float32 read transformed with FFT, shifted, and consumed."
               [ readFloat
                 "fft", "FFT", [ "type", "Float32"; "chunkX", "64"; "chunkY", "64"; "chunkZ", "16" ]
                 "shift", "ShiftFFT", [ "chunkX", "64"; "chunkY", "64"; "chunkZ", "16" ] ]
           ignoreTemplate
               $"bottomup-72-fft-invfft-ignore-{sizeSuffix}"
               "Float32 read transformed with FFT/invFFT and consumed."
               [ readFloat
                 "fft", "FFT", [ "type", "Float32"; "chunkX", "64"; "chunkY", "64"; "chunkZ", "16" ]
                 "ifft", "InvFFT", [ "chunkX", "64"; "chunkY", "64"; "chunkZ", "16" ] ]
           writeUInt8Template
               $"bottomup-73-fft-invfft-write-{sizeSuffix}"
               "Float32 read transformed with FFT/invFFT, cast, and written."
               [ readFloat
                 "fft", "FFT", [ "type", "Float32"; "chunkX", "64"; "chunkY", "64"; "chunkZ", "16" ]
                 "ifft", "InvFFT", [ "chunkX", "64"; "chunkY", "64"; "chunkZ", "16" ] ]
               "Float32"
           ignoreTemplate
               $"bottomup-74-fft-modulus-ignore-{sizeSuffix}"
               "Float32 read transformed with FFT, converted to modulus, and consumed."
               [ readFloat
                 "fft", "FFT", [ "type", "Float32"; "chunkX", "64"; "chunkY", "64"; "chunkZ", "16" ]
                 "modulus", "ComplexModulus", [] ]
           writeUInt8Template
               $"bottomup-75-fft-modulus-write-{sizeSuffix}"
               "Float32 read transformed with FFT, converted to modulus, cast, and written."
               [ readFloat
                 "fft", "FFT", [ "type", "Float32"; "chunkX", "64"; "chunkY", "64"; "chunkZ", "16" ]
                 "modulus", "ComplexModulus", [] ]
               "Float32"
           ignoreTemplate
               $"bottomup-76-gradient-vector-ignore-{sizeSuffix}"
               "Float32 read converted to a gradient vector field and consumed."
               [ readFloat; "gradient", "Gradient", [ "order", "1"; "windowSize", "3" ] ]
           ignoreTemplate
               $"bottomup-77-gradient-vectorElement-ignore-{sizeSuffix}"
               "Float32 read gradient x-component consumed."
               [ readFloat
                 "gradient", "Gradient", [ "order", "1"; "windowSize", "3" ]
                 "component", "VectorElement", [ "component", "0" ] ]
           writeUInt8Template
               $"bottomup-78-gradient-vectorElement-write-{sizeSuffix}"
               "Float32 read gradient x-component cast and written."
               [ readFloat
                 "gradient", "Gradient", [ "order", "1"; "windowSize", "3" ]
                 "component", "VectorElement", [ "component", "0" ] ]
               "Float32"
           ignoreTemplate
               $"bottomup-79-structureTensor-vector-ignore-{sizeSuffix}"
               "Float32 read structure tensor eigensystem consumed."
               [ readFloat
                 "tensor", "StructureTensor", [ "sigma", "1.0"; "rho", "2.0" ]
                 "eigensystem", "SymmetricMatrixEigensystem", [] ]
           ignoreTemplate
               $"bottomup-80-structureTensor-vectorElement-ignore-{sizeSuffix}"
               "Float32 read structure tensor first component consumed."
               [ readFloat
                 "tensor", "StructureTensor", [ "sigma", "1.0"; "rho", "2.0" ]
                 "eigensystem", "SymmetricMatrixEigensystem", []
                 "component", "VectorElement", [ "component", "0" ] ]
           writeUInt8Template
               $"bottomup-81-structureTensor-vectorElement-write-{sizeSuffix}"
               "Float32 read structure tensor first component cast and written."
               [ readFloat
                 "tensor", "StructureTensor", [ "sigma", "1.0"; "rho", "2.0" ]
                 "eigensystem", "SymmetricMatrixEigensystem", []
                 "component", "VectorElement", [ "component", "0" ] ]
               "Float32" |]

    let keypointAndDistanceLayer =
        [| ignoreTemplate
               $"bottomup-90-signedDistanceBand-uint8-ignore-{sizeSuffix}"
               "UInt8 threshold then signed distance band consumed."
               [ readUInt8; thresholdNode "UInt8" "128.0"; "distance", "SignedDistanceBand", [ "bandRadius", "8"; "stride", "8" ] ]
           writeUInt8Template
               $"bottomup-91-signedDistanceBand-uint8-write-{sizeSuffix}"
               "UInt8 threshold then signed distance band cast and written."
               [ readUInt8; thresholdNode "UInt8" "128.0"; "distance", "SignedDistanceBand", [ "bandRadius", "8"; "stride", "8" ] ]
               "Float32"
           ignoreTemplate
               $"bottomup-92-siftKeypoints-float-ignore-{sizeSuffix}"
               "Float32 read SIFT keypoints consumed."
               [ readFloat; "keypoints", "SiftKeypoints", [ "type", "Float32"; "sigma0", "1.0"; "scaleFactor", "1.6"; "scaleLevels", "4"; "contrastThreshold", "0.03"; "stride", "8" ] ]
           ignoreTemplate
               $"bottomup-93-hessianKeypoints-float-ignore-{sizeSuffix}"
               "Float32 read Hessian keypoints consumed."
               [ readFloat; "keypoints", "HessianKeypoints", [ "type", "Float32"; "sigma", "1.0"; "responseKind", "Blob"; "threshold", "0.03"; "stride", "8" ] ]
           ignoreTemplate
               $"bottomup-94-harrisKeypoints-float-ignore-{sizeSuffix}"
               "Float32 read Harris keypoints consumed."
               [ readFloat; "keypoints", "Harris3DKeypoints", [ "type", "Float32"; "sigma", "1.0"; "rho", "1.5"; "k", "0.04"; "threshold", "0.03"; "stride", "8" ] ]
           ignoreTemplate
               $"bottomup-95-forstnerKeypoints-float-ignore-{sizeSuffix}"
               "Float32 read Forstner keypoints consumed."
               [ readFloat; "keypoints", "Forstner3DKeypoints", [ "type", "Float32"; "sigma", "1.0"; "rho", "1.5"; "threshold", "0.03"; "stride", "8" ] ] |]

    let dependencyBreakerLayer =
        [| ignoreTemplate
               $"bottomup-100-read-float3232-ignore-{sizeSuffix}"
               "Float32 read consumed without writing, to anchor Float32 read/cast probes."
               [ readFloat32 ]
           ignoreTemplate
               $"bottomup-101-cast-float32-uint8-ignore-{sizeSuffix}"
               "Float32 read cast to UInt8 and consumed."
               [ readFloat32; castNode "Float32" "UInt8" ]
           writeTemplate
               $"bottomup-102-cast-float32-uint8-write-{sizeSuffix}"
               "Float32 read cast to UInt8 and written."
               [ readFloat32; castNode "Float32" "UInt8" ]
           ignoreTemplate
               $"bottomup-103-addNormalNoise-uint8-read-ignore-{sizeSuffix}"
               "UInt8 read plus addNormalNoise consumed without writing."
               [ readUInt8; "noise", "AddNormalNoise", [ "type", "UInt8"; "mean", "128.0"; "std", "50.0" ] ]
           writeTemplate
               $"bottomup-104-addNormalNoise-uint8-read-write-{sizeSuffix}"
               "UInt8 read plus addNormalNoise written."
               [ readUInt8; "noise", "AddNormalNoise", [ "type", "UInt8"; "mean", "128.0"; "std", "50.0" ] ]
           ignoreTemplate
               $"bottomup-105-fft-modulus-only-ignore-{sizeSuffix}"
               "Float32 read FFT then modulus, isolating ComplexModulus against FFT."
               [ readFloat
                 "fft", "FFT", [ "type", "Float32"; "chunkX", "64"; "chunkY", "64"; "chunkZ", "16" ]
                 "modulus", "ComplexModulus", [] ]
           ignoreTemplate
               $"bottomup-106-fft-shift-modulus-ignore-{sizeSuffix}"
               "Float32 read FFT, shifted spectrum, then modulus, isolating ShiftFFT."
               [ readFloat
                 "fft", "FFT", [ "type", "Float32"; "chunkX", "64"; "chunkY", "64"; "chunkZ", "16" ]
                 "shift", "ShiftFFT", [ "chunkX", "64"; "chunkY", "64"; "chunkZ", "16" ]
                 "modulus", "ComplexModulus", [] ]
           writeUInt8Template
               $"bottomup-107-fft-modulus-write-{sizeSuffix}"
               "Float32 read FFT modulus cast and written."
               [ readFloat
                 "fft", "FFT", [ "type", "Float32"; "chunkX", "64"; "chunkY", "64"; "chunkZ", "16" ]
                 "modulus", "ComplexModulus", [] ]
               "Float32"
           ignoreTemplate
               $"bottomup-108-sumProjection-only-ignore-{sizeSuffix}"
               "UInt8 read sumProjection consumed without cast/write."
               [ readUInt8; "projection", "SumProjection", [ "type", "UInt8"; "function", "Identity" ] ]
           ignoreTemplate
               $"bottomup-109-sumProjection-cast-ignore-{sizeSuffix}"
               "UInt8 read sumProjection cast to UInt8 and consumed."
               [ readUInt8
                 "projection", "SumProjection", [ "type", "UInt8"; "function", "Identity" ]
                 castNode "Float32" "UInt8" ]
           writeTemplate
               $"bottomup-110-sumProjection-cast-write-{sizeSuffix}"
               "UInt8 read sumProjection cast to UInt8 and written."
               [ readUInt8
                 "projection", "SumProjection", [ "type", "UInt8"; "function", "Identity" ]
                 castNode "Float32" "UInt8" ]
           ignoreTemplate
               $"bottomup-111-permuteAxes-only-ignore-{sizeSuffix}"
               "UInt8 read x/y axis swap consumed without writing."
               [ readUInt8; "permute", "PermuteAxes", [ "axes", "(1,0,2)"; "tileSize", "64" ] ]
           ignoreTemplate
               $"bottomup-112-permuteAxes-back-ignore-{sizeSuffix}"
               "UInt8 read x/y swap and swap back consumed, isolating repeated transpose behavior."
               [ readUInt8
                 "permuteA", "PermuteAxes", [ "axes", "(1,0,2)"; "tileSize", "64" ]
                 "permuteB", "PermuteAxes", [ "axes", "(1,0,2)"; "tileSize", "64" ] ]
           writeTemplate
               $"bottomup-113-permuteAxes-back-write-{sizeSuffix}"
               "UInt8 read x/y swap and swap back written."
               [ readUInt8
                 "permuteA", "PermuteAxes", [ "axes", "(1,0,2)"; "tileSize", "64" ]
                 "permuteB", "PermuteAxes", [ "axes", "(1,0,2)"; "tileSize", "64" ] ] |]

    let reducerLayer =
        [| ignoreTemplate
               $"bottomup-120-computeStats-uint8-ignore-{sizeSuffix}"
               "UInt8 read reduced to image statistics."
               [ readUInt8; "stats", "ComputeStats", [] ]
           ignoreTemplate
               $"bottomup-121-computeStats-float-ignore-{sizeSuffix}"
               "Float32 read reduced to image statistics."
               [ readFloat; "stats", "ComputeStats", [] ]
           ignoreTemplate
               $"bottomup-122-estimateHistogram-float-ignore-{sizeSuffix}"
               "Float32 noisy input sampled into a histogram estimate."
               [ "histogram",
                 "EstimateHistogram",
                 [ "availableMemory", string availableMemory + "UL"
                   "type", "Float32"
                   "slices", string (min 16u config.Shape.Depth)
                   "input", config.NoisyInput
                   "suffix", ".tiff"
                   "down", "4"
                   "estimator", "DKWAndHoldout"
                   "confidence", "0.95" ] ]
           ignoreTemplate
               $"bottomup-123-connectedComponents-uint8-ignore-{sizeSuffix}"
               "UInt8 threshold then connected components consumed."
               [ readUInt8
                 thresholdNode "UInt8" "128.0"
                 "components", "ConnectedComponents", [ "windowSize", "3" ] ]
           ignoreTemplate
               $"bottomup-124-objectSizeStats-uint8-ignore-{sizeSuffix}"
               "UInt8 threshold streamed as objects and reduced to object-size statistics."
               [ readUInt8
                 thresholdNode "UInt8" "128.0"
                 "objects", "StreamConnectedObjects", [ "connectivity", "Six" ]
                 "measure", "MeasureObjects", []
                 "sizes", "ObjectSizeStats", [] ] |]

    [| "00-empty", emptyLayer
       "01-starters", sourceLayer
       "02-io-casts", readCastLayer
       "02-sources", syntheticSourcesLayer
       "03-simple-unary", simpleUnaryLayer
       "04-window-slab", windowSlabLayer
       "04-windowed-unary", windowLayer
       "05-intensity-and-additive", intensityAndAdditiveLayer
       "06-geometry-and-projection", geometryAndProjectionLayer
       "07-fourier-and-vector", fourierAndVectorLayer
       "08-keypoint-and-distance", keypointAndDistanceLayer
       "09-dependency-breakers", dependencyBreakerLayer
       "10-reducers", reducerLayer |]

let private generatedGraphTemplates () =
    let readUInt8Feature =
        "Read:type=UInt8:format=Image stack:suffix=.tiff:thickDepth=1:multiscaleIndex=0:datasetIndex=0:timepoint=0:channel=0:maxParallelChunks=0:frameAxis=0"
    let readFloatFeature =
        "Read:type=Float32:format=Image stack:suffix=.tiff:thickDepth=1:multiscaleIndex=0:datasetIndex=0:timepoint=0:channel=0:maxParallelChunks=0:frameAxis=0"
    let writeFeature =
        "Write:format=Image stack:suffix=.tiff:depth=1:chunkX=64:chunkY=64:chunkZ=8:maxConcurrentWrites=0:frameAxis=0"
    let thresholdFeature = "Threshold:type=UInt8:lower=128.0:upper=infinity"
    let castFloatToUInt8 = "Cast:sourceType=Float32:targetType=UInt8"
    let stretchFloat0255 = "IntensityStretch:type=Float32:inputMinimum=0.0:inputMaximum=255.0:outputMinimum=0.0:outputMaximum=255.0"

    let template name description features nodes =
        { Name = name
          Description = description
          Features = features @ [ writeFeature; "intercept" ]
          Graph = graphForLinearPipeline name (nodes @ [ writeNode name ]) }

    let uint8Template name description features nodes =
        template
            name
            description
            (readUInt8Feature :: features)
            (readNode "UInt8" :: nodes)

    let floatTemplate name description features nodes =
        template
            name
            description
            (readFloatFeature :: features)
            (readNode "Float32" :: nodes)

    let binaryOp functionId parameters =
        "op", functionId, parameters

    let thresholdNode' = thresholdNode "UInt8" "128.0"
    let stretchCastNodes =
        [ intensityStretchNode "stretch" "Float32" "0.0" "255.0"
          castNode "Float32" "UInt8" ]

    let uint8Morphology =
        [ "opening", "Opening", [ "radius", "3" ], "Opening:radius=3"
          "closing", "Closing", [ "radius", "3" ], "Closing:radius=3"
          "erode", "Erode", [ "radius", "3" ], "Erode:radius=3"
          "dilate", "Dilate", [ "radius", "3" ], "Dilate:radius=3"
          "binaryMedian", "BinaryMedian", [ "radius", "3"; "windowSize", "7" ], "BinaryMedian:radius=3:windowSize=7"
          "binaryContour", "BinaryContour", [ "fullyConnected", "false"; "windowSize", "7" ], "BinaryContour:fullyConnected=false:windowSize=7"
          "fillSmallHoles", "FillSmallHoles", [ "maximumVolume", "128"; "connectivity", "TwentySix" ], "FillSmallHoles:maximumVolume=128:connectivity=TwentySix" ]

    let grayscaleMorphology =
        [ "grayErode", "GrayscaleErode", [ "type", "UInt8"; "radius", "3"; "windowSize", "7" ], "GrayscaleErode:type=UInt8:radius=3:windowSize=7"
          "grayDilate", "GrayscaleDilate", [ "type", "UInt8"; "radius", "3"; "windowSize", "7" ], "GrayscaleDilate:type=UInt8:radius=3:windowSize=7"
          "grayOpening", "GrayscaleOpening", [ "type", "UInt8"; "radius", "3"; "windowSize", "7" ], "GrayscaleOpening:type=UInt8:radius=3:windowSize=7"
          "grayClosing", "GrayscaleClosing", [ "type", "UInt8"; "radius", "3"; "windowSize", "7" ], "GrayscaleClosing:type=UInt8:radius=3:windowSize=7"
          "blackTopHat", "BlackTopHat", [ "type", "UInt8"; "radius", "3"; "windowSize", "7" ], "BlackTopHat:type=UInt8:radius=3:windowSize=7"
          "whiteTopHat", "WhiteTopHat", [ "type", "UInt8"; "radius", "3"; "windowSize", "7" ], "WhiteTopHat:type=UInt8:radius=3:windowSize=7"
          "morphGradient", "MorphologicalGradient", [ "type", "UInt8"; "radius", "3"; "windowSize", "7" ], "MorphologicalGradient:type=UInt8:radius=3:windowSize=7" ]

    let filterOps =
        [ "median", "SmoothWMedian", [ "type", "Float32"; "radius", "3"; "windowSize", "7" ], "SmoothWMedian:type=Float32:radius=3:windowSize=7"
          "bilateral", "SmoothWBilateral", [ "type", "Float32"; "domainSigma", "1.5"; "rangeSigma", "30.0"; "windowSize", "7" ], "SmoothWBilateral:type=Float32:domainSigma=1.5:rangeSigma=30.0:windowSize=7"
          "gradient", "GradientMagnitude", [ "type", "Float32"; "windowSize", "7" ], "GradientMagnitude:type=Float32:windowSize=7"
          "sobel", "SobelEdge", [ "type", "Float32"; "windowSize", "7" ], "SobelEdge:type=Float32:windowSize=7"
          "laplacian", "Laplacian", [ "type", "Float32"; "windowSize", "7" ], "Laplacian:type=Float32:windowSize=7"
          "gauss", "SmoothWGauss", [ "sigma", "3.0"; "outputRegionMode", "None"; "boundaryCondition", "None"; "windowSize", "None" ], "SmoothWGauss:sigma=3.0:outputRegionMode=None:boundaryCondition=None:windowSize=None" ]

    [|
        for id, functionId, parameters, feature in uint8Morphology do
            yield
                uint8Template
                    $"bio-threshold-{id}-64x64x64"
                    $"Threshold then {functionId} cleanup."
                    [ thresholdFeature; feature ]
                    [ thresholdNode'; binaryOp functionId parameters ]

        for idA, functionA, parametersA, featureA in uint8Morphology do
            for idB, functionB, parametersB, featureB in uint8Morphology do
                if idA <> idB then
                    yield
                        uint8Template
                            $"bio-threshold-{idA}-{idB}-64x64x64"
                            $"Threshold then {functionA} and {functionB} cleanup."
                            [ thresholdFeature; featureA; featureB ]
                            [ thresholdNode'
                              $"{idA}A", functionA, parametersA
                              $"{idB}B", functionB, parametersB ]

        for id, functionId, parameters, feature in grayscaleMorphology do
            yield
                uint8Template
                    $"bio-grayscale-{id}-64x64x64"
                    $"Grayscale morphology probe using {functionId}."
                    [ feature ]
                    [ binaryOp functionId parameters ]

        for idA, functionA, parametersA, featureA in grayscaleMorphology do
            for idB, functionB, parametersB, featureB in grayscaleMorphology do
                if idA <> idB then
                    yield
                        uint8Template
                            $"bio-grayscale-{idA}-{idB}-64x64x64"
                            $"Grayscale morphology chain using {functionA} and {functionB}."
                            [ featureA; featureB ]
                            [ $"{idA}A", functionA, parametersA
                              $"{idB}B", functionB, parametersB ]

        for id, functionId, parameters, feature in filterOps do
            yield
                floatTemplate
                    $"bio-filter-{id}-64x64x64"
                    $"Float filter probe using {functionId}."
                    [ feature; stretchFloat0255; castFloatToUInt8 ]
                    ([ binaryOp functionId parameters ] @ stretchCastNodes)

        for idA, functionA, parametersA, featureA in filterOps do
            for idB, functionB, parametersB, featureB in filterOps do
                if idA <> idB then
                    yield
                        floatTemplate
                            $"bio-filter-{idA}-{idB}-64x64x64"
                            $"Float filter chain using {functionA} and {functionB}."
                            [ featureA; featureB; stretchFloat0255; castFloatToUInt8 ]
                            ([ $"{idA}A", functionA, parametersA
                               $"{idB}B", functionB, parametersB ] @ stretchCastNodes)

        yield
            uint8Template
                "bio-projection-threshold-64x64x64"
                "Sum projection followed by stretch and threshold."
                [ "SumProjection:type=UInt8:function=Identity"; stretchFloat0255; castFloatToUInt8; thresholdFeature ]
                ([ "projection", "SumProjection", [ "type", "UInt8"; "function", "Identity" ] ] @ stretchCastNodes @ [ thresholdNode' ])

        yield
            uint8Template
                "bio-resize-threshold-write-64x64x64"
                "Resize to canonical 64^3 then threshold."
                [ "Resize:type=UInt8:width=64:height=64:depth=64:interpolation=Linear"; thresholdFeature ]
                [ "resize", "Resize", [ "type", "UInt8"; "width", "64"; "height", "64"; "depth", "64"; "interpolation", "Linear" ]
                  thresholdNode' ]

        yield
            floatTemplate
                "bio-gauss-threshold-morphology-64x64x64"
                "Gaussian smoothing, cast-to-mask, then morphology cleanup."
                [ "SmoothWGauss:sigma=3.0:outputRegionMode=None:boundaryCondition=None:windowSize=None"
                  stretchFloat0255
                  castFloatToUInt8
                  thresholdFeature
                  "Opening:radius=3"
                  "Closing:radius=3" ]
                ([ "gauss", "SmoothWGauss", [ "sigma", "3.0"; "outputRegionMode", "None"; "boundaryCondition", "None"; "windowSize", "None" ] ]
                 @ stretchCastNodes
                 @ [ thresholdNode'
                     "opening", "Opening", [ "radius", "3" ]
                     "closing", "Closing", [ "radius", "3" ] ])
    |]

let private operationGraph (probe: ProbeResultJson) =
    let op = operationName probe
    let name = probe.name
    let pixelType =
        match (probeParameter "pixelType" probe).ToLowerInvariant() with
        | "uint8" -> "UInt8"
        | "float32" -> "Float32"
        | other -> other

    let source = sourceNode probe
    let write = writeNode name
    let withWrite nodes = graphForLinearPipeline name (nodes @ [ write ])
    let withUInt8Write nodes =
        if pixelType = "UInt8" then
            withWrite nodes
        else
            graphForLinearPipeline name (nodes @ [ castNode pixelType "UInt8"; write ])

    match op with
    | "zero"
    | "zero-write" ->
        Some(withWrite [ source ])
    | "read-ignore"
    | "read-write" ->
        Some(withWrite [ readNode pixelType ])
    | "noise-write" ->
        Some(
            withUInt8Write
                [ source
                  "noise", "AddNormalNoise", [ "type", pixelType; "mean", "128.0"; "std", "50.0" ] ])
    | "add-salt-and-pepper-write" ->
        Some(
            withWrite
                [ source
                  "noise", "AddSaltAndPepperNoise", [ "type", "UInt8"; "probability", "0.02" ] ])
    | "salt-and-pepper-write" ->
        Some(
            withWrite
                [ "noise",
                  "SaltAndPepperNoise",
                  [ "availableMemory", string availableMemory + "UL"
                    "type", "UInt8"
                    "width", probeParameter "width" probe
                    "height", probeParameter "height" probe
                    "depth", probeParameter "depth" probe
                    "probability", "0.02" ] ])
    | "poisson-noise-write" ->
        Some(
            withWrite
                [ "noise",
                  "PoissonNoise",
                  [ "availableMemory", string availableMemory + "UL"
                    "type", "Float32"
                    "width", probeParameter "width" probe
                    "height", probeParameter "height" probe
                    "depth", probeParameter "depth" probe
                    "scale", "2.0" ]
                  castNode "Float32" "UInt8" ])
    | "speckle-noise-write" ->
        Some(
            withWrite
                [ "noise",
                  "SpeckleNoise",
                  [ "availableMemory", string availableMemory + "UL"
                    "type", "Float32"
                    "width", probeParameter "width" probe
                    "height", probeParameter "height" probe
                    "depth", probeParameter "depth" probe
                    "std", "0.5" ]
                  castNode "Float32" "UInt8" ])
    | "add-poisson-speckle-write" ->
        Some(
            withWrite
                [ source
                  "poisson", "AddPoissonNoise", [ "type", "Float32"; "lambda", "2.0" ]
                  "speckle", "AddSpeckleNoise", [ "type", "Float32"; "std", "0.5" ]
                  castNode "Float32" "UInt8" ])
    | "threshold-write" ->
        Some(
            withWrite
                [ source
                  "noise", "AddNormalNoise", [ "type", pixelType; "mean", "128.0"; "std", "50.0" ]
                  "threshold", "Threshold", [ "type", pixelType; "lower", "128.0"; "upper", "infinity" ]
                  "scale", "ImageOpScalar", [ "operation", "*"; "type", "UInt8"; "value", "255" ] ])
    | "erode-write"
    | "dilate-write"
    | "opening-write"
    | "closing-write" ->
        let functionId =
            match op with
            | "erode-write" -> "Erode"
            | "dilate-write" -> "Dilate"
            | "opening-write" -> "Opening"
            | _ -> "Closing"
        Some(withWrite [ source; "op", functionId, [ "radius", "3" ] ])
    | "grayscale-erode-write"
    | "grayscale-dilate-write"
    | "grayscale-opening-write"
    | "grayscale-closing-write" ->
        let functionId =
            match op with
            | "grayscale-erode-write" -> "GrayscaleErode"
            | "grayscale-dilate-write" -> "GrayscaleDilate"
            | "grayscale-opening-write" -> "GrayscaleOpening"
            | _ -> "GrayscaleClosing"
        Some(withWrite [ source; "op", functionId, [ "type", "UInt8"; "radius", "3" ] ])
    | "black-top-hat-write"
    | "white-top-hat-write"
    | "morphological-gradient-write" ->
        let functionId =
            match op with
            | "black-top-hat-write" -> "BlackTopHat"
            | "white-top-hat-write" -> "WhiteTopHat"
            | _ -> "MorphologicalGradient"
        Some(withWrite [ source; "op", functionId, [ "type", "UInt8"; "radius", "3" ] ])
    | "binary-median-write" ->
        Some(withWrite [ source; "op", "BinaryMedian", [ "radius", "3" ] ])
    | "binary-contour-write" ->
        Some(withWrite [ source; "op", "BinaryContour", [ "fullyConnected", "false" ] ])
    | "fill-small-holes-write" ->
        Some(withWrite [ source; "op", "FillSmallHoles", [ "maximumVolume", "128"; "connectivity", "TwentySix" ] ])
    | "resize-write"
    | "resize64-write" ->
        let side = if op = "resize-write" then "96" else "64"
        Some(withWrite [ source; "resize", "Resize", [ "type", "UInt8"; "width", side; "height", side; "depth", side; "interpolation", "Linear" ] ])
    | "resample-write" ->
        Some(
            withWrite
                [ source
                  "resample", "Resample", [ "type", "Float32"; "factorX", "1.5"; "factorY", "1.5"; "factorZ", "1.5"; "interpolation", "Linear" ]
                  castNode "Float32" "UInt8" ])
    | "bio-threshold-morphology-write" ->
        Some(
            withWrite
                [ readNode "UInt8"
                  thresholdNode "UInt8" "128.0"
                  "opening", "Opening", [ "radius", "3" ]
                  "closing", "Closing", [ "radius", "3" ] ])
    | "bio-edge-filter-write" ->
        Some(
            withWrite
                [ readNode "Float32"
                  "median", "SmoothWMedian", [ "type", "Float32"; "radius", "3"; "windowSize", "7" ]
                  "gradient", "GradientMagnitude", [ "type", "Float32"; "windowSize", "7" ]
                  intensityStretchNode "stretch" "Float32" "0.0" "255.0"
                  castNode "Float32" "UInt8" ])
    | "bio-background-mask-write" ->
        Some(
            withWrite
                [ readNode "Float32"
                  "gauss", "SmoothWGauss", [ "sigma", "3.0"; "outputRegionMode", "None"; "boundaryCondition", "None"; "windowSize", "None" ]
                  intensityStretchNode "stretch" "Float32" "0.0" "255.0"
                  castNode "Float32" "UInt8"
                  thresholdNode "UInt8" "128.0" ])
    | "bio-projection-inspection-write" ->
        Some(
            withWrite
                [ readNode "UInt8"
                  "projection", "SumProjection", [ "type", "UInt8"; "function", "Identity" ]
                  intensityStretchNode "stretch" "Float32" "0.0" "255.0"
                  castNode "Float32" "UInt8" ])
    | _ -> None

let private writeProbeGraphs outputDir analysisTokens probes graphTemplates =
    if Directory.Exists outputDir then
        Directory.Delete(outputDir, true)
    Directory.CreateDirectory outputDir |> ignore

    let measuredWritten =
        probes
        |> Array.choose (fun probe ->
            operationGraph probe
            |> Option.map (fun graph ->
                let path = Path.Combine(outputDir, probe.name + ".json")
                PipelineGraphStorage.save path graph
                path))

    let templateWritten =
        graphTemplates
        |> Array.filter (graphTemplateMatchesAnalysisTokens analysisTokens)
        |> Array.choose (fun template ->
            let path = Path.Combine(outputDir, template.Name + ".json")
            if measuredWritten |> Array.exists (fun written -> Path.GetFileNameWithoutExtension written = template.Name) then
                None
            else
                PipelineGraphStorage.save path template.Graph
                Some path)

    printfn "Wrote %d measured probing graph(s) and %d generated template graph(s) to %s." measuredWritten.Length templateWritten.Length outputDir

let graphTemplatesForCalibration () =
    generatedGraphTemplates ()

let graphTemplateLayersForBottomUp config =
    bottomUpGraphTemplates config

let graphTemplatesForBottomUp config =
    bottomUpGraphTemplates config |> Array.collect snd

let writeGraphTemplates outputDir (graphTemplates: GraphTemplate array) =
    if Directory.Exists outputDir then
        Directory.Delete(outputDir, true)

    Directory.CreateDirectory outputDir |> ignore

    let written =
        graphTemplates
        |> Array.map (fun template ->
            let path = Path.Combine(outputDir, template.Name + ".json")
            PipelineGraphStorage.save path template.Graph
            path)

    printfn "Wrote %d calibration graph(s) to %s." written.Length outputDir

let private cleanDirectory path =
    if Directory.Exists(path) then
        Directory.Delete(path, true)
    Directory.CreateDirectory(path) |> ignore

let private createMovingBoxes<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (_foreground: 'T) size output =
    cleanDirectory output

    source availableMemory
    |> zero<'T> size.Width size.Height size.Depth
    >=> write output ".tiff"
    |> sink

let private createMovingBoxesWithSuffix<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> foreground size output suffix =
    if suffix = ".tiff" || suffix = ".tif" then
        createMovingBoxes<'T> foreground size output
    else
        cleanDirectory output
        source availableMemory
        |> zero<'T> size.Width size.Height size.Depth
        >=> write output suffix
        |> sink

let private createNoisyFromShape<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> foreground mean std size output =
    let temp = output + "_shape_tmp"
    createMovingBoxes<'T> foreground size temp
    cleanDirectory output

    source availableMemory
    |> read<'T> temp ".tiff"
    >=> addNormalNoise mean std
    >=> write output ".tiff"
    |> sink

    if Directory.Exists temp then
        Directory.Delete(temp, true)

let private createNoisyMovingBoxes noisyPixelType size output =
    match noisyPixelType with
    | "UInt8" -> createNoisyFromShape<uint8> 255uy 128.0 50.0 size output
    | "UInt16" -> createNoisyFromShape<uint16> 4096us 2048.0 512.0 size output
    | "Float32" -> createNoisyFromShape<float32> 255.0f 128.0 50.0 size output
    | _ -> failwithf "Unsupported noisy input type '%s'. Use UInt8, UInt16, or Float32." noisyPixelType

let private createTypedMovingBoxes pixelType size output =
    match pixelType with
    | "UInt8" -> createMovingBoxesWithSuffix<uint8> 255uy size output (bottomUpIoSuffix pixelType)
    | "UInt16" -> createMovingBoxesWithSuffix<uint16> 4096us size output (bottomUpIoSuffix pixelType)
    | "Float32" -> createMovingBoxesWithSuffix<float32> 255.0f size output (bottomUpIoSuffix pixelType)
    | _ -> failwithf "Unsupported IO probe input type '%s'." pixelType

let private createTypedMovingBoxesWithSuffix pixelType suffix size output =
    match pixelType with
    | "UInt8" -> createMovingBoxesWithSuffix<uint8> 255uy size output suffix
    | "UInt16" -> createMovingBoxesWithSuffix<uint16> 4096us size output suffix
    | "Float32" -> createMovingBoxesWithSuffix<float32> 255.0f size output suffix
    | _ -> failwithf "Unsupported IO probe input type '%s'." pixelType

let private createTypedMovingBoxesZarr pixelType _size output =
    cleanDirectory output
    failwithf "Zarr Probe input generation is deferred for Chunk-native probing; requested '%s' at '%s'." pixelType output

let createBottomUpInputsForShape (imageSize: ImageSize) (noisyPixelType: string) (inputRoot: string) =
    let normalizedNoisyType =
        match noisyPixelType.ToLowerInvariant() with
        | "uint8" -> "UInt8"
        | "uint16" -> "UInt16"
        | "float32" | "float32" -> "Float32"
        | other -> failwithf "Unsupported noisy input type '%s'. Use UInt8, UInt16, or Float32." other

    cleanDirectory inputRoot
    let shapeInput = Path.Combine(inputRoot, "shapes").Replace('\\', '/')
    let noisyInput = Path.Combine(inputRoot, "noisy").Replace('\\', '/')
    let typedInputRoot = Path.Combine(inputRoot, "typed")

    createMovingBoxes<uint8> 255uy imageSize shapeInput
    createNoisyMovingBoxes normalizedNoisyType imageSize noisyInput
    let typedFormatInputs =
        bottomUpIoPixelTypes
        |> List.collect (fun pixelType ->
            bottomUpSupportedFormats pixelType
            |> List.map (fun suffix ->
                let output =
                    Path.Combine(typedInputRoot, pixelType.ToLowerInvariant(), formatLabel suffix).Replace('\\', '/')
                createTypedMovingBoxesWithSuffix pixelType suffix imageSize output
                (pixelType, suffix), output))
        |> Map.ofList

    let typedInputs =
        bottomUpIoPixelTypes
        |> List.choose (fun pixelType ->
            let suffix = bottomUpIoSuffix pixelType
            typedFormatInputs
            |> Map.tryFind (pixelType, suffix)
            |> Option.map (fun output -> pixelType, output))
        |> Map.ofList

    { Shape = imageSize
      Depth = imageSize.Depth
      ShapeInput = shapeInput
      NoisyInput = noisyInput
      NoisyPixelType = normalizedNoisyType
      TypedInputs = typedInputs
      TypedFormatInputs = typedFormatInputs }

let createBottomUpInputsWithDepth (size: uint) (depth: uint) (noisyPixelType: string) (inputRoot: string) =
    createBottomUpInputsForShape { Width = size; Height = size; Depth = depth } noisyPixelType inputRoot

let createBottomUpInputs (size: uint) (noisyPixelType: string) (inputRoot: string) =
    createBottomUpInputsWithDepth size size noisyPixelType inputRoot

let private createInputStack size inputDir =
    Directory.CreateDirectory(inputDir) |> ignore
    source availableMemory
    |> zero<uint8> size.Width size.Height size.Depth
    >=> addNormalNoise 128.0 50.0
    >=> write inputDir ".tiff"
    |> sink

type ProbeOptions =
    { ReportPath: string
      AnalysisFeaturesPath: string option
      EmitJsonDirectory: string option
      LowSupportThreshold: int option
      NonBoilerplate: bool
      SqrtOnly: bool
      StackUnstackOnly: bool
      ConvolutionBreakdownOnly: bool
      DiscreteGaussianBreakdownOnly: bool }

let private parseArgs (args: string array) =
    let mutable reportPath = None
    let mutable analysisFeaturesPath = None
    let mutable emitJsonDirectory = None
    let mutable lowSupportThreshold = Some 2
    let mutable nonBoilerplate = false
    let mutable sqrtOnly = false
    let mutable stackUnstackOnly = false
    let mutable convolutionBreakdownOnly = false
    let mutable discreteGaussianBreakdownOnly = false
    let mutable i = 0

    while i < args.Length do
        match args[i] with
        | "--analysis-features" ->
            if i + 1 >= args.Length then
                failwith "Expected a path after --analysis-features."
            analysisFeaturesPath <- Some(Path.GetFullPath args[i + 1])
            i <- i + 2
        | "--emit-json"
        | "--emit-json-dir"
        | "--write-json" ->
            if i + 1 >= args.Length then
                failwith "Expected a directory after --emit-json."
            emitJsonDirectory <- Some(Path.GetFullPath args[i + 1])
            i <- i + 2
        | "--low-support-threshold" ->
            if i + 1 >= args.Length then
                failwith "Expected an integer after --low-support-threshold."
            match Int32.TryParse args[i + 1] with
            | true, threshold when threshold >= 1 ->
                lowSupportThreshold <- Some threshold
                i <- i + 2
            | _ ->
                failwith "Expected --low-support-threshold to be a positive integer."
        | "--all-analysis-features" ->
            lowSupportThreshold <- None
            i <- i + 1
        | "--boilerplate-only"
        | "--only-boilerplate" ->
            nonBoilerplate <- false
            i <- i + 1
        | "--non-boilerplate" ->
            nonBoilerplate <- true
            i <- i + 1
        | "--sqrt-only"
        | "--only-sqrt" ->
            sqrtOnly <- true
            i <- i + 1
        | "--stack-unstack-only"
        | "--only-stack-unstack" ->
            stackUnstackOnly <- true
            i <- i + 1
        | "--convolution-breakdown-only"
        | "--only-convolution-breakdown" ->
            convolutionBreakdownOnly <- true
            i <- i + 1
        | "--discrete-gaussian-breakdown-only"
        | "--only-discrete-gaussian-breakdown" ->
            discreteGaussianBreakdownOnly <- true
            i <- i + 1
        | "--operation" ->
            if i + 1 >= args.Length then
                failwith "Expected an operation name after --operation."
            match args[i + 1].ToLowerInvariant() with
            | "sqrt" ->
                sqrtOnly <- true
                i <- i + 2
            | "boilerplate" ->
                nonBoilerplate <- false
                i <- i + 2
            | "non-boilerplate"
            | "nonboilerplate" ->
                nonBoilerplate <- true
                i <- i + 2
            | "stack-unstack"
            | "stackunstack" ->
                stackUnstackOnly <- true
                i <- i + 2
            | "convolution-breakdown"
            | "convolutionbreakdown" ->
                convolutionBreakdownOnly <- true
                i <- i + 2
            | "discrete-gaussian-breakdown"
            | "discretegaussianbreakdown" ->
                discreteGaussianBreakdownOnly <- true
                i <- i + 2
            | operation ->
                failwith $"Unsupported probing operation '{operation}'. Currently supported: boilerplate, sqrt, stack-unstack, convolution-breakdown, discrete-gaussian-breakdown."
        | arg when arg.StartsWith("-") ->
            failwith $"Unknown probing argument '{arg}'."
        | path ->
            match reportPath with
            | Some existing ->
                failwith $"Expected only one report path, got '{existing}' and '{path}'."
            | None ->
                reportPath <- Some path
                i <- i + 1

    let selectedModeCount =
        [ sqrtOnly; stackUnstackOnly; convolutionBreakdownOnly; discreteGaussianBreakdownOnly ]
        |> List.filter id
        |> List.length

    if selectedModeCount > 1 then
        failwith "Choose only one single-operation probe mode."

    { ReportPath =
        reportPath
        |> Option.defaultValue "stackprocessing-probing.json"
        |> Path.GetFullPath
      AnalysisFeaturesPath = analysisFeaturesPath
      EmitJsonDirectory = emitJsonDirectory
      LowSupportThreshold = lowSupportThreshold
      NonBoilerplate = nonBoilerplate
      SqrtOnly = sqrtOnly
      StackUnstackOnly = stackUnstackOnly
      ConvolutionBreakdownOnly = convolutionBreakdownOnly
      DiscreteGaussianBreakdownOnly = discreteGaussianBreakdownOnly }

let main args =
    let options = parseArgs args
    let reportPath = options.ReportPath
    let tempRoot =
        Path.Combine(
            Path.GetTempPath(),
            "StackProcessing.Probe",
            DateTimeOffset.UtcNow.ToString("yyyyMMddTHHmmssfff"))

    cleanDirectory tempRoot

    if options.StackUnstackOnly || options.ConvolutionBreakdownOnly || options.DiscreteGaussianBreakdownOnly then
        failwith "The old Image stack/unstack and convolution-breakdown Probe modes have been removed. Use Chunk-native sample/probe graphs instead."

    let oldFocusedMode =
        options.SqrtOnly

    let includeNonBoilerplate =
        options.NonBoilerplate

    let inputDirs : Map<ImageSize, string> =
        if oldFocusedMode then
            Map.empty
        else
            (if includeNonBoilerplate then inputSizes else boilerplateInputSizes)
            |> List.map (fun size ->
                let path = Path.Combine(tempRoot, $"input-{sizeName size}")
                createInputStack size path
                size, path)
            |> Map.ofList

    let outputDir size name =
        let path = Path.Combine(tempRoot, $"output-{sizeName size}", name)
        Directory.CreateDirectory(path) |> ignore
        path

    let probes =
        [| for xy in (if includeNonBoilerplate || oldFocusedMode then xySizes else boilerplateXySizes) do
               if not oldFocusedMode then
                   for depth in (if includeNonBoilerplate then defaultDepths else boilerplateDepths) do
                       let size = imageSize xy depth
                       let inputDir = inputDirs[size]
                       let suffix = sizeName size

                       yield runSinkProbe
                                 $"zero-uint8-{suffix}"
                                 $"Synthetic UInt8 {suffix} source consumed without writing."
                                 (let p = defaultImageParameters size "uint8" 1u
                                  p["operation"] <- "zero"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<uint8> size.Width size.Height size.Depth
                                     >=> ignoreSingles ())

                       yield runSinkProbe
                                 $"zero-uint8-write-{suffix}"
                                 $"Synthetic UInt8 {suffix} source written to TIFF."
                                 (let p = defaultImageParameters size "uint8" 1u
                                  p["operation"] <- "zero-write"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<uint8> size.Width size.Height size.Depth
                                     >=> write (outputDir size "zero-uint8-write") ".tiff")

                       if includeNonBoilerplate then
                           yield runSinkProbe
                                     $"add-normal-noise-uint8-write-{suffix}"
                                     $"Synthetic UInt8 {suffix} source, additive Gaussian noise, write."
                                     (let p = defaultImageParameters size "uint8" 1u
                                      p["operation"] <- "noise-write"
                                      p["stage"] <- "AddNormalNoise"
                                      p["mean"] <- "128.0"
                                      p["std"] <- "50.0"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> zero<uint8> size.Width size.Height size.Depth
                                         >=> addNormalNoise 128.0 50.0
                                         >=> write (outputDir size "add-normal-noise-uint8-write") ".tiff")

                       yield runSinkProbe
                                 $"read-uint8-ignore-{suffix}"
                                 $"Read UInt8 {suffix} TIFF stack and consume it without writing."
                                 (let p = defaultImageParameters size "uint8" 1u
                                  p["operation"] <- "read-ignore"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> read<uint8> inputDir ".tiff"
                                     >=> ignoreSingles ())

                       yield runSinkProbe
                                 $"read-uint8-write-{suffix}"
                                 $"Copy-style UInt8 {suffix} read and write pipeline."
                                 (let p = defaultImageParameters size "uint8" 1u
                                  p["operation"] <- "read-write"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> read<uint8> inputDir ".tiff"
                                     >=> write (outputDir size "read-uint8-write") ".tiff")

                       let readUInt8Feature =
                           "Read:type=UInt8:format=Image stack:suffix=.tiff:thickDepth=1:multiscaleIndex=0:datasetIndex=0:timepoint=0:channel=0:maxParallelChunks=0:frameAxis=0"
                       let readUInt8FeatureWithAxes =
                           "Read:type=UInt8:format=Image stack:suffix=.tiff:thickDepth=1:multiscaleIndex=0:datasetIndex=0:timepoint=0:channel=0:maxParallelChunks=0:frameAxis=0:yAxis=1:xAxis=2"
                       let readFloatFeature =
                           "Read:type=Float32:format=Image stack:suffix=.tiff:thickDepth=1:multiscaleIndex=0:datasetIndex=0:timepoint=0:channel=0:maxParallelChunks=0:frameAxis=0"
                       let readFloatFeatureWithAxes =
                           "Read:type=Float32:format=Image stack:suffix=.tiff:thickDepth=1:multiscaleIndex=0:datasetIndex=0:timepoint=0:channel=0:maxParallelChunks=0:frameAxis=0:yAxis=1:xAxis=2"
                       let writeStackFeature =
                           "Write:format=Image stack:suffix=.tiff:depth=1:chunkX=64:chunkY=64:chunkZ=8:maxConcurrentWrites=0:frameAxis=0"
                       let writeStackFeatureWithAxes =
                           "Write:format=Image stack:suffix=.tiff:depth=1:chunkX=64:chunkY=64:chunkZ=8:maxConcurrentWrites=0:frameAxis=0:yAxis=1:xAxis=2"

                       yield runSinkProbe
                                 $"bio-threshold-morphology-uint8-write-{suffix}"
                                 $"Community-inspired 3D nuclei-style threshold, morphology cleanup, and write for {suffix}."
                                 (let p = defaultImageParameters size "uint8" 7u
                                  p["operation"] <- "bio-threshold-morphology-write"
                                  p["features"] <-
                                      encodedFeatures
                                          [ readUInt8Feature
                                            readUInt8FeatureWithAxes
                                            "Threshold:type=UInt8:lower=128.0:upper=infinity"
                                            "Opening:radius=3"
                                            "Closing:radius=3"
                                            writeStackFeature
                                            writeStackFeatureWithAxes ]
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> read<uint8> inputDir ".tiff"
                                     >=> threshold<uint8> 128.0 infinity
                                     >=> opening 3u
                                     >=> closing 3u
                                     >=> write (outputDir size "bio-threshold-morphology-uint8-write") ".tiff")

                       yield runSinkProbe
                                 $"bio-edge-filter-float-write-{suffix}"
                                 $"Community-inspired denoise, edge enhancement, stretch, cast, and write for {suffix}."
                                 (let p = defaultImageParameters size "float32" 7u
                                  p["operation"] <- "bio-edge-filter-write"
                                  p["features"] <-
                                      encodedFeatures
                                          [ readFloatFeature
                                            readFloatFeatureWithAxes
                                            "GradientMagnitude:type=Float32:windowSize=7"
                                            "IntensityStretch:type=Float32:inputMinimum=0.0:inputMaximum=255.0:outputMinimum=0.0:outputMaximum=255.0"
                                            "Cast:sourceType=Float32:targetType=UInt8"
                                            writeStackFeature
                                            writeStackFeatureWithAxes ]
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> read<float32> inputDir ".tiff"
                                     >=> gradientMagnitude<float32> (Some 7u)
                                     >=> intensityStretch 0.0 255.0 0.0 255.0
                                     >=> cast<_, uint8>
                                     >=> write (outputDir size "bio-edge-filter-float-write") ".tiff")

                       yield runSinkProbe
                                 $"bio-background-mask-float-write-{suffix}"
                                 $"Community-inspired smooth background mask workflow for {suffix}."
                                 (let p = defaultImageParameters size "float32" 7u
                                  p["operation"] <- "bio-background-mask-write"
                                  p["features"] <-
                                      encodedFeatures
                                          [ readFloatFeature
                                            readFloatFeatureWithAxes
                                            "SmoothWGauss:sigma=3.0:outputRegionMode=None:boundaryCondition=None:windowSize=None"
                                            "IntensityStretch:type=Float32:inputMinimum=0.0:inputMaximum=255.0:outputMinimum=0.0:outputMaximum=255.0"
                                            "Cast:sourceType=Float32:targetType=UInt8"
                                            "Threshold:type=UInt8:lower=128.0:upper=infinity"
                                            writeStackFeature
                                            writeStackFeatureWithAxes ]
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> read<float32> inputDir ".tiff"
                                     >=> smoothWGauss canonicalSigma None None None
                                     >=> intensityStretch 0.0 255.0 0.0 255.0
                                     >=> cast<_, uint8>
                                     >=> threshold<uint8> 128.0 infinity
                                     >=> write (outputDir size "bio-background-mask-float-write") ".tiff")

                       yield runSinkProbe
                                 $"bio-projection-inspection-uint8-write-{suffix}"
                                 $"Community-inspired 3D-to-2D projection inspection workflow for {suffix}."
                                 (let p = defaultImageParameters size "uint8" 1u
                                  p["operation"] <- "bio-projection-inspection-write"
                                  p["features"] <-
                                      encodedFeatures
                                          [ readUInt8Feature
                                            readUInt8FeatureWithAxes
                                            "SumProjection:type=UInt8:function=Identity"
                                            "IntensityStretch:type=Float32:inputMinimum=0.0:inputMaximum=255.0:outputMinimum=0.0:outputMaximum=255.0"
                                            "Cast:sourceType=Float32:targetType=UInt8"
                                            writeStackFeature
                                            writeStackFeatureWithAxes ]
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> read<uint8> inputDir ".tiff"
                                     >=> sumProjection "Identity"
                                     >=> intensityStretch 0.0 255.0 0.0 255.0
                                     >=> cast<_, uint8>
                                     >=> write (outputDir size "bio-projection-inspection-uint8-write") ".tiff")

                       if includeNonBoilerplate then
                           yield runSinkProbe
                                     $"threshold-float32-write-{suffix}"
                                     $"Synthetic Float32 {suffix} source, threshold to UInt8, write."
                                     (let p = defaultImageParameters size "float32" 1u
                                      p["operation"] <- "threshold-write"
                                      p["mean"] <- "128.0"
                                      p["std"] <- "50.0"
                                      p["threshold"] <- "128.0"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> zero<float32> size.Width size.Height size.Depth
                                         >=> addNormalNoise 128.0 50.0
                                         >=> threshold<float32> 128.0 infinity
                                         >=> imageMulScalar 255.0
                                         >=> write (outputDir size "threshold-float32-write") ".tiff")

                           yield runDrainProbe
                                     $"compute-stats-read-float32-{suffix}"
                                     $"Read {suffix} stack as Float32 and drain computeStats reducer."
                                     (let p = defaultImageParameters size "float32" 1u
                                      p["operation"] <- "compute-stats"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<float32> inputDir ".tiff"
                                         >=> computeStats ())

                           yield runSinkProbe
                                     $"add-salt-and-pepper-uint8-write-{suffix}"
                                     $"Synthetic UInt8 {suffix} source, add salt-and-pepper noise, write."
                                     (let p = defaultImageParameters size "uint8" 1u
                                      p["operation"] <- "add-salt-and-pepper-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> zero<uint8> size.Width size.Height size.Depth
                                         >=> addSaltAndPepperNoise 0.02 None None
                                         >=> write (outputDir size "add-salt-and-pepper-uint8-write") ".tiff")

                           yield runSinkProbe
                                     $"salt-and-pepper-uint8-write-{suffix}"
                                     $"Synthetic UInt8 {suffix} salt-and-pepper source, write."
                                     (let p = defaultImageParameters size "uint8" 1u
                                      p["operation"] <- "salt-and-pepper-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> saltAndPepperNoise<uint8> size.Width size.Height size.Depth 0.02
                                         >=> write (outputDir size "salt-and-pepper-uint8-write") ".tiff")

                           yield runSinkProbe
                                     $"poisson-noise-float-write-{suffix}"
                                     $"Synthetic Float32 {suffix} Poisson noise source, cast to UInt8, write."
                                     (let p = defaultImageParameters size "float32" 1u
                                      p["operation"] <- "poisson-noise-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> poissonNoise<float32> size.Width size.Height size.Depth 2.0
                                         >=> cast<_, uint8>
                                         >=> write (outputDir size "poisson-noise-float-write") ".tiff")

                           yield runSinkProbe
                                     $"fill-small-holes-uint8-write-{suffix}"
                                     $"Read UInt8 {suffix}, fill small holes, write."
                                     (let p = defaultImageParameters size "uint8" 1u
                                      p["operation"] <- "fill-small-holes-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<uint8> inputDir ".tiff"
                                         >=> fillSmallHoles 128UL ObjectConnectivity.TwentySix
                                         >=> write (outputDir size "fill-small-holes-uint8-write") ".tiff")

                           yield runSinkProbe
                                     $"binary-contour-uint8-write-{suffix}"
                                     $"Read UInt8 {suffix}, binary contour, write."
                                     (let p = defaultImageParameters size "uint8" 7u
                                      p["operation"] <- "binary-contour-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<uint8> inputDir ".tiff"
                                         >=> binaryContour false (Some 7)
                                         >=> write (outputDir size "binary-contour-uint8-write") ".tiff")

                           yield runSinkProbe
                                     $"erode-uint8-write-{suffix}"
                                     $"Read UInt8 {suffix}, binary erode, write."
                                     (let p = defaultImageParameters size "uint8" 1u
                                      p["operation"] <- "erode-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<uint8> inputDir ".tiff"
                                         >=> erode 3u
                                         >=> write (outputDir size "erode-uint8-write") ".tiff")

                           yield runSinkProbe
                                     $"dilate-uint8-write-{suffix}"
                                     $"Read UInt8 {suffix}, binary dilate, write."
                                     (let p = defaultImageParameters size "uint8" 1u
                                      p["operation"] <- "dilate-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<uint8> inputDir ".tiff"
                                         >=> dilate 3u
                                         >=> write (outputDir size "dilate-uint8-write") ".tiff")

                           yield runSinkProbe
                                     $"opening-uint8-write-{suffix}"
                                     $"Read UInt8 {suffix}, binary opening, write."
                                     (let p = defaultImageParameters size "uint8" 1u
                                      p["operation"] <- "opening-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<uint8> inputDir ".tiff"
                                         >=> opening 3u
                                         >=> write (outputDir size "opening-uint8-write") ".tiff")

                           yield runSinkProbe
                                     $"closing-uint8-write-{suffix}"
                                     $"Read UInt8 {suffix}, binary closing, write."
                                     (let p = defaultImageParameters size "uint8" 1u
                                      p["operation"] <- "closing-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<uint8> inputDir ".tiff"
                                         >=> closing 3u
                                         >=> write (outputDir size "closing-uint8-write") ".tiff")

                           yield runSinkProbe
                                     $"black-top-hat-uint8-write-{suffix}"
                                     $"Read UInt8 {suffix}, black top-hat, write."
                                     (let p = defaultImageParameters size "uint8" 7u
                                      p["operation"] <- "black-top-hat-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<uint8> inputDir ".tiff"
                                         >=> blackTopHat<uint8> 3u (Some 7)
                                         >=> write (outputDir size "black-top-hat-uint8-write") ".tiff")

                           yield runSinkProbe
                                     $"white-top-hat-uint8-write-{suffix}"
                                     $"Read UInt8 {suffix}, white top-hat, write."
                                     (let p = defaultImageParameters size "uint8" 7u
                                      p["operation"] <- "white-top-hat-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<uint8> inputDir ".tiff"
                                         >=> whiteTopHat<uint8> 3u (Some 7)
                                         >=> write (outputDir size "white-top-hat-uint8-write") ".tiff")

                           yield runSinkProbe
                                     $"morphological-gradient-uint8-write-{suffix}"
                                     $"Read UInt8 {suffix}, morphological gradient, write."
                                     (let p = defaultImageParameters size "uint8" 7u
                                      p["operation"] <- "morphological-gradient-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<uint8> inputDir ".tiff"
                                         >=> morphologicalGradient<uint8> 3u (Some 7)
                                         >=> write (outputDir size "morphological-gradient-uint8-write") ".tiff")

                           yield runSinkProbe
                                     $"filters-float-write-{suffix}"
                                     $"Read Float32 {suffix}, common image filters, cast, write."
                                     (let p = defaultImageParameters size "float32" 7u
                                      p["operation"] <- "filters-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<float32> inputDir ".tiff"
                                         >=> gradientMagnitude<float32> (Some 7u)
                                         >=> sobelEdge<float32> (Some 7u)
                                         >=> laplacian<float32> (Some 7u)
                                         >=> intensityStretch 0.0 255.0 0.0 255.0
                                         >=> cast<_, uint8>
                                         >=> write (outputDir size "filters-float-write") ".tiff")

                           let readUInt8Feature =
                               "Read:type=UInt8:format=Image stack:suffix=.tiff:thickDepth=1:multiscaleIndex=0:datasetIndex=0:timepoint=0:channel=0:maxParallelChunks=0:frameAxis=0"
                           let readUInt8FeatureWithAxes =
                               "Read:type=UInt8:format=Image stack:suffix=.tiff:thickDepth=1:multiscaleIndex=0:datasetIndex=0:timepoint=0:channel=0:maxParallelChunks=0:frameAxis=0:yAxis=1:xAxis=2"
                           let writeStackFeature =
                               "Write:format=Image stack:suffix=.tiff:depth=1:chunkX=64:chunkY=64:chunkZ=8:maxConcurrentWrites=0:frameAxis=0"
                           let writeStackFeatureWithAxes =
                               "Write:format=Image stack:suffix=.tiff:depth=1:chunkX=64:chunkY=64:chunkZ=8:maxConcurrentWrites=0:frameAxis=0:yAxis=1:xAxis=2"
                           let mixedPairs =
                               [ "erode-dilate", "Erode:radius=3", erode 3u, "Dilate:radius=3", dilate 3u
                                 "dilate-erode", "Dilate:radius=3", dilate 3u, "Erode:radius=3", erode 3u
                                 "opening-closing", "Opening:radius=3", opening 3u, "Closing:radius=3", closing 3u
                                 "closing-opening", "Closing:radius=3", closing 3u, "Opening:radius=3", opening 3u
                                 "black-white-top-hat", "BlackTopHat:type=UInt8:radius=3:windowSize=7", blackTopHat<uint8> 3u (Some 7), "WhiteTopHat:type=UInt8:radius=3:windowSize=7", whiteTopHat<uint8> 3u (Some 7)
                                 "white-black-top-hat", "WhiteTopHat:type=UInt8:radius=3:windowSize=7", whiteTopHat<uint8> 3u (Some 7), "BlackTopHat:type=UInt8:radius=3:windowSize=7", blackTopHat<uint8> 3u (Some 7)
                                 "morph-gradient-contour", "MorphologicalGradient:type=UInt8:radius=3:windowSize=7", morphologicalGradient<uint8> 3u (Some 7), "BinaryContour:fullyConnected=false:windowSize=7", binaryContour false (Some 7)
                                 "fill-holes-contour", "FillSmallHoles:maximumVolume=128:connectivity=TwentySix", fillSmallHoles 128UL ObjectConnectivity.TwentySix, "BinaryContour:fullyConnected=false:windowSize=7", binaryContour false (Some 7) ]

                           for name, featureA, stageA, featureB, stageB in mixedPairs do
                               yield runSinkProbe
                                         $"mixed-{name}-uint8-write-{suffix}"
                                         $"Read UInt8 {suffix}, run {name}, write."
                                         (let p = defaultImageParameters size "uint8" 7u
                                          p["operation"] <- "mixed-uint8-write"
                                          p["features"] <-
                                              encodedFeatures
                                                  [ readUInt8Feature
                                                    readUInt8FeatureWithAxes
                                                    featureA
                                                    featureB
                                                    writeStackFeature
                                                    writeStackFeatureWithAxes ]
                                          p)
                                         (fun () ->
                                             source availableMemory
                                             |> read<uint8> inputDir ".tiff"
                                             >=> stageA
                                             >=> stageB
                                             >=> write (outputDir size $"mixed-{name}-uint8-write") ".tiff")

                           yield runSinkProbe
                                     $"resize-uint8-write-{suffix}"
                                     $"Read UInt8 {suffix}, resize to 96^3, write."
                                     (let p = defaultImageParameters size "uint8" 1u
                                      p["operation"] <- "resize-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<uint8> inputDir ".tiff"
                                         |> resize 96u 96u 96u "Linear"
                                         >=> write (outputDir size "resize-uint8-write") ".tiff")

                           yield runSinkProbe
                                     $"resize64-uint8-write-{suffix}"
                                     $"Read UInt8 {suffix}, resize to 64^3, write."
                                     (let p = defaultImageParameters size "uint8" 1u
                                      p["operation"] <- "resize64-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<uint8> inputDir ".tiff"
                                         |> resize 64u 64u 64u "Linear"
                                         >=> write (outputDir size "resize64-uint8-write") ".tiff")

                           yield runSinkProbe
                                     $"resample-float32-write-{suffix}"
                                     $"Synthetic Float32 {suffix} normal noise, resample by 1.5, cast, write."
                                     (let p = defaultImageParameters size "float32" 1u
                                      p["operation"] <- "resample-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> normalNoise<float32> size.Width size.Height size.Depth 128.0 25.0
                                         |> resample 1.5 1.5 1.5 "Linear"
                                         >=> cast<_, uint8>
                                         >=> write (outputDir size "resample-float32-write") ".tiff")

               if
                   (includeNonBoilerplate || options.SqrtOnly)
                   && not options.StackUnstackOnly
                   && not options.ConvolutionBreakdownOnly
                   && not options.DiscreteGaussianBreakdownOnly
               then
                   for depth in singletonDepths do
                       let size = imageSize xy depth
                       let suffix = sizeName size

                       if options.SqrtOnly then
                           yield runSinkProbe
                                     $"zero-uint8-{suffix}"
                                     $"Synthetic UInt8 {suffix} source consumed without writing."
                                     (let p = singletonImageParameters size "uint8" 1u
                                      p["operation"] <- "zero"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> zero<uint8> size.Width size.Height size.Depth
                                         >=> ignoreSingles ())

                           yield runSinkProbe
                                     $"zero-uint8-write-{suffix}"
                                     $"Synthetic UInt8 {suffix} source written to TIFF."
                                     (let p = singletonImageParameters size "uint8" 1u
                                      p["operation"] <- "zero-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> zero<uint8> size.Width size.Height size.Depth
                                         >=> write (outputDir size "zero-uint8-write") ".tiff")

                       yield runSinkProbe
                                 $"sqrt-input-float-{suffix}"
                                 $"Synthetic Float32 {suffix} source prepared for sqrt and consumed without applying sqrt."
                                 (let p = singletonImageParameters size "float32" 1u
                                  p["operation"] <- "sqrt-input"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float32> size.Width size.Height size.Depth
                                     >=> imageAddScalar 4.0
                                     >=> ignoreSingles ())

                       yield runSinkProbe
                                 $"sqrt-float-{suffix}"
                                 $"Synthetic Float32 {suffix} source, sqrt, consumed without writing."
                                 (let p = singletonImageParameters size "float32" 1u
                                  p["operation"] <- "sqrt"
                                  p["baselineOperation"] <- "sqrt-input"
                                  p["stage"] <- "sqrt"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float32> size.Width size.Height size.Depth
                                     >=> imageAddScalar 4.0
                                     >=> sqrt<float32>
                                     >=> ignoreSingles ())

                       yield runSinkProbe
                                 $"sqrt-float-write-{suffix}"
                                 $"Synthetic Float32 {suffix} source, sqrt, cast to UInt8, write."
                                 (let p = singletonImageParameters size "float32" 1u
                                  p["operation"] <- "sqrt-write"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float32> size.Width size.Height size.Depth
                                     >=> imageAddScalar 4.0
                                     >=> sqrt<float32>
                                     >=> cast<_, uint8>
                                     >=> write (outputDir size "sqrt-float-write") ".tiff")

                       for windowSize in unaryWindowSizes do
                           yield runSinkProbe
                                     $"sqrt-windowed-float-{suffix}-win-{windowSize}"
                                     $"Synthetic Float32 {suffix} source, windowed sqrt with window size {windowSize}, consumed without writing."
                                     (let p = singletonImageParameters size "float32" windowSize
                                      p["operation"] <- "sqrt-windowed"
                                      p["baselineOperation"] <- "sqrt-input"
                                      p["stage"] <- "sqrt-windowed"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> zero<float32> size.Width size.Height size.Depth
                                         >=> imageAddScalar 4.0
                                         >=> sqrtWindowed<float32> windowSize
                                         >=> ignoreSingles ())

                           yield runSinkProbe
                                     $"sqrt-windowed-float-write-{suffix}-win-{windowSize}"
                                     $"Synthetic Float32 {suffix} source, windowed sqrt with window size {windowSize}, cast to UInt8, write."
                                     (let p = singletonImageParameters size "float32" windowSize
                                      p["operation"] <- "sqrt-windowed-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> zero<float32> size.Width size.Height size.Depth
                                         >=> imageAddScalar 4.0
                                         >=> sqrtWindowed<float32> windowSize
                                         >=> cast<_, uint8>
                                         >=> write (outputDir size $"sqrt-windowed-float-write-win-{windowSize}") ".tiff")

                       if not options.SqrtOnly then
                           for windowSize in gaussianWindowSizes do
                               let inputDir = inputDirs[size]
                               yield runSinkProbe
                                        $"smoothWGauss-read-float32-cast-write-{suffix}-win-{windowSize}"
                                         $"Read {suffix} stack as Float32, smoothWGauss windowed convolution with window size {windowSize}, cast to UInt8, write."
                                         (let p = singletonImageParameters size "float32" windowSize
                                          p["operation"] <- "smoothWGauss-write"
                                          p["sigma"] <- canonicalSigmaText
                                          p["kernelSize"] <- string canonicalKernelSize
                                          p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<float32> inputDir ".tiff"
                                             >=> smoothWGauss canonicalSigma None None (Some windowSize)
                                             >=> cast<_, uint8>
                                            >=> write (outputDir size $"smoothWGauss-read-float32-cast-write-win-{windowSize}") ".tiff")

               () |]

    let calibrations = buildCalibrations probes
    let probes = attachPredictions calibrations probes
    let probes = filterProbesByAnalysisFeatures options.LowSupportThreshold options.AnalysisFeaturesPath probes
    let analysisTokens =
        options.AnalysisFeaturesPath
        |> Option.map (loadAnalysisFeatureTokens options.LowSupportThreshold)
        |> Option.defaultValue Set.empty

    options.EmitJsonDirectory
    |> Option.iter (fun outputDir -> writeProbeGraphs outputDir analysisTokens probes (generatedGraphTemplates ()))

    let report =
        { generatedUtc = DateTimeOffset.UtcNow.ToString("O")
          osDescription = RuntimeInformation.OSDescription
          processArchitecture = RuntimeInformation.ProcessArchitecture.ToString()
          frameworkDescription = RuntimeInformation.FrameworkDescription
          configuration = "Default"
          sampleIntervalMilliseconds = sampleIntervalMs
          warmupCount = warmupCount
          repetitionCount = repetitionCount
          workingDirectory = Environment.CurrentDirectory
          tempDirectory = tempRoot
          calibrations = calibrations
          probes = probes }

    let options = JsonSerializerOptions(WriteIndented = true)
    options.Converters.Add(JsonStringEnumConverter())
    let json = JsonSerializer.Serialize(report, options)
    File.WriteAllText(reportPath, json)
    writeProbeAnalysisCsvs reportPath calibrations probes

    printfn "Wrote probing report to %s" reportPath
    0
