module StackProcessing.Probing

open System
open System.Collections.Generic
open System.Diagnostics
open System.IO
open System.Runtime.InteropServices
open System.Text.Json
open System.Text.Json.Serialization
open SlimPipeline
open StackProcessing

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
type ProbeWorkJson =
    { calibrationKey: string
      cpuWorkUnits: float
      nativeWorkUnits: float
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
      costPeakWork: ProbeWorkJson option
      costWorks: ProbeWorkJson array
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
      calibrations: Dictionary<string, StageCostCoefficients>
      probes: ProbeResultJson array }

let private availableMemory = 2UL * 1024UL * 1024UL * 1024UL
let private sampleIntervalMs = 10
let private warmupCount = 1
let private repetitionCount = 5

type ImageSize =
    { Width: uint
      Height: uint
      Depth: uint }

let private xySizes =
    [ 128u; 256u; 512u ]

let private imageSize xy depth =
    { Width = xy
      Height = xy
      Depth = depth }

let private depthsFrom baseDepth =
    [ baseDepth; baseDepth + 1u; baseDepth + 2u ]

let private defaultDepths = depthsFrom 3u
let private singletonDepths = depthsFrom 1u

let private inputDepths =
    defaultDepths @ singletonDepths
    |> List.distinct
    |> List.sort

let private inputSizes =
    [ for xy in xySizes do
          for depth in inputDepths do
              imageSize xy depth ]

let private convolutionBreakdownSizes =
    [ 16u; 32u; 64u ]
    |> List.map (fun side -> imageSize side side)

let private gaussianWindowSizes =
    [ 5u; 9u; 15u ]

let private unaryWindowSizes =
    [ 1u; 5u; 9u; 15u ]

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

let private workToJson (work: StageWorkPressure) =
    work.CalibrationKey
    |> Option.map (fun key ->
        { calibrationKey = key
          cpuWorkUnits = work.CpuWorkUnits
          nativeWorkUnits = work.NativeWorkUnits
          ioReadBytes = work.IoReadBytes
          ioWriteBytes = work.IoWriteBytes
          ioReadOps = work.IoReadOps
          ioWriteOps = work.IoWriteOps })

let private workFromJson (work: ProbeWorkJson) =
    { CpuWorkUnits = work.cpuWorkUnits
      NativeWorkUnits = work.nativeWorkUnits
      IoReadBytes = work.ioReadBytes
      IoWriteBytes = work.ioWriteBytes
      IoReadOps = work.ioReadOps
      IoWriteOps = work.ioWriteOps
      CalibrationKey = Some work.calibrationKey }

let private coefficientsFromWork elapsedMilliseconds (work: StageWorkPressure) =
    let elapsed = elapsedMilliseconds
    { CpuMillisecondsPerUnit =
        if work.CpuWorkUnits > 0.0 then elapsed / work.CpuWorkUnits else 0.0
      NativeMillisecondsPerUnit =
        if work.NativeWorkUnits > 0.0 then elapsed / work.NativeWorkUnits else 0.0
      IoReadMillisecondsPerByte =
        if work.IoReadBytes > 0UL then elapsed / float work.IoReadBytes else 0.0
      IoWriteMillisecondsPerByte =
        if work.IoWriteBytes > 0UL then elapsed / float work.IoWriteBytes else 0.0
      IoReadMillisecondsPerOp =
        if work.IoReadBytes = 0UL && work.IoReadOps > 0UL then elapsed / float work.IoReadOps else 0.0
      IoWriteMillisecondsPerOp =
        if work.IoWriteBytes = 0UL && work.IoWriteOps > 0UL then elapsed / float work.IoWriteOps else 0.0 }

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
    let costPeakWork = metadataPlan.costPeak |> Option.bind (fun cost -> workToJson cost.Work)
    let costWorks =
        metadataPlan.costObservations
        |> List.rev
        |> List.choose (fun cost -> workToJson cost.Work)
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
      costPeakWork = costPeakWork
      costWorks = costWorks
      iterations = iterations }

let private runSinkProbe name description probeParameters (planFactory: unit -> Plan<unit, _>) =
    let execute plan = sink plan
    runProbe name description probeParameters execute planFactory

let private runDrainProbe name description probeParameters (planFactory: unit -> Plan<unit, _>) =
    let execute plan = drain plan |> ignore
    runProbe name description probeParameters execute planFactory

let private tryStageWork (probe: ProbeResultJson) =
    match tryParameter "stage" probe with
    | Some stage ->
        let normalizedStage = normalizedIdentifier stage
        probe.costWorks
        |> Array.tryFind (fun work ->
            (normalizedIdentifier work.calibrationKey).Contains(normalizedStage))
    | None ->
        match probe.costWorks with
        | [| work |] -> Some work
        | _ -> None

let private coefficientObservations (probes: ProbeResultJson array) =
    let probesByOperationAndShape =
        probes
        |> Array.groupBy (fun probe -> operationName probe, calibrationParameterKey probe)
        |> Array.map (fun (key, matches) -> key, matches[0])
        |> Map.ofArray

    seq {
        for probe in probes do
            match tryParameter "baselineOperation" probe, tryStageWork probe with
            | Some baselineOperation, Some work ->
                let baselineKey = baselineOperation, calibrationParameterKey probe

                match probesByOperationAndShape |> Map.tryFind baselineKey with
                | Some baseline ->
                    let elapsed =
                        max 0.0 (probe.elapsedMedianMilliseconds - baseline.elapsedMedianMilliseconds)

                    if elapsed > 0.0 then
                        yield work.calibrationKey, coefficientsFromWork elapsed (workFromJson work)
                | None -> ()
            | None, Some work when probe.costWorks.Length = 1 ->
                if probe.elapsedMedianMilliseconds > 0.0 then
                    yield work.calibrationKey, coefficientsFromWork probe.elapsedMedianMilliseconds (workFromJson work)
            | _ -> ()
    }
    |> Seq.toList

let private averageCoefficients (items: StageCostCoefficients list) =
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
    let calibrations = Dictionary<string, StageCostCoefficients>()

    probes
    |> coefficientObservations
    |> List.groupBy fst
    |> List.iter (fun (key, observations) ->
        calibrations[key] <-
            observations
            |> List.map snd
            |> averageCoefficients)

    calibrations

let private tryPredictElapsedMilliseconds (calibrations: Dictionary<string, StageCostCoefficients>) (probe: ProbeResultJson) =
    let mutable any = false
    let mutable total = 0.0

    for work in probe.costWorks do
        match calibrations.TryGetValue work.calibrationKey with
        | true, coefficients ->
            any <- true
            total <- total + StageCostCoefficients.estimateMilliseconds coefficients (workFromJson work)
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

let private releaseImages (images: Image<float> list) =
    images |> List.iter (fun image -> image.decRefCount())

let private releaseConsumedImages (window: Window<Image<float>>) =
    let _, emitCount = window.EmitRange
    window.Items
    |> List.take (min (int emitCount) window.Items.Length)
    |> List.iter (fun image -> image.decRefCount())

let private stackUnstackWindow (window: Window<Image<float>>) =
    match window.Items with
    | [] -> []
    | items ->
        let stack = ImageFunctions.stack items
        releaseConsumedImages window
        let outputCount = min (snd window.EmitRange) (stack.GetDepth())
        let result =
            if outputCount = 0u then
                []
            else
                ImageFunctions.unstackSkipNTakeM 0u outputCount stack
        stack.decRefCount()
        result

let private stackUnstackInputStage (windowSize: uint) : Stage<Image<float>, Image<float>> =
    StackCore.window windowSize 0u windowSize
    --> StackCore.flatten ()

let private stackUnstackStage (windowSize: uint) : Stage<Image<float>, Image<float>> =
    StackCore.window windowSize 0u windowSize
    --> StackCore.mapWindow "stack-unstack" (fun _ window -> stackUnstackWindow window) id id
    --> StackCore.flattenList ()

let private stackOnlyWindow (window: Window<Image<float>>) =
    match window.Items with
    | [] -> []
    | items ->
        let stack = ImageFunctions.stack items
        releaseConsumedImages window
        [ stack ]

let private stackConvolveWindow (kernel: Image<float>) (window: Window<Image<float>>) =
    match window.Items with
    | [] -> []
    | items ->
        let stack = ImageFunctions.stack items
        releaseConsumedImages window
        let convolved = ImageFunctions.convolve (Some ImageFunctions.Valid) None stack kernel
        stack.decRefCount()
        [ convolved ]

let private stackConvolveUnstackWindow (kernel: Image<float>) (window: Window<Image<float>>) =
    match stackConvolveWindow kernel window with
    | [] -> []
    | [ convolved ] ->
        let result = ImageFunctions.unstackSkipNTakeM 0u (convolved.GetDepth()) convolved
        convolved.decRefCount()
        result
    | images ->
        releaseImages images
        failwith "stackConvolveWindow returned more than one volume."

let private stackDiscreteGaussianWindow (kernelSize: uint) (window: Window<Image<float>>) =
    match window.Items with
    | [] -> []
    | items ->
        let stack = ImageFunctions.stack items
        releaseConsumedImages window
        let filtered = ImageFunctions.discreteGaussian 3u 1.0 (Some kernelSize) None None stack
        stack.decRefCount()
        [ filtered ]

let private stackDiscreteGaussianUnstackWindow (kernelSize: uint) (window: Window<Image<float>>) =
    match stackDiscreteGaussianWindow kernelSize window with
    | [] -> []
    | [ filtered ] ->
        let result = ImageFunctions.unstackSkipNTakeM 0u (filtered.GetDepth()) filtered
        filtered.decRefCount()
        result
    | images ->
        releaseImages images
        failwith "stackDiscreteGaussianWindow returned more than one volume."

let private convolutionBreakdownInputStage (windowSize: uint) : Stage<Image<float>, Image<float>> =
    StackCore.window windowSize 0u windowSize
    --> StackCore.flatten ()

let private convolutionBreakdownStackStage (windowSize: uint) : Stage<Image<float>, Image<float>> =
    StackCore.window windowSize 0u windowSize
    --> StackCore.mapWindow "convolution-breakdown-stack" (fun _ window -> stackOnlyWindow window) id id
    --> StackCore.flattenList ()

let private convolutionBreakdownStackConvolveStage (windowSize: uint) (kernel: Image<float>) : Stage<Image<float>, Image<float>> =
    StackCore.window windowSize 0u windowSize
    --> StackCore.mapWindow "convolution-breakdown-stack-convolve" (fun _ window -> stackConvolveWindow kernel window) id id
    --> StackCore.flattenList ()

let private convolutionBreakdownStackConvolveUnstackStage (windowSize: uint) (kernel: Image<float>) : Stage<Image<float>, Image<float>> =
    StackCore.window windowSize 0u windowSize
    --> StackCore.mapWindow "convolution-breakdown-stack-convolve-unstack" (fun _ window -> stackConvolveUnstackWindow kernel window) id id
    --> StackCore.flattenList ()

let private discreteGaussianBreakdownStackStage (windowSize: uint) =
    convolutionBreakdownStackStage windowSize

let private discreteGaussianBreakdownStackFilterStage (windowSize: uint) (kernelSize: uint) : Stage<Image<float>, Image<float>> =
    StackCore.window windowSize 0u windowSize
    --> StackCore.mapWindow "discrete-gaussian-breakdown-stack-filter" (fun _ window -> stackDiscreteGaussianWindow kernelSize window) id id
    --> StackCore.flattenList ()

let private discreteGaussianBreakdownStackFilterUnstackStage (windowSize: uint) (kernelSize: uint) : Stage<Image<float>, Image<float>> =
    StackCore.window windowSize 0u windowSize
    --> StackCore.mapWindow "discrete-gaussian-breakdown-stack-filter-unstack" (fun _ window -> stackDiscreteGaussianUnstackWindow kernelSize window) id id
    --> StackCore.flattenList ()

let private sizeName size =
    $"{size.Width}x{size.Height}x{size.Depth}"

let private createInputStack size inputDir =
    Directory.CreateDirectory(inputDir) |> ignore
    source availableMemory
    |> zero<uint8> size.Width size.Height size.Depth
    >=> addNormalNoise 128.0 50.0
    >=> write inputDir ".tiff"
    |> sink

let private cleanDirectory path =
    if Directory.Exists(path) then
        Directory.Delete(path, true)
    Directory.CreateDirectory(path) |> ignore

type ProbeOptions =
    { ReportPath: string
      SqrtOnly: bool
      StackUnstackOnly: bool
      ConvolutionBreakdownOnly: bool
      DiscreteGaussianBreakdownOnly: bool }

let private parseArgs (args: string array) =
    let mutable reportPath = None
    let mutable sqrtOnly = false
    let mutable stackUnstackOnly = false
    let mutable convolutionBreakdownOnly = false
    let mutable discreteGaussianBreakdownOnly = false
    let mutable i = 0

    while i < args.Length do
        match args[i] with
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
                failwith $"Unsupported probing operation '{operation}'. Currently supported: sqrt, stack-unstack, convolution-breakdown, discrete-gaussian-breakdown."
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
      SqrtOnly = sqrtOnly
      StackUnstackOnly = stackUnstackOnly
      ConvolutionBreakdownOnly = convolutionBreakdownOnly
      DiscreteGaussianBreakdownOnly = discreteGaussianBreakdownOnly }

[<EntryPoint>]
let main args =
    let options = parseArgs args
    let reportPath = options.ReportPath
    let tempRoot =
        Path.Combine(
            Path.GetTempPath(),
            "StackProcessing.Probing",
            DateTimeOffset.UtcNow.ToString("yyyyMMddTHHmmssfff"))

    cleanDirectory tempRoot

    let inputDirs : Map<ImageSize, string> =
        if options.SqrtOnly || options.StackUnstackOnly || options.ConvolutionBreakdownOnly || options.DiscreteGaussianBreakdownOnly then
            Map.empty
        else
            inputSizes
            |> List.map (fun size ->
                let path = Path.Combine(tempRoot, $"input-{sizeName size}")
                createInputStack size path
                size, path)
            |> Map.ofList

    let outputDir size name =
        let path = Path.Combine(tempRoot, $"output-{sizeName size}", name)
        Directory.CreateDirectory(path) |> ignore
        path

    let convolutionBreakdownKernel =
        if options.ConvolutionBreakdownOnly then
            let kernel: Image<float> = ImageFunctions.gauss 3u 1.0 (Some 8u)
            Some kernel
        else
            None

    let probes =
        [| for xy in xySizes do
               if not options.SqrtOnly && not options.StackUnstackOnly && not options.ConvolutionBreakdownOnly && not options.DiscreteGaussianBreakdownOnly then
                   for depth in defaultDepths do
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

                       yield runSinkProbe
                                 $"add-normal-noise-uint8-write-{suffix}"
                                 $"Synthetic UInt8 {suffix} source, additive Gaussian noise, write."
                                 (let p = defaultImageParameters size "uint8" 1u
                                  p["operation"] <- "noise-write"
                                  p["mean"] <- "128"
                                  p["sigma"] <- "50"
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

                       yield runSinkProbe
                                 $"threshold-float-write-{suffix}"
                                 $"Synthetic Float64 {suffix} source, threshold to UInt8, write."
                                 (let p = defaultImageParameters size "float" 1u
                                  p["operation"] <- "threshold-write"
                                  p["threshold"] <- "128"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float> size.Width size.Height size.Depth
                                     >=> addNormalNoise 128.0 50.0
                                     >=> threshold 128.0 infinity
                                     >=> imageMulScalar 255uy
                                     >=> write (outputDir size "threshold-float-write") ".tiff")

                       yield runDrainProbe
                                 $"compute-stats-read-float-{suffix}"
                                 $"Read {suffix} stack as Float64 and drain computeStats reducer."
                                 (let p = defaultImageParameters size "float" 1u
                                  p["operation"] <- "compute-stats"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> read<float> inputDir ".tiff"
                                     >=> computeStats ())

               if not options.StackUnstackOnly && not options.ConvolutionBreakdownOnly && not options.DiscreteGaussianBreakdownOnly then
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
                                 $"Synthetic Float64 {suffix} source prepared for sqrt and consumed without applying sqrt."
                                 (let p = singletonImageParameters size "float" 1u
                                  p["operation"] <- "sqrt-input"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float> size.Width size.Height size.Depth
                                     >=> imageAddScalar 4.0
                                     >=> ignoreSingles ())

                       yield runSinkProbe
                                 $"sqrt-float-{suffix}"
                                 $"Synthetic Float64 {suffix} source, sqrt, consumed without writing."
                                 (let p = singletonImageParameters size "float" 1u
                                  p["operation"] <- "sqrt"
                                  p["baselineOperation"] <- "sqrt-input"
                                  p["stage"] <- "sqrt"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float> size.Width size.Height size.Depth
                                     >=> imageAddScalar 4.0
                                     >=> sqrt<float>
                                     >=> ignoreSingles ())

                       yield runSinkProbe
                                 $"sqrt-float-write-{suffix}"
                                 $"Synthetic Float64 {suffix} source, sqrt, cast to UInt8, write."
                                 (let p = singletonImageParameters size "float" 1u
                                  p["operation"] <- "sqrt-write"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float> size.Width size.Height size.Depth
                                     >=> imageAddScalar 4.0
                                     >=> sqrt<float>
                                     >=> cast<float, uint8>
                                     >=> write (outputDir size "sqrt-float-write") ".tiff")

                       for windowSize in unaryWindowSizes do
                           yield runSinkProbe
                                     $"sqrt-windowed-float-{suffix}-win-{windowSize}"
                                     $"Synthetic Float64 {suffix} source, windowed sqrt with window size {windowSize}, consumed without writing."
                                     (let p = singletonImageParameters size "float" windowSize
                                      p["operation"] <- "sqrt-windowed"
                                      p["baselineOperation"] <- "sqrt-input"
                                      p["stage"] <- "sqrt-windowed"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> zero<float> size.Width size.Height size.Depth
                                         >=> imageAddScalar 4.0
                                         >=> sqrtWindowed<float> windowSize
                                         >=> ignoreSingles ())

                           yield runSinkProbe
                                     $"sqrt-windowed-float-write-{suffix}-win-{windowSize}"
                                     $"Synthetic Float64 {suffix} source, windowed sqrt with window size {windowSize}, cast to UInt8, write."
                                     (let p = singletonImageParameters size "float" windowSize
                                      p["operation"] <- "sqrt-windowed-write"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> zero<float> size.Width size.Height size.Depth
                                         >=> imageAddScalar 4.0
                                         >=> sqrtWindowed<float> windowSize
                                         >=> cast<float, uint8>
                                         >=> write (outputDir size $"sqrt-windowed-float-write-win-{windowSize}") ".tiff")

                       if not options.SqrtOnly then
                           for windowSize in gaussianWindowSizes do
                               let inputDir = inputDirs[size]
                               yield runSinkProbe
                                         $"convolve3d-read-float-cast-write-{suffix}-win-{windowSize}"
                                         $"Read {suffix} stack as Float64, discreteGaussian windowed convolution with window size {windowSize}, cast to UInt8, write."
                                         (let p = singletonImageParameters size "float" windowSize
                                          p["operation"] <- "discreteGaussian-write"
                                          p["sigma"] <- "1"
                                          p["kernelSize"] <- "5"
                                          p)
                                         (fun () ->
                                             source availableMemory
                                             |> read<float> inputDir ".tiff"
                                             >=> discreteGaussian 1.0 None None (Some windowSize)
                                             >=> cast<float, uint8>
                                             >=> write (outputDir size $"convolve3d-read-float-cast-write-win-{windowSize}") ".tiff")

               if options.StackUnstackOnly then
                   for depth in inputDepths do
                       let size = imageSize xy depth
                       let suffix = sizeName size

                       for windowSize in stackUnstackWindowSizes do
                           yield runSinkProbe
                                     $"stack-unstack-input-float-{suffix}-win-{windowSize}"
                                     $"Synthetic Float64 {suffix} source, windowed and flattened with window size {windowSize} without stacking."
                                     (let p = imageParameters size "float" windowSize 1u
                                      p["operation"] <- "stack-unstack-input"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> zero<float> size.Width size.Height size.Depth
                                         >=> stackUnstackInputStage windowSize
                                         >=> ignoreSingles ())

                           yield runSinkProbe
                                     $"stack-unstack-float-{suffix}-win-{windowSize}"
                                     $"Synthetic Float64 {suffix} source, stack then unstack with window size {windowSize}."
                                     (let p = imageParameters size "float" windowSize 1u
                                      p["operation"] <- "stack-unstack"
                                      p["baselineOperation"] <- "stack-unstack-input"
                                      p["stage"] <- "stack-unstack"
                                      p)
                                     (fun () ->
                                         source availableMemory
                                         |> zero<float> size.Width size.Height size.Depth
                                         >=> stackUnstackStage windowSize
                                         >=> ignoreSingles ())

               if options.ConvolutionBreakdownOnly && xy = List.head xySizes then
                   let kernel = convolutionBreakdownKernel |> Option.get
                   let kernelSize = 8u

                   for size in convolutionBreakdownSizes do
                       let suffix = sizeName size
                       let windowSize = size.Depth

                       yield runSinkProbe
                                 $"convolution-breakdown-input-float-{suffix}"
                                 $"Synthetic Float64 {suffix} source, windowed and flattened without stack or convolution."
                                 (let p = imageParameters size "float" windowSize 1u
                                  p["operation"] <- "convolution-breakdown-input"
                                  p["kernelSize"] <- string kernelSize
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float> size.Width size.Height size.Depth
                                     >=> convolutionBreakdownInputStage windowSize
                                     >=> ignoreSingles ())

                       yield runSinkProbe
                                 $"convolution-breakdown-stack-float-{suffix}"
                                 $"Synthetic Float64 {suffix} source, stack only."
                                 (let p = imageParameters size "float" windowSize 1u
                                  p["operation"] <- "convolution-breakdown-stack"
                                  p["baselineOperation"] <- "convolution-breakdown-input"
                                  p["stage"] <- "stack"
                                  p["kernelSize"] <- string kernelSize
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float> size.Width size.Height size.Depth
                                     >=> convolutionBreakdownStackStage windowSize
                                     >=> ignoreSingles ())

                       yield runSinkProbe
                                 $"convolution-breakdown-stack-convolve-float-{suffix}"
                                 $"Synthetic Float64 {suffix} source, stack and SimpleITK Valid convolution with an 8^3 Gaussian kernel."
                                 (let p = imageParameters size "float" windowSize 1u
                                  p["operation"] <- "convolution-breakdown-stack-convolve"
                                  p["baselineOperation"] <- "convolution-breakdown-stack"
                                  p["stage"] <- "convolve"
                                  p["kernelSize"] <- string kernelSize
                                  p["outputRegionMode"] <- "Valid"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float> size.Width size.Height size.Depth
                                     >=> convolutionBreakdownStackConvolveStage windowSize kernel
                                     >=> ignoreSingles ())

                       yield runSinkProbe
                                 $"convolution-breakdown-stack-convolve-unstack-float-{suffix}"
                                 $"Synthetic Float64 {suffix} source, stack, SimpleITK Valid convolution with an 8^3 Gaussian kernel, then unstack."
                                 (let p = imageParameters size "float" windowSize 1u
                                  p["operation"] <- "convolution-breakdown-stack-convolve-unstack"
                                  p["baselineOperation"] <- "convolution-breakdown-stack-convolve"
                                  p["stage"] <- "unstack"
                                  p["kernelSize"] <- string kernelSize
                                  p["outputRegionMode"] <- "Valid"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float> size.Width size.Height size.Depth
                                     >=> convolutionBreakdownStackConvolveUnstackStage windowSize kernel
                                     >=> ignoreSingles ())

               if options.DiscreteGaussianBreakdownOnly && xy = List.head xySizes then
                   let kernelSize = 8u

                   for size in convolutionBreakdownSizes do
                       let suffix = sizeName size
                       let windowSize = size.Depth

                       yield runSinkProbe
                                 $"discrete-gaussian-breakdown-input-float-{suffix}"
                                 $"Synthetic Float64 {suffix} source, windowed and flattened without stack or discreteGaussian."
                                 (let p = imageParameters size "float" windowSize 1u
                                  p["operation"] <- "discrete-gaussian-breakdown-input"
                                  p["kernelSize"] <- string kernelSize
                                  p["sigma"] <- "1"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float> size.Width size.Height size.Depth
                                     >=> convolutionBreakdownInputStage windowSize
                                     >=> ignoreSingles ())

                       yield runSinkProbe
                                 $"discrete-gaussian-breakdown-stack-float-{suffix}"
                                 $"Synthetic Float64 {suffix} source, stack only."
                                 (let p = imageParameters size "float" windowSize 1u
                                  p["operation"] <- "discrete-gaussian-breakdown-stack"
                                  p["baselineOperation"] <- "discrete-gaussian-breakdown-input"
                                  p["stage"] <- "stack"
                                  p["kernelSize"] <- string kernelSize
                                  p["sigma"] <- "1"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float> size.Width size.Height size.Depth
                                     >=> discreteGaussianBreakdownStackStage windowSize
                                     >=> ignoreSingles ())

                       yield runSinkProbe
                                 $"discrete-gaussian-breakdown-stack-filter-float-{suffix}"
                                 $"Synthetic Float64 {suffix} source, stack and discreteGaussian with an 8^3 Gaussian kernel."
                                 (let p = imageParameters size "float" windowSize 1u
                                  p["operation"] <- "discrete-gaussian-breakdown-stack-filter"
                                  p["baselineOperation"] <- "discrete-gaussian-breakdown-stack"
                                  p["stage"] <- "discreteGaussian"
                                  p["kernelSize"] <- string kernelSize
                                  p["sigma"] <- "1"
                                  p["outputRegionMode"] <- "Default"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float> size.Width size.Height size.Depth
                                     >=> discreteGaussianBreakdownStackFilterStage windowSize kernelSize
                                     >=> ignoreSingles ())

                       yield runSinkProbe
                                 $"discrete-gaussian-breakdown-stack-filter-unstack-float-{suffix}"
                                 $"Synthetic Float64 {suffix} source, stack, discreteGaussian with an 8^3 Gaussian kernel, then unstack."
                                 (let p = imageParameters size "float" windowSize 1u
                                  p["operation"] <- "discrete-gaussian-breakdown-stack-filter-unstack"
                                  p["baselineOperation"] <- "discrete-gaussian-breakdown-stack-filter"
                                  p["stage"] <- "unstack"
                                  p["kernelSize"] <- string kernelSize
                                  p["sigma"] <- "1"
                                  p["outputRegionMode"] <- "Default"
                                  p)
                                 (fun () ->
                                     source availableMemory
                                     |> zero<float> size.Width size.Height size.Depth
                                     >=> discreteGaussianBreakdownStackFilterUnstackStage windowSize kernelSize
                                     >=> ignoreSingles ()) |]

    convolutionBreakdownKernel |> Option.iter (fun kernel -> kernel.decRefCount())
    let calibrations = buildCalibrations probes
    let probes = attachPredictions calibrations probes

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

    printfn "Wrote probing report to %s" reportPath
    0
