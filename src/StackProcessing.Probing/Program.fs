module StackProcessing.Probing

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
let private canonicalKernelSize = canonicalWindowSize

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
    [ canonicalWindowSize ]

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
        | "float" -> "Float64"
        | other -> other

    let readFeature =
        $"Read:type={capitalizedType}:format=Image stack:suffix=.tiff:slabDepth=1:multiscaleIndex=0:datasetIndex=0:timepoint=0:channel=0:maxParallelChunks=0:frameAxis=0:yAxis=1:xAxis=2"

    let writeFeature =
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
        match operationName probe with
        | "zero" ->
            yield zeroFeature
        | "zero-write" ->
            yield zeroFeature
            yield writeFeature
        | "noise-write" ->
            yield zeroFeature
            yield $"AddNormalNoise:type={capitalizedType}:mean={mean}:std={std}"
            yield writeFeature
        | "read-ignore" ->
            yield readFeature
        | "read-write" ->
            yield readFeature
            yield writeFeature
        | "threshold-write" ->
            yield zeroFeature
            yield $"AddNormalNoise:type={capitalizedType}:mean={mean}:std={std}"
            yield $"Threshold:type={capitalizedType}:lower={thresholdValue}:upper=infinity"
            yield "ImageOpScalar:operation=*:type=UInt8:value=255"
            yield writeFeature
        | "compute-stats" ->
            yield readFeature
            yield "ComputeStats"
        | "sqrt" ->
            yield zeroFeature
            yield $"ImageOpScalar:operation=+:type={capitalizedType}:value=4.0"
            yield $"Sqrt:type={capitalizedType}"
        | "sqrt-write" ->
            yield zeroFeature
            yield $"ImageOpScalar:operation=+:type={capitalizedType}:value=4.0"
            yield $"Sqrt:type={capitalizedType}"
            yield "Cast:sourceType=Float64:targetType=UInt8"
            yield writeFeature
        | "sqrt-windowed"
        | "sqrt-windowed-write" ->
            yield zeroFeature
            yield $"ImageOpScalar:operation=+:type={capitalizedType}:value=4.0"
            yield $"SqrtWindowed:type={capitalizedType}:windowSize={windowSize}"
            if operationName probe = "sqrt-windowed-write" then
                yield "Cast:sourceType=Float64:targetType=UInt8"
                yield writeFeature
        | "smoothWGauss-write" ->
            yield readFeature
            yield $"SmoothWGauss:type={capitalizedType}:sigma={sigma}:kernelSize={kernelSize}:boundary=<empty>:windowSize={windowSize}"
            yield "Cast:sourceType=Float64:targetType=UInt8"
            yield writeFeature
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

let private timeCostFromJson (timeCost: ProbeTimeCostJson) =
    { CpuCostUnits = timeCost.cpuCostUnits
      NativeCostUnits = timeCost.nativeCostUnits
      IoReadBytes = timeCost.ioReadBytes
      IoWriteBytes = timeCost.ioWriteBytes
      IoReadOps = timeCost.ioReadOps
      IoWriteOps = timeCost.ioWriteOps
      CalibrationKey = Some timeCost.calibrationKey }

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
    let costPeakTime = metadataPlan.costPeak |> Option.bind (fun cost -> timeCostToJson cost.Time)
    let costTimes =
        metadataPlan.costObservations
        |> List.rev
        |> List.choose (fun cost -> timeCostToJson cost.Time)
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
                    yield [ probe.name; $"cost:{timeCost.calibrationKey}:present"; "1"; "costPresence" ]

                    if timeCost.cpuCostUnits <> 0.0 then
                        yield [ probe.name; $"cost:{timeCost.calibrationKey}:cpuCostUnits"; invariant timeCost.cpuCostUnits; "costUnits" ]

                    if timeCost.nativeCostUnits <> 0.0 then
                        yield [ probe.name; $"cost:{timeCost.calibrationKey}:nativeCostUnits"; invariant timeCost.nativeCostUnits; "costUnits" ]

                    if timeCost.ioReadBytes <> 0UL then
                        yield [ probe.name; $"cost:{timeCost.calibrationKey}:ioReadBytes"; invariantUInt64 timeCost.ioReadBytes; "costUnits" ]

                    if timeCost.ioWriteBytes <> 0UL then
                        yield [ probe.name; $"cost:{timeCost.calibrationKey}:ioWriteBytes"; invariantUInt64 timeCost.ioWriteBytes; "costUnits" ]

                    if timeCost.ioReadOps <> 0UL then
                        yield [ probe.name; $"cost:{timeCost.calibrationKey}:ioReadOps"; invariantUInt64 timeCost.ioReadOps; "costUnits" ]

                    if timeCost.ioWriteOps <> 0UL then
                        yield [ probe.name; $"cost:{timeCost.calibrationKey}:ioWriteOps"; invariantUInt64 timeCost.ioWriteOps; "costUnits" ]
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

let private releaseImages (images: Image<float> list) =
    images |> List.iter (fun image -> image.decRefCount())

let private releaseConsumedImages (window: Window<Image<float>>) =
    window.Items
    |> List.take (min (int window.ReleaseCount) window.Items.Length)
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
        let filtered = ImageFunctions.discreteGaussian 3u canonicalSigma (Some kernelSize) None None stack
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
      AnalysisFeaturesPath: string option
      LowSupportThreshold: int option
      NonBoilerplate: bool
      SqrtOnly: bool
      StackUnstackOnly: bool
      ConvolutionBreakdownOnly: bool
      DiscreteGaussianBreakdownOnly: bool }

let private parseArgs (args: string array) =
    let mutable reportPath = None
    let mutable analysisFeaturesPath = None
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
      LowSupportThreshold = lowSupportThreshold
      NonBoilerplate = nonBoilerplate
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

    let oldFocusedMode =
        options.SqrtOnly
        || options.StackUnstackOnly
        || options.ConvolutionBreakdownOnly
        || options.DiscreteGaussianBreakdownOnly

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

    let convolutionBreakdownKernel =
        if options.ConvolutionBreakdownOnly then
            let kernel: Image<float> = ImageFunctions.gauss 3u canonicalSigma (Some canonicalKernelSize)
            Some kernel
        else
            None

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

                       if includeNonBoilerplate then
                           yield runSinkProbe
                                     $"threshold-float-write-{suffix}"
                                     $"Synthetic Float64 {suffix} source, threshold to UInt8, write."
                                     (let p = defaultImageParameters size "float" 1u
                                      p["operation"] <- "threshold-write"
                                      p["mean"] <- "128.0"
                                      p["std"] <- "50.0"
                                      p["threshold"] <- "128.0"
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
                                         $"Read {suffix} stack as Float64, smoothWGauss windowed convolution with window size {windowSize}, cast to UInt8, write."
                                         (let p = singletonImageParameters size "float" windowSize
                                          p["operation"] <- "smoothWGauss-write"
                                          p["sigma"] <- canonicalSigmaText
                                          p["kernelSize"] <- string canonicalKernelSize
                                          p)
                                     (fun () ->
                                         source availableMemory
                                         |> read<float> inputDir ".tiff"
                                             >=> smoothWGauss canonicalSigma None None (Some windowSize)
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
                   let kernelSize = canonicalKernelSize

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
                   let kernelSize = canonicalKernelSize

                   for size in convolutionBreakdownSizes do
                       let suffix = sizeName size
                       let windowSize = size.Depth

                       yield runSinkProbe
                                 $"discrete-gaussian-breakdown-input-float-{suffix}"
                                 $"Synthetic Float64 {suffix} source, windowed and flattened without stack or discreteGaussian."
                                 (let p = imageParameters size "float" windowSize 1u
                                  p["operation"] <- "discrete-gaussian-breakdown-input"
                                  p["kernelSize"] <- string kernelSize
                                  p["sigma"] <- canonicalSigmaText
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
                                  p["sigma"] <- canonicalSigmaText
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
                                  p["sigma"] <- canonicalSigmaText
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
                                  p["sigma"] <- canonicalSigmaText
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
    let probes = filterProbesByAnalysisFeatures options.LowSupportThreshold options.AnalysisFeaturesPath probes

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
