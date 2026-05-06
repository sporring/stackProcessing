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
    [ 32u; 64u; 128u ]

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

let private gaussianWindowSizes =
    [ 5u; 9u; 15u ]

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

let private reportPathFromArgs (args: string array) =
    if args.Length > 0 then
        Path.GetFullPath(args[0])
    else
        Path.GetFullPath("stackprocessing-probing.json")

[<EntryPoint>]
let main args =
    let reportPath = reportPathFromArgs args
    let tempRoot =
        Path.Combine(
            Path.GetTempPath(),
            "StackProcessing.Probing",
            DateTimeOffset.UtcNow.ToString("yyyyMMddTHHmmssfff"))

    cleanDirectory tempRoot

    let inputDirs =
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

    let probes =
        [| for xy in xySizes do
               for depth in defaultDepths do
                   let size = imageSize xy depth
                   let inputDir = inputDirs[size]
                   let suffix = sizeName size

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

               for depth in singletonDepths do
                   let size = imageSize xy depth
                   let inputDir = inputDirs[size]
                   let suffix = sizeName size

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

                   for windowSize in gaussianWindowSizes do
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
                                     >=> write (outputDir size $"convolve3d-read-float-cast-write-win-{windowSize}") ".tiff") |]
    let calibrations = Dictionary<string, StageCostCoefficients>()
    for probe in probes do
        match probe.costWorks with
        | [| work |] ->
            let pressure =
                { CpuWorkUnits = work.cpuWorkUnits
                  NativeWorkUnits = work.nativeWorkUnits
                  IoReadBytes = work.ioReadBytes
                  IoWriteBytes = work.ioWriteBytes
                  IoReadOps = work.ioReadOps
                  IoWriteOps = work.ioWriteOps
                  CalibrationKey = Some work.calibrationKey }
            calibrations[work.calibrationKey] <- coefficientsFromWork probe.elapsedMedianMilliseconds pressure
        | _ -> ()

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
