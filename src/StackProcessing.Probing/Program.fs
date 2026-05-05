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
      rssBaselineBytes: uint64
      rssPeakBytes: uint64
      rssDeltaBytes: uint64
      rssPostGcBytes: uint64
      rssRetainedDeltaBytes: uint64
      elapsedMilliseconds: int64 }

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
let private width = 64u
let private height = 64u
let private depth = 64u
let private sampleIntervalMs = 10
let private warmupCount = 1
let private repetitionCount = 5

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

let private median (values: uint64 array) =
    if values.Length = 0 then 0UL
    else
        let sorted = Array.sort values
        sorted[sorted.Length / 2]

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
    let elapsed = float elapsedMilliseconds
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

let private runProbe name description execute (planFactory: unit -> Plan<unit, _>) =
    printfn "Probing %s" name
    let metadataPlan = planFactory()

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
               forceFullGc()
               let postGcRss = MemoryProbe.currentRssBytes()
               let retained =
                   if postGcRss > snapshot.Baseline then
                       postGcRss - snapshot.Baseline
                   else
                       0UL

               { iteration = i
                 rssBaselineBytes = snapshot.Baseline
                 rssPeakBytes = snapshot.Peak
                 rssDeltaBytes = snapshot.Delta
                 rssPostGcBytes = postGcRss
                 rssRetainedDeltaBytes = retained
                 elapsedMilliseconds = stopwatch.ElapsedMilliseconds } |]

    let medianDelta = iterations |> Array.map _.rssDeltaBytes |> median
    let medianRetained = iterations |> Array.map _.rssRetainedDeltaBytes |> median
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
            deltaDistance + retainedDistance)
        |> Array.head
    let costPeakWork = metadataPlan.costPeak |> Option.bind (fun cost -> workToJson cost.Work)
    let costWorks =
        metadataPlan.costObservations
        |> List.rev
        |> List.choose (fun cost -> workToJson cost.Work)
        |> List.toArray

    { name = name
      description = description
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
      warmupCount = warmupCount
      repetitionCount = repetitionCount
      source = sourceToJson metadataPlan.sourcePeek
      costPeakWork = costPeakWork
      costWorks = costWorks
      iterations = iterations }

let private runSinkProbe name description (planFactory: unit -> Plan<unit, _>) =
    let execute plan = sink plan
    runProbe name description execute planFactory

let private runDrainProbe name description (planFactory: unit -> Plan<unit, _>) =
    let execute plan = drain plan |> ignore
    runProbe name description execute planFactory

let private createInputStack inputDir =
    Directory.CreateDirectory(inputDir) |> ignore
    source availableMemory
    |> zero<uint8> width height depth
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

    let inputDir = Path.Combine(tempRoot, "input")
    createInputStack inputDir

    let outputDir name =
        let path = Path.Combine(tempRoot, name)
        Directory.CreateDirectory(path) |> ignore
        path

    let probes =
        [| runSinkProbe
               "zero-uint8-write"
               "Synthetic UInt8 source written to TIFF."
               (fun () ->
                   source availableMemory
                   |> zero<uint8> width height depth
                   >=> write (outputDir "zero-uint8-write") ".tiff")

           runSinkProbe
               "add-normal-noise-uint8-write"
               "Synthetic UInt8 source, additive Gaussian noise, write."
               (fun () ->
                   source availableMemory
                   |> zero<uint8> width height depth
                   >=> addNormalNoise 128.0 50.0
                   >=> write (outputDir "add-normal-noise-uint8-write") ".tiff")

           runSinkProbe
               "read-uint8-ignore"
               "Read UInt8 TIFF stack and consume it without writing."
               (fun () ->
                   source availableMemory
                   |> read<uint8> inputDir ".tiff"
                   >=> ignoreSingles ())

           runSinkProbe
               "read-uint8-write"
               "Copy-style UInt8 read and write pipeline."
               (fun () ->
                   source availableMemory
                   |> read<uint8> inputDir ".tiff"
                   >=> write (outputDir "read-uint8-write") ".tiff")

           runSinkProbe
               "threshold-float-write"
               "Synthetic Float64 source, threshold to UInt8, write."
               (fun () ->
                   source availableMemory
                   |> zero<float> width height depth
                   >=> addNormalNoise 128.0 50.0
                   >=> threshold 128.0 infinity
                   >=> imageMulScalar 255uy
                   >=> write (outputDir "threshold-float-write") ".tiff")

           runSinkProbe
               "convolve3d-read-float-cast-write"
               "Read stack as Float64, discreteGaussian windowed convolution, cast to UInt8, write."
               (fun () ->
                   source availableMemory
                   |> read<float> inputDir ".tiff"
                   >=> discreteGaussian 1.0 None None (Some 15u)
                   >=> cast<float, uint8>
                   >=> write (outputDir "convolve3d-read-float-cast-write") ".tiff")

           runDrainProbe
               "compute-stats-read-float"
               "Read stack as Float64 and drain computeStats reducer."
               (fun () ->
                   source availableMemory
                   |> read<float> inputDir ".tiff"
                   >=> computeStats ()) |]

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
            calibrations[work.calibrationKey] <- coefficientsFromWork probe.elapsedMilliseconds pressure
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
