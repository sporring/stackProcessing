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
type ProbeResultJson =
    { name: string
      description: string
      predictedImagePeakBytes: uint64
      rssBaselineBytes: uint64
      rssPeakBytes: uint64
      rssDeltaBytes: uint64
      elapsedMilliseconds: int64
      source: ProbeSourceJson }

[<CLIMutable>]
type ProbeReportJson =
    { generatedUtc: string
      osDescription: string
      processArchitecture: string
      frameworkDescription: string
      configuration: string
      sampleIntervalMilliseconds: int
      workingDirectory: string
      tempDirectory: string
      probes: ProbeResultJson array }

let private availableMemory = 2UL * 1024UL * 1024UL * 1024UL
let private width = 64u
let private height = 64u
let private depth = 64u
let private sampleIntervalMs = 10

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

let private runSinkProbe name description (planFactory: unit -> Plan<unit, _>) =
    forceFullGc()
    let plan = planFactory()
    let stopwatch = Stopwatch.StartNew()
    let _, snapshot =
        MemoryProbe.measure sampleIntervalMs (fun () ->
            sink plan)
    stopwatch.Stop()
    { name = name
      description = description
      predictedImagePeakBytes = plan.memPeak
      rssBaselineBytes = snapshot.Baseline
      rssPeakBytes = snapshot.Peak
      rssDeltaBytes = snapshot.Delta
      elapsedMilliseconds = stopwatch.ElapsedMilliseconds
      source = sourceToJson plan.sourcePeek }

let private runDrainProbe name description (planFactory: unit -> Plan<unit, _>) =
    forceFullGc()
    let plan = planFactory()
    let stopwatch = Stopwatch.StartNew()
    let _, snapshot =
        MemoryProbe.measure sampleIntervalMs (fun () ->
            drain plan |> ignore)
    stopwatch.Stop()
    { name = name
      description = description
      predictedImagePeakBytes = plan.memPeak
      rssBaselineBytes = snapshot.Baseline
      rssPeakBytes = snapshot.Peak
      rssDeltaBytes = snapshot.Delta
      elapsedMilliseconds = stopwatch.ElapsedMilliseconds
      source = sourceToJson plan.sourcePeek }

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
               "read-uint8-write"
               "Read UInt8 TIFF stack and write it through."
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

    let report =
        { generatedUtc = DateTimeOffset.UtcNow.ToString("O")
          osDescription = RuntimeInformation.OSDescription
          processArchitecture = RuntimeInformation.ProcessArchitecture.ToString()
          frameworkDescription = RuntimeInformation.FrameworkDescription
          configuration = "Default"
          sampleIntervalMilliseconds = sampleIntervalMs
          workingDirectory = Environment.CurrentDirectory
          tempDirectory = tempRoot
          probes = probes }

    let options = JsonSerializerOptions(WriteIndented = true)
    options.Converters.Add(JsonStringEnumConverter())
    let json = JsonSerializer.Serialize(report, options)
    File.WriteAllText(reportPath, json)

    printfn "Wrote probing report to %s" reportPath
    0
