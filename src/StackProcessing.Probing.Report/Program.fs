module StackProcessing.Probing.Report

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open Plotly.NET

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
      parameters: Dictionary<string, string>
      observedElements: uint64
      observedBytes: uint64
      predictedImagePeakBytes: uint64
      rssDeltaMedianBytes: uint64
      elapsedMilliseconds: float
      elapsedMedianMilliseconds: float
      throughputMedianElementsPerSecond: float
      throughputMedianBytesPerSecond: float
      source: ProbeSourceJson }

[<CLIMutable>]
type ProbeReportJson =
    { generatedUtc: string
      probes: ProbeResultJson array }

let private bytesToMiB bytes =
    float bytes / 1024.0 / 1024.0

let private bytesPerSecondToMiBPerSecond value =
    value / 1024.0 / 1024.0

let private tryGetParameter key (probe: ProbeResultJson) =
    if isNull probe.parameters then
        None
    else
        match probe.parameters.TryGetValue key with
        | true, value -> Some value
        | false, _ -> None

let private tryParseDouble (value: string) =
    match Double.TryParse(value, Globalization.NumberStyles.Float, Globalization.CultureInfo.InvariantCulture) with
    | true, parsed -> Some parsed
    | false, _ -> None

let private tryParseBool (value: string) =
    match Boolean.TryParse(value) with
    | true, parsed -> Some parsed
    | false, _ -> None

let private nPlus2Probes (probes: ProbeResultJson array) =
    let tagged =
        probes
        |> Array.filter (fun probe ->
            probe
            |> tryGetParameter "isNPlus2"
            |> Option.bind tryParseBool
            |> Option.defaultValue false)

    if tagged.Length > 0 then tagged else probes

let private relevantParameter preferredKeys (index: int) (probe: ProbeResultJson) =
    preferredKeys
    |> List.tryPick (fun key ->
        probe
        |> tryGetParameter key
        |> Option.bind tryParseDouble
        |> Option.map (fun value -> key, value))
    |> Option.defaultValue ("probeIndex", float index)

let private operationName (probe: ProbeResultJson) =
    probe
    |> tryGetParameter "operation"
    |> Option.defaultValue probe.name

let private label (probe: ProbeResultJson) =
    let parameterText =
        if isNull probe.parameters || probe.parameters.Count = 0 then
            ""
        else
            probe.parameters
            |> Seq.map (fun kv -> $"{kv.Key}={kv.Value}")
            |> String.concat ", "

    if String.IsNullOrWhiteSpace parameterText then
        probe.name
    else
        $"{probe.name}<br>{parameterText}"

let private measuredBytes (probe: ProbeResultJson) =
    if probe.observedBytes > 0UL then
        probe.observedBytes
    elif not (isNull (box probe.source)) && probe.source.hasLength && probe.source.elementBytes > 0UL then
        probe.source.length * probe.source.elementBytes
    else
        0UL

let private elapsedMilliseconds (probe: ProbeResultJson) =
    if probe.elapsedMedianMilliseconds > 0.0 then
        probe.elapsedMedianMilliseconds
    else
        probe.elapsedMilliseconds

let private medianMiBPerSecond (probe: ProbeResultJson) =
    if probe.throughputMedianBytesPerSecond > 0.0 then
        bytesPerSecondToMiBPerSecond probe.throughputMedianBytesPerSecond
    else
        let bytes = measuredBytes probe
        let elapsed = elapsedMilliseconds probe
        if bytes > 0UL && elapsed > 0.0 then
            bytesPerSecondToMiBPerSecond (float bytes / (elapsed / 1000.0))
        else
            0.0

let private makeScatter title xCandidates yTitle ySelector (probes: ProbeResultJson array) =
    let rows =
        probes
        |> Array.mapi (fun index probe ->
            let parameterName, x = relevantParameter xCandidates index probe
            parameterName, operationName probe, x, ySelector probe, label probe)

    let xTitle =
        rows
        |> Array.tryPick (fun (parameterName, _, _, _, _) ->
            if parameterName <> "probeIndex" then Some parameterName else None)
        |> Option.defaultValue "probe"

    rows
    |> Array.groupBy (fun (_, operation, _, _, _) -> operation)
    |> Array.map (fun (operation, points) ->
        let ordered = points |> Array.sortBy (fun (_, _, x, _, _) -> x)
        let x = ordered |> Array.map (fun (_, _, value, _, _) -> value)
        let y = ordered |> Array.map (fun (_, _, _, value, _) -> value)
        let labels = ordered |> Array.map (fun (_, _, _, _, text) -> text)
        Chart.Scatter(x = x, y = y, mode = StyleParam.Mode.Markers, MultiText = labels)
        |> Chart.withTraceInfo(Name = operation))
    |> Chart.combine
    |> Chart.withTitle (Title.init(Text = title))
    |> Chart.withXAxisStyle xTitle
    |> Chart.withYAxisStyle yTitle

let private usage () =
    printfn "Usage: dotnet run --project src/StackProcessing.Probing.Report -- <probing-report.json> [output-directory]"

[<EntryPoint>]
let main args =
    if args.Length < 1 then
        usage ()
        1
    else
        let inputPath = Path.GetFullPath args[0]
        let outputDirectory =
            if args.Length > 1 then
                Path.GetFullPath args[1]
            else
                Path.GetDirectoryName(inputPath)

        Directory.CreateDirectory(outputDirectory) |> ignore

        let json = File.ReadAllText inputPath
        let options = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
        let report = JsonSerializer.Deserialize<ProbeReportJson>(json, options)

        if isNull (box report) || isNull report.probes || report.probes.Length = 0 then
            failwith $"No probes found in {inputPath}"

        let memoryChart =
            makeScatter
                "StackProcessing memory by window size"
                [ "windowSize" ]
                "RSS median delta (MiB)"
                (fun probe -> bytesToMiB probe.rssDeltaMedianBytes)
                report.probes

        let speedChart =
            makeScatter
                "StackProcessing speed by window size"
                [ "windowSize" ]
                "Throughput (MiB/s)"
                medianMiBPerSecond
                report.probes

        let memoryByImageChart =
            makeScatter
                "StackProcessing memory by image size (n+2 depth)"
                [ "imagePixels"; "imageVoxels"; "width"; "height"; "depth" ]
                "RSS median delta (MiB)"
                (fun probe -> bytesToMiB probe.rssDeltaMedianBytes)
                (nPlus2Probes report.probes)

        let speedByImageChart =
            makeScatter
                "StackProcessing speed by image size (n+2 depth)"
                [ "imagePixels"; "imageVoxels"; "width"; "height"; "depth" ]
                "Throughput (MiB/s)"
                medianMiBPerSecond
                (nPlus2Probes report.probes)

        let memoryPath = Path.Combine(outputDirectory, "stackprocessing-probing-memory-by-window.html")
        let speedPath = Path.Combine(outputDirectory, "stackprocessing-probing-speed-by-window.html")
        let memoryByImagePath = Path.Combine(outputDirectory, "stackprocessing-probing-memory-by-image-size.html")
        let speedByImagePath = Path.Combine(outputDirectory, "stackprocessing-probing-speed-by-image-size.html")

        memoryChart |> Chart.saveHtml(memoryPath)
        speedChart |> Chart.saveHtml(speedPath)
        memoryByImageChart |> Chart.saveHtml(memoryByImagePath)
        speedByImageChart |> Chart.saveHtml(speedByImagePath)

        printfn "Wrote %s" memoryPath
        printfn "Wrote %s" speedPath
        printfn "Wrote %s" memoryByImagePath
        printfn "Wrote %s" speedByImagePath
        0
