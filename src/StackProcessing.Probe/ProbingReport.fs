module ProbeProbingReport

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

let private isWritePipelineOperation (operation: string) =
    operation.EndsWith("-write", StringComparison.Ordinal)

let private displayOperationName (probe: ProbeResultJson) =
    let operation = operationName probe
    if isWritePipelineOperation operation then
        let trimmed = operation.Substring(0, operation.Length - "-write".Length)
        $"{trimmed} pipeline"
    else
        operation

let private plottedRawProbes (probes: ProbeResultJson array) =
    probes

let private copyParameters (parameters: Dictionary<string, string>) =
    let copy = Dictionary<string, string>()

    if not (isNull parameters) then
        parameters
        |> Seq.iter (fun kv -> copy[kv.Key] <- kv.Value)

    copy

let private parameterKey keys (probe: ProbeResultJson) =
    keys
    |> List.map (fun key ->
        probe
        |> tryGetParameter key
        |> Option.defaultValue "")
    |> String.concat "|"

let private label (probe: ProbeResultJson) =
    let parameterText =
        if isNull probe.parameters || probe.parameters.Count = 0 then
            ""
        else
            probe.parameters
            |> Seq.filter (fun kv -> kv.Key <> "operation")
            |> Seq.map (fun kv -> $"{kv.Key}={kv.Value}")
            |> String.concat ", "

    if String.IsNullOrWhiteSpace parameterText then
        displayOperationName probe
    else
        $"{displayOperationName probe}<br>{parameterText}"

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

let private estimateWriteTimeProbes (probes: ProbeResultJson array) =
    let keyFields = [ "width"; "height"; "depth"; "pixelType" ]

    let zeroByKey =
        probes
        |> Array.filter (fun probe -> operationName probe = "zero")
        |> Array.map (fun probe -> parameterKey keyFields probe, probe)
        |> Map.ofArray

    probes
    |> Array.filter (fun probe -> operationName probe = "zero-write")
    |> Array.choose (fun writeProbe ->
        let key = parameterKey keyFields writeProbe

        match zeroByKey |> Map.tryFind key with
        | None -> None
        | Some zeroProbe ->
            let writeElapsed = elapsedMilliseconds writeProbe
            let zeroElapsed = elapsedMilliseconds zeroProbe
            let elapsed = max 0.0 (writeElapsed - zeroElapsed)
            let rssDelta =
                let delta = int64 writeProbe.rssDeltaMedianBytes - int64 zeroProbe.rssDeltaMedianBytes
                if delta > 0L then uint64 delta else 0UL
            let bytes = measuredBytes writeProbe
            let throughput =
                if bytes > 0UL && elapsed > 0.0 then
                    float bytes / (elapsed / 1000.0)
                else
                    0.0
            let p = copyParameters writeProbe.parameters
            p["operation"] <- "write-estimate"
            p["derivedFrom"] <- "zero-write - zero"
            let nameKey = key.Replace("|", "x")
            Some
                { name = $"write-estimate-{nameKey}"
                  description = $"Estimated write time for {key}, derived from zero-write minus zero."
                  parameters = p
                  observedElements = writeProbe.observedElements
                  observedBytes = bytes
                  predictedImagePeakBytes = 0UL
                  rssDeltaMedianBytes = rssDelta
                  elapsedMilliseconds = elapsed
                  elapsedMedianMilliseconds = elapsed
                  throughputMedianElementsPerSecond = 0.0
                  throughputMedianBytesPerSecond = throughput
                  source = writeProbe.source })

let private estimateStageTimeProbes (probes: ProbeResultJson array) =
    let keyFields = [ "width"; "height"; "depth"; "pixelType" ]

    let baselineByKey =
        probes
        |> Array.map (fun probe -> operationName probe, parameterKey keyFields probe, probe)
        |> Array.groupBy (fun (operation, key, _) -> operation, key)
        |> Array.map (fun ((operation, key), matches) ->
            let _, _, probe = matches |> Array.head
            (operation, key), probe)
        |> Map.ofArray

    probes
    |> Array.choose (fun probe ->
        match probe |> tryGetParameter "baselineOperation" with
        | None -> None
        | Some baselineOperation ->
            let key = parameterKey keyFields probe

            match baselineByKey |> Map.tryFind (baselineOperation, key) with
            | None -> None
            | Some baselineProbe ->
                let probeElapsed = elapsedMilliseconds probe
                let baselineElapsed = elapsedMilliseconds baselineProbe
                let elapsed = max 0.0 (probeElapsed - baselineElapsed)
                let rssDelta =
                    let delta = int64 probe.rssDeltaMedianBytes - int64 baselineProbe.rssDeltaMedianBytes
                    if delta > 0L then uint64 delta else 0UL
                let bytes = measuredBytes probe
                let throughput =
                    if bytes > 0UL && elapsed > 0.0 then
                        float bytes / (elapsed / 1000.0)
                    else
                        0.0
                let p = copyParameters probe.parameters
                let stageName =
                    probe
                    |> tryGetParameter "stage"
                    |> Option.defaultValue (operationName probe)
                p["operation"] <- $"{stageName}-estimate"
                p["derivedFrom"] <- $"{operationName probe} - {baselineOperation}"
                let nameKey = key.Replace("|", "x")
                Some
                    { name = $"{stageName}-estimate-{nameKey}"
                      description = $"Estimated stage time for {stageName} and {key}, derived from {operationName probe} minus {baselineOperation}."
                      parameters = p
                      observedElements = probe.observedElements
                      observedBytes = bytes
                      predictedImagePeakBytes = 0UL
                      rssDeltaMedianBytes = rssDelta
                      elapsedMilliseconds = elapsed
                      elapsedMedianMilliseconds = elapsed
                      throughputMedianElementsPerSecond = 0.0
                      throughputMedianBytesPerSecond = throughput
                      source = probe.source })

let private makeScatter title xCandidates yTitle ySelector (probes: ProbeResultJson array) =
    let rows =
        probes
        |> Array.mapi (fun index probe ->
            let parameterName, x = relevantParameter xCandidates index probe
            parameterName, displayOperationName probe, x, ySelector probe, label probe)

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
    printfn "Usage: dotnet run --project src/StackProcessing.Probe -- report <probing-report.json> [output-directory]"

let main (args: string array) =
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

        let writeTimeProbes = estimateWriteTimeProbes report.probes
        let stageTimeProbes = estimateStageTimeProbes report.probes

        if stageTimeProbes.Length = 0 then
            failwith "No stage-estimate probes could be derived. Re-run StackProcessing.Probe probing so the report includes probes with matching baselineOperation metadata."

        let rawProbes = plottedRawProbes report.probes

        let memoryChart =
            makeScatter
                "StackProcessing memory by window size"
                [ "windowSize" ]
                "RSS median delta (MiB)"
                (fun probe -> bytesToMiB probe.rssDeltaMedianBytes)
                rawProbes

        let speedChart =
            makeScatter
                "StackProcessing speed by window size"
                [ "windowSize" ]
                "Throughput (MiB/s)"
                medianMiBPerSecond
                rawProbes

        let memoryByImageChart =
            makeScatter
                "StackProcessing memory by image size (n+2 depth)"
                [ "imagePixels"; "imageVoxels"; "width"; "height"; "depth" ]
                "RSS median delta (MiB)"
                (fun probe -> bytesToMiB probe.rssDeltaMedianBytes)
                (nPlus2Probes rawProbes)

        let speedByImageChart =
            makeScatter
                "StackProcessing speed by image size (n+2 depth)"
                [ "imagePixels"; "imageVoxels"; "width"; "height"; "depth" ]
                "Throughput (MiB/s)"
                medianMiBPerSecond
                (nPlus2Probes rawProbes)

        let memoryPath = Path.Combine(outputDirectory, "stackprocessing-probing-memory-by-window.html")
        let speedPath = Path.Combine(outputDirectory, "stackprocessing-probing-speed-by-window.html")
        let memoryByImagePath = Path.Combine(outputDirectory, "stackprocessing-probing-memory-by-image-size.html")
        let speedByImagePath = Path.Combine(outputDirectory, "stackprocessing-probing-speed-by-image-size.html")
        let writeTimePath = Path.Combine(outputDirectory, "stackprocessing-probing-write-time-by-image-size.html")
        let writeThroughputPath = Path.Combine(outputDirectory, "stackprocessing-probing-write-throughput-by-image-size.html")
        let stageMemoryByWindowPath = Path.Combine(outputDirectory, "stackprocessing-probing-stage-memory-by-window.html")
        let stageSpeedByWindowPath = Path.Combine(outputDirectory, "stackprocessing-probing-stage-speed-by-window.html")
        let stageMemoryByImagePath = Path.Combine(outputDirectory, "stackprocessing-probing-stage-memory-by-image-size.html")
        let stageSpeedByImagePath = Path.Combine(outputDirectory, "stackprocessing-probing-stage-speed-by-image-size.html")

        memoryChart |> Chart.saveHtml(memoryPath)
        speedChart |> Chart.saveHtml(speedPath)
        memoryByImageChart |> Chart.saveHtml(memoryByImagePath)
        speedByImageChart |> Chart.saveHtml(speedByImagePath)

        let stageMemoryByWindowChart =
            makeScatter
                "StackProcessing estimated stage memory by window size"
                [ "windowSize" ]
                "Estimated stage RSS median delta (MiB)"
                (fun probe -> bytesToMiB probe.rssDeltaMedianBytes)
                stageTimeProbes

        let stageSpeedByWindowChart =
            makeScatter
                "StackProcessing estimated stage speed by window size"
                [ "windowSize" ]
                "Estimated stage throughput (MiB/s)"
                medianMiBPerSecond
                stageTimeProbes

        let stageMemoryByImageChart =
            makeScatter
                "StackProcessing estimated stage memory by image size (n+2 depth)"
                [ "imagePixels"; "imageVoxels"; "width"; "height"; "depth" ]
                "Estimated stage RSS median delta (MiB)"
                (fun probe -> bytesToMiB probe.rssDeltaMedianBytes)
                (nPlus2Probes stageTimeProbes)

        let stageSpeedByImageChart =
            makeScatter
                "StackProcessing estimated stage speed by image size (n+2 depth)"
                [ "imagePixels"; "imageVoxels"; "width"; "height"; "depth" ]
                "Estimated stage throughput (MiB/s)"
                medianMiBPerSecond
                (nPlus2Probes stageTimeProbes)

        stageMemoryByWindowChart |> Chart.saveHtml(stageMemoryByWindowPath)
        stageSpeedByWindowChart |> Chart.saveHtml(stageSpeedByWindowPath)
        stageMemoryByImageChart |> Chart.saveHtml(stageMemoryByImagePath)
        stageSpeedByImageChart |> Chart.saveHtml(stageSpeedByImagePath)

        if writeTimeProbes.Length > 0 then
            let writeTimeChart =
                makeScatter
                    "StackProcessing estimated write time by image size"
                    [ "imagePixels"; "imageVoxels"; "width"; "height"; "depth" ]
                    "Estimated write elapsed (ms)"
                    elapsedMilliseconds
                    (nPlus2Probes writeTimeProbes)

            let writeThroughputChart =
                makeScatter
                    "StackProcessing estimated write throughput by image size"
                    [ "imagePixels"; "imageVoxels"; "width"; "height"; "depth" ]
                    "Estimated write throughput (MiB/s)"
                    medianMiBPerSecond
                    (nPlus2Probes writeTimeProbes)

            writeTimeChart |> Chart.saveHtml(writeTimePath)
            writeThroughputChart |> Chart.saveHtml(writeThroughputPath)

        printfn "Wrote %s" memoryPath
        printfn "Wrote %s" speedPath
        printfn "Wrote %s" memoryByImagePath
        printfn "Wrote %s" speedByImagePath
        printfn "Wrote %s" stageMemoryByWindowPath
        printfn "Wrote %s" stageSpeedByWindowPath
        printfn "Wrote %s" stageMemoryByImagePath
        printfn "Wrote %s" stageSpeedByImagePath
        if writeTimeProbes.Length > 0 then
            printfn "Wrote %s" writeTimePath
            printfn "Wrote %s" writeThroughputPath
        0
