module ProbeAnalysis

open System
open System.Globalization
open System.IO
open System.Text
open System.Text.RegularExpressions
open Studio.Graph
open StackProcessingCost

type Options =
    { SamplesRoot: string
      OutputDirectory: string
      ModelOutputPath: string
      ExtraJsonRoots: string list
      ProbingPrefixes: string list
      Ridge: float
      MinSupport: int
      DiagnosticsOnly: bool
      IncludeSamples: bool }

type AnalysisRow =
    { RowId: string
      Source: string
      ItemPath: string
      FeatureValues: Map<string, float> }

type RuntimeCostTerm =
    { StageName: string
      InputLength: uint64 option
      OutputLength: uint64 option
      Multiplicity: uint64
      MemoryPeakBytes: uint64 option
      Tags: Map<string, string> }

type Measurement =
    { RowId: string
      Name: string
      Value: float
      SourcePath: string
      RuntimeCostTerms: RuntimeCostTerm list }

type RunSummary =
    { EstimatedPeakMemoryKB: float option
      RssPeakDeltaKB: float option
      ActualRunSeconds: float option
      ProcessElapsedSeconds: float option
      RuntimeCostTerms: RuntimeCostTerm list }

let private usage () =
    printfn "Usage: dotnet run --project src/StackProcessing.Probe -- analysis [--samples-root PATH] [--output PATH] [--probing-prefix PATH] [--ridge VALUE] [--diagnostics-only]"
    printfn ""
    printfn "Extracts Studio JSON stage keys, optionally merges Probing CSV exports,"
    printfn "writes a system matrix, and fits non-negative least-squares coefficients for each measurement vector."
    printfn ""
    printfn "Defaults:"
    printfn "  --samples-root samples"
    printfn "  --output       tmp/analysis"
    printfn "  --model-output models/fitted/stackprocessing.operator-cost.json"
    printfn "                 Write the fitted operator-term model outside tmp."
    printfn "  --extra-json-root PATH"
    printfn "                 Include generated Studio JSON graphs from PATH."
    printfn "  --ridge        1e-8"
    printfn "  --min-support  N"
    printfn "                 Minimum feature support for fitted operator terms. Defaults to 3."
    printfn "  --diagnostics-only"
    printfn "                 Write matrix/coverage diagnostics without parsing run logs or fitting coefficients."
    printfn "  --no-samples   Analyze only extra/probe JSON roots."

let private defaultSamplesRoot () =
    let cwd = Directory.GetCurrentDirectory()

    if Directory.Exists(Path.Combine(cwd, "samples")) then
        Path.Combine(cwd, "samples")
    elif String.Equals(Path.GetFileName cwd, "samples", StringComparison.OrdinalIgnoreCase) then
        cwd
    else
        Path.GetFullPath(Path.Combine(cwd, "..", "..", "samples"))

let private repositoryRootFromSamplesRoot samplesRoot =
    let cwd = Directory.GetCurrentDirectory()

    if File.Exists(Path.Combine(cwd, "StackProcessing.sln")) then
        cwd
    elif String.Equals(Path.GetFileName(Path.GetFullPath samplesRoot), "samples", StringComparison.OrdinalIgnoreCase) then
        Directory.GetParent(Path.GetFullPath samplesRoot).FullName
    else
        cwd

let private probeOutputRoot samplesRoot =
    Path.Combine(repositoryRootFromSamplesRoot samplesRoot, "tmp")

let private defaultOptions () =
    let samplesRoot = defaultSamplesRoot ()
    let repositoryRoot = repositoryRootFromSamplesRoot samplesRoot

    { SamplesRoot = samplesRoot
      OutputDirectory = Path.Combine(probeOutputRoot samplesRoot, "analysis")
      ModelOutputPath = Path.Combine(repositoryRoot, "models", "fitted", "stackprocessing.operator-cost.json")
      ExtraJsonRoots = []
      ProbingPrefixes = []
      Ridge = 1e-8
      MinSupport = 3
      DiagnosticsOnly = false
      IncludeSamples = true }

let rec private parseArgs options args =
    match args with
    | [] -> Ok options
    | ("-h" | "--help") :: _ ->
        usage ()
        Error 0
    | "--samples-root" :: value :: rest ->
        parseArgs { options with SamplesRoot = Path.GetFullPath value } rest
    | "--output" :: value :: rest ->
        parseArgs { options with OutputDirectory = Path.GetFullPath value } rest
    | "--model-output" :: value :: rest ->
        parseArgs { options with ModelOutputPath = Path.GetFullPath value } rest
    | "--extra-json-root" :: value :: rest
    | "--generated-json-root" :: value :: rest
    | "--probe-json-root" :: value :: rest ->
        parseArgs { options with ExtraJsonRoots = options.ExtraJsonRoots @ [ Path.GetFullPath value ] } rest
    | "--probing-prefix" :: value :: rest ->
        parseArgs { options with ProbingPrefixes = options.ProbingPrefixes @ [ Path.GetFullPath value ] } rest
    | "--ridge" :: value :: rest ->
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, ridge when ridge >= 0.0 -> parseArgs { options with Ridge = ridge } rest
        | _ ->
            eprintfn "analysis: --ridge expects a non-negative floating-point value"
            Error 2
    | "--min-support" :: value :: rest ->
        match Int32.TryParse value with
        | true, minSupport when minSupport > 0 -> parseArgs { options with MinSupport = minSupport } rest
        | _ ->
            eprintfn "analysis: --min-support expects a positive integer"
            Error 2
    | "--diagnostics-only" :: rest ->
        parseArgs { options with DiagnosticsOnly = true } rest
    | "--no-samples" :: rest ->
        parseArgs { options with IncludeSamples = false } rest
    | option :: _ ->
        eprintfn "analysis: unknown option %s" option
        usage ()
        Error 2

let private relativePath root path =
    Path.GetRelativePath(root, path).Replace(Path.DirectorySeparatorChar, '/')

let private csvEscape (value: string) =
    if value.Contains(',') || value.Contains('"') || value.Contains('\n') then
        "\"" + value.Replace("\"", "\"\"") + "\""
    else
        value

let private writeCsv (path: string) (rows: seq<string list>) =
    Directory.CreateDirectory(Path.GetDirectoryName path) |> ignore
    rows
    |> Seq.map (List.map csvEscape >> String.concat ",")
    |> fun lines -> File.WriteAllLines(path, lines)

let private parseCsvLine (line: string) =
    let values = ResizeArray<string>()
    let current = StringBuilder()
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

let private readCsv path =
    if File.Exists path then
        File.ReadAllLines path
        |> Array.toList
        |> List.map parseCsvLine
    else
        []

let private columnIndex name (header: string list) =
    header |> List.tryFindIndex ((=) name)

let private columnValue index (row: string list) =
    index
    |> Option.bind (fun index ->
        if index < row.Length then Some row[index] else None)
    |> Option.defaultValue ""

let private tryParseCsvFloat (value: string) =
    match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
    | true, parsed -> Some parsed
    | _ ->
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.CurrentCulture) with
        | true, parsed -> Some parsed
        | _ -> None

let private normalizeValue (value: string) =
    Regex.Replace(value.Trim(), @"\s+", " ")

let private preservedInputParameterKeys =
    // Studio can mark size parameters as graph inputs while still saving the
    // concrete sample value. For timing features, those values matter more than
    // the fact that they were exposed as inputs.
    set [ "width"; "height"; "depth"; "polygon" ]

let private parameterValues (node: SavedNode) =
    let definition = BuiltInCatalog.find node.FunctionId
    let saved =
        node.Parameters
        |> Array.map (fun parameter ->
            let value =
                if parameter.UseInput && not (preservedInputParameterKeys.Contains parameter.Key) then
                    "<input>"
                else
                    normalizeValue parameter.Value

            parameter.Key, value)
        |> Map.ofArray

    definition,
    [ for parameter in definition.Parameters do
          let value =
              saved
              |> Map.tryFind parameter.Key
              |> Option.defaultValue (normalizeValue parameter.DefaultValue)

          parameter.Key, value ]

let private ignoredTimingParameterKeys =
    set
        [ "availableMemory"
          "input"
          "output"
          "suffix"
          "title"
          "xAxis"
          "yAxis"
          "label"
          "name"
          "datasetPath"
          "physicalSizeX"
          "physicalSizeY"
          "physicalSizeZ" ]

let private ignoredTimingFunctionIds =
    set [ "Expand"; "FileDirectory"; "Scalar"; "Tap" ]

let private collapsedTimingFunctionIds =
    set [ "Print" ]

let private featureKey (node: SavedNode) =
    let definition, parameters = parameterValues node
    if collapsedTimingFunctionIds.Contains definition.Id then
        definition.Id
    else
        let parameters =
            parameters
            |> List.filter (fun (key, _) -> not (ignoredTimingParameterKeys.Contains key))

        match parameters with
        | [] -> definition.Id
        | _ ->
            parameters
            |> List.map (fun (key, value) -> $"{key}={value}")
            |> String.concat ":"
            |> fun suffix -> $"{definition.Id}:{suffix}"

let private safeName (value: string) =
    Regex.Replace(value.Replace('\\', '/'), @"[^A-Za-z0-9_.-]+", "_")

let private generatedPrefixForRoot (root: string) =
    let rootName =
        root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
        |> Path.GetFileName

    if rootName.StartsWith("calibration_", StringComparison.OrdinalIgnoreCase)
       || rootName.StartsWith("bottomup_", StringComparison.OrdinalIgnoreCase)
       || rootName.Equals("probingGraphs", StringComparison.OrdinalIgnoreCase) then
        "generated"
    else
        "generated/" + safeName rootName

let private stripGeneratedSessionPrefix (rowId: string) =
    let parts = rowId.Split([| '/' |], StringSplitOptions.RemoveEmptyEntries)

    if parts.Length > 1
       && (parts[0].StartsWith("calibration_", StringComparison.OrdinalIgnoreCase)
           || parts[0].StartsWith("bottomup_", StringComparison.OrdinalIgnoreCase)) then
        String.concat "/" parts[1..]
    else
        rowId

let private discoverGraphsInRoot (samplesRoot: string) (scanRoot: string) includeTmp namePrefix =
    Directory.EnumerateFiles(scanRoot, "*.json", SearchOption.AllDirectories)
    |> Seq.filter (fun path ->
        let relative = relativePath scanRoot path
        let parts = relative.Split('/')

        (includeTmp || not (relative.StartsWith("tmp/", StringComparison.OrdinalIgnoreCase)))
        && not (relative.StartsWith("RunAll/", StringComparison.OrdinalIgnoreCase))
        && not (relative.StartsWith("RunJson/", StringComparison.OrdinalIgnoreCase))
        && not (parts |> Array.exists (fun part ->
            part.Equals("bin", StringComparison.OrdinalIgnoreCase)
            || part.Equals("obj", StringComparison.OrdinalIgnoreCase))))
    |> Seq.sort
    |> Seq.map (fun path ->
        let relative = relativePath scanRoot path
        let localRowId =
            relative.Substring(0, relative.Length - Path.GetExtension(relative).Length)
            |> fun rowId -> if includeTmp then stripGeneratedSessionPrefix rowId else rowId

        let rowId =
            if String.IsNullOrWhiteSpace namePrefix then
                localRowId
            else
                namePrefix.TrimEnd('/') + "/" + localRowId
        let graph = PipelineGraphStorage.load path
        let features =
            graph.Nodes
            |> Array.filter (fun node -> not (ignoredTimingFunctionIds.Contains node.FunctionId))
            |> Array.map featureKey
            |> Array.distinct
            |> Array.sort
            |> Array.map (fun feature -> feature, 1.0)
            |> Map.ofArray

        { RowId = rowId
          Source = "runJson"
          ItemPath = path
          FeatureValues = features |> Map.add "intercept" 1.0 })
    |> Seq.toList

let private discoverGraphs (samplesRoot: string) includeSamples extraJsonRoots =
    let sampleGraphs =
        if includeSamples then
            discoverGraphsInRoot samplesRoot samplesRoot false ""
        else
            []

    let generatedGraphs =
        extraJsonRoots
        |> List.collect (fun root ->
            if Directory.Exists root then
                let prefix = generatedPrefixForRoot root
                discoverGraphsInRoot samplesRoot root true prefix
            else
                eprintfn "analysis: extra JSON root does not exist: %s" root
                [])

    sampleGraphs @ generatedGraphs

let private isGeneratedRow (rowId: string) =
    rowId.StartsWith("generated/", StringComparison.Ordinal)

let private generatedRowContains (marker: string) (rowId: string) =
    isGeneratedRow rowId && rowId.Contains(marker, StringComparison.Ordinal)

let private probingRowId rowId =
    $"probing/{rowId}"

let private discoverProbingRows prefixes =
    prefixes
    |> List.collect (fun prefix ->
        let rowsCsv = readCsv (prefix + "-rows.csv")
        let featuresCsv = readCsv (prefix + "-features.csv")

        let rowPaths =
            match rowsCsv with
            | header :: rows ->
                let rowIdIndex = columnIndex "rowId" header
                let descriptionIndex = columnIndex "description" header

                rows
                |> List.choose (fun row ->
                    let rowId = columnValue rowIdIndex row
                    if String.IsNullOrWhiteSpace rowId then
                        None
                    else
                        Some(probingRowId rowId, columnValue descriptionIndex row))
                |> Map.ofList
            | [] -> Map.empty

        let featureGroups =
            match featuresCsv with
            | header :: rows ->
                let rowIdIndex = columnIndex "rowId" header
                let featureIndex = columnIndex "featureKey" header
                let valueIndex = columnIndex "value" header
                let featureKindIndex = columnIndex "featureKind" header
                let includedFeatureKinds =
                    set [ "sampleCompatible" ]

                rows
                |> List.choose (fun row ->
                    let rowId = columnValue rowIdIndex row
                    let feature = columnValue featureIndex row
                    let value = columnValue valueIndex row
                    let featureKind = columnValue featureKindIndex row

                    match rowId, feature, includedFeatureKinds.Contains featureKind, tryParseCsvFloat value with
                    | "", _, _, _
                    | _, "", _, _
                    | _, _, false, _
                    | _, _, _, None -> None
                    | rowId, feature, true, Some value -> Some(probingRowId rowId, feature, value))
                |> List.groupBy (fun (rowId, _, _) -> rowId)
                |> List.map (fun (rowId, values) ->
                    let features =
                        values
                        |> List.map (fun (_, feature, value) -> feature, value)
                        |> Map.ofList

                    { RowId = rowId
                      Source = "probing"
                      ItemPath = rowPaths |> Map.tryFind rowId |> Option.defaultValue prefix
                      FeatureValues = features |> Map.add "intercept" 1.0 })
            | [] -> []

        featureGroups)

let private tryRegex (pattern: string) (line: string) =
    let m = Regex.Match(line, pattern)
    if m.Success then Some m else None

let private groupValue (name: string) (m: Match) =
    let group = m.Groups[name]
    if group.Success then group.Value else ""

let private tryParseFloat (value: string) =
    match Double.TryParse(value, NumberStyles.Float, CultureInfo.CurrentCulture) with
    | true, parsed -> Some parsed
    | _ ->
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, parsed -> Some parsed
        | _ -> None

let private secondsFromRunSummary (value: string) (unit: string) =
    match tryParseFloat value, unit with
    | Some milliseconds, "ms" -> Some(milliseconds / 1000.0)
    | Some seconds, "s" -> Some seconds
    | _ -> None

let private elapsedSecondsFromLog (lines: string array) =
    lines
    |> Array.rev
    |> Array.tryPick (fun line ->
        if line.StartsWith("Run finished in ", StringComparison.Ordinal) then
            let value = line.Substring("Run finished in ".Length).Trim().TrimEnd('.')

            match TimeSpan.TryParse value with
            | true, elapsed -> Some elapsed.TotalSeconds
            | _ -> None
        else
            None)

let private tryParseInvariantUInt64 (value: string) =
    match UInt64.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture) with
    | true, parsed -> Some parsed
    | _ -> None

let private parseRuntimeTerm (text: string) =
    let fields =
        text.Split(';', StringSplitOptions.RemoveEmptyEntries)
        |> Array.choose (fun part ->
            let index = part.IndexOf('=')
            if index <= 0 then
                None
            else
                Some(part.Substring(0, index), part.Substring(index + 1)))
        |> Map.ofArray

    if fields.IsEmpty then
        None
    else
        let field key = fields |> Map.tryFind key

        let tags =
            fields
            |> Map.remove "stage"
            |> Map.remove "inputLength"
            |> Map.remove "outputLength"
            |> Map.remove "multiplicity"
            |> Map.remove "memoryPeakBytes"
            |> Map.remove "estimatedMilliseconds"

        Some
            { StageName = field "stage" |> Option.defaultValue ""
              InputLength = field "inputLength" |> Option.bind tryParseInvariantUInt64
              OutputLength = field "outputLength" |> Option.bind tryParseInvariantUInt64
              Multiplicity = field "multiplicity" |> Option.bind tryParseInvariantUInt64 |> Option.defaultValue 1UL
              MemoryPeakBytes = field "memoryPeakBytes" |> Option.bind tryParseInvariantUInt64
              Tags = tags }

let private parseRuntimeTermsText (text: string) =
    if String.IsNullOrWhiteSpace text then
        []
    else
        text.Split('|', StringSplitOptions.RemoveEmptyEntries)
        |> Array.choose parseRuntimeTerm
        |> Array.toList

let private runtimeCostTermsFromLog (lines: string array) =
    let marker = "Pipeline cost terms:"

    lines
    |> Array.rev
    |> Array.tryPick (fun line ->
        let index = line.IndexOf(marker, StringComparison.Ordinal)
        if index < 0 then
            None
        else
            Some(line.Substring(index + marker.Length).Trim() |> parseRuntimeTermsText))
    |> Option.defaultValue []

let private parseRunSummary (logPath: string) =
    if not (File.Exists logPath) then
        None
    else
        let lines = File.ReadAllLines logPath
        let runtimeCostTerms = runtimeCostTermsFromLog lines

        lines
        |> Array.indexed
        |> Array.rev
        |> Array.tryPick (fun (index, line) ->
            if line.Contains("Run summary:", StringComparison.Ordinal) && index + 3 < lines.Length then
                match
                    tryRegex @"^estimated peak / available memory (?<estimatedPeakMemory>\S+) KB / (?<availableMemory>\S+) KB$" lines[index + 1],
                    tryRegex @"^Measured peak delta, baseline, peak: (?<peakMemory>\S+) KB \(baseline (?<baselineMemory>\S+) KB, peak (?<processPeakMemory>\S+) KB\)$" lines[index + 2],
                    tryRegex @"^Estimated/actual time .+ / (?<actualTime>\S+) (?<actualTimeUnit>ms|s)\.?$" lines[index + 3]
                with
                | Some estimated, Some measured, Some timing ->
                    Some
                        { EstimatedPeakMemoryKB = groupValue "estimatedPeakMemory" estimated |> tryParseFloat
                          RssPeakDeltaKB = groupValue "peakMemory" measured |> tryParseFloat
                          ActualRunSeconds = secondsFromRunSummary (groupValue "actualTime" timing) (groupValue "actualTimeUnit" timing)
                          ProcessElapsedSeconds = elapsedSecondsFromLog lines
                          RuntimeCostTerms = runtimeCostTerms }
                | _ -> None
            else
                None)

let private runOutputDirectories samplesRoot prefix =
    let tmpRoots =
        [ probeOutputRoot samplesRoot
          Path.Combine(samplesRoot, "tmp") ]
        |> List.distinct

    seq {
        for tmp in tmpRoots do
            let legacy = Path.Combine(tmp, prefix)
            if Directory.Exists legacy then
                yield legacy

            if Directory.Exists tmp then
                yield!
                    Directory.EnumerateDirectories(tmp, prefix + "_*", SearchOption.TopDirectoryOnly)
                    |> Seq.collect (fun batch ->
                        Directory.EnumerateDirectories(batch, "repeat_*", SearchOption.TopDirectoryOnly))
    }
    |> Seq.distinct
    |> Seq.toList

let private logPathsForRow samplesRoot rowId =
    let rowIdCandidates =
        let m = Regex.Match(rowId, @"^generated/size_[^/]+/(.+)$")
        if m.Success then
            [ rowId; "generated/" + m.Groups[1].Value ]
        else
            [ rowId ]

    [ "runJson"; "runAll" ]
    |> List.collect (fun prefix ->
        runOutputDirectories samplesRoot prefix
        |> List.choose (fun outputDir ->
            rowIdCandidates
            |> List.tryPick (fun candidate ->
                let logPath = Path.Combine(outputDir, candidate + ".out")
                if File.Exists logPath then Some logPath else None)))
    |> List.distinct

let private measurementsForLog rowId logPath =
    match parseRunSummary logPath with
    | None -> []
    | Some summary ->
        [ "elapsedMilliseconds", summary.ActualRunSeconds |> Option.map (fun seconds -> seconds * 1000.0)
          "actualRunSeconds", summary.ActualRunSeconds
          "processElapsedMilliseconds", summary.ProcessElapsedSeconds |> Option.map (fun seconds -> seconds * 1000.0)
          "processElapsedSeconds", summary.ProcessElapsedSeconds
          "rssDeltaBytes", summary.RssPeakDeltaKB |> Option.map (fun kb -> kb * 1024.0)
          "rssPeakDeltaKB", summary.RssPeakDeltaKB
          "predictedMemoryPeakBytes", summary.EstimatedPeakMemoryKB |> Option.map (fun kb -> kb * 1024.0)
          "estimatedPeakMemoryKB", summary.EstimatedPeakMemoryKB ]
        |> List.choose (fun (name, value) ->
            value
            |> Option.map (fun value ->
                { RowId = rowId
                  Name = name
                  Value = value
                  SourcePath = logPath
                  RuntimeCostTerms = summary.RuntimeCostTerms }))

let private measurementsForRow (samplesRoot: string) (row: AnalysisRow) =
    if row.Source <> "runJson" then
        []
    else
        logPathsForRow samplesRoot row.RowId
        |> List.collect (measurementsForLog row.RowId)

let private probingMeasurementAliases measurement value =
    seq {
        yield measurement, value

        match measurement with
        | "actualElapsedMedianMilliseconds" -> yield "elapsedMilliseconds", value
        | "rssDeltaMedianBytes" -> yield "rssDeltaBytes", value
        | _ -> ()
    }
    |> Seq.toList

let private probingMeasurements prefixes =
    prefixes
    |> List.collect (fun prefix ->
        match readCsv (prefix + "-vectors.csv") with
        | header :: rows ->
            let rowIdIndex = columnIndex "rowId" header
            let measurementIndex = columnIndex "measurement" header
            let valueIndex = columnIndex "value" header

            rows
            |> List.choose (fun row ->
                let rowId = columnValue rowIdIndex row
                let measurement = columnValue measurementIndex row
                let value = columnValue valueIndex row

                match rowId, measurement, tryParseCsvFloat value with
                | "", _, _
                | _, "", _
                | _, _, None -> None
                | rowId, measurement, Some value ->
                    Some
                        [ for name, value in probingMeasurementAliases measurement value ->
                            { RowId = probingRowId rowId
                              Name = name
                              Value = value
                              SourcePath = prefix + "-vectors.csv"
                              RuntimeCostTerms = [] } ])
            |> List.concat
        | [] -> [])

let private matrix (rows: AnalysisRow list) (features: string list) =
    let featureIndex =
        features
        |> List.mapi (fun index feature -> feature, index)
        |> Map.ofList

    let a = Array2D.zeroCreate<float> rows.Length features.Length

    rows
    |> List.iteri (fun rowIndex row ->
        row.FeatureValues
        |> Map.iter (fun feature value ->
            match featureIndex |> Map.tryFind feature with
            | Some col -> a[rowIndex, col] <- value
            | None -> ()))

    a

let private matrixRank tolerance (a: float[,]) =
    let rows = a.GetLength(0)
    let cols = a.GetLength(1)
    let m = Array2D.copy a
    let mutable rank = 0
    let mutable row = 0
    let mutable col = 0

    while row < rows && col < cols do
        let mutable pivotRow = row
        let mutable pivotValue = abs m[row, col]

        for candidateRow in row + 1 .. rows - 1 do
            let candidate = abs m[candidateRow, col]
            if candidate > pivotValue then
                pivotRow <- candidateRow
                pivotValue <- candidate

        if pivotValue <= tolerance then
            col <- col + 1
        else
            if pivotRow <> row then
                for k in col .. cols - 1 do
                    let tmp = m[row, k]
                    m[row, k] <- m[pivotRow, k]
                    m[pivotRow, k] <- tmp

            let pivot = m[row, col]
            for k in col .. cols - 1 do
                m[row, k] <- m[row, k] / pivot

            for eliminateRow in row + 1 .. rows - 1 do
                let factor = m[eliminateRow, col]
                if factor <> 0.0 then
                    for k in col .. cols - 1 do
                        m[eliminateRow, k] <- m[eliminateRow, k] - factor * m[row, k]

            rank <- rank + 1
            row <- row + 1
            col <- col + 1

    rank

let private subMatrix (rowIndexes: int list) (colIndexes: int list) (a: float[,]) =
    let b = Array2D.zeroCreate<float> rowIndexes.Length colIndexes.Length

    rowIndexes
    |> List.iteri (fun targetRow sourceRow ->
        colIndexes
        |> List.iteri (fun targetCol sourceCol ->
            b[targetRow, targetCol] <- a[sourceRow, sourceCol]))

    b

let private activeColumnIndexes (rowIndexes: int list) (a: float[,]) =
    [ for col in 0 .. a.GetLength(1) - 1 do
          if rowIndexes |> List.exists (fun row -> a[row, col] <> 0.0) then
              col ]

let private gramMatrix (a: float[,]) =
    let rows = a.GetLength(0)
    let cols = a.GetLength(1)
    let gram = Array2D.zeroCreate<float> cols cols

    for row in 0 .. rows - 1 do
        for i in 0 .. cols - 1 do
            let ai = a[row, i]

            if ai <> 0.0 then
                for j in i .. cols - 1 do
                    gram[i, j] <- gram[i, j] + ai * a[row, j]

    for i in 0 .. cols - 1 do
        for j in i + 1 .. cols - 1 do
            gram[j, i] <- gram[i, j]

    gram

let private symmetricEigenvalues tolerance (a: float[,]) =
    let n = a.GetLength(0)
    let m = Array2D.copy a
    let maxIterations = max 100 (20 * n * n)
    let mutable iteration = 0
    let mutable doneRotating = false

    while iteration < maxIterations && not doneRotating do
        let mutable p = 0
        let mutable q = 0
        let mutable largest = 0.0

        for i in 0 .. n - 1 do
            for j in i + 1 .. n - 1 do
                let value = abs m[i, j]
                if value > largest then
                    largest <- value
                    p <- i
                    q <- j

        if largest <= tolerance then
            doneRotating <- true
        else
            let app = m[p, p]
            let aqq = m[q, q]
            let apq = m[p, q]
            let tau = (aqq - app) / (2.0 * apq)
            let sign = if tau >= 0.0 then 1.0 else -1.0
            let t = sign / (abs tau + sqrt (1.0 + tau * tau))
            let c = 1.0 / sqrt (1.0 + t * t)
            let s = t * c

            for k in 0 .. n - 1 do
                if k <> p && k <> q then
                    let mkp = m[k, p]
                    let mkq = m[k, q]
                    m[k, p] <- c * mkp - s * mkq
                    m[p, k] <- m[k, p]
                    m[k, q] <- s * mkp + c * mkq
                    m[q, k] <- m[k, q]

            m[p, p] <- c * c * app - 2.0 * s * c * apq + s * s * aqq
            m[q, q] <- s * s * app + 2.0 * s * c * apq + c * c * aqq
            m[p, q] <- 0.0
            m[q, p] <- 0.0

        iteration <- iteration + 1

    [ for i in 0 .. n - 1 -> m[i, i] ]
    |> List.sortDescending

let private conditionNumbers tolerance (a: float[,]) =
    let cols = a.GetLength(1)

    if cols = 0 then
        0, Double.PositiveInfinity, Double.PositiveInfinity
    else
        let eigenvalues = gramMatrix a |> symmetricEigenvalues tolerance
        let maxEigenvalue = eigenvalues |> List.max
        let eigenTolerance = max tolerance (float (max (a.GetLength(0)) cols) * Double.Epsilon * maxEigenvalue)
        let positive = eigenvalues |> List.filter (fun value -> value > eigenTolerance)
        let rank = positive.Length

        if rank = 0 then
            rank, Double.PositiveInfinity, Double.PositiveInfinity
        else
            let effectiveCondition = sqrt (maxEigenvalue / (positive |> List.min))
            let fullCondition =
                if rank = cols then
                    effectiveCondition
                else
                    Double.PositiveInfinity

            rank, fullCondition, effectiveCondition

let private measurementMatrix (rows: AnalysisRow list) (features: string list) (measurements: Measurement list) =
    let rowIndex =
        rows
        |> List.mapi (fun index row -> row.RowId, index)
        |> Map.ofList

    let fullMatrix = matrix rows features
    let usable =
        measurements
        |> List.choose (fun measurement ->
            rowIndex
            |> Map.tryFind measurement.RowId
            |> Option.map (fun index -> index, measurement))

    let a = Array2D.zeroCreate<float> usable.Length features.Length
    let y = Array.zeroCreate<float> usable.Length

    usable
    |> List.iteri (fun fitRow (sourceRow, measurement) ->
        y[fitRow] <- measurement.Value

        for col in 0 .. features.Length - 1 do
            a[fitRow, col] <- fullMatrix[sourceRow, col])

    usable, a, y

let private emptyInterceptValue (rows: AnalysisRow list) (usable: (int * Measurement) list) =
    usable
    |> List.choose (fun (rowIndex, measurement) ->
        let row = rows[rowIndex]

        if row.FeatureValues.Count = 1 && row.FeatureValues.ContainsKey "intercept" then
            Some measurement.Value
        else
            None)
    |> function
        | [] -> None
        | values -> Some(List.average values)

let private fixedCoefficientsForFit (rows: AnalysisRow list) (features: string list) (usable: (int * Measurement) list) =
    seq {
        if features |> List.contains "Ignore" then
            yield "Ignore", 0.0

        match emptyInterceptValue rows usable with
        | Some value when features |> List.contains "intercept" ->
            yield "intercept", value
        | _ -> ()
    }
    |> Map.ofSeq

let private solveWithFixedCoefficients ridge rows features usable (a: float[,]) (y: float[]) =
    // Calibration anchors: empty measures common startup/shutdown, while Ignore
    // is defined to have zero stage cost.
    let fixedCoefficients = fixedCoefficientsForFit rows features usable
    let featureIndexes = features |> List.mapi (fun index feature -> feature, index) |> Map.ofList
    let solvedFeatures = features |> List.filter (fun feature -> not (fixedCoefficients.ContainsKey feature))
    let solvedFeatureIndexes = solvedFeatures |> List.map (fun feature -> featureIndexes[feature])
    let adjustedY = Array.copy y

    for row in 0 .. a.GetLength(0) - 1 do
        let fixedContribution =
            fixedCoefficients
            |> Seq.sumBy (fun (KeyValue(feature, coefficient)) ->
                let col = featureIndexes[feature]
                a[row, col] * coefficient)

        adjustedY[row] <- adjustedY[row] - fixedContribution

    let solvedMatrix = Array2D.zeroCreate<float> y.Length solvedFeatures.Length

    solvedFeatureIndexes
    |> List.iteri (fun targetCol sourceCol ->
        for row in 0 .. y.Length - 1 do
            solvedMatrix[row, targetCol] <- a[row, sourceCol])

    let solvedCoefficients =
        if solvedFeatures.IsEmpty then
            [||]
        else
            TinyLinAlg.Dense.nonNegativeLeastSquares ridge 20000 1e-10 solvedMatrix adjustedY

    let coefficientByFeature =
        seq {
            yield! fixedCoefficients |> Seq.map (fun (KeyValue(feature, coefficient)) -> feature, coefficient)
            yield! solvedFeatures |> List.mapi (fun index feature -> feature, solvedCoefficients[index])
        }
        |> Map.ofSeq

    features
    |> List.map (fun feature -> coefficientByFeature |> Map.tryFind feature |> Option.defaultValue 0.0)
    |> List.toArray

let private fitMeasurements (ridge: float) (rows: AnalysisRow list) (features: string list) (measurements: Measurement list) =
    measurements
    |> List.groupBy _.Name
    |> List.collect (fun (measurementName, values) ->
        let usable, a, y = measurementMatrix rows features values

        if usable.Length = 0 then
            []
        else
            let coefficients = solveWithFixedCoefficients ridge rows features usable a y
            let predicted = TinyLinAlg.Dense.predict a coefficients
            let mean = y |> Array.average
            let residuals = Array.map2 (fun actual fit -> actual - fit) y predicted
            let sse = residuals |> Array.sumBy (fun value -> value * value)
            let sst = y |> Array.sumBy (fun value -> let d = value - mean in d * d)
            let rmse = sqrt (sse / float y.Length)
            let r2 = if sst > 0.0 then 1.0 - sse / sst else 1.0
            let summary = measurementName, y.Length, features.Length, rmse, r2

            [ yield! features |> List.mapi (fun index feature -> Choice1Of2(summary, feature, coefficients[index]))
              yield! usable |> List.mapi (fun fitRow (_, measurement) ->
                  Choice2Of2(summary, measurement, predicted[fitRow], residuals[fitRow])) ])

let private invariant (value: float) =
    value.ToString("G17", CultureInfo.InvariantCulture)

type private FeatureMetadata =
    { Operator: string
      Parameters: Map<string, string> }

type private EvidenceContext =
    { PixelType: string option
      Width: uint64 option
      Height: uint64 option
      Depth: uint64 option }

let private parseFeatureMetadata (feature: string) : FeatureMetadata =
    let parts = feature.Split(':', StringSplitOptions.RemoveEmptyEntries)

    let operator =
        if parts.Length = 0 then feature else parts[0]

    let parameters =
        parts
        |> Seq.skip 1
        |> Seq.choose (fun part ->
            let index = part.IndexOf('=')
            if index <= 0 then
                None
            else
                Some(part.Substring(0, index), part.Substring(index + 1)))
        |> Map.ofSeq

    { Operator = operator
      Parameters = parameters }

let private tryParseUInt64Text (value: string) =
    let m = Regex.Match(value, @"^\s*(\d+)")

    if m.Success then
        match UInt64.TryParse(m.Groups[1].Value, NumberStyles.Integer, CultureInfo.InvariantCulture) with
        | true, parsed -> Some parsed
        | _ -> None
    else
        None

let private tryParseFloatText (value: string) =
    let m = Regex.Match(value, @"^\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

    if m.Success then
        match Double.TryParse(m.Value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, parsed -> Some parsed
        | _ -> None
    else
        None

let private parameterUInt64 key (metadata: FeatureMetadata) =
    metadata.Parameters
    |> Map.tryFind key
    |> Option.bind tryParseUInt64Text

let private parameterFloat key (metadata: FeatureMetadata) =
    metadata.Parameters
    |> Map.tryFind key
    |> Option.bind tryParseFloatText

let private normalizePixelType (value: string) =
    match value with
    | null -> None
    | text ->
        match text.Trim().ToLowerInvariant() with
        | "" -> None
        | "byte" | "uint8" -> Some "UInt8"
        | "sbyte" | "int8" -> Some "Int8"
        | "uint16" -> Some "UInt16"
        | "int16" -> Some "Int16"
        | "uint32" -> Some "UInt32"
        | "int32" -> Some "Int32"
        | "uint64" -> Some "UInt64"
        | "int64" -> Some "Int64"
        | "single" | "float32" -> Some "Float32"
        | "double" | "float" | "float64" -> Some "Float64"
        | "complex" -> Some "Complex"
        | other -> Some other

let private pixelTypeBytes (pixelType: string) =
    match pixelType with
    | "UInt8" | "Int8" -> Some 1UL
    | "UInt16" | "Int16" -> Some 2UL
    | "UInt32" | "Int32" | "Float32" -> Some 4UL
    | "UInt64" | "Int64" | "Float64" -> Some 8UL
    | "Complex" -> Some 16UL
    | _ -> None

let private featureSize metadata =
    match parameterUInt64 "width" metadata, parameterUInt64 "height" metadata, parameterUInt64 "depth" metadata with
    | Some width, Some height, Some depth -> Some(width, height, depth)
    | _ -> None

let private rowIdSize (rowId: string) =
    let sizePrefix = Regex.Match(rowId, @"(?:^|/)size_(\d+)x(\d+)x(\d+)(?:/|$)")

    let toSize (m: Match) =
        match tryParseUInt64Text m.Groups[1].Value, tryParseUInt64Text m.Groups[2].Value, tryParseUInt64Text m.Groups[3].Value with
        | Some width, Some height, Some depth -> Some(width, height, depth)
        | _ -> None

    if sizePrefix.Success then
        toSize sizePrefix
    else
        let matches = Regex.Matches(rowId, @"(\d+)x(\d+)x(\d+)")
        if matches.Count = 0 then
            None
        else
            toSize (matches.Item(matches.Count - 1))

let private inferEvidenceContext (row: AnalysisRow) =
    let metadata =
        row.FeatureValues
        |> Map.toList
        |> List.map (fst >> parseFeatureMetadata)

    let size =
        rowIdSize row.RowId
        |> Option.orElseWith (fun () -> metadata |> List.tryPick featureSize)

    let pixelType =
        metadata
        |> List.tryPick (fun metadata ->
            metadata.Parameters
            |> Map.tryFind "type"
            |> Option.bind normalizePixelType)

    match size with
    | Some(width, height, depth) ->
        { PixelType = pixelType
          Width = Some width
          Height = Some height
          Depth = Some depth }
    | None ->
        { PixelType = pixelType
          Width = None
          Height = None
          Depth = None }

let private evidenceRow (rowContext: EvidenceContext) rowId measurementName measurementValue sourcePath feature featureValue : Fitting.EvidenceRow =
    let metadata = parseFeatureMetadata feature
    let featurePixelType =
        metadata.Parameters
        |> Map.tryFind "type"
        |> Option.bind normalizePixelType

    let featureSize = featureSize metadata

    let width, height, depth =
        match rowContext.Width, rowContext.Height, rowContext.Depth with
        | Some width, Some height, Some depth -> Some width, Some height, Some depth
        | _ ->
            match featureSize with
            | Some(width, height, depth) -> Some width, Some height, Some depth
            | None -> rowContext.Width, rowContext.Height, rowContext.Depth

    let pixelType =
        featurePixelType |> Option.orElse rowContext.PixelType

    let slicePixels =
        match width, height with
        | Some width, Some height -> Some(width * height)
        | _ -> None

    let voxels =
        match slicePixels, depth with
        | Some slicePixels, Some depth -> Some(slicePixels * depth)
        | _ -> None

    let bytesPerPixel = pixelType |> Option.bind pixelTypeBytes

    let sliceBytes =
        match slicePixels, bytesPerPixel with
        | Some slicePixels, Some bytesPerPixel -> Some(slicePixels * bytesPerPixel)
        | _ -> None

    let volumeBytes =
        match voxels, bytesPerPixel with
        | Some voxels, Some bytesPerPixel -> Some(voxels * bytesPerPixel)
        | _ -> None

    { RowId = rowId
      Measurement = measurementName
      Value = measurementValue
      SourcePath = sourcePath
      FeatureKey = feature
      FeatureValue = featureValue
      Operator = metadata.Operator
      PixelType = pixelType
      Width = width
      Height = height
      Depth = depth
      Voxels = voxels
      SlicePixels = slicePixels
      SliceBytes = sliceBytes
      VolumeBytes = volumeBytes
      WindowSize = parameterFloat "windowSize" metadata
      Radius = parameterFloat "radius" metadata }

let private runtimeTermEvidenceRow (measurement: Measurement) (term: RuntimeCostTerm) : Fitting.EvidenceRow option =
    let tag key = term.Tags |> Map.tryFind key
    let operator =
        tag "operator"
        |> Option.orElseWith (fun () ->
            if String.IsNullOrWhiteSpace term.StageName then None else Some term.StageName)

    operator
    |> Option.map (fun operator ->
        let pixelType = tag "pixelType" |> Option.bind normalizePixelType
        let voxels = tag "voxels" |> Option.bind tryParseUInt64Text
        let bytesPerPixel = pixelType |> Option.bind pixelTypeBytes

        let volumeBytes =
            match voxels, bytesPerPixel with
            | Some voxels, Some bytesPerPixel -> Some(voxels * bytesPerPixel)
            | _ -> None

        let featureKey =
            let tags =
                term.Tags
                |> Map.toList
                |> List.sortBy fst
                |> List.map (fun (key, value) -> key + "=" + value)

            match tags with
            | [] -> operator
            | _ -> operator + ":" + String.concat ":" tags

        { RowId = measurement.RowId
          Measurement = measurement.Name
          Value = measurement.Value
          SourcePath = measurement.SourcePath
          FeatureKey = featureKey
          FeatureValue = float term.Multiplicity
          Operator = operator
          PixelType = pixelType
          Width = None
          Height = None
          Depth = None
          Voxels = voxels
          SlicePixels = None
          SliceBytes = None
          VolumeBytes = volumeBytes
          WindowSize = tag "windowSize" |> Option.bind tryParseFloatText
          Radius = tag "radius" |> Option.bind tryParseFloatText })

let private writeOutputs (options: Options) (rows: AnalysisRow list) =
    let features =
        rows
        |> List.collect (fun row -> row.FeatureValues |> Map.toList |> List.map fst)
        |> List.distinct
        |> List.sort

    let measurements =
        if options.DiagnosticsOnly then
            []
        else
            (rows |> List.collect (measurementsForRow options.SamplesRoot))
            @ probingMeasurements options.ProbingPrefixes

    let matrixValues = matrix rows features
    let featureSupport =
        features
        |> List.map (fun feature ->
            let count =
                rows
                |> List.sumBy (fun row -> if row.FeatureValues.ContainsKey feature then 1 else 0)

            feature, count)
        |> Map.ofList

    writeCsv
        (Path.Combine(options.OutputDirectory, "rows.csv"))
        (seq {
            yield [ "rowId"; "source"; "path"; "featureCount" ]
            for row in rows do
                let path =
                    if Path.IsPathFullyQualified row.ItemPath then
                        relativePath options.SamplesRoot row.ItemPath
                    else
                        row.ItemPath

                yield [ row.RowId; row.Source; path; string row.FeatureValues.Count ]
        })

    writeCsv
        (Path.Combine(options.OutputDirectory, "features.csv"))
        (seq {
            yield [ "rowId"; "featureKey"; "value"; "path" ]
            for row in rows do
                for KeyValue(feature, value) in row.FeatureValues do
                    let path =
                        if Path.IsPathFullyQualified row.ItemPath then
                            relativePath options.SamplesRoot row.ItemPath
                        else
                            row.ItemPath

                    yield [ row.RowId; feature; invariant value; path ]
        })

    writeCsv
        (Path.Combine(options.OutputDirectory, "matrix.csv"))
        (seq {
            yield "rowId" :: features
            for rowIndex in 0 .. rows.Length - 1 do
                yield
                    rows[rowIndex].RowId
                    :: [ for col in 0 .. features.Length - 1 ->
                           if matrixValues[rowIndex, col] = 0.0 then "0" else "1" ]
        })

    writeCsv
        (Path.Combine(options.OutputDirectory, "featureDiagnostics.csv"))
        (seq {
            yield [ "featureKey"; "supportCount"; "supportRatio" ]
            for feature in features do
                let count = featureSupport[feature]
                yield
                    [ feature
                      string count
                      invariant (float count / float rows.Length) ]
        })

    writeCsv
        (Path.Combine(options.OutputDirectory, "vectors.csv"))
        (seq {
            yield [ "rowId"; "measurement"; "value"; "log" ]
            for measurement in measurements do
                yield
                    [ measurement.RowId
                      measurement.Name
                      invariant measurement.Value
                      if Path.IsPathFullyQualified measurement.SourcePath then
                          relativePath options.SamplesRoot measurement.SourcePath
                      else
                          measurement.SourcePath ]
        })

    let costEvidencePath = Path.Combine(options.OutputDirectory, "costEvidence.csv")

    Fitting.writeEvidenceCsv
        costEvidencePath
        (seq {
            let rowsById =
                rows
                |> List.map (fun row -> row.RowId, row)
                |> Map.ofList

            for measurement in measurements do
                match rowsById |> Map.tryFind measurement.RowId with
                | None -> ()
                | Some row ->
                    let sourcePath =
                        if Path.IsPathFullyQualified measurement.SourcePath then
                            relativePath options.SamplesRoot measurement.SourcePath
                        else
                            measurement.SourcePath

                    match measurement.RuntimeCostTerms with
                    | _ :: _ ->
                        let measurement = { measurement with SourcePath = sourcePath }

                        for term in measurement.RuntimeCostTerms do
                            match runtimeTermEvidenceRow measurement term with
                            | Some row -> yield row
                            | None -> ()
                    | [] ->
                        let rowContext = inferEvidenceContext row

                        for KeyValue(feature, value) in row.FeatureValues do
                            yield evidenceRow rowContext measurement.RowId measurement.Name measurement.Value sourcePath feature value
        })

    if not options.DiagnosticsOnly then
        Fitting.fitOperatorTermsFromCsv costEvidencePath options.Ridge options.MinSupport options.OutputDirectory options.ModelOutputPath
        |> ignore

    writeCsv
        (Path.Combine(options.OutputDirectory, "diagnostics.csv"))
        (seq {
            yield [ "measurement"; "rowCount"; "columnCount"; "rank"; "ridge" ]
            yield
                [ "allGraphs"
                  string rows.Length
                  string features.Length
                  string (matrixRank 1e-10 matrixValues)
                  invariant options.Ridge ]

            for measurementName, values in measurements |> List.groupBy _.Name do
                let _, a, _ = measurementMatrix rows features values

                yield
                    [ measurementName
                      string (a.GetLength(0))
                      string (a.GetLength(1))
                      string (matrixRank 1e-10 a)
                      invariant options.Ridge ]
        })

    let subsetDefinitions: (string * (AnalysisRow -> bool)) list =
        [ "generated-all", fun row -> isGeneratedRow row.RowId
          "generated-bio-filter", fun row -> generatedRowContains "/bio-filter" row.RowId
          "generated-bio-grayscale", fun row -> generatedRowContains "/bio-grayscale" row.RowId
          "generated-bio-threshold", fun row -> generatedRowContains "/bio-threshold" row.RowId
          "generated-bio-threshold+grayscale",
          fun row ->
              generatedRowContains "/bio-threshold" row.RowId
              || generatedRowContains "/bio-grayscale" row.RowId
          "generated-bio-filter+projection",
          fun row ->
              generatedRowContains "/bio-filter" row.RowId
              || generatedRowContains "/bio-projection" row.RowId
          "generated-boilerplate",
          fun row ->
              generatedRowContains "/zero-" row.RowId
              || generatedRowContains "/read-" row.RowId ]

    writeCsv
        (Path.Combine(options.OutputDirectory, "subsetDiagnostics.csv"))
        (seq {
            yield
                [ "subset"
                  "rowCount"
                  "activeColumnCount"
                  "rank"
                  "rankRatio"
                  "fullColumnConditionNumber"
                  "effectiveConditionNumber"
                  "density" ]

            for name, predicate in subsetDefinitions do
                let rowIndexes =
                    rows
                    |> List.mapi (fun index row -> index, row)
                    |> List.choose (fun (index, row) -> if predicate row then Some index else None)

                if not rowIndexes.IsEmpty then
                    let activeColumns = activeColumnIndexes rowIndexes matrixValues

                    if not activeColumns.IsEmpty then
                        let subsetMatrix = subMatrix rowIndexes activeColumns matrixValues
                        let rank, fullCondition, effectiveCondition = conditionNumbers 1e-10 subsetMatrix
                        let nonzero =
                            seq {
                                for row in 0 .. subsetMatrix.GetLength(0) - 1 do
                                    for col in 0 .. subsetMatrix.GetLength(1) - 1 do
                                        subsetMatrix[row, col]
                            }
                            |> Seq.sumBy (fun value -> if value <> 0.0 then 1 else 0)

                        let density =
                            float nonzero
                            / float (subsetMatrix.GetLength(0) * subsetMatrix.GetLength(1))

                        yield
                            [ name
                              string rowIndexes.Length
                              string activeColumns.Length
                              string rank
                              invariant (float rank / float activeColumns.Length)
                              invariant fullCondition
                              invariant effectiveCondition
                              invariant density ]
        })

    let fitRows = fitMeasurements options.Ridge rows features measurements

    writeCsv
        (Path.Combine(options.OutputDirectory, "coefficients.csv"))
        (seq {
            yield [ "measurement"; "featureKey"; "coefficient"; "supportCount"; "rowCount"; "columnCount"; "rmse"; "r2"; "ridge"; "solver" ]
            for fit in fitRows do
                match fit with
                | Choice1Of2((measurement, rowCount, columnCount, rmse, r2), feature, coefficient) ->
                    yield
                        [ measurement
                          feature
                          invariant coefficient
                          string featureSupport[feature]
                          string rowCount
                          string columnCount
                          invariant rmse
                          invariant r2
                          invariant options.Ridge
                          "nonNegativeLeastSquares" ]
                | Choice2Of2 _ -> ()
        })

    writeCsv
        (Path.Combine(options.OutputDirectory, "predictions.csv"))
        (seq {
            yield [ "rowId"; "measurement"; "actual"; "predicted"; "residual"; "rowCount"; "columnCount"; "rmse"; "r2"; "ridge"; "solver" ]
            for fit in fitRows do
                match fit with
                | Choice1Of2 _ -> ()
                | Choice2Of2((measurement, rowCount, columnCount, rmse, r2), observed, predicted, residual) ->
                    yield
                        [ observed.RowId
                          measurement
                          invariant observed.Value
                          invariant predicted
                          invariant residual
                          string rowCount
                          string columnCount
                          invariant rmse
                          invariant r2
                          invariant options.Ridge
                          "nonNegativeLeastSquares" ]
        })

    Fitting.writeCoefficientModel
        (Path.Combine(options.OutputDirectory, "stackprocessing.cost.json"))
        "StackProcessing fitted cost model"
        [ "Coefficients are fitted from Probe and RunSamples evidence."
          "Empty graph defines the intercept row."
          "Ignore sinks are treated as zero-cost sinks by the analysis constraints." ]
        (seq {
            for fit in fitRows do
                match fit with
                | Choice1Of2((measurement, rowCount, _columnCount, rmse, r2), feature, coefficient) ->
                    let costCoefficient: CostCoefficient =
                        { Measurement = measurement
                          FeatureKey = feature
                          Coefficient = coefficient
                          SupportCount = featureSupport[feature]
                          RowCount = rowCount
                          Rmse = rmse
                          R2 = r2
                          Solver = "nonNegativeLeastSquares" }

                    yield costCoefficient
                | Choice2Of2 _ -> ()
        })

    printfn "wrote %d rows, %d features, %d measurements to %s" rows.Length features.Length measurements.Length options.OutputDirectory

let main argv =
    match parseArgs (defaultOptions ()) (Array.toList argv) with
    | Error exitCode -> exitCode
    | Ok options ->
        let rows =
            discoverGraphs options.SamplesRoot options.IncludeSamples options.ExtraJsonRoots
            @ discoverProbingRows options.ProbingPrefixes

        writeOutputs options rows
        0
