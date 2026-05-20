module ProbeLocalUpdate

open System
open System.Globalization
open System.IO
open System.Text
open System.Text.RegularExpressions
open StackProcessingCost

type Options =
    { SamplesRoot: string
      AnalysisDirectory: string
      ProbeJsonRoot: string
      InputDirectory: string
      ModelInputPath: string option
      ModelOutputPath: string
      DiscrepancyCsv: string option
      Operators: string list
      Sizes: uint list
      Depth: uint option
      NoisyType: string
      Radius: uint option
      WindowSize: uint option
      Sigma: float option
      Repeat: int
      Jobs: int
      MaxOperators: int
      MinSupport: int
      RunProbes: bool
      FitModel: bool }

type Suspect =
    { Operator: string
      Count: int
      Parameters: Map<string, string> }

let private usage () =
    printfn "Usage: dotnet run --project src/StackProcessing.Probe -- local-update [options]"
    printfn ""
    printfn "Emits targeted probe graphs around operators implicated by current discrepancy evidence,"
    printfn "optionally runs them, fits a local model, and overlays it on the existing model."
    printfn ""
    printfn "Options:"
    printfn "  --operators A,B        Explicit operator/function ids to probe."
    printfn "  --sizes 64,128,256     Image sizes to probe. Defaults to 128."
    printfn "  --depth N              Probe depth. Defaults to each size, e.g. --size 64 --depth 32."
    printfn "  --radius N             Override radius for local radius-dependent templates."
    printfn "  --window-size N        Override windowSize for local window-dependent templates."
    printfn "  --sigma VALUE          Override sigma for local Gaussian templates."
    printfn "  --repeat N             Probe repeats. Defaults to 3."
    printfn "  -j, --jobs N           Parallel probe runs. Defaults to 1."
    printfn "  --max-operators N      Maximum inferred suspects. Defaults to 8."
    printfn "  --min-support N        Minimum support for local fitted terms. Defaults to 1."
    printfn "  --cost-model PATH      Existing fitted model to overlay. Defaults to models/fitted/stackprocessing.operator-cost.json."
    printfn "  --model-output PATH    Updated model path. Defaults to models/local/stackprocessing.operator-cost.json."
    printfn "  --discrepancies PATH   Runtime discrepancy CSV. Defaults to tmp/costDiscrepancies.csv when present."
    printfn "  --no-run-probes        Emit graphs and stop before running them."
    printfn "  --no-fit               Do not fit or write the updated model."

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

let private timestamp () =
    DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture)

let private defaultSamplesRoot () =
    let cwd = Directory.GetCurrentDirectory()
    let samples = Path.Combine(cwd, "samples")
    if Directory.Exists samples then samples else Path.GetFullPath "samples"

let private defaultOptions () =
    let samplesRoot = defaultSamplesRoot ()
    let repositoryRoot = repositoryRootFromSamplesRoot samplesRoot
    let tmpRoot = probeOutputRoot samplesRoot

    { SamplesRoot = samplesRoot
      AnalysisDirectory = Path.Combine(tmpRoot, "analysis-local")
      ProbeJsonRoot = Path.Combine(tmpRoot, "probingGraphs")
      InputDirectory = Path.Combine(tmpRoot, "probeInputs")
      ModelInputPath = Some(Path.Combine(repositoryRoot, "models", "fitted", "stackprocessing.operator-cost.json"))
      ModelOutputPath = Path.Combine(repositoryRoot, "models", "local", "stackprocessing.operator-cost.json")
      DiscrepancyCsv =
        let path = Path.Combine(tmpRoot, "costDiscrepancies.csv")
        if File.Exists path then Some path else None
      Operators = []
      Sizes = [ 128u ]
      Depth = None
      NoisyType = "Float32"
      Radius = None
      WindowSize = None
      Sigma = None
      Repeat = 3
      Jobs = 1
      MaxOperators = 8
      MinSupport = 1
      RunProbes = true
      FitModel = true }

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

let private readCsvMaps path =
    if String.IsNullOrWhiteSpace path || not (File.Exists path) then
        []
    else
        match File.ReadLines path |> Seq.toList with
        | [] -> []
        | headerLine :: rows ->
            let header = parseCsvLine headerLine
            rows
            |> List.choose (fun line ->
                if String.IsNullOrWhiteSpace line then
                    None
                else
                    let values = parseCsvLine line
                    header
                    |> List.mapi (fun index name ->
                        let value = if index < values.Length then values[index] else ""
                        name, value)
                    |> Map.ofList
                    |> Some)

let private field name (row: Map<string, string>) =
    row |> Map.tryFind name |> Option.defaultValue ""

let private normalizeToken (value: string) =
    Regex.Replace(value, @"[^A-Za-z0-9]+", "").ToLowerInvariant()

let private canonicalOperatorName (value: string) =
    let trimmed = value.Trim()
    if String.IsNullOrWhiteSpace trimmed then
        ""
    else
        let firstToken =
            trimmed.Split([| ' '; '\t'; '\r'; '\n' |], StringSplitOptions.RemoveEmptyEntries)
            |> Array.tryHead
            |> Option.defaultValue trimmed

        firstToken.Trim('"', '\'')

let private splitCsvList (value: string) =
    value.Split([| ','; ';' |], StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
    |> Array.toList

let private parseSizes value =
    splitCsvList value
    |> List.choose (fun text ->
        match UInt32.TryParse text with
        | true, size when size > 0u -> Some size
        | _ -> None)

let rec private parseArgs options args =
    match args with
    | [] -> Ok options
    | ("-h" | "--help") :: _ ->
        usage ()
        Error 0
    | "--samples-root" :: value :: rest ->
        parseArgs { options with SamplesRoot = Path.GetFullPath value } rest
    | "--analysis-dir" :: value :: rest
    | "--analysis-output" :: value :: rest ->
        parseArgs { options with AnalysisDirectory = Path.GetFullPath value } rest
    | "--probe-json-root" :: value :: rest
    | "--emit-json" :: value :: rest ->
        parseArgs { options with ProbeJsonRoot = Path.GetFullPath value } rest
    | "--input-dir" :: value :: rest ->
        parseArgs { options with InputDirectory = Path.GetFullPath value } rest
    | "--cost-model" :: value :: rest
    | "--model-input" :: value :: rest ->
        parseArgs { options with ModelInputPath = Some(Path.GetFullPath value) } rest
    | "--no-base-model" :: rest ->
        parseArgs { options with ModelInputPath = None } rest
    | "--model-output" :: value :: rest ->
        parseArgs { options with ModelOutputPath = Path.GetFullPath value } rest
    | "--discrepancies" :: value :: rest
    | "--flags" :: value :: rest ->
        parseArgs { options with DiscrepancyCsv = Some(Path.GetFullPath value) } rest
    | "--operators" :: value :: rest
    | "--operator" :: value :: rest ->
        parseArgs { options with Operators = options.Operators @ splitCsvList value } rest
    | "--sizes" :: value :: rest ->
        match parseSizes value with
        | [] ->
            eprintfn "local-update: --sizes expects a comma-separated list of positive integers"
            Error 2
        | sizes -> parseArgs { options with Sizes = sizes } rest
    | "--size" :: value :: rest ->
        match UInt32.TryParse value with
        | true, size when size > 0u -> parseArgs { options with Sizes = [ size ] } rest
        | _ ->
            eprintfn "local-update: --size expects a positive integer"
            Error 2
    | "--noisy-type" :: value :: rest ->
        parseArgs { options with NoisyType = value } rest
    | "--depth" :: value :: rest ->
        match UInt32.TryParse value with
        | true, depth when depth > 0u -> parseArgs { options with Depth = Some depth } rest
        | _ ->
            eprintfn "local-update: --depth expects a positive integer"
            Error 2
    | "--radius" :: value :: rest ->
        match UInt32.TryParse value with
        | true, radius when radius > 0u -> parseArgs { options with Radius = Some radius } rest
        | _ ->
            eprintfn "local-update: --radius expects a positive integer"
            Error 2
    | "--window-size" :: value :: rest
    | "--window" :: value :: rest ->
        match UInt32.TryParse value with
        | true, windowSize when windowSize > 0u -> parseArgs { options with WindowSize = Some windowSize } rest
        | _ ->
            eprintfn "local-update: --window-size expects a positive integer"
            Error 2
    | "--sigma" :: value :: rest ->
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, sigma when sigma > 0.0 -> parseArgs { options with Sigma = Some sigma } rest
        | _ ->
            eprintfn "local-update: --sigma expects a positive floating-point value"
            Error 2
    | "--repeat" :: value :: rest
    | "--repeats" :: value :: rest ->
        match Int32.TryParse value with
        | true, repeat when repeat > 0 -> parseArgs { options with Repeat = repeat } rest
        | _ ->
            eprintfn "local-update: --repeat expects a positive integer"
            Error 2
    | ("-j" | "--jobs") :: value :: rest ->
        match Int32.TryParse value with
        | true, jobs when jobs > 0 -> parseArgs { options with Jobs = jobs } rest
        | _ ->
            eprintfn "local-update: -j/--jobs expects a positive integer"
            Error 2
    | "--max-operators" :: value :: rest ->
        match Int32.TryParse value with
        | true, count when count > 0 -> parseArgs { options with MaxOperators = count } rest
        | _ ->
            eprintfn "local-update: --max-operators expects a positive integer"
            Error 2
    | "--min-support" :: value :: rest ->
        match Int32.TryParse value with
        | true, count when count > 0 -> parseArgs { options with MinSupport = count } rest
        | _ ->
            eprintfn "local-update: --min-support expects a positive integer"
            Error 2
    | "--no-run-probes" :: rest ->
        parseArgs { options with RunProbes = false } rest
    | "--run-probes" :: rest ->
        parseArgs { options with RunProbes = true } rest
    | "--no-fit" :: rest ->
        parseArgs { options with FitModel = false } rest
    | "--fit" :: rest ->
        parseArgs { options with FitModel = true } rest
    | option :: _ ->
        eprintfn "local-update: unknown option %s" option
        usage ()
        Error 2

let private parseTaggedContext (context: string) =
    context.Split([| ';' |], StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
    |> Array.choose (fun part ->
        let pieces = part.Split([| '=' |], 2, StringSplitOptions.TrimEntries)
        if pieces.Length = 2 && not (String.IsNullOrWhiteSpace pieces[0]) then
            Some(pieces[0], pieces[1])
        else
            None)
    |> Map.ofArray

let private suspectObservationsFromRuntimeFlags (path: string option) =
    match path with
    | None -> []
    | Some path ->
        readCsvMaps path
        |> List.collect (fun row ->
            let contextObservations =
                field "costContexts" row
                |> fun value -> value.Split([| '|' |], StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
                |> Array.toList
                |> List.choose (fun context ->
                    let tags = parseTaggedContext context
                    match tags |> Map.tryFind "operator" with
                    | Some operator when not (String.IsNullOrWhiteSpace operator) ->
                        let parameters =
                            tags
                            |> Map.remove "operator"
                            |> Map.remove "pixelType"

                        Some(canonicalOperatorName operator, parameters)
                    | _ -> None)

            if contextObservations.IsEmpty then
                field "graphNodes" row
                |> fun value -> value.Split([| '|' |], StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
                |> Array.toList
                |> List.map (fun operator -> canonicalOperatorName operator, Map.empty)
            else
                contextObservations)

let private suspectObservationsFromAnalysis analysisDirectory =
    let discrepanciesPath = Path.Combine(analysisDirectory, "operatorModelDiscrepancies.csv")
    let evidencePath = Path.Combine(analysisDirectory, "costEvidence.csv")
    let flaggedRowIds =
        readCsvMaps discrepanciesPath
        |> List.choose (fun row ->
            match field "flagged" row with
            | value when String.Equals(value, "true", StringComparison.OrdinalIgnoreCase) -> Some(field "rowId" row)
            | _ -> None)
        |> Set.ofList

    if Set.isEmpty flaggedRowIds then
        []
    else
        readCsvMaps evidencePath
        |> List.choose (fun row ->
            if flaggedRowIds.Contains(field "rowId" row) then
                let operator = field "operator" row
                if String.IsNullOrWhiteSpace operator then
                    None
                else
                    let parameters =
                        [ "windowSize", field "windowSize" row
                          "radius", field "radius" row ]
                        |> List.choose (fun (key, value) ->
                            if String.IsNullOrWhiteSpace value then None else Some(key, value))
                        |> Map.ofList

                    Some(canonicalOperatorName operator, parameters)
            else
                None)

let private nonSuspectOperators =
    set
        ([ "intercept"
           "Ignore"
           "Read"
           "Write"
           "Zero"
           "Scalar"
           "Cast"
           "Print" ]
         |> List.map normalizeToken)

let private inferSuspects options =
    let observations =
        if not options.Operators.IsEmpty then
            options.Operators |> List.map (fun operator -> canonicalOperatorName operator, Map.empty)
        else
            suspectObservationsFromRuntimeFlags options.DiscrepancyCsv
            @ suspectObservationsFromAnalysis options.AnalysisDirectory

    observations
    |> List.filter (fun (operator, _) -> not (String.IsNullOrWhiteSpace operator))
    |> List.filter (fun (operator, _) -> not (nonSuspectOperators.Contains(normalizeToken operator)))
    |> List.groupBy fst
    |> List.map (fun (operator, rows) ->
        let parameters =
            rows
            |> List.collect (snd >> Map.toList)
            |> List.groupBy fst
            |> List.choose (fun (key, values) ->
                values
                |> List.map snd
                |> List.filter (String.IsNullOrWhiteSpace >> not)
                |> List.countBy id
                |> List.sortByDescending snd
                |> List.tryHead
                |> Option.map (fun (value, _) -> key, value))
            |> Map.ofList

        { Operator = operator
          Count = rows.Length
          Parameters = parameters })
    |> List.sortByDescending _.Count
    |> List.truncate options.MaxOperators

let private templateTokens (template: ProbeProbing.GraphTemplate) =
    seq {
        yield template.Name
        yield template.Description
        for feature in template.Features do
            yield feature
        for node in template.Graph.Nodes do
            yield node.FunctionId
            for parameter in node.Parameters do
                yield parameter.Value
    }
    |> Seq.map normalizeToken
    |> Seq.toList

let private templateMatches suspects (template: ProbeProbing.GraphTemplate) =
    let tokens = templateTokens template
    suspects
    |> List.exists (fun suspect ->
        let suspectToken = normalizeToken suspect
        tokens |> List.exists (fun token -> token = suspectToken || token.Contains(suspectToken, StringComparison.Ordinal)))

let private uniqueTemplates (templates: ProbeProbing.GraphTemplate seq) =
    templates
    |> Seq.groupBy _.Name
    |> Seq.map (fun (_, group) -> group |> Seq.head)
    |> Seq.sortBy _.Name
    |> Seq.toArray

let private parameterValue key (parameters: Map<string, string>) =
    parameters
    |> Map.tryFind key
    |> Option.filter (String.IsNullOrWhiteSpace >> not)

let private optionOrInferred explicitValue key parameters =
    explicitValue |> Option.map string |> Option.orElseWith (fun () -> parameterValue key parameters)

let private floatOptionOrInferred (explicitValue: float option) key parameters =
    explicitValue
    |> Option.map (fun value -> Convert.ToString(value, CultureInfo.InvariantCulture))
    |> Option.orElseWith (fun () -> parameterValue key parameters)

let private parameterOverridesForFunction options parameters functionId =
    let radius = optionOrInferred options.Radius "radius" parameters
    let windowSize = optionOrInferred options.WindowSize "windowSize" parameters

    match functionId with
    | "SmoothWMedian" ->
        [ radius |> Option.map (fun value -> "radius", value)
          windowSize |> Option.map (fun value -> "windowSize", value) ]
        |> List.choose id
    | "BinaryMedian"
    | "GrayscaleErode"
    | "GrayscaleDilate"
    | "GrayscaleOpening"
    | "GrayscaleClosing"
    | "WhiteTopHat"
    | "BlackTopHat"
    | "MorphologicalGradient" ->
        [ radius |> Option.map (fun value -> "radius", value)
          windowSize |> Option.map (fun value -> "windowSize", value) ]
        |> List.choose id
    | "SmoothWGauss" ->
        [ floatOptionOrInferred options.Sigma "sigma" parameters |> Option.map (fun value -> "sigma", value)
          windowSize |> Option.map (fun value -> "windowSize", value) ]
        |> List.choose id
    | "GradientMagnitude"
    | "SobelEdge"
    | "Laplacian"
    | "BinaryContour" ->
        [ windowSize |> Option.map (fun value -> "windowSize", value) ]
        |> List.choose id
    | _ -> []

let private parametersForFunction suspects functionId =
    suspects
    |> List.tryFind (fun suspect -> String.Equals(normalizeToken suspect.Operator, normalizeToken functionId, StringComparison.Ordinal))
    |> Option.map _.Parameters
    |> Option.defaultValue Map.empty

let private applyLocalParameterOverrides options suspects (template: ProbeProbing.GraphTemplate) =
    let mutable changed = false
    let applied = ResizeArray<string * string>()

    let rewriteParameter functionId (parameter: Studio.Graph.SavedParameter) =
        let parameters = parametersForFunction suspects functionId
        parameterOverridesForFunction options parameters functionId
        |> List.tryFind (fun (key, _) -> String.Equals(key, parameter.Key, StringComparison.OrdinalIgnoreCase))
        |> function
            | Some(_, value) when parameter.Value <> value ->
                changed <- true
                applied.Add(parameter.Key, value)
                { parameter with Value = value }
            | _ -> parameter

    let nodes =
        template.Graph.Nodes
        |> Array.map (fun node ->
            let parameters = node.Parameters |> Array.map (rewriteParameter node.FunctionId)
            { node with Parameters = parameters })

    if changed then
        let suffix =
            applied
            |> Seq.distinct
            |> Seq.map (fun (key, value) ->
                let safeValue = value.Replace(".", "p").Replace(",", "p")
                match key with
                | key when String.Equals(key, "radius", StringComparison.OrdinalIgnoreCase) -> "r" + safeValue
                | key when String.Equals(key, "windowSize", StringComparison.OrdinalIgnoreCase) -> "w" + safeValue
                | key when String.Equals(key, "sigma", StringComparison.OrdinalIgnoreCase) -> "s" + safeValue
                | _ -> normalizeToken key + safeValue)
            |> String.concat "-"

        { template with
            Name = if String.IsNullOrWhiteSpace suffix then template.Name else template.Name + "-" + suffix
            Description = template.Description + " Local parameter override."
            Graph = { template.Graph with Nodes = nodes } }
    else
        template

let private sourceLayerTemplates (layers: (string * ProbeProbing.GraphTemplate array) array) =
    layers
    |> Array.tryFind (fun (name, _) -> name.StartsWith("01-", StringComparison.Ordinal))
    |> Option.map snd
    |> Option.defaultValue [||]

let private localTemplatesForSize options suspects probeRunRoot size =
    let depth = options.Depth |> Option.defaultValue size
    let inputDir =
        Path.Combine(options.InputDirectory, $"local_{timestamp()}", $"size_{size}x{size}x{depth}")

    let inputConfig = ProbeProbing.createBottomUpInputsWithDepth size depth options.NoisyType inputDir
    let layers = ProbeProbing.graphTemplateLayersForBottomUp inputConfig
    let sourceTemplates = sourceLayerTemplates layers
    let suspectTemplates =
        layers
        |> Array.collect snd
        |> Array.filter (templateMatches (suspects |> List.map _.Operator))
        |> Array.map (applyLocalParameterOverrides options suspects)

    Array.append sourceTemplates suspectTemplates
    |> uniqueTemplates

let private runSamplesJson options probeRunRoot =
    let args =
        [| yield options.SamplesRoot
           yield "--json"
           yield "--extra-json-root"
           yield probeRunRoot
           yield "--extra-json-only"
           yield "--optimize"
           yield "false"
           yield "--repeat"
           yield string options.Repeat
           yield "-j"
           yield string options.Jobs
           match options.ModelInputPath with
           | Some path when File.Exists path ->
               yield "--cost-model"
               yield path
           | _ -> () |]

    RunSamples.main args

let private runAnalysis options probeRunRoot localAnalysisDir localModelPath =
    let args =
        [| yield "--samples-root"
           yield options.SamplesRoot
           yield "--no-samples"
           yield "--extra-json-root"
           yield probeRunRoot
           yield "--output"
           yield localAnalysisDir
           yield "--model-output"
           yield localModelPath
           yield "--min-support"
           yield string options.MinSupport |]

    ProbeAnalysis.main args

let private mergeModels basePath localPath outputPath =
    let baseModel =
        basePath
        |> Option.filter File.Exists
        |> Option.map Fitting.OperatorCostModel.loadOrDefault
        |> Option.defaultValue Fitting.OperatorCostModel.empty

    let localModel = Fitting.OperatorCostModel.loadOrDefault localPath

    let localKeys =
        localModel.Coefficients
        |> Array.map (fun row -> row.Measurement, row.TermKey)
        |> Set.ofArray

    let merged =
        { baseModel with
            SchemaVersion = max baseModel.SchemaVersion localModel.SchemaVersion
            Name = "StackProcessing locally updated operator cost model"
            CreatedUtc = DateTimeOffset.UtcNow
            Assumptions =
                Array.append
                    baseModel.Assumptions
                    [| $"Local update merged from {localPath}."
                       "Coefficients measured by local probes override matching base measurement/term keys." |]
            Coefficients =
                Array.append
                    (baseModel.Coefficients
                     |> Array.filter (fun row -> not (localKeys.Contains(row.Measurement, row.TermKey))))
                    localModel.Coefficients }

    Fitting.OperatorCostModel.save outputPath merged

let main argv =
    match parseArgs (defaultOptions ()) (Array.toList argv) with
    | Error code -> code
    | Ok options ->
        try
            let suspects = inferSuspects options

            if suspects.IsEmpty then
                eprintfn "local-update: no suspect operators found. Use --operators A,B or run analysis/discrepancy reporting first."
                1
            else
                printfn "Local update suspects: %s" (suspects |> List.map (fun suspect -> $"{suspect.Operator}({suspect.Count})") |> String.concat ", ")

                let runId = "local_" + timestamp()
                let probeRunRoot = Path.Combine(options.ProbeJsonRoot, runId)
                Directory.CreateDirectory probeRunRoot |> ignore

                let templates =
                    options.Sizes
                    |> List.toArray
                    |> Array.collect (localTemplatesForSize options suspects probeRunRoot)
                    |> uniqueTemplates

                let selected =
                    templates
                    |> Array.filter (fun template ->
                        template.Name.Contains("bottomup-00-empty", StringComparison.OrdinalIgnoreCase)
                        || template.Name.Contains("zero-", StringComparison.OrdinalIgnoreCase)
                        || template.Name.Contains("read-", StringComparison.OrdinalIgnoreCase)
                        || templateMatches (suspects |> List.map _.Operator) template)

                if selected.Length = 0 then
                    eprintfn "local-update: no matching probe templates found for %s" (suspects |> List.map _.Operator |> String.concat ", ")
                    1
                else
                    ProbeProbing.writeGraphTemplates probeRunRoot selected

                    let mutable exitCode = 0

                    if options.RunProbes then
                        exitCode <- runSamplesJson options probeRunRoot

                    if exitCode = 0 && options.FitModel then
                        let localAnalysisDir = Path.Combine(options.AnalysisDirectory, runId)
                        let localModelPath = Path.Combine(localAnalysisDir, "stackprocessing.operator-cost.local-only.json")
                        exitCode <- runAnalysis options probeRunRoot localAnalysisDir localModelPath

                        if exitCode = 0 then
                            mergeModels options.ModelInputPath localModelPath options.ModelOutputPath
                            printfn "Wrote locally updated cost model to %s" options.ModelOutputPath

                    if exitCode = 0 then
                        printfn "Local update probe graphs: %s" probeRunRoot

                    exitCode
        with ex ->
            eprintfn "local-update failed: %s" ex.Message
            1
