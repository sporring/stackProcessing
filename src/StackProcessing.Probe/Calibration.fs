module ProbeCalibration

open System
open System.Globalization
open System.IO
open System.Text

type Options =
    { SamplesRoot: string
      AnalysisDirectory: string
      ProbeJsonRoot: string
      Iterations: int
      MinSupport: int
      MaxCondition: float
      MinRankRatio: float
      MinR2: float
      ResidualMultiple: float
      MaxUnknowns: int
      MaxGraphs: int
      RunProbes: bool
      ProbeRepeats: int
      Jobs: int
      SkipAnalysis: bool }

type RowInfo =
    { RowId: string
      Source: string
      Path: string }

type FeatureDiagnostic =
    { FeatureKey: string
      SupportCount: int
      SupportRatio: float }

type SubsetDiagnostic =
    { Name: string
      RowCount: int
      ActiveColumnCount: int
      Rank: int
      RankRatio: float
      EffectiveConditionNumber: float }

type Coefficient =
    { Measurement: string
      FeatureKey: string
      Coefficient: float
      SupportCount: int
      RowCount: int
      ColumnCount: int
      Rmse: float
      R2: float }

type Prediction =
    { RowId: string
      Measurement: string
      Actual: float
      Predicted: float
      Residual: float }

type MeasurementValue =
    { RowId: string
      Measurement: string
      Value: float
      Log: string }

let private usage () =
    printfn "Usage: dotnet run --project src/StackProcessing.Probe -- calibrate [options]"
    printfn ""
    printfn "Runs the greedy calibration loop over existing sample/probe measurements."
    printfn "It analyzes the current matrix, freezes reliable coefficients, emits and runs"
    printfn "one-unknown or small triangular JSON probe batches, then writes final estimates."
    printfn ""
    printfn "Options:"
    printfn "  --samples-root PATH     Sample root. Defaults to samples."
    printfn "  --analysis-dir PATH     Analysis output/read directory. Defaults to tmp/analysis."
    printfn "  --probe-json-root PATH  Probe JSON output directory. Defaults to tmp/probingGraphs."
    printfn "  --iterations N          Greedy loop iterations. Defaults to 5."
    printfn "  --no-run-probes         Emit probe graphs without running them."
    printfn "  --repeat N              Repeat emitted probe runs. Defaults to 1."
    printfn "  --probe-repeats N       Alias for --repeat."
    printfn "  -j, --jobs N            Run up to N emitted probe graphs at once. Defaults to 1."
    printfn "  --skip-analysis         Use existing analysis CSVs without rerunning analysis first."
    printfn "  --min-support N         Minimum feature support for freezing. Defaults to 3."
    printfn "  --max-condition VALUE   Maximum effective condition number for a stable subset. Defaults to 50."
    printfn "  --min-rank-ratio VALUE  Minimum subset rank ratio for a stable subset. Defaults to 0.8."
    printfn "  --min-r2 VALUE          Minimum global fit R2 for freezing a measurement. Defaults to 0."
    printfn "  --residual-multiple X   Median residual must be <= X * RMSE. Defaults to 2."
    printfn "  --max-unknowns N        Max non-boilerplate features in emitted triangular probes. Defaults to 3."
    printfn "  --max-graphs N          Max emitted probe graphs per iteration. Defaults to 200."

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

let private defaultOptions () =
    let samplesRoot = defaultSamplesRoot ()
    let root = repositoryRootFromSamplesRoot samplesRoot

    { SamplesRoot = samplesRoot
      AnalysisDirectory = Path.Combine(root, "tmp", "analysis")
      ProbeJsonRoot = Path.Combine(root, "tmp", "probingGraphs")
      Iterations = 5
      MinSupport = 3
      MaxCondition = 50.0
      MinRankRatio = 0.8
      MinR2 = 0.0
      ResidualMultiple = 2.0
      MaxUnknowns = 3
      MaxGraphs = 200
      RunProbes = true
      ProbeRepeats = 1
      Jobs = 1
      SkipAnalysis = false }

let private timestamp () =
    DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture)

let rec private parseArgs options args =
    match args with
    | [] -> Ok options
    | ("-h" | "--help") :: _ ->
        usage ()
        Error 0
    | "--samples-root" :: value :: rest ->
        let samplesRoot = Path.GetFullPath value
        let root = repositoryRootFromSamplesRoot samplesRoot
        parseArgs
            { options with
                SamplesRoot = samplesRoot
                AnalysisDirectory =
                    if options.AnalysisDirectory = (defaultOptions ()).AnalysisDirectory then
                        Path.Combine(root, "tmp", "analysis")
                    else
                        options.AnalysisDirectory
                ProbeJsonRoot =
                    if options.ProbeJsonRoot = (defaultOptions ()).ProbeJsonRoot then
                        Path.Combine(root, "tmp", "probingGraphs")
                    else
                        options.ProbeJsonRoot }
            rest
    | "--analysis-dir" :: value :: rest
    | "--output" :: value :: rest ->
        parseArgs { options with AnalysisDirectory = Path.GetFullPath value } rest
    | "--probe-json-root" :: value :: rest
    | "--emit-json" :: value :: rest ->
        parseArgs { options with ProbeJsonRoot = Path.GetFullPath value } rest
    | "--iterations" :: value :: rest ->
        match Int32.TryParse value with
        | true, n when n > 0 -> parseArgs { options with Iterations = n } rest
        | _ -> Error 2
    | "--min-support" :: value :: rest ->
        match Int32.TryParse value with
        | true, n when n > 0 -> parseArgs { options with MinSupport = n } rest
        | _ -> Error 2
    | "--max-condition" :: value :: rest ->
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, n when n > 0.0 -> parseArgs { options with MaxCondition = n } rest
        | _ -> Error 2
    | "--min-rank-ratio" :: value :: rest ->
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, n when n >= 0.0 && n <= 1.0 -> parseArgs { options with MinRankRatio = n } rest
        | _ -> Error 2
    | "--min-r2" :: value :: rest ->
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, n -> parseArgs { options with MinR2 = n } rest
        | _ -> Error 2
    | "--residual-multiple" :: value :: rest ->
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, n when n > 0.0 -> parseArgs { options with ResidualMultiple = n } rest
        | _ -> Error 2
    | "--max-unknowns" :: value :: rest ->
        match Int32.TryParse value with
        | true, n when n > 0 -> parseArgs { options with MaxUnknowns = n } rest
        | _ -> Error 2
    | "--max-graphs" :: value :: rest ->
        match Int32.TryParse value with
        | true, n when n > 0 -> parseArgs { options with MaxGraphs = n } rest
        | _ -> Error 2
    | "--run-probes" :: rest ->
        parseArgs { options with RunProbes = true } rest
    | "--no-run-probes" :: rest
    | "--emit-only" :: rest ->
        parseArgs { options with RunProbes = false } rest
    | ("-j" | "--jobs") :: value :: rest ->
        match Int32.TryParse value with
        | true, n when n > 0 -> parseArgs { options with Jobs = n } rest
        | _ -> Error 2
    | "--repeat" :: value :: rest
    | "--repeats" :: value :: rest
    | "--probe-repeats" :: value :: rest ->
        match Int32.TryParse value with
        | true, n when n > 0 -> parseArgs { options with ProbeRepeats = n } rest
        | _ -> Error 2
    | "--skip-analysis" :: rest ->
        parseArgs { options with SkipAnalysis = true } rest
    | value :: _ ->
        eprintfn "calibrate: unknown argument '%s'" value
        Error 2

let private splitCsvLine (line: string) =
    let cells = ResizeArray<string>()
    let current = StringBuilder()
    let mutable quoted = false
    let mutable i = 0

    while i < line.Length do
        let ch = line[i]

        if quoted then
            if ch = '"' then
                if i + 1 < line.Length && line[i + 1] = '"' then
                    current.Append '"' |> ignore
                    i <- i + 1
                else
                    quoted <- false
            else
                current.Append ch |> ignore
        else
            match ch with
            | '"' -> quoted <- true
            | ',' ->
                cells.Add(current.ToString())
                current.Clear() |> ignore
            | _ -> current.Append ch |> ignore

        i <- i + 1

    cells.Add(current.ToString())
    cells.ToArray()

let private readCsvMaps path =
    if not (File.Exists path) then
        []
    else
        let lines = File.ReadAllLines path

        if lines.Length = 0 then
            []
        else
            let headers = splitCsvLine lines[0]

            lines
            |> Array.skip 1
            |> Array.filter (String.IsNullOrWhiteSpace >> not)
            |> Array.map (fun line ->
                let cells = splitCsvLine line

                headers
                |> Array.mapi (fun i header -> header, if i < cells.Length then cells[i] else "")
                |> Map.ofArray)
            |> Array.toList

let private csvCell (value: string) =
    if value.Contains(',') || value.Contains('"') || value.Contains('\n') || value.Contains('\r') then
        "\"" + value.Replace("\"", "\"\"") + "\""
    else
        value

let private writeCsv (path: string) rows =
    Directory.CreateDirectory(Path.GetDirectoryName(path: string)) |> ignore
    File.WriteAllLines(path, rows |> Seq.map (List.map csvCell >> String.concat ","))

let private field name (row: Map<string, string>) =
    row |> Map.tryFind name |> Option.defaultValue ""

let private intField name row =
    match Int32.TryParse(field name row) with
    | true, value -> value
    | _ -> 0

let private floatField name row =
    match Double.TryParse(field name row, NumberStyles.Float, CultureInfo.InvariantCulture) with
    | true, value -> value
    | _ -> Double.NaN

let private invariant (value: float) =
    value.ToString("G17", CultureInfo.InvariantCulture)

let private normalizeFeatureKey (feature: string) =
    feature.Split(':')
    |> Array.filter (fun part -> not (part.StartsWith("suffix=", StringComparison.Ordinal)))
    |> String.concat ":"

let private loadRows dir =
    readCsvMaps (Path.Combine(dir, "rows.csv"))
    |> List.map (fun row ->
        { RowId = field "rowId" row
          Source = field "source" row
          Path = field "path" row })

let private loadFeatures dir =
    readCsvMaps (Path.Combine(dir, "features.csv"))
    |> List.groupBy (fun row -> field "rowId" row)
    |> List.map (fun (rowId, values) ->
        rowId,
        (values
         |> List.map (fun row -> normalizeFeatureKey (field "featureKey" row), floatField "value" row)
         |> Map.ofList))
    |> Map.ofList

let private loadFeatureDiagnostics dir =
    readCsvMaps (Path.Combine(dir, "featureDiagnostics.csv"))
    |> List.map (fun row ->
        { FeatureKey = normalizeFeatureKey (field "featureKey" row)
          SupportCount = intField "supportCount" row
          SupportRatio = floatField "supportRatio" row })

let private loadSubsets dir =
    readCsvMaps (Path.Combine(dir, "subsetDiagnostics.csv"))
    |> List.map (fun row ->
        { Name = field "subset" row
          RowCount = intField "rowCount" row
          ActiveColumnCount = intField "activeColumnCount" row
          Rank = intField "rank" row
          RankRatio = floatField "rankRatio" row
          EffectiveConditionNumber = floatField "effectiveConditionNumber" row })

let private loadCoefficients dir =
    readCsvMaps (Path.Combine(dir, "coefficients.csv"))
    |> List.map (fun row ->
        { Measurement = field "measurement" row
          FeatureKey = normalizeFeatureKey (field "featureKey" row)
          Coefficient = floatField "coefficient" row
          SupportCount = intField "supportCount" row
          RowCount = intField "rowCount" row
          ColumnCount = intField "columnCount" row
          Rmse = floatField "rmse" row
          R2 = floatField "r2" row })

let private loadPredictions dir =
    readCsvMaps (Path.Combine(dir, "predictions.csv"))
    |> List.map (fun row ->
        { RowId = field "rowId" row
          Measurement = field "measurement" row
          Actual = floatField "actual" row
          Predicted = floatField "predicted" row
          Residual = floatField "residual" row })

let private loadMeasurements dir =
    readCsvMaps (Path.Combine(dir, "vectors.csv"))
    |> List.map (fun row ->
        { RowId = field "rowId" row
          Measurement = field "measurement" row
          Value = floatField "value" row
          Log = field "log" row })

let private median values =
    let sorted = values |> Seq.filter Double.IsFinite |> Seq.sort |> Seq.toArray

    if sorted.Length = 0 then
        Double.NaN
    elif sorted.Length % 2 = 1 then
        sorted[sorted.Length / 2]
    else
        let right = sorted.Length / 2
        0.5 * (sorted[right - 1] + sorted[right])

let private isGeneratedRow (rowId: string) =
    rowId.StartsWith("generated/", StringComparison.Ordinal)

let private generatedRowContains (marker: string) (rowId: string) =
    isGeneratedRow rowId && rowId.Contains(marker, StringComparison.Ordinal)

let private rowMatchesSubset subsetName (rowId: string) =
    match subsetName with
    | "generated-all" -> isGeneratedRow rowId
    | "generated-bio-filter" -> generatedRowContains "/bio-filter" rowId
    | "generated-bio-grayscale" -> generatedRowContains "/bio-grayscale" rowId
    | "generated-bio-threshold" -> generatedRowContains "/bio-threshold" rowId
    | "generated-bio-threshold+grayscale" ->
        generatedRowContains "/bio-threshold" rowId
        || generatedRowContains "/bio-grayscale" rowId
    | "generated-bio-filter+projection" ->
        generatedRowContains "/bio-filter" rowId
        || generatedRowContains "/bio-projection" rowId
    | "generated-boilerplate" ->
        generatedRowContains "/zero-" rowId
        || generatedRowContains "/read-" rowId
    | _ -> false

let private measurementKind (measurement: string) =
    let text = measurement.ToLowerInvariant()

    if text.Contains("memory") || text.Contains("rss") || text.Contains("bytes") || text.Contains("kb") then
        "memory"
    elif text.Contains("time") || text.Contains("second") || text.Contains("elapsed") then
        "time"
    else
        "other"

let private isPlanningBoilerplateFeature (feature: string) =
    feature = "intercept"
    || feature.StartsWith("Read:", StringComparison.Ordinal)
    || feature.StartsWith("Write:", StringComparison.Ordinal)
    || feature.StartsWith("Zero:", StringComparison.Ordinal)

let private runAnalysis options =
    let args =
        ResizeArray<string>(
            [| "--samples-root"
               options.SamplesRoot
               "--output"
               options.AnalysisDirectory |]
        )

    if Directory.Exists options.ProbeJsonRoot then
        args.Add "--extra-json-root"
        args.Add options.ProbeJsonRoot

    ProbeAnalysis.main (args.ToArray())

let private runProbeGraphs options probeJsonRoot =
    RunSamples.main
        [| options.SamplesRoot
           "--json"
           "--extra-json-root"
           probeJsonRoot
           "--extra-json-only"
           "--repeat"
           string options.ProbeRepeats
           "-j"
           string options.Jobs |]

let private analyzeIteration options iteration emitProbes =
    if not options.SkipAnalysis || iteration > 1 then
        let exitCode = runAnalysis options
        if exitCode <> 0 then
            failwithf "analysis failed with exit code %d" exitCode

    let rows = loadRows options.AnalysisDirectory
    let rowFeatures = loadFeatures options.AnalysisDirectory
    let featureDiagnostics = loadFeatureDiagnostics options.AnalysisDirectory
    let subsets = loadSubsets options.AnalysisDirectory
    let coefficients = loadCoefficients options.AnalysisDirectory
    let predictions = loadPredictions options.AnalysisDirectory
    let measurements = loadMeasurements options.AnalysisDirectory

    let supportByFeature =
        featureDiagnostics
        |> List.map (fun diagnostic -> diagnostic.FeatureKey, diagnostic.SupportCount)
        |> Map.ofList

    let stableSubsets =
        subsets
        |> List.filter (fun subset ->
            subset.ActiveColumnCount > 0
            && subset.RankRatio >= options.MinRankRatio
            && Double.IsFinite subset.EffectiveConditionNumber
            && subset.EffectiveConditionNumber <= options.MaxCondition)

    let stableRows =
        rows
        |> List.filter (fun row -> stableSubsets |> List.exists (fun subset -> rowMatchesSubset subset.Name row.RowId))
        |> List.map _.RowId
        |> Set.ofList

    let stableFeatures =
        stableRows
        |> Seq.collect (fun rowId ->
            rowFeatures
            |> Map.tryFind rowId
            |> Option.map (Map.toSeq >> Seq.map fst)
            |> Option.defaultValue Seq.empty)
        |> Seq.append [ "intercept" ]
        |> Set.ofSeq

    let medianAbsResidualByMeasurement =
        predictions
        |> List.groupBy _.Measurement
        |> List.map (fun (measurement, rows) -> measurement, rows |> List.map (fun row -> abs row.Residual) |> median)
        |> Map.ofList

    let frozen =
        coefficients
        |> List.choose (fun coefficient ->
            let support = supportByFeature |> Map.tryFind coefficient.FeatureKey |> Option.defaultValue coefficient.SupportCount
            let medianResidual =
                medianAbsResidualByMeasurement
                |> Map.tryFind coefficient.Measurement
                |> Option.defaultValue Double.NaN

            let residualOk =
                if not (Double.IsFinite coefficient.Rmse) || coefficient.Rmse = 0.0 then
                    true
                elif not (Double.IsFinite medianResidual) then
                    true
                else
                    medianResidual <= options.ResidualMultiple * coefficient.Rmse

            let accepted =
                stableFeatures.Contains coefficient.FeatureKey
                && support >= options.MinSupport
                && Double.IsFinite coefficient.Coefficient
                && coefficient.Coefficient >= 0.0
                && coefficient.R2 >= options.MinR2
                && residualOk

            if accepted then
                Some(coefficient, medianResidual)
            else
                None)

    let solvedFeatureKinds =
        frozen
        |> List.groupBy (fun (coefficient, _) -> coefficient.FeatureKey)
        |> List.map (fun (feature, coefficients) ->
            feature,
            (coefficients |> List.map (fun (coefficient, _) -> measurementKind coefficient.Measurement) |> Set.ofList))
        |> Map.ofList

    let solvedFeatures =
        solvedFeatureKinds
        |> Map.toSeq
        |> Seq.choose (fun (feature, kinds) ->
            if kinds.Contains "time" && kinds.Contains "memory" then Some feature else None)
        |> Set.ofSeq

    let sampleRows =
        rows
        |> List.filter (fun row -> not (isGeneratedRow row.RowId))

    let sampleVocabulary =
        sampleRows
        |> Seq.collect (fun row ->
            rowFeatures
            |> Map.tryFind row.RowId
            |> Option.map (Map.toSeq >> Seq.map fst)
            |> Option.defaultValue Seq.empty)
        |> Seq.filter ((<>) "intercept")
        |> Set.ofSeq

    let sampleStageVocabulary =
        sampleVocabulary
        |> Set.filter (isPlanningBoilerplateFeature >> not)

    let remainingTargets =
        sampleStageVocabulary
        |> Set.filter (fun feature -> not (solvedFeatures.Contains feature))

    let planningSolvedFeatures =
        sampleVocabulary
        |> Set.filter isPlanningBoilerplateFeature
        |> Set.union solvedFeatures
        |> Set.add "intercept"

    let templates =
        ProbeProbing.graphTemplatesForCalibration ()
        |> Array.map (fun template ->
            let normalizedFeatures =
                template.Features
                |> List.map normalizeFeatureKey
                |> List.distinct
                |> Set.ofList

            template, normalizedFeatures)

    let templatePlans =
        templates
        |> Array.choose (fun (template, templateFeatures) ->
            let unknowns =
                templateFeatures
                |> Set.filter (fun feature -> not (planningSolvedFeatures.Contains feature))

            let targetUnknowns = Set.intersect unknowns remainingTargets

            if targetUnknowns.IsEmpty then
                None
            elif unknowns.Count = 1 then
                Some(template, "one-unknown", unknowns, targetUnknowns)
            elif unknowns.Count <= options.MaxUnknowns then
                Some(template, "triangular", unknowns, targetUnknowns)
            else
                None)
        |> Array.sortBy (fun (_, mode, unknowns, targets) ->
            let modeRank = if mode = "one-unknown" then 0 else 1
            modeRank, unknowns.Count, targets.Count)

    let emitted =
        if emitProbes then
            templatePlans
            |> Seq.distinctBy (fun (template, _, _, _) -> template.Name)
            |> Seq.truncate options.MaxGraphs
            |> Seq.toArray
        else
            [||]

    let coveredByEmitted =
        emitted
        |> Seq.collect (fun (_, _, _, targets) -> targets)
        |> Set.ofSeq

    writeCsv
        (Path.Combine(options.AnalysisDirectory, "frozenCoefficients.csv"))
        (seq {
            yield
                [ "measurement"
                  "featureKey"
                  "coefficient"
                  "supportCount"
                  "rmse"
                  "r2"
                  "medianAbsResidual"
                  "reason" ]

            for coefficient, medianResidual in frozen do
                yield
                    [ coefficient.Measurement
                      coefficient.FeatureKey
                      invariant coefficient.Coefficient
                      string coefficient.SupportCount
                      invariant coefficient.Rmse
                      invariant coefficient.R2
                      invariant medianResidual
                      "stable-subset+support+residual" ]
        })

    let frozenCoefficientMap =
        frozen
        |> List.map (fun (coefficient, _) -> (coefficient.Measurement, coefficient.FeatureKey), coefficient.Coefficient)
        |> Map.ofList

    let sampleRowIds = sampleRows |> List.map _.RowId |> Set.ofList

    writeCsv
        (Path.Combine(options.AnalysisDirectory, "sampleEstimates.csv"))
        (seq {
            yield
                [ "rowId"
                  "measurement"
                  "actual"
                  "estimated"
                  "residual"
                  "relativeResidual"
                  "knownFeatureCount"
                  "missingFeatureCount"
                  "missingFeatures"
                  "log" ]

            for measurement in measurements do
                if sampleRowIds.Contains measurement.RowId then
                    let features =
                        rowFeatures
                        |> Map.tryFind measurement.RowId
                        |> Option.defaultValue Map.empty
                        |> Map.add "intercept" 1.0

                    let mutable estimated = 0.0
                    let mutable knownFeatureCount = 0
                    let missingFeatures = ResizeArray<string>()

                    for KeyValue(feature, value) in features do
                        match frozenCoefficientMap |> Map.tryFind (measurement.Measurement, feature) with
                        | Some coefficient ->
                            estimated <- estimated + value * coefficient
                            knownFeatureCount <- knownFeatureCount + 1
                        | None ->
                            missingFeatures.Add feature

                    let residual = measurement.Value - estimated
                    let relativeResidual =
                        if measurement.Value = 0.0 then
                            Double.NaN
                        else
                            residual / measurement.Value

                    yield
                        [ measurement.RowId
                          measurement.Measurement
                          invariant measurement.Value
                          invariant estimated
                          invariant residual
                          invariant relativeResidual
                          string knownFeatureCount
                          string missingFeatures.Count
                          String.concat ";" (missingFeatures |> Seq.sort)
                          measurement.Log ]
        })

    writeCsv
        (Path.Combine(options.AnalysisDirectory, "greedyCoverage.csv"))
        (seq {
            yield [ "kind"; "count" ]
            yield [ "sampleVocabulary"; string sampleVocabulary.Count ]
            yield [ "sampleStageVocabulary"; string sampleStageVocabulary.Count ]
            yield [ "stableSubsets"; string stableSubsets.Length ]
            yield [ "featuresSolvedForTimeAndMemory"; string solvedFeatures.Count ]
            yield [ "remainingSampleFeatures"; string remainingTargets.Count ]
            yield [ "targetsCoveredByEmittedProbes"; string coveredByEmitted.Count ]
            yield [ "emittedProbeGraphs"; string emitted.Length ]
        })

    writeCsv
        (Path.Combine(options.AnalysisDirectory, "probeTargets.csv"))
        (seq {
            yield [ "featureKey"; "supportCount"; "status" ]

            for feature in sampleVocabulary |> Seq.sort do
                let status =
                    if solvedFeatures.Contains feature then
                        "frozen"
                    elif isPlanningBoilerplateFeature feature then
                        "calibration-boilerplate"
                    elif coveredByEmitted.Contains feature then
                        "probe-emitted"
                    else
                        "needs-probe-pattern-or-more-context"

                yield
                    [ feature
                      string (supportByFeature |> Map.tryFind feature |> Option.defaultValue 0)
                      status ]
        })

    writeCsv
        (Path.Combine(options.AnalysisDirectory, "probePlan.csv"))
        (seq {
            yield [ "template"; "mode"; "unknownCount"; "unknownFeatures"; "targetFeatures"; "emitted" ]

            for template, mode, unknowns, targets in templatePlans do
                let isEmitted = emitted |> Array.exists (fun (selected, _, _, _) -> selected.Name = template.Name)

                yield
                    [ template.Name
                      mode
                      string unknowns.Count
                      String.concat ";" (unknowns |> Seq.sort)
                      String.concat ";" (targets |> Seq.sort)
                      string isEmitted ]
        })

    let emitDirectory =
        if options.RunProbes || options.Iterations > 1 then
            Path.Combine(options.ProbeJsonRoot, sprintf "iteration_%03d" iteration)
        else
            options.ProbeJsonRoot

    if not emitProbes then
        printfn "Final estimate pass: no new calibration probes emitted."
    elif emitted.Length > 0 then
        ProbeProbing.writeGraphTemplates emitDirectory (emitted |> Array.map (fun (template, _, _, _) -> template))
    else
        printfn "No calibration probes could be emitted from the current template set."

    printfn
        "Greedy calibration: %d/%d sample features frozen for time+memory; %d target feature(s) covered by %d emitted probe graph(s)."
        solvedFeatures.Count
        sampleStageVocabulary.Count
        coveredByEmitted.Count
        emitted.Length

    sampleStageVocabulary.Count, solvedFeatures.Count, remainingTargets.Count, emitted.Length, emitDirectory

let main args =
    match parseArgs (defaultOptions ()) (Array.toList args) with
    | Error exitCode -> exitCode
    | Ok options ->
        try
            let options =
                if options.RunProbes then
                    { options with ProbeJsonRoot = Path.Combine(options.ProbeJsonRoot, "calibration_" + timestamp ()) }
                else
                    options

            if options.RunProbes then
                printfn "Calibration probe root: %s" options.ProbeJsonRoot

            let mutable doneLoop = false
            let mutable iteration = 1
            let mutable exitCode = 0
            let mutable ranProbeBatch = false

            while not doneLoop && iteration <= options.Iterations do
                printfn "Calibration iteration %d/%d" iteration options.Iterations
                let vocabularyCount, solvedCount, remainingCount, emittedCount, emitDirectory = analyzeIteration options iteration true

                if remainingCount = 0 then
                    printfn "Calibration complete: all %d sample feature(s) have frozen time and memory estimates." vocabularyCount
                    doneLoop <- true
                elif emittedCount = 0 then
                    printfn "Calibration stopped: %d feature(s) still need a new probe pattern or manual intervention." remainingCount
                    doneLoop <- true
                elif options.RunProbes then
                    let runExit = runProbeGraphs options emitDirectory
                    if runExit <> 0 then
                        eprintfn "probe graph run failed with exit code %d" runExit
                        exitCode <- runExit
                        doneLoop <- true
                    else
                        ranProbeBatch <- true
                        iteration <- iteration + 1
                else
                    if not options.RunProbes then
                        printfn "Development mode: emitted probes only. Run them with Probe samples --json --extra-json-root %s --extra-json-only." emitDirectory
                    doneLoop <- true

                if options.RunProbes && not doneLoop && solvedCount >= vocabularyCount then
                    doneLoop <- true

            if exitCode = 0 && ranProbeBatch then
                printfn "Final calibration estimate pass"
                let vocabularyCount, _, remainingCount, _, _ = analyzeIteration options iteration false
                if remainingCount = 0 then
                    printfn "Calibration complete: all %d sample feature(s) have frozen time and memory estimates." vocabularyCount
                else
                    printfn "Calibration finished with %d sample feature(s) still unresolved." remainingCount

            exitCode
        with ex ->
            eprintfn "calibrate failed: %s" ex.Message
            1
