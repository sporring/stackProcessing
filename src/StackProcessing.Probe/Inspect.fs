module ProbeInspect

open System
open System.Globalization
open System.IO
open System.Text.Json

type Options =
    { MeasurementStorePath: string
      OutputDirectory: string
      MaxStep: string option
      MinRepeats: int
      MinTimeR2: float
      MaxFlaggedRatio: float
      Selector: ProbeSelection.EvidenceSelector
      RequestOutputPath: string option }

type CollectionRequest =
    { SchemaVersion: int
      CreatedUtc: DateTimeOffset
      Families: string array
      Members: string array
      MinRepeats: int
      Reason: string
      ExtraArgs: string array }

let private usage () =
    printfn "Usage: dotnet run --project src/StackProcessing.Probe -- inspect [options]"
    printfn ""
    printfn "Inspects durable measurement coverage and suggests the next collection step."
    printfn ""
    printfn "Options:"
    printfn "  --measurement-store PATH  Measurement JSONL store. Defaults to measurements/stackprocessing-probe.jsonl."
    printfn "  --output PATH             Inspection output directory. Defaults to tmp/inspect."
    printfn "  --max-step FAMILY         Highest ladder family to consider."
    printfn "  --family LIST             Restrict inspected families."
    printfn "  --member LIST             Restrict inspected members/operators."
    printfn "  --min-repeats N           Desired repeats per graph/measurement. Defaults to 3."
    printfn "  --min-time-r2 VALUE       Minimum elapsedMilliseconds R2. Defaults to 0.8."
    printfn "  --max-flagged-ratio VALUE Maximum operator discrepancy flagged ratio. Defaults to 0.1."
    printfn "  --suggest PATH            Write a collection request JSON."

let private repositoryRoot () =
    let cwd = Directory.GetCurrentDirectory()
    if File.Exists(Path.Combine(cwd, "StackProcessing.sln")) then cwd else cwd

let private defaultOptions () =
    let root = repositoryRoot ()
    { MeasurementStorePath = Path.Combine(root, "measurements", "stackprocessing-probe.jsonl")
      OutputDirectory = Path.Combine(root, "tmp", "inspect")
      MaxStep = None
      MinRepeats = 3
      MinTimeR2 = 0.8
      MaxFlaggedRatio = 0.1
      Selector = { Families = [ "all" ]; Members = []; UpTo = None }
      RequestOutputPath = None }

let rec private parseArgs options args =
    match args with
    | [] -> Ok options
    | ("-h" | "--help") :: _ ->
        usage ()
        Error 0
    | "--measurement-store" :: value :: rest ->
        parseArgs { options with MeasurementStorePath = Path.GetFullPath value } rest
    | "--output" :: value :: rest ->
        parseArgs { options with OutputDirectory = Path.GetFullPath value } rest
    | "--max-step" :: value :: rest
    | "--up-to" :: value :: rest ->
        match ProbeSelection.normalizeFamily value with
        | Some family -> parseArgs { options with MaxStep = Some family; Selector = { options.Selector with UpTo = Some family } } rest
        | None ->
            eprintfn "inspect: unknown ladder family '%s'" value
            Error 2
    | "--family" :: value :: rest
    | "--families" :: value :: rest ->
        match ProbeSelection.parseFamilies value with
        | Some families -> parseArgs { options with Selector = { options.Selector with Families = families; UpTo = None } } rest
        | None ->
            eprintfn "inspect: --family expects io,io-cast,singleton,neighbourhood,geometry,fourier,keypoints,dependency,reducers, or all"
            Error 2
    | "--member" :: value :: rest
    | "--members" :: value :: rest
    | "--operator" :: value :: rest
    | "--operators" :: value :: rest ->
        parseArgs { options with Selector = { options.Selector with Members = options.Selector.Members @ ProbeSelection.splitCsvList value } } rest
    | "--min-repeats" :: value :: rest ->
        match Int32.TryParse value with
        | true, minRepeats when minRepeats > 0 -> parseArgs { options with MinRepeats = minRepeats } rest
        | _ ->
            eprintfn "inspect: --min-repeats expects a positive integer"
            Error 2
    | "--min-time-r2" :: value :: rest ->
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, value -> parseArgs { options with MinTimeR2 = value } rest
        | _ ->
            eprintfn "inspect: --min-time-r2 expects a floating-point value"
            Error 2
    | "--max-flagged-ratio" :: value :: rest ->
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, value when value >= 0.0 -> parseArgs { options with MaxFlaggedRatio = value } rest
        | _ ->
            eprintfn "inspect: --max-flagged-ratio expects a non-negative floating-point value"
            Error 2
    | "--suggest" :: value :: rest
    | "--request-output" :: value :: rest ->
        parseArgs { options with RequestOutputPath = Some(Path.GetFullPath value) } rest
    | option :: _ ->
        eprintfn "inspect: unknown option %s" option
        usage ()
        Error 2

let private csvEscape (value: string) =
    if value.Contains(',') || value.Contains('"') || value.Contains('\n') then
        "\"" + value.Replace("\"", "\"\"") + "\""
    else
        value

let private writeCsv (path: string) rows =
    Directory.CreateDirectory(Path.GetDirectoryName path) |> ignore
    rows
    |> Seq.map (List.map csvEscape >> String.concat ",")
    |> fun lines -> File.WriteAllLines(path, lines)

let private invariant (value: float) =
    value.ToString("G17", CultureInfo.InvariantCulture)

let private familyRecords selector (records: ProbeAnalysis.StoredMeasurementRecord list) =
    records
    |> List.filter (ProbeFit.selectorMatchesRecord selector)
    |> List.groupBy (fun record -> ProbeSelection.familyForRowId record.RowId |> Option.defaultValue "unknown")
    |> Map.ofList

let private familyCoverage (records: ProbeAnalysis.StoredMeasurementRecord list) =
    let groups =
        records
        |> List.groupBy (fun record -> record.RowId, record.Measurement)
        |> List.map (fun ((rowId, measurement), rows) -> rowId, measurement, rows.Length)

    let graphCount = groups |> List.map (fun (rowId, _, _) -> rowId) |> List.distinct |> List.length
    let measurementCount = groups.Length
    let minRepeats =
        groups
        |> List.map (fun (_, _, count) -> count)
        |> function
            | [] -> 0
            | counts -> counts |> List.min
    let medianRepeats =
        groups
        |> List.map (fun (_, _, count) -> count)
        |> List.sort
        |> function
            | [] -> 0.0
            | counts when counts.Length % 2 = 1 -> float counts[counts.Length / 2]
            | counts -> 0.5 * float (counts[counts.Length / 2 - 1] + counts[counts.Length / 2])

    graphCount, measurementCount, minRepeats, medianRepeats

let private nextRequest options recordsByFamily =
    let families =
        ProbeSelection.selectedFamilies options.Selector
        |> List.filter ((<>) "all")

    families
    |> List.tryPick (fun family ->
        let records = recordsByFamily |> Map.tryFind family |> Option.defaultValue []
        let graphCount, _, minRepeats, _ = familyCoverage records
        if graphCount = 0 then
            Some(family, $"No measurements found for family '{family}'.")
        elif minRepeats < options.MinRepeats then
            Some(family, $"Family '{family}' has minimum repeat count {minRepeats}, below requested {options.MinRepeats}.")
        else
            None)

type private FitQuality =
    { TimeR2: float option
      FlaggedRows: int
      PredictionRows: int
      Failed: bool
      Reason: string option
      SuggestedFamilies: string list
      SuggestedMembers: string list
      SuggestedExtraArgs: string list }

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

let private csvRows path =
    if File.Exists path then
        File.ReadAllLines path |> Array.toList |> List.map parseCsvLine
    else
        []

let private columnIndex name header =
    header |> List.tryFindIndex ((=) name)

let private columnValue index row =
    index
    |> Option.bind (fun index -> if index < List.length row then Some row[index] else None)
    |> Option.defaultValue ""

let private tryParseFloat (value: string) =
    match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
    | true, parsed -> Some parsed
    | _ -> None

let private earliestFamily families =
    let order =
        ProbeSelection.ladder
        |> List.mapi (fun index family -> family, index)
        |> Map.ofList

    families
    |> List.distinct
    |> List.sortBy (fun family -> order |> Map.tryFind family |> Option.defaultValue Int32.MaxValue)
    |> List.tryHead

let private graphNameFromRowId (rowId: string) =
    let normalized = rowId.Replace('\\', '/')
    let slash = normalized.LastIndexOf('/')
    if slash >= 0 then normalized.Substring(slash + 1) else normalized

let private shapeFromRowId (rowId: string) =
    let normalized = rowId.Replace('\\', '/')
    let marker = "/size_"
    let markerIndex = normalized.IndexOf(marker, StringComparison.OrdinalIgnoreCase)
    if markerIndex >= 0 then
        let start = markerIndex + marker.Length
        let finish = normalized.IndexOf('/', start)
        if finish > start then Some(normalized.Substring(start, finish - start)) else None
    else
        let graphName = graphNameFromRowId rowId
        let parts = graphName.Split('-', StringSplitOptions.RemoveEmptyEntries)
        parts
        |> Array.tryLast
        |> Option.filter (fun value -> value.Contains('x'))

type private DiscrepancySuggestion =
    { Families: string list
      Members: string list
      ExtraArgs: string list
      Reason: string option }

let private suggestionForFlaggedRows reason rows rowIdIndex =
    let rowIds =
        rows
        |> List.map (columnValue rowIdIndex)
        |> List.filter (String.IsNullOrWhiteSpace >> not)

    let members =
        rowIds
        |> List.map graphNameFromRowId
        |> List.distinct
        |> List.sort

    let shapes =
        rowIds
        |> List.choose shapeFromRowId
        |> List.distinct
        |> List.sort

    let extraArgs =
        match shapes with
        | [] -> []
        | _ -> [ "--shapes"; String.concat "," shapes ]

    { Families =
        rowIds
        |> List.choose ProbeSelection.familyForRowId
        |> List.distinct
      Members = members
      ExtraArgs = extraArgs
      Reason = reason }

let private diagnoseDiscrepancies discrepanciesPath =
    match csvRows discrepanciesPath with
    | header :: rows ->
        let rowIdIndex = columnIndex "rowId" header
        let measurementIndex = columnIndex "measurement" header
        let predictedIndex = columnIndex "predicted" header
        let flaggedIndex = columnIndex "flagged" header

        let elapsedFlagged =
            rows
            |> List.filter (fun row ->
                columnValue measurementIndex row = "elapsedMilliseconds"
                && String.Equals(columnValue flaggedIndex row, "True", StringComparison.OrdinalIgnoreCase))

        let familyForRow row =
            columnValue rowIdIndex row
            |> ProbeSelection.familyForRowId

        let predictedZeroRows =
            elapsedFlagged
            |> List.filter (fun row ->
                match columnValue predictedIndex row |> tryParseFloat with
                | Some predicted -> predicted = 0.0
                | _ -> false)

        match predictedZeroRows |> List.choose familyForRow |> earliestFamily with
        | Some family ->
            let flaggedRows =
                predictedZeroRows
                |> List.filter (fun row -> familyForRow row = Some family)

            let reason =
                Some $"elapsedMilliseconds has predicted-zero flagged rows in '{family}', so the lower-level time basis needs repair before adding higher stages."

            suggestionForFlaggedRows reason flaggedRows rowIdIndex
        | None ->
            let mostFlagged =
                elapsedFlagged
                |> List.choose familyForRow
                |> List.countBy id
                |> List.sortByDescending snd
                |> List.tryHead

            match mostFlagged with
            | Some(family, count) ->
                let flaggedRows =
                    elapsedFlagged
                    |> List.filter (fun row -> familyForRow row = Some family)

                let reason =
                    Some $"elapsedMilliseconds discrepancies are concentrated in '{family}' ({count} flagged rows)."

                suggestionForFlaggedRows reason flaggedRows rowIdIndex
            | None ->
                { Families = []
                  Members = []
                  ExtraArgs = []
                  Reason = None }
    | _ ->
        { Families = []
          Members = []
          ExtraArgs = []
          Reason = None }

let private inspectFitQuality (options: Options) =
    let fitOutput = Path.Combine(options.OutputDirectory, "fit")
    let modelOutput = Path.Combine(options.OutputDirectory, "inspect-model.json")
    Directory.CreateDirectory fitOutput |> ignore

    let fitOptions : ProbeFit.Options =
        { MeasurementStorePath = options.MeasurementStorePath
          OutputDirectory = fitOutput
          ModelOutputPath = modelOutput
          Ridge = 1e-8
          MinSupport = 3
          Selector = options.Selector }

    let _, _, fits = ProbeFit.fit fitOptions
    if fits = 0 then
        { TimeR2 = None
          FlaggedRows = 0
          PredictionRows = 0
          Failed = true
          Reason = Some "No fit could be produced from the selected evidence."
          SuggestedFamilies = []
          SuggestedMembers = []
          SuggestedExtraArgs = [] }
    else
        let diagnosticsPath = Path.Combine(fitOutput, "operatorModelDiagnostics.csv")
        let timeR2 =
            match csvRows diagnosticsPath with
            | header :: rows ->
                let measurementIndex = columnIndex "measurement" header
                let r2Index = columnIndex "r2" header
                rows
                |> List.tryPick (fun row ->
                    if columnValue measurementIndex row = "elapsedMilliseconds" then
                        columnValue r2Index row |> tryParseFloat
                    else
                        None)
            | [] -> None

        let discrepanciesPath = Path.Combine(fitOutput, "operatorModelDiscrepancies.csv")
        let discrepancySuggestion = diagnoseDiscrepancies discrepanciesPath
        let flaggedRows, predictionRows =
            match csvRows discrepanciesPath with
            | header :: rows ->
                let flaggedIndex = columnIndex "flagged" header
                let flagged =
                    rows
                    |> List.sumBy (fun row ->
                        if String.Equals(columnValue flaggedIndex row, "True", StringComparison.OrdinalIgnoreCase) then 1 else 0)
                flagged, rows.Length
            | [] -> 0, 0

        let flaggedRatio =
            if predictionRows > 0 then float flaggedRows / float predictionRows else 0.0

        let failureReason =
            match timeR2 with
            | Some r2 when r2 < options.MinTimeR2 ->
                discrepancySuggestion.Reason
                |> Option.orElse (Some $"elapsedMilliseconds R2 {invariant r2} is below requested {invariant options.MinTimeR2}.")
            | None ->
                Some "elapsedMilliseconds R2 is unavailable."
            | _ when flaggedRatio > options.MaxFlaggedRatio ->
                discrepancySuggestion.Reason
                |> Option.orElse (Some $"operator discrepancy flagged ratio {invariant flaggedRatio} is above requested {invariant options.MaxFlaggedRatio}.")
            | _ ->
                None

        { TimeR2 = timeR2
          FlaggedRows = flaggedRows
          PredictionRows = predictionRows
          Failed = failureReason.IsSome
          Reason = failureReason
          SuggestedFamilies = discrepancySuggestion.Families
          SuggestedMembers = discrepancySuggestion.Members
          SuggestedExtraArgs = discrepancySuggestion.ExtraArgs }

let private writeRequest (path: string) (options: Options) (families: string list) (members: string list) (reason: string) (extraArgs: string list) =
    Directory.CreateDirectory(Path.GetDirectoryName path) |> ignore
    let request =
        { SchemaVersion = 1
          CreatedUtc = DateTimeOffset.UtcNow
          Families = families |> List.toArray
          Members = members |> List.toArray
          MinRepeats = options.MinRepeats
          Reason = reason
          ExtraArgs = extraArgs |> List.toArray }
    let json = JsonSerializer.Serialize(request, JsonSerializerOptions(WriteIndented = true))
    File.WriteAllText(path, json)
    printfn "wrote collection request to %s" path

let main argv =
    match parseArgs (defaultOptions ()) (Array.toList argv) with
    | Error exitCode -> exitCode
    | Ok options ->
        try
            let records = ProbeFit.readMeasurementStore options.MeasurementStorePath
            let recordsByFamily = familyRecords options.Selector records
            let families = ProbeSelection.selectedFamilies options.Selector |> List.filter ((<>) "all")

            Directory.CreateDirectory options.OutputDirectory |> ignore
            writeCsv
                (Path.Combine(options.OutputDirectory, "coverage.csv"))
                (seq {
                    yield [ "family"; "records"; "graphs"; "measurements"; "minRepeats"; "medianRepeats" ]
                    for family in families do
                        let familyRecords = recordsByFamily |> Map.tryFind family |> Option.defaultValue []
                        let graphs, measurements, minRepeats, medianRepeats = familyCoverage familyRecords
                        yield [ family; string familyRecords.Length; string graphs; string measurements; string minRepeats; invariant medianRepeats ]
                })

            printfn "inspect read %d measurement record(s) from %s." records.Length options.MeasurementStorePath

            for family in families do
                let familyRecords = recordsByFamily |> Map.tryFind family |> Option.defaultValue []
                let graphs, measurements, minRepeats, medianRepeats = familyCoverage familyRecords
                printfn "%-14s records=%d graphs=%d measurements=%d minRepeats=%d medianRepeats=%s" family familyRecords.Length graphs measurements minRepeats (invariant medianRepeats)

            match nextRequest options recordsByFamily with
            | Some(family, reason) ->
                printfn "next collection suggestion: --family %s (%s)" family reason
                match options.RequestOutputPath with
                | Some path -> writeRequest path options [ family ] options.Selector.Members reason []
                | None -> ()
            | None ->
                printfn "coverage looks sufficient through the selected ladder scope."
                let quality = inspectFitQuality options
                match quality.TimeR2 with
                | Some r2 -> printfn "fit quality elapsedMilliseconds R2=%s" (invariant r2)
                | None -> printfn "fit quality elapsedMilliseconds R2=<unavailable>"
                printfn "fit quality flagged predictions=%d/%d" quality.FlaggedRows quality.PredictionRows

                if quality.Failed then
                    let reason = quality.Reason |> Option.defaultValue "Fit quality gate failed."
                    let suggestedFamilies =
                        match quality.SuggestedFamilies with
                        | _ :: _ as families -> families
                        | [] ->
                            options.MaxStep
                            |> Option.map (fun family -> [ family ])
                            |> Option.defaultValue (ProbeSelection.selectedFamilies options.Selector)
                            |> List.filter ((<>) "all")
                    let suggestedMembers =
                        match quality.SuggestedMembers with
                        | _ :: _ as members -> members
                        | [] -> options.Selector.Members

                    let memberText =
                        match suggestedMembers with
                        | [] -> ""
                        | members ->
                            let text = String.concat "," members
                            $" members={text}"

                    printfn "next collection suggestion: revisit %s%s (%s)" (String.concat "," suggestedFamilies) memberText reason
                    match options.RequestOutputPath with
                    | Some path ->
                        writeRequest path options suggestedFamilies suggestedMembers reason quality.SuggestedExtraArgs
                    | None -> ()
                else
                    printfn "fit quality looks sufficient through the selected ladder scope."

            0
        with ex ->
            eprintfn "inspect failed: %s" ex.Message
            1
