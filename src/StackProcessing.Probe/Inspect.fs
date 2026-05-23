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
      Selector: ProbeSelection.EvidenceSelector
      RequestOutputPath: string option }

type CollectionRequest =
    { SchemaVersion: int
      CreatedUtc: DateTimeOffset
      Families: string array
      Members: string array
      MinRepeats: int
      Reason: string }

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
                | Some path ->
                    Directory.CreateDirectory(Path.GetDirectoryName path) |> ignore
                    let request =
                        { SchemaVersion = 1
                          CreatedUtc = DateTimeOffset.UtcNow
                          Families = [| family |]
                          Members = options.Selector.Members |> List.toArray
                          MinRepeats = options.MinRepeats
                          Reason = reason }
                    let json = JsonSerializer.Serialize(request, JsonSerializerOptions(WriteIndented = true))
                    File.WriteAllText(path, json)
                    printfn "wrote collection request to %s" path
                | None -> ()
            | None ->
                printfn "coverage looks sufficient through the selected ladder scope."

            0
        with ex ->
            eprintfn "inspect failed: %s" ex.Message
            1
