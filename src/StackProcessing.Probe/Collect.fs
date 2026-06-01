module ProbeCollect

open System
open System.IO
open System.Text.Json

type Options =
    { Families: string list
      Members: string list
      All: bool
      RequestPath: string option
      ExtraArgs: string list }

type CollectionRequest =
    { SchemaVersion: int
      CreatedUtc: DateTimeOffset
      Families: string array
      Members: string array
      MinRepeats: int
      Reason: string
      ExtraArgs: string array }

let private usage () =
    printfn "Usage: dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll collect [options]"
    printfn ""
    printfn "Collects Probe measurements for selected ladder families or members."
    printfn "Family collection delegates to the controlled probe ladder; member collection delegates to local-update."
    printfn ""
    printfn "Selection:"
    printfn "  --all                 Collect every ladder family."
    printfn "  --family LIST         Collect families, e.g. io,io-cast,singleton."
    printfn "  --member LIST         Collect member/operator probes, e.g. SmoothWGauss."
    printfn "  --request PATH        Execute an inspect-generated collection request."
    printfn ""
    printfn "Common pass-through options:"
    printfn "  --shapes LIST --shape WxHxD --sizes LIST --repeat N -j N --keep-tmp --no-run-probes"

let private defaultOptions =
    { Families = []
      Members = []
      All = false
      RequestPath = None
      ExtraArgs = [] }

let private hasFitChoice args =
    args |> List.exists (fun arg -> arg = "--fit" || arg = "--no-fit")

let private withNoFitDefault args =
    if hasFitChoice args then args else args @ [ "--no-fit" ]

let private hasRepeatChoice args =
    args |> List.exists (fun arg -> arg = "--repeat" || arg = "--repeats")

let private withRepeatDefault repeat args =
    if hasRepeatChoice args then args else args @ [ "--repeat"; string repeat ]

let rec private parseArgs options args =
    match args with
    | [] -> Ok options
    | ("-h" | "--help") :: _ ->
        usage ()
        Error 0
    | "--all" :: rest ->
        parseArgs { options with All = true; Families = [ "all" ] } rest
    | "--family" :: value :: rest
    | "--families" :: value :: rest ->
        match ProbeSelection.parseFamilies value with
        | Some families -> parseArgs { options with Families = options.Families @ families } rest
        | None ->
            eprintfn "collect: --family expects io,io-cast,singleton,window-slab,neighbourhood,geometry,fourier,keypoints,dependency,reducers, or all"
            Error 2
    | "--member" :: value :: rest
    | "--members" :: value :: rest
    | "--operator" :: value :: rest
    | "--operators" :: value :: rest ->
        parseArgs { options with Members = options.Members @ ProbeSelection.splitCsvList value } rest
    | "--request" :: value :: rest
    | "--request-file" :: value :: rest ->
        parseArgs { options with RequestPath = Some(Path.GetFullPath value) } rest
    | option :: value :: rest when option.StartsWith("-", StringComparison.Ordinal) && not (value.StartsWith("-", StringComparison.Ordinal)) ->
        parseArgs { options with ExtraArgs = options.ExtraArgs @ [ option; value ] } rest
    | option :: rest when option.StartsWith("-", StringComparison.Ordinal) ->
        parseArgs { options with ExtraArgs = options.ExtraArgs @ [ option ] } rest
    | value :: _ ->
        eprintfn "collect: unknown positional argument '%s'" value
        usage ()
        Error 2

let private runFamilies (options: Options) =
    let families =
        if options.All || options.Families.IsEmpty then
            [ "all" ]
        else
            options.Families |> List.distinct

    let args =
        withNoFitDefault options.ExtraArgs
        @ [ "--phases"; String.concat "," families ]
        @ if options.Members.IsEmpty then
              []
          else
              [ "--members"; String.concat "," (options.Members |> List.distinct) ]

    ProbeBottomUpCalibration.main (args |> List.toArray)

let private runMembers (options: Options) =
    let args =
        withNoFitDefault options.ExtraArgs
        @ [ "--operators"; String.concat "," (options.Members |> List.distinct) ]

    ProbeLocalUpdate.main (args |> List.toArray)

let private readRequest (path: string) =
    let jsonOptions = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
    JsonSerializer.Deserialize<CollectionRequest>(File.ReadAllText path, jsonOptions)

let private runRequest (options: Options) (path: string) =
    let request = readRequest path
    printfn "collect request: %s" request.Reason

    let requestOptions =
        let extraArgs =
            options.ExtraArgs
            @ (request.ExtraArgs |> Array.toList)
            |> withRepeatDefault request.MinRepeats

        { options with
            Families = request.Families |> Array.toList
            Members = request.Members |> Array.toList
            ExtraArgs = extraArgs }

    if not requestOptions.Families.IsEmpty then
        runFamilies requestOptions
    elif not requestOptions.Members.IsEmpty then
        runMembers requestOptions
    else
        runFamilies requestOptions

let main argv =
    match parseArgs defaultOptions (Array.toList argv) with
    | Error exitCode -> exitCode
    | Ok options ->
        match options.RequestPath with
        | Some path ->
            runRequest options path
        | None when not options.Members.IsEmpty ->
            runMembers options
        | None ->
            runFamilies options
