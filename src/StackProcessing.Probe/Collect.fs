module ProbeCollect

open System

type Options =
    { Families: string list
      Members: string list
      All: bool
      ExtraArgs: string list }

let private usage () =
    printfn "Usage: dotnet run --project src/StackProcessing.Probe -- collect [options]"
    printfn ""
    printfn "Collects Probe measurements for selected ladder families or members."
    printfn "Family collection delegates to the controlled probe ladder; member collection delegates to local-update."
    printfn ""
    printfn "Selection:"
    printfn "  --all                 Collect every ladder family."
    printfn "  --family LIST         Collect families, e.g. io,io-cast,singleton."
    printfn "  --member LIST         Collect member/operator probes, e.g. SmoothWGauss."
    printfn ""
    printfn "Common pass-through options:"
    printfn "  --shapes LIST --shape WxHxD --sizes LIST --repeat N -j N --keep-tmp --no-run-probes"

let private defaultOptions =
    { Families = []
      Members = []
      All = false
      ExtraArgs = [] }

let private hasFitChoice args =
    args |> List.exists (fun arg -> arg = "--fit" || arg = "--no-fit")

let private withNoFitDefault args =
    if hasFitChoice args then args else args @ [ "--no-fit" ]

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
            eprintfn "collect: --family expects io,io-cast,singleton,neighbourhood,geometry,fourier,keypoints,dependency,reducers, or all"
            Error 2
    | "--member" :: value :: rest
    | "--members" :: value :: rest
    | "--operator" :: value :: rest
    | "--operators" :: value :: rest ->
        parseArgs { options with Members = options.Members @ ProbeSelection.splitCsvList value } rest
    | option :: value :: rest when option.StartsWith("-", StringComparison.Ordinal) && not (value.StartsWith("-", StringComparison.Ordinal)) ->
        parseArgs { options with ExtraArgs = options.ExtraArgs @ [ option; value ] } rest
    | option :: rest when option.StartsWith("-", StringComparison.Ordinal) ->
        parseArgs { options with ExtraArgs = options.ExtraArgs @ [ option ] } rest
    | value :: _ ->
        eprintfn "collect: unknown positional argument '%s'" value
        usage ()
        Error 2

let private runFamilies options =
    let families =
        if options.All || options.Families.IsEmpty then
            [ "all" ]
        else
            options.Families |> List.distinct

    let args =
        withNoFitDefault options.ExtraArgs
        @ [ "--phases"; String.concat "," families ]

    ProbeBottomUpCalibration.main (args |> List.toArray)

let private runMembers options =
    let args =
        withNoFitDefault options.ExtraArgs
        @ [ "--operators"; String.concat "," (options.Members |> List.distinct) ]

    ProbeLocalUpdate.main (args |> List.toArray)

let main argv =
    match parseArgs defaultOptions (Array.toList argv) with
    | Error exitCode -> exitCode
    | Ok options ->
        if not options.Members.IsEmpty then
            runMembers options
        else
            runFamilies options
