module ProbeLadderClimb

open System
open System.Globalization
open System.IO
open System.Text.Json

type Options =
    { Families: string list
      Shapes: string list
      Repeat: int
      Jobs: int
      NoisyType: string
      MinRepeats: int
      MaxRequestRounds: int
      MeasurementStorePath: string
      ModelOutputPath: string
      OutputDirectory: string
      MinTimeR2: float option
      MaxFlaggedRatio: float option
      StopOnPlateau: bool
      Cleanup: bool
      ExtraCollectArgs: string list }

let private usage () =
    printfn "Usage: dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll climb [options]"
    printfn ""
    printfn "Runs the collect-fit-inspect ladder protocol family by family."
    printfn ""
    printfn "Options:"
    printfn "  --families LIST          Explicit families to climb."
    printfn "  --from FAMILY            Start at FAMILY in the implicit ladder."
    printfn "  --through FAMILY         Climb through FAMILY in the implicit ladder."
    printfn "  --include-window-slab    Include the experimental window-slab family."
    printfn "  --shapes LIST            Shape scope. Defaults to 256x256x256,512x512x128,1024x1024x64."
    printfn "  --repeat N               Initial family collection repeats. Defaults to 6."
    printfn "  --min-repeats N          Inspect repeat gate. Defaults to --repeat."
    printfn "  -j N                     Collection parallelism. Defaults to 1."
    printfn "  --noisy-type TYPE        Probe noisy input type. Defaults to Float32."
    printfn "  --max-request-rounds N   Max inspect-request collection rounds per family. Defaults to 3."
    printfn "  --measurement-store PATH Measurement JSONL store."
    printfn "  --model-output PATH      Fitted model path."
    printfn "  --output PATH            Climb output directory. Defaults to tmp/climb."
    printfn "  --min-time-r2 VALUE      Pass-through inspect R2 gate."
    printfn "  --max-flagged-ratio V    Pass-through inspect flagged-ratio gate."
    printfn "  --stop-on-plateau        Stop instead of moving to next family after plateau."
    printfn "  --no-cleanup             Do not delete generated scratch between families."
    printfn "  --collect-arg ARG        Append one extra argument to family collect."

let private repositoryRoot () =
    let cwd = Directory.GetCurrentDirectory()
    if File.Exists(Path.Combine(cwd, "StackProcessing.sln")) then cwd else cwd

let private defaultShapes =
    [ "256x256x256"; "512x512x128"; "1024x1024x64" ]

let private defaultOptions () =
    let root = repositoryRoot ()
    { Families = ProbeSelection.implicitLadder
      Shapes = defaultShapes
      Repeat = 6
      Jobs = 1
      NoisyType = "Float32"
      MinRepeats = 6
      MaxRequestRounds = 3
      MeasurementStorePath = Path.Combine(root, "measurements", "stackprocessing-probe.jsonl")
      ModelOutputPath = Path.Combine(root, "models", "fitted", "stackprocessing.operator-cost.json")
      OutputDirectory = Path.Combine(root, "tmp", "climb")
      MinTimeR2 = None
      MaxFlaggedRatio = None
      StopOnPlateau = false
      Cleanup = true
      ExtraCollectArgs = [] }

let private familiesFrom fromFamily throughFamily includeWindowSlab =
    let ladder =
        if includeWindowSlab then ProbeSelection.ladder else ProbeSelection.implicitLadder

    let startIndex =
        fromFamily
        |> Option.bind ProbeSelection.normalizeFamily
        |> Option.bind (fun family -> ladder |> List.tryFindIndex ((=) family))
        |> Option.defaultValue 0

    let endIndex =
        throughFamily
        |> Option.bind ProbeSelection.normalizeFamily
        |> Option.bind (fun family -> ladder |> List.tryFindIndex ((=) family))
        |> Option.defaultValue (ladder.Length - 1)

    if startIndex > endIndex then
        []
    else
        ladder |> List.skip startIndex |> List.take (endIndex - startIndex + 1)

let rec private parseArgs (options: Options) (fromFamily: string option) (throughFamily: string option) includeWindowSlab args =
    match args with
    | [] ->
        let families =
            if options.Families = ProbeSelection.implicitLadder && (fromFamily.IsSome || throughFamily.IsSome || includeWindowSlab) then
                familiesFrom fromFamily throughFamily includeWindowSlab
            else
                options.Families

        Ok { options with Families = families; MinRepeats = if options.MinRepeats <= 0 then options.Repeat else options.MinRepeats }
    | ("-h" | "--help") :: _ ->
        usage ()
        Error 0
    | "--families" :: value :: rest
    | "--family" :: value :: rest ->
        match ProbeSelection.parseFamilies value with
        | Some families -> parseArgs { options with Families = families |> List.filter ((<>) "all") } fromFamily throughFamily includeWindowSlab rest
        | None ->
            eprintfn "climb: --families expects known ladder families"
            Error 2
    | "--from" :: value :: rest ->
        match ProbeSelection.normalizeFamily value with
        | Some family when family <> "all" -> parseArgs options (Some family) throughFamily includeWindowSlab rest
        | _ ->
            eprintfn "climb: --from expects a concrete ladder family"
            Error 2
    | "--through" :: value :: rest
    | "--up-to" :: value :: rest ->
        match ProbeSelection.normalizeFamily value with
        | Some family when family <> "all" -> parseArgs options fromFamily (Some family) includeWindowSlab rest
        | _ ->
            eprintfn "climb: --through expects a concrete ladder family"
            Error 2
    | "--include-window-slab" :: rest ->
        parseArgs options fromFamily throughFamily true rest
    | "--shapes" :: value :: rest ->
        match ProbeSelection.parseShapes value with
        | Some shapes -> parseArgs { options with Shapes = shapes } fromFamily throughFamily includeWindowSlab rest
        | None ->
            eprintfn "climb: --shapes expects comma-separated shapes"
            Error 2
    | "--shape" :: value :: rest ->
        match ProbeSelection.normalizeShape value with
        | Some shape -> parseArgs { options with Shapes = [ shape ] } fromFamily throughFamily includeWindowSlab rest
        | None ->
            eprintfn "climb: --shape expects WxHxD"
            Error 2
    | "--repeat" :: value :: rest
    | "--repeats" :: value :: rest ->
        match Int32.TryParse value with
        | true, repeat when repeat > 0 ->
            let minRepeats =
                if options.MinRepeats = options.Repeat then repeat else options.MinRepeats
            parseArgs { options with Repeat = repeat; MinRepeats = minRepeats } fromFamily throughFamily includeWindowSlab rest
        | _ ->
            eprintfn "climb: --repeat expects a positive integer"
            Error 2
    | "--min-repeats" :: value :: rest ->
        match Int32.TryParse value with
        | true, minRepeats when minRepeats > 0 ->
            parseArgs { options with MinRepeats = minRepeats } fromFamily throughFamily includeWindowSlab rest
        | _ ->
            eprintfn "climb: --min-repeats expects a positive integer"
            Error 2
    | "-j" :: value :: rest
    | "--jobs" :: value :: rest ->
        match Int32.TryParse value with
        | true, jobs when jobs > 0 -> parseArgs { options with Jobs = jobs } fromFamily throughFamily includeWindowSlab rest
        | _ ->
            eprintfn "climb: -j expects a positive integer"
            Error 2
    | "--noisy-type" :: value :: rest ->
        parseArgs { options with NoisyType = value } fromFamily throughFamily includeWindowSlab rest
    | "--max-request-rounds" :: value :: rest
    | "--max-iterations" :: value :: rest ->
        match Int32.TryParse value with
        | true, rounds when rounds >= 0 ->
            parseArgs { options with MaxRequestRounds = rounds } fromFamily throughFamily includeWindowSlab rest
        | _ ->
            eprintfn "climb: --max-request-rounds expects a non-negative integer"
            Error 2
    | "--measurement-store" :: value :: rest ->
        parseArgs { options with MeasurementStorePath = Path.GetFullPath value } fromFamily throughFamily includeWindowSlab rest
    | "--model-output" :: value :: rest ->
        parseArgs { options with ModelOutputPath = Path.GetFullPath value } fromFamily throughFamily includeWindowSlab rest
    | "--output" :: value :: rest ->
        parseArgs { options with OutputDirectory = Path.GetFullPath value } fromFamily throughFamily includeWindowSlab rest
    | "--min-time-r2" :: value :: rest ->
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, parsed -> parseArgs { options with MinTimeR2 = Some parsed } fromFamily throughFamily includeWindowSlab rest
        | _ ->
            eprintfn "climb: --min-time-r2 expects a floating-point value"
            Error 2
    | "--max-flagged-ratio" :: value :: rest ->
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, parsed when parsed >= 0.0 -> parseArgs { options with MaxFlaggedRatio = Some parsed } fromFamily throughFamily includeWindowSlab rest
        | _ ->
            eprintfn "climb: --max-flagged-ratio expects a non-negative floating-point value"
            Error 2
    | "--stop-on-plateau" :: rest ->
        parseArgs { options with StopOnPlateau = true } fromFamily throughFamily includeWindowSlab rest
    | "--no-cleanup" :: rest ->
        parseArgs { options with Cleanup = false } fromFamily throughFamily includeWindowSlab rest
    | "--collect-arg" :: value :: rest ->
        parseArgs { options with ExtraCollectArgs = options.ExtraCollectArgs @ [ value ] } fromFamily throughFamily includeWindowSlab rest
    | option :: _ ->
        eprintfn "climb: unknown option %s" option
        usage ()
        Error 2

let private runStep label command args =
    printfn ""
    printfn "=== %s: %s %s" label command (String.concat " " args)
    let exitCode =
        match command with
        | "collect" -> ProbeCollect.main (args |> List.toArray)
        | "fit" -> ProbeFit.main (args |> List.toArray)
        | "inspect" -> ProbeInspect.main (args |> List.toArray)
        | _ -> 2

    if exitCode <> 0 then
        failwith $"{command} failed with exit code {exitCode}."

let private shapesArgs options =
    match options.Shapes with
    | [] -> []
    | shapes -> [ "--shapes"; String.concat "," shapes ]

let private familyShapesArgs family options =
    if family = "empty" then
        [ "--shape"; "64x64x64" ]
    else
        shapesArgs options

let private optionalInspectArgs options =
    [ match options.MinTimeR2 with
      | Some value -> yield "--min-time-r2"; yield value.ToString("G17", CultureInfo.InvariantCulture)
      | None -> ()
      match options.MaxFlaggedRatio with
      | Some value -> yield "--max-flagged-ratio"; yield value.ToString("G17", CultureInfo.InvariantCulture)
      | None -> () ]

let private requestSignature (request: ProbeCollect.CollectionRequest) =
    [ "families=" + String.concat "," (request.Families |> Array.sort)
      "members=" + String.concat "," (request.Members |> Array.sort)
      "extra=" + String.concat " " request.ExtraArgs
      "reason=" + request.Reason ]
    |> String.concat "|"

let private tryReadRequest path =
    if File.Exists path then
        let jsonOptions = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
        try
            JsonSerializer.Deserialize<ProbeCollect.CollectionRequest>(File.ReadAllText path, jsonOptions)
            |> Option.ofObj
        with _ ->
            None
    else
        None

let private cleanupTmp root =
    let tmp = Path.Combine(root, "tmp")
    let deleteDirectory path =
        if Directory.Exists path then
            Directory.Delete(path, true)

    if Directory.Exists tmp then
        Directory.GetDirectories(tmp, "runJson_*")
        |> Array.iter deleteDirectory

    deleteDirectory (Path.Combine(tmp, "probingGraphs"))
    deleteDirectory (Path.Combine(tmp, "probeInputs"))
    deleteDirectory (Path.Combine(tmp, "analysis"))

let private climbFamily root options family =
    printfn ""
    printfn "############################################################"
    printfn "climb family: %s" family
    printfn "############################################################"

    let requestPath = Path.Combine(options.OutputDirectory, $"{family}-request.json")
    if File.Exists requestPath then File.Delete requestPath

    runStep
        family
        "collect"
        ([ "--family"; family
           "--repeat"; string options.Repeat
           "-j"; string options.Jobs
           "--noisy-type"; options.NoisyType ]
         @ familyShapesArgs family options
         @ options.ExtraCollectArgs)

    let rec loop requestRounds previousSignature repeatedCount =
        if File.Exists requestPath then File.Delete requestPath

        runStep
            family
            "fit"
            ([ "--up-to"; family
               "--measurement-store"; options.MeasurementStorePath
               "--model-output"; options.ModelOutputPath ]
             @ shapesArgs options)

        runStep
            family
            "inspect"
            ([ "--max-step"; family
               "--min-repeats"; string options.MinRepeats
               "--measurement-store"; options.MeasurementStorePath
               "--suggest"; requestPath ]
             @ shapesArgs options
             @ optionalInspectArgs options)

        match tryReadRequest requestPath with
        | None ->
            printfn "climb family accepted: %s" family
            "accepted"
        | Some request ->
            let signature = requestSignature request
            let repeatedCount =
                match previousSignature with
                | Some previous when previous = signature -> repeatedCount + 1
                | _ -> 0

            if requestRounds >= options.MaxRequestRounds then
                printfn "climb family plateau: %s reached max request rounds (%d)." family options.MaxRequestRounds
                "plateau:max-rounds"
            elif repeatedCount >= 1 then
                printfn "climb family plateau: %s repeated the same request signature." family
                "plateau:repeated-request"
            else
                printfn "climb request round %d/%d for %s: %s" (requestRounds + 1) options.MaxRequestRounds family request.Reason
                runStep
                    family
                    "collect"
                    ([ "--request"; requestPath; "-j"; string options.Jobs ]
                     @ familyShapesArgs family options)
                loop (requestRounds + 1) (Some signature) repeatedCount

    let status = loop 0 None 0

    if options.Cleanup then
        cleanupTmp root
        printfn "cleaned generated probe scratch below tmp/ for family %s." family

    status

let main argv =
    match parseArgs (defaultOptions ()) None None false (Array.toList argv) with
    | Error exitCode -> exitCode
    | Ok options ->
        try
            let root = repositoryRoot ()
            Directory.CreateDirectory options.OutputDirectory |> ignore
            Directory.CreateDirectory(Path.GetDirectoryName options.ModelOutputPath) |> ignore

            printfn "probe ladder climb"
            printfn "families: %s" (String.concat "," options.Families)
            printfn "shapes: %s" (String.concat "," options.Shapes)
            printfn "repeat=%d minRepeats=%d maxRequestRounds=%d jobs=%d" options.Repeat options.MinRepeats options.MaxRequestRounds options.Jobs

            let mutable failed = false
            for family in options.Families do
                if not failed then
                    let status = climbFamily root options family
                    if status.StartsWith("plateau", StringComparison.Ordinal) && options.StopOnPlateau then
                        failed <- true

            if failed then 1 else 0
        with ex ->
            eprintfn "climb failed: %s" ex.Message
            1
