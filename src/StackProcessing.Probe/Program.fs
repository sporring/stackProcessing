module StackProcessing.Probe

open System
open System.IO
open System.Text

let private usage () =
    printfn "Usage: dotnet run --project src/StackProcessing.Probe -- [--log PATH] <command> [args]"
    printfn ""
    printfn "Global options:"
    printfn "  --log PATH  Tee stdout and stderr to PATH while still printing to the console."
    printfn ""
    printfn "Commands:"
    printfn "  samples    Run sample F# projects, or Studio JSON graphs with --json."
    printfn "  analysis   Extract feature matrices, diagnostics, coefficients, and predictions."
    printfn "  collect    Collect durable measurement evidence for families or members."
    printfn "  fit        Fit a cost model from the durable measurement store."
    printfn "  inspect    Inspect stored evidence and suggest the next collection step."
    printfn "  climb      Run collect-fit-inspect ladder climbing family by family."
    printfn "  probing    Measure or emit probe graphs."
    printfn "  calibrate  Run the greedy analysis/freezing/probe-emission loop."
    printfn "  bottom-up  Run controlled bottom-up calibration probe layers."
    printfn "  local-update"
    printfn "             Emit targeted probes for flagged pipelines and write a local model overlay."
    printfn "  report     Render legacy probing HTML reports."
    printfn ""
    printfn "Examples:"
    printfn "  dotnet run --project src/StackProcessing.Probe -- samples --repeat 3"
    printfn "  dotnet run --project src/StackProcessing.Probe -- samples --json --extra-json-root tmp/probingGraphs"
    printfn "  dotnet run --project src/StackProcessing.Probe -- analysis --extra-json-root tmp/probingGraphs"
    printfn "  dotnet run --project src/StackProcessing.Probe -- collect --family io --repeat 6"
    printfn "  dotnet run --project src/StackProcessing.Probe -- fit --up-to io-cast"
    printfn "  dotnet run --project src/StackProcessing.Probe -- inspect --max-step singleton"
    printfn "  dotnet run --project src/StackProcessing.Probe -- climb --through singleton"
    printfn "  dotnet run --project src/StackProcessing.Probe -- probing tmp/analysis/probing-boilerplate.json --emit-json tmp/probingGraphs"
    printfn "  dotnet run --project src/StackProcessing.Probe -- bottom-up --repeat 3 -j 1"
    printfn "  dotnet run --project src/StackProcessing.Probe -- calibrate --repeat 3 -j 1"
    printfn "  dotnet run --project src/StackProcessing.Probe -- local-update --operators SmoothWGauss --sizes 128"

type private TeeTextWriter(primary: TextWriter, secondary: TextWriter) =
    inherit TextWriter()

    override _.Encoding: Encoding = primary.Encoding

    override _.Write(value: char) =
        primary.Write value
        secondary.Write value

    override _.Write(value: string) =
        primary.Write value
        secondary.Write value

    override _.Write(buffer: char array, index: int, count: int) =
        primary.Write(buffer, index, count)
        secondary.Write(buffer, index, count)

    override _.WriteLine() =
        primary.WriteLine()
        secondary.WriteLine()

    override _.WriteLine(value: string) =
        primary.WriteLine value
        secondary.WriteLine value

    override _.Flush() =
        primary.Flush()
        secondary.Flush()

let private extractLogPath (argv: string array) =
    let rec loop kept logPath args =
        match args with
        | [] -> Ok(logPath, kept |> List.rev |> List.toArray)
        | "--log" :: path :: rest -> loop kept (Some path) rest
        | "--log" :: [] ->
            eprintfn "probe: --log expects a path"
            Error 2
        | arg :: rest -> loop (arg :: kept) logPath rest

    loop [] None (Array.toList argv)

let private runWithOptionalLog logPath run =
    match logPath with
    | None -> run ()
    | Some path ->
        let fullPath = Path.GetFullPath path
        let directory = Path.GetDirectoryName fullPath

        if not (String.IsNullOrWhiteSpace directory) then
            Directory.CreateDirectory directory |> ignore

        let originalOut = Console.Out
        let originalError = Console.Error

        use writer = new StreamWriter(fullPath, false, Encoding.UTF8)
        writer.AutoFlush <- true

        use outTee = new TeeTextWriter(originalOut, writer)
        use errorTee = new TeeTextWriter(originalError, writer)

        Console.SetOut outTee
        Console.SetError errorTee

        try
            printfn "probe log: %s" fullPath
            run ()
        finally
            Console.Out.Flush()
            Console.Error.Flush()
            Console.SetOut originalOut
            Console.SetError originalError

let private dispatch command args =
    match command with
    | "samples"
    | "run-samples"
    | "runsamples" ->
        RunSamples.main args
    | "run-json"
    | "runjson"
    | "json" ->
        RunSamples.main (Array.append [| "--json" |] args)
    | "run-all"
    | "runall"
    | "run" ->
        RunSamples.main args
    | "analysis"
    | "analyze"
    | "analyse" ->
        ProbeAnalysis.main args
    | "collect"
    | "collection" ->
        ProbeCollect.main args
    | "fit"
    | "fit-model"
    | "model" ->
        ProbeFit.main args
    | "inspect"
    | "inspection"
    | "coverage" ->
        ProbeInspect.main args
    | "climb"
    | "climb-ladder"
    | "ladder-climb" ->
        ProbeLadderClimb.main args
    | "probing"
    | "probe"
    | "emit" ->
        ProbeProbing.main args
    | "calibrate"
    | "calibration"
    | "loop" ->
        ProbeCalibration.main args
    | "bottom-up"
    | "bottomup"
    | "ladder" ->
        ProbeBottomUpCalibration.main args
    | "local-update"
    | "localupdate"
    | "update-local"
    | "repair" ->
        ProbeLocalUpdate.main args
    | "report"
    | "reports" ->
        ProbeProbingReport.main args
    | "-h"
    | "--help"
    | "help" ->
        usage ()
        0
    | _ ->
        eprintfn "probe: unknown command '%s'" command
        usage ()
        2

[<EntryPoint>]
let main (argv: string array) =
    match extractLogPath argv with
    | Error exitCode -> exitCode
    | Ok(logPath, argv) ->
        runWithOptionalLog logPath (fun () ->
            match argv with
            | [||] ->
                usage ()
                2
            | _ ->
                dispatch argv[0] argv[1..])
