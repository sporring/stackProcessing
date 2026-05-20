module StackProcessing.Probe

open System

let private usage () =
    printfn "Usage: dotnet run --project src/StackProcessing.Probe -- <command> [args]"
    printfn ""
    printfn "Commands:"
    printfn "  samples    Run sample F# projects, or Studio JSON graphs with --json."
    printfn "  analysis   Extract feature matrices, diagnostics, coefficients, and predictions."
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
    printfn "  dotnet run --project src/StackProcessing.Probe -- probing tmp/analysis/probing-boilerplate.json --emit-json tmp/probingGraphs"
    printfn "  dotnet run --project src/StackProcessing.Probe -- bottom-up --repeat 3 -j 1"
    printfn "  dotnet run --project src/StackProcessing.Probe -- calibrate --repeat 3 -j 1"
    printfn "  dotnet run --project src/StackProcessing.Probe -- local-update --operators SmoothWGauss --sizes 128"

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
    match argv with
    | [||] ->
        usage ()
        2
    | _ ->
        dispatch argv[0] argv[1..]
