module RunSamples

open System
open System.IO

let private usage () =
    printfn "Usage: dotnet run --project src/StackProcessing.RunSamples -- [DIRECTORY] [options]"
    printfn ""
    printfn "Runs sample pipelines below DIRECTORY. DIRECTORY defaults to samples."
    printfn "By default, runs F# sample projects. Use --json to compile and run Studio JSON graphs."
    printfn ""
    printfn "Common options:"
    printfn "  --json              Run Studio JSON graphs instead of F# sample projects."
    printfn "  --samples-root PATH Explicitly set the sample root directory."
    printfn "  -j, --jobs N        Run up to N samples at once."
    printfn "  --repeat N          Run the full set N times."
    printfn "  --help              Show this help."
    printfn ""
    printfn "Other options are forwarded to the selected runner."

let private isOption (value: string) =
    value.StartsWith("-", StringComparison.Ordinal)

let private optionTakesValue value =
    match value with
    | "-j"
    | "--jobs"
    | "--debug-level"
    | "--optimize"
    | "--optimizer"
    | "--repeat"
    | "--repeats"
    | "--run-id"
    | "--timeout"
    | "--timeout-minutes"
    | "--cost-model"
    | "--samples-root"
    | "--extra-json-root"
    | "--extra-samples-root"
    | "--generated-samples-root"
    | "--probe-samples-root" -> true
    | _ -> false

let private normalizeArgs (argv: string array) =
    let mutable json = false
    let mutable samplesRoot: string option = None
    let forwarded = ResizeArray<string>()
    let mutable i = 0

    while i < argv.Length do
        match argv[i] with
        | "--json" ->
            json <- true
            i <- i + 1
        | "--samples-root" when i + 1 < argv.Length ->
            samplesRoot <- Some argv[i + 1]
            forwarded.Add argv[i]
            forwarded.Add argv[i + 1]
            i <- i + 2
        | value when optionTakesValue value && i + 1 < argv.Length ->
            forwarded.Add value
            forwarded.Add argv[i + 1]
            i <- i + 2
        | "-h"
        | "--help" ->
            forwarded.Add argv[i]
            i <- i + 1
        | value when samplesRoot.IsNone && not (isOption value) ->
            samplesRoot <- Some value
            forwarded.Add "--samples-root"
            forwarded.Add value
            i <- i + 1
        | value ->
            forwarded.Add value
            i <- i + 1

    if samplesRoot.IsNone && not (forwarded |> Seq.exists ((=) "--samples-root")) then
        forwarded.Insert(0, "samples")
        forwarded.Insert(0, "--samples-root")

    json, forwarded.ToArray()

[<EntryPoint>]
let main (argv: string array) =
    if argv |> Array.exists (fun arg -> arg = "--run-samples-help") then
        usage ()
        0
    else
        let json, args = normalizeArgs argv

        if args.Length > 0 && (args[0] = "-h" || args[0] = "--help") then
            usage ()
            printfn ""

        if json then
            RunJson.main args
        else
            RunAll.main args
