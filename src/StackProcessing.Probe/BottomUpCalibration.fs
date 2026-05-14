module ProbeBottomUpCalibration

open System
open System.Globalization
open System.IO

type Options =
    { SamplesRoot: string
      AnalysisDirectory: string
      ProbeJsonRoot: string
      Repeat: int
      Jobs: int
      Layers: int
      RunProbes: bool }

let private usage () =
    printfn "Usage: dotnet run --project src/StackProcessing.Probe -- bottom-up [options]"
    printfn ""
    printfn "Runs controlled bottom-up calibration batches with optimizer off."
    printfn "The batches build from intercept/traversal/read/write probes toward sources"
    printfn "and one-more stage probes. Sample workloads are left for validation."
    printfn ""
    printfn "Options:"
    printfn "  --samples-root PATH     Sample root. Defaults to samples."
    printfn "  --analysis-dir PATH     Analysis output/read directory. Defaults to tmp/analysis."
    printfn "  --probe-json-root PATH  Probe JSON output directory. Defaults to tmp/probingGraphs."
    printfn "  --repeat N              Repeat emitted probe runs. Defaults to 3."
    printfn "  -j, --jobs N            Run up to N emitted probe graphs at once. Defaults to 1."
    printfn "  --layers N              Number of bottom-up layers to run. Defaults to all."
    printfn "  --no-run-probes         Emit probe graphs and analyze only."

let private defaultSamplesRoot () =
    let cwd = Directory.GetCurrentDirectory()

    if File.Exists(Path.Combine(cwd, "StackProcessing.sln")) then
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

let private timestamp () =
    DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture)

let private defaultOptions () =
    let samplesRoot = defaultSamplesRoot ()
    let root = repositoryRootFromSamplesRoot samplesRoot

    { SamplesRoot = samplesRoot
      AnalysisDirectory = Path.Combine(root, "tmp", "analysis")
      ProbeJsonRoot = Path.Combine(root, "tmp", "probingGraphs")
      Repeat = 3
      Jobs = 1
      Layers = Int32.MaxValue
      RunProbes = true }

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
    | "--repeat" :: value :: rest
    | "--repeats" :: value :: rest ->
        match Int32.TryParse value with
        | true, n when n > 0 -> parseArgs { options with Repeat = n } rest
        | _ ->
            eprintfn "bottom-up: --repeat expects a positive integer"
            Error 2
    | ("-j" | "--jobs") :: value :: rest ->
        match Int32.TryParse value with
        | true, n when n > 0 -> parseArgs { options with Jobs = n } rest
        | _ ->
            eprintfn "bottom-up: -j/--jobs expects a positive integer"
            Error 2
    | "--layers" :: value :: rest ->
        match Int32.TryParse value with
        | true, n when n > 0 -> parseArgs { options with Layers = n } rest
        | _ ->
            eprintfn "bottom-up: --layers expects a positive integer"
            Error 2
    | "--run-probes" :: rest ->
        parseArgs { options with RunProbes = true } rest
    | "--no-run-probes" :: rest
    | "--emit-only" :: rest ->
        parseArgs { options with RunProbes = false } rest
    | value :: _ ->
        eprintfn "bottom-up: unknown argument '%s'" value
        usage ()
        Error 2

let private runAnalysis options probeRoot =
    ProbeAnalysis.main
        [| "--samples-root"
           options.SamplesRoot
           "--output"
           options.AnalysisDirectory
           "--extra-json-root"
           probeRoot
           "--no-samples" |]

let private runProbeGraphs options layerDir =
    RunSamples.main
        [| options.SamplesRoot
           "--json"
           "--extra-json-root"
           layerDir
           "--extra-json-only"
           "--optimize"
           "false"
           "--repeat"
           string options.Repeat
           "-j"
           string options.Jobs |]

let private writePlan (path: string) (layers: (string * ProbeProbing.GraphTemplate array) array) =
    Directory.CreateDirectory(Path.GetDirectoryName path) |> ignore

    let lines =
        seq {
            yield "layer,graph"
            for layerName, templates in layers do
                for template in templates do
                    yield $"{layerName},{template.Name}"
        }

    File.WriteAllLines(path, lines)

let main argv =
    match parseArgs (defaultOptions ()) (Array.toList argv) with
    | Error exitCode -> exitCode
    | Ok options ->
        try
            let allLayers = ProbeProbing.graphTemplateLayersForBottomUp ()
            let selectedLayers = allLayers |> Array.truncate options.Layers
            let probeRoot = Path.Combine(options.ProbeJsonRoot, "bottomup_" + timestamp ())

            printfn "Bottom-up probe root: %s" probeRoot
            writePlan (Path.Combine(probeRoot, "bottomUpPlan.csv")) selectedLayers

            let mutable exitCode = 0

            for layerIndex, (layerName, templates) in selectedLayers |> Array.indexed do
                if exitCode = 0 then
                    let layerDir = Path.Combine(probeRoot, sprintf "layer_%03d_%s" (layerIndex + 1) layerName)
                    printfn "Bottom-up layer %d/%d: %s (%d graph(s))" (layerIndex + 1) selectedLayers.Length layerName templates.Length
                    ProbeProbing.writeGraphTemplates layerDir templates

                    if options.RunProbes then
                        let runExit = runProbeGraphs options layerDir
                        if runExit <> 0 then
                            eprintfn "bottom-up probe graph run failed with exit code %d" runExit
                            exitCode <- runExit

                    if exitCode = 0 then
                        let analysisExit = runAnalysis options probeRoot
                        if analysisExit <> 0 then
                            eprintfn "bottom-up analysis failed with exit code %d" analysisExit
                            exitCode <- analysisExit

            if exitCode = 0 then
                printfn "Bottom-up calibration pass complete. Analysis written to %s." options.AnalysisDirectory

            exitCode
        with ex ->
            eprintfn "bottom-up failed: %s" ex.Message
            1
