module ProbeBottomUpCalibration

open System
open System.Globalization
open System.IO

type Options =
    { SamplesRoot: string
      AnalysisDirectory: string
      ProbeJsonRoot: string
      InputDirectory: string
      ImageSizes: uint list
      NoisyType: string
      Repeat: int
      Jobs: int
      Layers: int
      CleanTmp: bool
      RunProbes: bool }

let private usage () =
    printfn "Usage: dotnet run --project src/StackProcessing.Probe -- bottom-up [options]"
    printfn ""
    printfn "Runs a fresh controlled bottom-up calibration with optimizer off."
    printfn "The command clears repository tmp by default, generates calibration input"
    printfn "stacks, emits probe graphs, measures them, and writes analysis CSVs."
    printfn ""
    printfn "Options:"
    printfn "  --samples-root PATH     Sample root. Defaults to samples."
    printfn "  --analysis-dir PATH     Analysis output/read directory. Defaults to tmp/analysis."
    printfn "  --probe-json-root PATH  Probe JSON output directory. Defaults to tmp/probingGraphs."
    printfn "  --input-dir PATH        Generated calibration input root. Defaults to tmp/probeInputs."
    printfn "  --size N                Width/height/depth of generated probe images. Defaults to 64."
    printfn "  --sizes A,B,C           Run the same probe layers for multiple cubic image sizes."
    printfn "  --noisy-type TYPE       TIFF-compatible noisy image type: UInt8, UInt16, or Float32. Defaults to Float32."
    printfn "  --repeat N              Repeat emitted probe runs. Defaults to 3."
    printfn "  -j, --jobs N            Run up to N emitted probe graphs at once. Defaults to 1."
    printfn "  --layers N              Number of bottom-up layers to run. Defaults to all."
    printfn "  --keep-tmp              Do not clear repository tmp before starting."
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
      InputDirectory = Path.Combine(root, "tmp", "probeInputs")
      ImageSizes = [ 64u ]
      NoisyType = "Float32"
      Repeat = 3
      Jobs = 1
      Layers = Int32.MaxValue
      CleanTmp = true
      RunProbes = true }

let private normalizeNoisyType (value: string) =
    match value.ToLowerInvariant() with
    | "uint8" -> Some "UInt8"
    | "uint16" -> Some "UInt16"
    | "float32" | "float" -> Some "Float32"
    | _ -> None

let private parseSizes (value: string) =
    let parts =
        value.Split([| ',' |], StringSplitOptions.RemoveEmptyEntries)
        |> Array.map _.Trim()

    if parts.Length = 0 then
        None
    else
        let parsed =
            parts
            |> Array.map UInt32.TryParse

        if parsed |> Array.forall (fun (ok, n) -> ok && n > 0u) then
            parsed
            |> Array.map snd
            |> Array.distinct
            |> Array.sort
            |> Array.toList
            |> Some
        else
            None

let rec private parseArgs options args =
    match args with
    | [] -> Ok options
    | ("-h" | "--help") :: _ ->
        usage ()
        Error 0
    | "--samples-root" :: value :: rest ->
        let samplesRoot = Path.GetFullPath value
        let root = repositoryRootFromSamplesRoot samplesRoot
        let defaults = defaultOptions ()
        parseArgs
            { options with
                SamplesRoot = samplesRoot
                AnalysisDirectory =
                    if options.AnalysisDirectory = defaults.AnalysisDirectory then
                        Path.Combine(root, "tmp", "analysis")
                    else
                        options.AnalysisDirectory
                ProbeJsonRoot =
                    if options.ProbeJsonRoot = defaults.ProbeJsonRoot then
                        Path.Combine(root, "tmp", "probingGraphs")
                    else
                        options.ProbeJsonRoot
                InputDirectory =
                    if options.InputDirectory = defaults.InputDirectory then
                        Path.Combine(root, "tmp", "probeInputs")
                    else
                        options.InputDirectory }
            rest
    | "--analysis-dir" :: value :: rest
    | "--output" :: value :: rest ->
        parseArgs { options with AnalysisDirectory = Path.GetFullPath value } rest
    | "--probe-json-root" :: value :: rest
    | "--emit-json" :: value :: rest ->
        parseArgs { options with ProbeJsonRoot = Path.GetFullPath value } rest
    | "--input-dir" :: value :: rest ->
        parseArgs { options with InputDirectory = Path.GetFullPath value } rest
    | "--size" :: value :: rest
    | "--image-size" :: value :: rest ->
        match UInt32.TryParse value with
        | true, n when n > 0u -> parseArgs { options with ImageSizes = [ n ] } rest
        | _ ->
            eprintfn "bottom-up: --size expects a positive integer"
            Error 2
    | "--sizes" :: value :: rest
    | "--image-sizes" :: value :: rest ->
        match parseSizes value with
        | Some sizes -> parseArgs { options with ImageSizes = sizes } rest
        | None ->
            eprintfn "bottom-up: --sizes expects a comma-separated list of positive integers, for example 64,128,256"
            Error 2
    | "--noisy-type" :: value :: rest
    | "--gray-type" :: value :: rest ->
        match normalizeNoisyType value with
        | Some noisyType -> parseArgs { options with NoisyType = noisyType } rest
        | None ->
            eprintfn "bottom-up: --noisy-type expects UInt8, UInt16, or Float32"
            Error 2
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
    | "--keep-tmp" :: rest ->
        parseArgs { options with CleanTmp = false } rest
    | "--run-probes" :: rest ->
        parseArgs { options with RunProbes = true } rest
    | "--no-run-probes" :: rest
    | "--emit-only" :: rest ->
        parseArgs { options with RunProbes = false } rest
    | value :: _ ->
        eprintfn "bottom-up: unknown argument '%s'" value
        usage ()
        Error 2

let private cleanTmp options =
    let root = repositoryRootFromSamplesRoot options.SamplesRoot
    let tmpRoot = Path.Combine(root, "tmp")

    if options.CleanTmp && Directory.Exists tmpRoot then
        Directory.Delete(tmpRoot, true)

    Directory.CreateDirectory tmpRoot |> ignore

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

let private writePlan (path: string) (layersBySize: (uint * (string * ProbeProbing.GraphTemplate array) array) array) =
    Directory.CreateDirectory(Path.GetDirectoryName path) |> ignore

    let lines =
        seq {
            yield "size,layer,graph"
            for size, layers in layersBySize do
                for layerName, templates in layers do
                    for template in templates do
                        yield $"{size},{layerName},{template.Name}"
        }

    File.WriteAllLines(path, lines)

let main argv =
    match parseArgs (defaultOptions ()) (Array.toList argv) with
    | Error exitCode -> exitCode
    | Ok options ->
        try
            cleanTmp options

            let probeRoot = Path.Combine(options.ProbeJsonRoot, "bottomup_" + timestamp ())
            let layersBySize =
                options.ImageSizes
                |> List.map (fun size ->
                    let inputDir = Path.Combine(options.InputDirectory, sprintf "%ux%ux%u" size size size)

                    printfn
                        "Generating calibration inputs in %s (%ux%ux%u, noisy %s)."
                        inputDir
                        size
                        size
                        size
                        options.NoisyType

                    let inputConfig = ProbeProbing.createBottomUpInputs size options.NoisyType inputDir
                    let selectedLayers =
                        ProbeProbing.graphTemplateLayersForBottomUp inputConfig
                        |> Array.truncate options.Layers

                    size, selectedLayers)
                |> List.toArray

            printfn "Bottom-up probe root: %s" probeRoot
            writePlan (Path.Combine(probeRoot, "bottomUpPlan.csv")) layersBySize

            let mutable exitCode = 0

            for size, selectedLayers in layersBySize do
                for layerIndex, (layerName, templates) in selectedLayers |> Array.indexed do
                    if exitCode = 0 then
                        let layerNumber = layerIndex + 1
                        let layerDir =
                            Path.Combine(
                                probeRoot,
                                sprintf "size_%ux%ux%u" size size size,
                                sprintf "layer_%03d_%s" layerNumber layerName)

                        printfn
                            "Bottom-up size %ux%ux%u layer %d/%d: %s (%d graph(s))"
                            size
                            size
                            size
                            layerNumber
                            selectedLayers.Length
                            layerName
                            templates.Length

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
