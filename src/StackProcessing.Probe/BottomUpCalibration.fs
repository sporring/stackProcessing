module ProbeBottomUpCalibration

open System
open System.Globalization
open System.IO

type Options =
    { SamplesRoot: string
      AnalysisDirectory: string
      ProbeJsonRoot: string
      InputDirectory: string
      ImageShapes: ProbeProbing.ImageSize list
      NoisyType: string
      Repeat: int
      Jobs: int
      Layers: int
      Phases: string list
      Members: string list
      CleanTmp: bool
      RunProbes: bool
      FitModel: bool }

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
    printfn "  --size N                Width/height/depth of generated cubic probe images. Defaults to 64."
    printfn "  --sizes A,B,C           Run the same probe layers for multiple cubic image sizes."
    printfn "  --shape WxHxD           Run the probe layers for one rectangular image shape."
    printfn "  --shapes LIST           Comma-separated shapes, e.g. 256x256x256,512x512x128,1024x1024x64."
    printfn "  --noisy-type TYPE       TIFF-compatible noisy image type: UInt8, UInt16, or Float32. Defaults to Float32."
    printfn "  --repeat N              Repeat emitted probe runs. Defaults to 3."
    printfn "  -j, --jobs N            Run up to N emitted probe graphs at once. Defaults to 1."
    printfn "  --layers N              Number of bottom-up layers to run. Defaults to all."
    printfn "  --phase NAME            Probe phase: io, io-cast, sources, singleton, window-slab, neighbourhood, geometry, fourier, keypoints, dependency, reducers, or all."
    printfn "  --phases LIST           Comma-separated phases. Defaults to all."
    printfn "  --member LIST           Restrict generated probe graphs by graph/member/operator name."
    printfn "  --keep-tmp              Do not clear repository tmp before starting."
    printfn "                          Generated input stacks are still removed after each measured shape."
    printfn "  --no-run-probes         Emit probe graphs and analyze only."
    printfn "  --no-fit                Append evidence/measurements without fitting a model."

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
      ImageShapes = [ { Width = 64u; Height = 64u; Depth = 64u } ]
      NoisyType = "Float32"
      Repeat = 3
      Jobs = 1
      Layers = Int32.MaxValue
      Phases = [ "all" ]
      Members = []
      CleanTmp = true
      RunProbes = true
      FitModel = true }

let private normalizeNoisyType (value: string) =
    match value.ToLowerInvariant() with
    | "uint8" -> Some "UInt8"
    | "uint16" -> Some "UInt16"
    | "float32" | "float" -> Some "Float32"
    | _ -> None

let private splitCsvList (value: string) =
    value.Split([| ','; ';' |], StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
    |> Array.toList

let private parseSizes (value: string) =
    let parts =
        splitCsvList value
        |> List.toArray

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

let private mkShape width height depth : ProbeProbing.ImageSize =
    { Width = width
      Height = height
      Depth = depth }

let private cubeShape size =
    mkShape size size size

let private tryParsePositiveUInt (text: string) =
    match UInt32.TryParse text with
    | true, value when value > 0u -> Some value
    | _ -> None

let private tryParseShape (value: string) =
    let text = value.Trim().ToLowerInvariant().Replace(" ", "")

    let parts (separator: char) =
        text.Split([| separator |], StringSplitOptions.RemoveEmptyEntries)
        |> Array.toList

    match parts 'x' with
    | [ w; h; d ] ->
        match tryParsePositiveUInt w, tryParsePositiveUInt h, tryParsePositiveUInt d with
        | Some width, Some height, Some depth -> Some(mkShape width height depth)
        | _ -> None
    | _ ->
        match parts '*' with
        | [ xyPower; d ] when xyPower.EndsWith("^2", StringComparison.Ordinal) ->
            let sideText = xyPower.Substring(0, xyPower.Length - 2)
            match tryParsePositiveUInt sideText, tryParsePositiveUInt d with
            | Some side, Some depth -> Some(mkShape side side depth)
            | _ -> None
        | [ w; h; d ] ->
            match tryParsePositiveUInt w, tryParsePositiveUInt h, tryParsePositiveUInt d with
            | Some width, Some height, Some depth -> Some(mkShape width height depth)
            | _ -> None
        | _ when text.EndsWith("^3", StringComparison.Ordinal) ->
            text.Substring(0, text.Length - 2)
            |> tryParsePositiveUInt
            |> Option.map cubeShape
        | _ ->
            None

let private parseShapes value =
    let tokens = splitCsvList value
    if tokens.IsEmpty then
        None
    else
        let parsed = tokens |> List.map tryParseShape
        if parsed |> List.forall Option.isSome then
            parsed
            |> List.choose id
            |> List.distinct
            |> Some
        else
            None

let private shapeName (shape: ProbeProbing.ImageSize) =
    $"{shape.Width}x{shape.Height}x{shape.Depth}"

let private phaseForLayer layerName =
    match layerName with
    | "01-starters" -> "io"
    | "02-io-casts" -> "io-cast"
    | "02-sources" -> "sources"
    | "03-simple-unary" -> "singleton"
    | "04-window-slab" -> "window-slab"
    | "04-windowed-unary" -> "neighbourhood"
    | "05-intensity-and-additive" -> "singleton"
    | "06-geometry-and-projection" -> "geometry"
    | "07-fourier-and-vector" -> "fourier"
    | "08-keypoint-and-distance" -> "keypoints"
    | "09-dependency-breakers" -> "dependency"
    | "10-reducers" -> "reducers"
    | _ -> "other"

let private normalizePhase (value: string) =
    match value.Trim().ToLowerInvariant().Replace("_", "-") with
    | "all" -> Some "all"
    | "io" | "read-write" | "readwrite" -> Some "io"
    | "io-cast" | "io-casts" | "read-cast" | "readcast" | "conversion" | "conversions" -> Some "io-cast"
    | "sources" | "source" -> Some "sources"
    | "singleton" | "singletons" | "simple" | "simple-unary" -> Some "singleton"
    | "window-slab" | "windowslab" | "slab" | "slabs" | "z-agnostic-slab" | "zagnostic-slab" -> Some "window-slab"
    | "neighbourhood" | "neighborhood" | "window" | "windowed" | "windowed-unary" -> Some "neighbourhood"
    | "geometry" | "projection" | "geometry-and-projection" -> Some "geometry"
    | "fourier" | "vector" | "fourier-and-vector" -> Some "fourier"
    | "keypoints" | "keypoint" | "distance" | "keypoint-and-distance" -> Some "keypoints"
    | "dependency" | "dependency-breakers" -> Some "dependency"
    | "reducers" | "reducer" -> Some "reducers"
    | _ -> None

let private parsePhases value =
    let tokens = splitCsvList value
    if tokens.IsEmpty then
        None
    else
        let parsed = tokens |> List.map normalizePhase
        if parsed |> List.forall Option.isSome then
            parsed |> List.choose id |> List.distinct |> Some
        else
            None

let private selectPhaseLayers phases (layers: (string * ProbeProbing.GraphTemplate array) array) =
    if phases |> List.exists ((=) "all") then
        layers
    else
        let phaseSet = phases |> Set.ofList
        layers
        |> Array.filter (fun (layerName, _) ->
            phaseSet.Contains(phaseForLayer layerName))

let private templateMatchesMembers members (template: ProbeProbing.GraphTemplate) =
    if members |> List.isEmpty then
        true
    else
        let candidates =
            seq {
                template.Name
                template.Description
                for feature in template.Features do
                    feature
                for node in template.Graph.Nodes do
                    node.Id
                    node.FunctionId
            }
            |> Seq.map ProbeSelection.normalizeMember
            |> Seq.toList

        members
        |> List.map ProbeSelection.normalizeMember
        |> List.exists (fun memberName ->
            candidates
            |> List.exists (fun candidate ->
                candidate.Contains(memberName, StringComparison.Ordinal)))

let private selectMemberTemplates members (layers: (string * ProbeProbing.GraphTemplate array) array) =
    layers
    |> Array.choose (fun (layerName, templates) ->
        let selected =
            templates
            |> Array.filter (templateMatchesMembers members)

        if selected.Length = 0 then
            None
        else
            Some(layerName, selected))

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
        | true, n when n > 0u -> parseArgs { options with ImageShapes = [ cubeShape n ] } rest
        | _ ->
            eprintfn "bottom-up: --size expects a positive integer"
            Error 2
    | "--sizes" :: value :: rest
    | "--image-sizes" :: value :: rest ->
        match parseSizes value with
        | Some sizes -> parseArgs { options with ImageShapes = sizes |> List.map cubeShape } rest
        | None ->
            eprintfn "bottom-up: --sizes expects a comma-separated list of positive integers, for example 64,128,256"
            Error 2
    | "--shape" :: value :: rest ->
        match tryParseShape value with
        | Some shape -> parseArgs { options with ImageShapes = [ shape ] } rest
        | None ->
            eprintfn "bottom-up: --shape expects WxHxD, N^3, or W^2*D, for example 512x512x128"
            Error 2
    | "--shapes" :: value :: rest ->
        match parseShapes value with
        | Some shapes -> parseArgs { options with ImageShapes = shapes } rest
        | None ->
            eprintfn "bottom-up: --shapes expects comma-separated shapes, for example 256x256x256,512x512x128,1024x1024x64"
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
    | "--phase" :: value :: rest ->
        match normalizePhase value with
        | Some phase -> parseArgs { options with Phases = [ phase ] } rest
        | None ->
            eprintfn "bottom-up: unknown phase '%s'" value
            Error 2
    | "--phases" :: value :: rest ->
        match parsePhases value with
        | Some phases -> parseArgs { options with Phases = phases } rest
        | None ->
            eprintfn "bottom-up: --phases expects io,io-cast,sources,singleton,window-slab,neighbourhood,geometry,fourier,keypoints,dependency,reducers, or all"
            Error 2
    | "--member" :: value :: rest
    | "--members" :: value :: rest
    | "--operator" :: value :: rest
    | "--operators" :: value :: rest ->
        parseArgs { options with Members = options.Members @ ProbeSelection.splitCsvList value } rest
    | "--keep-tmp" :: rest ->
        parseArgs { options with CleanTmp = false } rest
    | "--run-probes" :: rest ->
        parseArgs { options with RunProbes = true } rest
    | "--no-run-probes" :: rest
    | "--emit-only" :: rest ->
        parseArgs { options with RunProbes = false } rest
    | "--no-fit" :: rest ->
        parseArgs { options with FitModel = false } rest
    | "--fit" :: rest ->
        parseArgs { options with FitModel = true } rest
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

let private runAnalysis options =
    let args =
        [ yield "--samples-root"
          yield options.SamplesRoot
          yield "--output"
          yield options.AnalysisDirectory
          yield "--extra-json-root"
          yield options.ProbeJsonRoot
          yield "--no-samples"
          if not options.FitModel then
              yield "--no-fit" ]

    ProbeAnalysis.main (args |> List.toArray)

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
           "--shuffle"
           "-j"
           string options.Jobs |]

let private writePlan (path: string) (layersBySize: (ProbeProbing.ImageSize * (string * ProbeProbing.GraphTemplate array) array) array) =
    Directory.CreateDirectory(Path.GetDirectoryName path) |> ignore

    let lines =
        seq {
            yield "width,height,depth,shape,layer,graph"
            for shape, layers in layersBySize do
                for layerName, templates in layers do
                    for template in templates do
                        yield $"{shape.Width},{shape.Height},{shape.Depth},{shapeName shape},{layerName},{template.Name}"
        }

    File.WriteAllLines(path, lines)

let private cleanupInputDirectory (inputDir: string) =
    if Directory.Exists inputDir then
        Directory.Delete(inputDir, true)
        printfn "Removed generated calibration inputs %s." inputDir

let main argv =
    match parseArgs (defaultOptions ()) (Array.toList argv) with
    | Error exitCode -> exitCode
    | Ok options ->
        try
            cleanTmp options

            let probeRoot = Path.Combine(options.ProbeJsonRoot, "bottomup_" + timestamp ())
            printfn "Bottom-up probe root: %s" probeRoot

            let plannedLayers = ResizeArray<ProbeProbing.ImageSize * (string * ProbeProbing.GraphTemplate array) array>()
            let mutable exitCode = 0

            for shape in options.ImageShapes do
                if exitCode = 0 then
                    let shapeText = shapeName shape
                    let inputDir = Path.Combine(options.InputDirectory, shapeText)

                    printfn
                        "Generating calibration inputs in %s (%ux%ux%u, noisy %s)."
                        inputDir
                        shape.Width
                        shape.Height
                        shape.Depth
                        options.NoisyType

                    let inputConfig = ProbeProbing.createBottomUpInputsForShape shape options.NoisyType inputDir
                    let selectedLayers =
                        ProbeProbing.graphTemplateLayersForBottomUp inputConfig
                        |> selectPhaseLayers options.Phases
                        |> selectMemberTemplates options.Members
                        |> Array.truncate options.Layers

                    plannedLayers.Add(shape, selectedLayers)
                    writePlan (Path.Combine(probeRoot, "bottomUpPlan.csv")) (plannedLayers.ToArray())

                    try
                        for layerIndex, (layerName, templates) in selectedLayers |> Array.indexed do
                            if exitCode = 0 then
                                let layerNumber = layerIndex + 1
                                let layerDir =
                                    Path.Combine(
                                        probeRoot,
                                        sprintf "size_%s" (shapeName shape),
                                        sprintf "layer_%03d_%s" layerNumber layerName)

                                printfn
                                    "Bottom-up size %ux%ux%u layer %d/%d: %s (%d graph(s))"
                                    shape.Width
                                    shape.Height
                                    shape.Depth
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
                                    let analysisExit = runAnalysis options
                                    if analysisExit <> 0 then
                                        eprintfn "bottom-up analysis failed with exit code %d" analysisExit
                                        exitCode <- analysisExit
                    finally
                        if options.RunProbes then
                            cleanupInputDirectory inputDir

            if exitCode = 0 then
                printfn "Bottom-up calibration pass complete. Analysis written to %s." options.AnalysisDirectory

            exitCode
        with ex ->
            eprintfn "bottom-up failed: %s" ex.Message
            1
