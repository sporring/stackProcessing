module RunJson

open System
open System.Diagnostics
open System.Globalization
open System.IO
open System.Security
open System.Text.RegularExpressions
open System.Threading
open System.Threading.Tasks
open Studio.Compiler
open Studio.Graph
open RunSamplesCommon

type Options =
    { SamplesRoot: string
      ExtraJsonRoots: string list
      IncludeSamples: bool
      Jobs: int
      DebugLevel: int
      Optimize: bool
      SkipBuild: bool
      CompileOnly: bool
      GatherOnly: bool
      Repeat: int
      RunId: string option
      Timeout: TimeSpan option }

type GraphJob =
    { Name: string
      JsonPath: string
      WorkingDirectory: string
      RunDirectory: string
      LogPath: string }

type ProcessResult =
    { ExitCode: int
      Elapsed: TimeSpan
      TimedOut: bool }

type GraphOutcome =
    { Job: GraphJob
      ExitCode: int
      Elapsed: TimeSpan }

type RunSummary =
    { EstimatedPeakMemory: string
      PeakMemory: string
      ActualRunSeconds: string }

let private outputDirectoryName = "runJson"
let private buildDirectoryName = "runJson-build"

let private timestampRunId () =
    DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture)

let private batchDirectoryName runId =
    $"{outputDirectoryName}_{runId}"

let private batchBuildDirectoryName runId =
    $"{buildDirectoryName}_{runId}"

let private outputDirectory samplesRoot =
    Path.Combine(runOutputRoot samplesRoot, outputDirectoryName)

let private buildDirectory samplesRoot =
    Path.Combine(sampleTempRoot samplesRoot, buildDirectoryName)

let private batchDirectory samplesRoot runId =
    Path.Combine(runOutputRoot samplesRoot, batchDirectoryName runId)

let private batchBuildDirectory samplesRoot runId =
    Path.Combine(sampleTempRoot samplesRoot, batchBuildDirectoryName runId)

let private repeatDirectory batchDir repeatIndex =
    Path.Combine(batchDir, sprintf "repeat_%03d" repeatIndex)

let usage () =
    printfn "Usage: ./runJson.sh [-j jobs] [--skip-build] [--compile-only] [--debug-level N] [--optimize true|false]"
    printfn ""
    printfn "Compiles sample Studio JSON graphs to F#, runs them, and stores logs below"
    printfn $"tmp/{outputDirectoryName}. The default is sequential."
    printfn ""
    printfn "Options:"
    printfn "  -j, --jobs N       Run up to N generated graphs at once."
    printfn "  -p, --parallel    Run with one job per logical CPU."
    printfn "  --skip-build      Run generated graph projects without building them first."
    printfn "  --compile-only    Generate and build F#, but do not run the programs."
    printfn "  --extra-json-root PATH"
    printfn "                    Also run JSON graphs from PATH, e.g. Probe-generated graphs."
    printfn "  --extra-json-only"
    printfn "                    Run only graphs from --extra-json-root."
    printfn "  --gather-only     Regenerate gather.csv files from existing graph logs."
    printfn "  --repeat N        Run the full graph set N times. Defaults to 1."
    printfn "  --run-id VALUE    Use VALUE in tmp/runJson_VALUE. Defaults to a timestamp."
    printfn "  --debug-level N   Accepted for symmetry with runAll; generated graphs carry their saved debug settings."
    printfn "  --optimize BOOL   Enable or disable optimizer use while running generated graphs. Defaults to false."
    printfn "  --no-optimize     Shortcut for --optimize false."
    printfn "  --timeout N       Stop a build or run after N minutes. Defaults to 30. Use 0 to disable."
    printfn "  -h, --help        Show this help."

let private defaultSamplesRoot () =
    let cwd = Directory.GetCurrentDirectory()

    if String.Equals(Path.GetFileName cwd, "samples", StringComparison.OrdinalIgnoreCase) then
        cwd
    else
        let samples = Path.Combine(cwd, "samples")
        if Directory.Exists samples then samples else cwd

let rec private parseArgs options args =
    match args with
    | [] -> Ok options
    | ("-h" | "--help") :: _ ->
        usage ()
        Error 0
    | ("-j" | "--jobs") :: value :: rest ->
        match Int32.TryParse value with
        | true, jobs when jobs > 0 -> parseArgs { options with Jobs = jobs } rest
        | _ ->
            eprintfn "runJson: -j/--jobs expects a positive integer"
            Error 2
    | ("-p" | "--parallel") :: rest ->
        parseArgs { options with Jobs = Environment.ProcessorCount } rest
    | "--debug-level" :: value :: rest ->
        match Int32.TryParse value with
        | true, level when level >= 0 -> parseArgs { options with DebugLevel = level } rest
        | _ ->
            eprintfn "runJson: --debug-level expects a non-negative integer"
            Error 2
    | "--optimize" :: value :: rest
    | "--optimizer" :: value :: rest ->
        match Boolean.TryParse value with
        | true, optimize -> parseArgs { options with Optimize = optimize } rest
        | _ ->
            eprintfn "runJson: --optimize expects true or false"
            Error 2
    | ("--no-optimize" | "--no-optimizer") :: rest ->
        parseArgs { options with Optimize = false } rest
    | "--skip-build" :: rest -> parseArgs { options with SkipBuild = true } rest
    | "--compile-only" :: rest -> parseArgs { options with CompileOnly = true } rest
    | "--gather-only" :: rest -> parseArgs { options with GatherOnly = true } rest
    | "--repeat" :: value :: rest
    | "--repeats" :: value :: rest ->
        match Int32.TryParse value with
        | true, repeat when repeat > 0 -> parseArgs { options with Repeat = repeat } rest
        | _ ->
            eprintfn "runJson: --repeat expects a positive integer"
            Error 2
    | "--run-id" :: value :: rest -> parseArgs { options with RunId = Some value } rest
    | "--timeout" :: value :: rest
    | "--timeout-minutes" :: value :: rest ->
        match Double.TryParse value with
        | true, minutes when minutes = 0.0 -> parseArgs { options with Timeout = None } rest
        | true, minutes when minutes > 0.0 -> parseArgs { options with Timeout = Some(TimeSpan.FromMinutes minutes) } rest
        | _ ->
            eprintfn "runJson: --timeout expects a non-negative number of minutes"
            Error 2
    | "--samples-root" :: value :: rest -> parseArgs { options with SamplesRoot = Path.GetFullPath value } rest
    | "--extra-json-root" :: value :: rest
    | "--generated-json-root" :: value :: rest
    | "--probe-json-root" :: value :: rest ->
        parseArgs { options with ExtraJsonRoots = options.ExtraJsonRoots @ [ Path.GetFullPath value ] } rest
    | "--extra-json-only" :: rest
    | "--generated-json-only" :: rest
    | "--probe-json-only" :: rest ->
        parseArgs { options with IncludeSamples = false } rest
    | option :: _ ->
        eprintfn "runJson: unknown option %s" option
        usage ()
        Error 2

let private discoverGraphsInRoot samplesRoot scanRoot outputDir buildDir includeTmp namePrefix =
    Directory.EnumerateFiles(scanRoot, "*.json", SearchOption.AllDirectories)
    |> Seq.filter (fun path ->
        let relative = relativePath scanRoot path
        let parts = relative.Split('/')

        (includeTmp || not (relative.StartsWith("tmp/", StringComparison.OrdinalIgnoreCase)))
        && not (parts |> Array.exists (fun part ->
            part.Equals("bin", StringComparison.OrdinalIgnoreCase)
            || part.Equals("obj", StringComparison.OrdinalIgnoreCase)))
        && not (relative.StartsWith("RunAll/", StringComparison.OrdinalIgnoreCase))
        && not (relative.StartsWith("RunJson/", StringComparison.OrdinalIgnoreCase)))
    |> Seq.map (fun path ->
        let relative = relativePath scanRoot path
        let localName = relative.Substring(0, relative.Length - Path.GetExtension(relative).Length)
        let name =
            if String.IsNullOrWhiteSpace namePrefix then
                localName
            else
                namePrefix.TrimEnd('/') + "/" + localName

        { Name = name
          JsonPath = path
          WorkingDirectory = Path.GetDirectoryName path
          RunDirectory = Path.Combine(buildDir, safeName name)
          LogPath = Path.Combine(outputDir, name + ".out") })
    |> Seq.sortBy _.Name
    |> Seq.toArray

let private discoverGraphs samplesRoot extraJsonRoots includeSamples outputDir buildDir =
    let sampleGraphs =
        if includeSamples then
            discoverGraphsInRoot samplesRoot samplesRoot outputDir buildDir false ""
        else
            [||]

    let generatedGraphs =
        extraJsonRoots
        |> List.toArray
        |> Array.collect (fun root ->
            if Directory.Exists root then
                let prefix = "generated/" + safeName (Path.GetFileName(root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)))
                discoverGraphsInRoot samplesRoot root outputDir buildDir true prefix
            else
                eprintfn "runJson: extra JSON root does not exist: %s" root
                [||])

    Array.append sampleGraphs generatedGraphs
    |> Array.sortBy _.Name

let private xml value =
    SecurityElement.Escape(value)

let private processNameForGraph (graphName: string) =
    let parts =
        graphName.Split([| '/'; '\\' |], StringSplitOptions.RemoveEmptyEntries)

    let compactName =
        match parts with
        | [| only |] -> only
        | _ when parts.Length >= 2 && String.Equals(parts[parts.Length - 1], parts[parts.Length - 2], StringComparison.OrdinalIgnoreCase) ->
            parts[parts.Length - 1]
        | _ -> String.concat "_" parts

    "GraphRun_" + safeName compactName

let private ensureRunProject repositoryRoot job =
    Directory.CreateDirectory job.RunDirectory |> ignore

    let projectPath = Path.Combine(job.RunDirectory, "GraphRun.fsproj")
    let assemblyName = processNameForGraph job.Name
    let probingOutput = Path.Combine(repositoryRoot, "src", "StackProcessing.Probe", "bin", "Debug", "net10.0")
    let simpleItkManaged = Path.Combine(repositoryRoot, "lib", "SimpleITKCSharpManaged.dll")
    let simpleItkWindowsNative = Path.Combine(repositoryRoot, "lib", "SimpleITKCSharpNative.dll")
    let simpleItkLinuxNative = Path.Combine(repositoryRoot, "lib", "libSimpleITKCSharpNative.so")
    let simpleItkMacNative = Path.Combine(repositoryRoot, "lib", "libSimpleITKCSharpNative.dylib")

    let referenceItems =
        if Directory.Exists probingOutput then
            Directory.GetFiles(probingOutput, "*.dll")
            |> Array.sort
            |> Array.map (fun dll ->
                let includeName = Path.GetFileNameWithoutExtension dll
                $"""    <Reference Include="{xml includeName}">
      <HintPath>{xml dll}</HintPath>
      <Private>true</Private>
    </Reference>""")
            |> String.concat Environment.NewLine
        else
            $"""    <Reference Include="SimpleITKCSharp">
      <HintPath>{xml simpleItkManaged}</HintPath>
      <Private>true</Private>
    </Reference>"""

    let projectFile =
        $"""<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <AssemblyName>{xml assemblyName}</AssemblyName>
  </PropertyGroup>
  <ItemGroup>
{referenceItems}
  </ItemGroup>
  <ItemGroup Condition="$([MSBuild]::IsOSPlatform('Windows'))">
    <None Include="{xml simpleItkWindowsNative}">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
      <TargetPath>libSimpleITKCSharpNative.dll</TargetPath>
    </None>
  </ItemGroup>
  <ItemGroup Condition="$([MSBuild]::IsOSPlatform('Linux'))">
    <None Include="{xml simpleItkLinuxNative}" CopyToOutputDirectory="PreserveNewest" />
  </ItemGroup>
  <ItemGroup Condition="$([MSBuild]::IsOSPlatform('OSX'))">
    <None Include="{xml simpleItkMacNative}" CopyToOutputDirectory="PreserveNewest" />
  </ItemGroup>
  <ItemGroup>
    <Compile Include="Program.fs" />
  </ItemGroup>
</Project>
"""

    if not (File.Exists projectPath) || File.ReadAllText(projectPath) <> projectFile then
        File.WriteAllText(projectPath, projectFile)

    projectPath

let private writeGeneratedProgram job =
    let graph = PipelineGraphStorage.load job.JsonPath
    let code = PipelineCodeGenerator.generateSavedGraph graph
    let programPath = Path.Combine(job.RunDirectory, "Program.fs")

    File.WriteAllText(programPath, code)
    code, programPath

let private prependEnvironmentPath (psi: ProcessStartInfo) name value =
    let separator = string Path.PathSeparator

    match psi.Environment.TryGetValue name with
    | true, existing when not (String.IsNullOrWhiteSpace existing) ->
        psi.Environment[name] <- value + separator + existing
    | _ -> psi.Environment[name] <- value

let private runProcessAsync
    (cancellationToken: CancellationToken)
    (logPath: string)
    (workingDirectory: string)
    (fileName: string)
    (args: string list)
    (envPath: string option)
    (timeout: TimeSpan option)
    (optimizer: string option)
    =
    task {
        Directory.CreateDirectory(Path.GetDirectoryName logPath) |> ignore

        use log =
            new StreamWriter(File.Open(logPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite))

        let writeLock = obj ()

        let writeLine (line: string) =
            lock writeLock (fun () ->
                log.WriteLine line
                log.Flush())

        let psi =
            ProcessStartInfo(
                FileName = fileName,
                WorkingDirectory = workingDirectory,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            )

        args |> List.iter (fun arg -> psi.ArgumentList.Add arg)
        optimizer |> Option.iter (fun value -> psi.Environment["STACKPROCESSING_OPTIMIZE"] <- value)

        match envPath with
        | Some path ->
            prependEnvironmentPath psi "DYLD_LIBRARY_PATH" path
            prependEnvironmentPath psi "LD_LIBRARY_PATH" path
        | None -> ()

        use proc = new Process(StartInfo = psi, EnableRaisingEvents = true)
        let stopwatch = Stopwatch.StartNew()

        if not (proc.Start()) then
            return { ExitCode = 1; Elapsed = stopwatch.Elapsed; TimedOut = false }
        else
            use linkedCancellation = CancellationTokenSource.CreateLinkedTokenSource cancellationToken
            let mutable timedOut = false

            match timeout with
            | Some timeout -> linkedCancellation.CancelAfter timeout
            | None -> ()

            let copyOutput (reader: StreamReader) =
                task {
                    let mutable finished = false

                    while not finished do
                        let! line = reader.ReadLineAsync()

                        if isNull line then
                            finished <- true
                        else
                            writeLine line
                }

            let stdout = copyOutput proc.StandardOutput
            let stderr = copyOutput proc.StandardError

            try
                do! proc.WaitForExitAsync(linkedCancellation.Token)
            with :? OperationCanceledException ->
                timedOut <- not cancellationToken.IsCancellationRequested

                if not proc.HasExited then
                    try
                        proc.Kill(entireProcessTree = true)
                    with _ ->
                        ()

                do! proc.WaitForExitAsync()

            let! _ = Task.WhenAll([| stdout; stderr |])
            stopwatch.Stop()
            return
                { ExitCode = if timedOut then 124 else proc.ExitCode
                  Elapsed = stopwatch.Elapsed
                  TimedOut = timedOut }
    }

let private runGraph (cancellationToken: CancellationToken) (repositoryRoot: string) (options: Options) (job: GraphJob) =
    task {
        Directory.CreateDirectory(Path.GetDirectoryName job.LogPath) |> ignore
        File.WriteAllText(job.LogPath, $"== JSON {job.Name} =={Environment.NewLine}Source: {job.JsonPath}{Environment.NewLine}")
        printfn "json %s" job.Name

        let projectPath = ensureRunProject repositoryRoot job
        let generated, programPath = writeGeneratedProgram job
        File.AppendAllText(job.LogPath, $"Generated: {programPath}{Environment.NewLine}{Environment.NewLine}")

        if generated.StartsWith("// Cannot generate F#:", StringComparison.Ordinal) then
            File.AppendAllText(job.LogPath, generated + Environment.NewLine)
            return
                { Job = job
                  ExitCode = 1
                  Elapsed = TimeSpan.Zero }
        else
            let mutable exitCode = 0
            let mutable elapsed = TimeSpan.Zero

            if not options.SkipBuild then
                File.AppendAllText(job.LogPath, $"== Build {job.Name} =={Environment.NewLine}")
                let! buildResult =
                    runProcessAsync
                        cancellationToken
                        job.LogPath
                        job.RunDirectory
                        "dotnet"
                        [ "build"; projectPath; "--verbosity"; "q"; "--disable-build-servers" ]
                        None
                        options.Timeout
                        None

                exitCode <- buildResult.ExitCode
                elapsed <- elapsed + buildResult.Elapsed
                File.AppendAllText(job.LogPath, $"Build finished in {buildResult.Elapsed}.{Environment.NewLine}")
                if buildResult.TimedOut then
                    File.AppendAllText(job.LogPath, $"Build timed out after {buildResult.Elapsed}.{Environment.NewLine}")

            if exitCode = 0 && not options.CompileOnly then
                File.AppendAllText(job.LogPath, $"{Environment.NewLine}== Run {job.Name} =={Environment.NewLine}")

                let optimizerValue = if options.Optimize then "1" else "0"
                let! runResult =
                    runProcessAsync
                        cancellationToken
                        job.LogPath
                        job.WorkingDirectory
                        "dotnet"
                        [ "run"; "--project"; projectPath; "--no-build"; "--verbosity"; "q" ]
                        (Some(Path.Combine(job.WorkingDirectory, "lib")))
                        options.Timeout
                        (Some optimizerValue)

                exitCode <- runResult.ExitCode
                elapsed <- elapsed + runResult.Elapsed
                File.AppendAllText(job.LogPath, $"Run finished in {runResult.Elapsed}.{Environment.NewLine}")
                if runResult.TimedOut then
                    File.AppendAllText(job.LogPath, $"Run timed out after {runResult.Elapsed}.{Environment.NewLine}")

            return
                { Job = job
                  ExitCode = exitCode
                  Elapsed = elapsed }
    }

let private runWithParallelism (cancellationToken: CancellationToken) (repositoryRoot: string) (options: Options) (jobs: GraphJob array) =
    task {
        use gate = new SemaphoreSlim(options.Jobs)

        let runOne job =
            task {
                do! gate.WaitAsync(cancellationToken)

                try
                    return! runGraph cancellationToken repositoryRoot options job
                finally
                    gate.Release() |> ignore
            }

        let tasks = jobs |> Array.map runOne
        let! results = Task.WhenAll tasks
        return results
    }

let private csvEscape (value: string) =
    if value.Contains(',') || value.Contains('"') || value.Contains('\n') then
        "\"" + value.Replace("\"", "\"\"") + "\""
    else
        value

let private tryRegex (pattern: string) (line: string) =
    let m = Regex.Match(line, pattern)
    if m.Success then Some m else None

let private groupValue (name: string) (m: Match) =
    let group = m.Groups[name]
    if group.Success then group.Value else ""

let private formatElapsedSeconds (elapsed: TimeSpan) =
    elapsed.TotalSeconds.ToString("F3", CultureInfo.InvariantCulture)

let private elapsedSecondsFromLog (lines: string array) =
    lines
    |> Array.rev
    |> Array.tryPick (fun line ->
        if line.StartsWith("Run finished in ", StringComparison.Ordinal) then
            let value =
                line.Substring("Run finished in ".Length).Trim().TrimEnd('.')

            match TimeSpan.TryParse value with
            | true, elapsed -> Some(formatElapsedSeconds elapsed)
            | _ -> None
        else
            None)

let private tryParseFloat (value: string) =
    match Double.TryParse(value, NumberStyles.Float, CultureInfo.CurrentCulture) with
    | true, parsed -> Some parsed
    | _ ->
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, parsed -> Some parsed
        | _ -> None

let private formatSeconds (seconds: float) =
    seconds.ToString("F6", CultureInfo.InvariantCulture)

let private secondsFromRunSummary (value: string) (unit: string) =
    match tryParseFloat value, unit with
    | Some milliseconds, "ms" -> formatSeconds (milliseconds / 1000.0)
    | Some seconds, "s" -> formatSeconds seconds
    | _ -> value

let private tryParseCurrentRunSummary (lines: string array) =
    lines
    |> Array.indexed
    |> Array.rev
    |> Array.tryPick (fun (index, line) ->
        if line.Contains("Run summary:", StringComparison.Ordinal) && index + 3 < lines.Length then
            let estimatedLine = lines[index + 1]
            let measuredLine = lines[index + 2]
            let timeLine = lines[index + 3]

            match
                tryRegex
                    @"^estimated peak / available memory (?<estimatedPeakMemory>\S+) KB / (?<availableMemory>\S+) KB$"
                    estimatedLine,
                tryRegex
                    @"^Measured peak delta, baseline, peak: (?<peakMemory>\S+) KB \(baseline (?<baselineMemory>\S+) KB, peak (?<processPeakMemory>\S+) KB\)$"
                    measuredLine,
                tryRegex
                    @"^Estimated/actual time .+ / (?<actualTime>\S+) (?<actualTimeUnit>ms|s)\.?$"
                    timeLine
            with
            | Some estimated, Some measured, Some timing ->
                Some
                    { EstimatedPeakMemory = groupValue "estimatedPeakMemory" estimated
                      PeakMemory = groupValue "peakMemory" measured
                      ActualRunSeconds = secondsFromRunSummary (groupValue "actualTime" timing) (groupValue "actualTimeUnit" timing) }
            | _ -> None
        else
            None)

let private tryParseLegacyRunSummary (lines: string array) =
    lines
    |> Array.rev
    |> Array.tryPick (
        tryRegex
            @"estimated peak memory (?<estimatedPeakMemory>\S+) KB / .*actual process RSS peak delta (?<peakMemory>\S+) KB .*actual time (?<actualTime>\S+) (?<actualTimeUnit>ms|s)"
    )
    |> Option.map (fun summary ->
        { EstimatedPeakMemory = groupValue "estimatedPeakMemory" summary
          PeakMemory = groupValue "peakMemory" summary
          ActualRunSeconds = secondsFromRunSummary (groupValue "actualTime" summary) (groupValue "actualTimeUnit" summary) })

let private logStats logPath =
    let lines =
        if File.Exists logPath then
            File.ReadAllLines logPath
        else
            [||]

    let errorLine =
        lines
        |> Array.tryFind (fun line ->
            line.Contains("Unhandled exception", StringComparison.OrdinalIgnoreCase)
            || line.Contains("Build FAILED", StringComparison.OrdinalIgnoreCase)
            || line.Contains(" error FS", StringComparison.OrdinalIgnoreCase)
            || line.Contains(" error MSB", StringComparison.OrdinalIgnoreCase)
            || line.Contains("Cannot generate F#", StringComparison.OrdinalIgnoreCase)
            || line.Contains("timed out", StringComparison.OrdinalIgnoreCase))

    let hasFinished =
        lines |> Array.exists (fun line -> line.StartsWith("Run finished in ", StringComparison.Ordinal))

    let status =
        match errorLine with
        | Some line when line.Contains("timed out", StringComparison.OrdinalIgnoreCase) -> "timeout"
        | Some _ -> "failed"
        | None when hasFinished -> "completed"
        | None when lines.Length > 0 -> "incomplete"
        | None -> "missing-log"

    let runSummary =
        tryParseCurrentRunSummary lines
        |> Option.orElseWith (fun () -> tryParseLegacyRunSummary lines)

    let estimatedPeakMemory =
        runSummary |> Option.map _.EstimatedPeakMemory |> Option.defaultValue ""

    let peakMemory =
        runSummary |> Option.map _.PeakMemory |> Option.defaultValue ""

    let actualRunSeconds =
        runSummary |> Option.map _.ActualRunSeconds |> Option.defaultValue ""

    let processElapsedSeconds =
        elapsedSecondsFromLog lines |> Option.defaultValue ""

    status, estimatedPeakMemory, peakMemory, "", actualRunSeconds, processElapsedSeconds, (errorLine |> Option.defaultValue "")

let private writeCsv samplesRoot outputDir (results: GraphOutcome array) =
    let path = Path.Combine(outputDir, "gather.csv")

    let lines =
        seq {
            yield
                [ "name"
                  "status"
                  "estimatedPeakMemoryKB"
                  "peakMemoryKB"
                  "peakImages"
                  "actualRunSeconds"
                  "processElapsedSeconds"
                  "exitCode"
                  "log"
                  "message" ]

            for result in results do
                let status, estimatedPeakMemory, peakMemory, peakImages, actualRunSeconds, logElapsedSeconds, message =
                    logStats result.Job.LogPath

                let processElapsedSeconds =
                    if String.IsNullOrWhiteSpace logElapsedSeconds then
                        formatElapsedSeconds result.Elapsed
                    else
                        logElapsedSeconds

                yield
                    [ result.Job.Name
                      status
                      estimatedPeakMemory
                      peakMemory
                      peakImages
                      actualRunSeconds
                      processElapsedSeconds
                      string result.ExitCode
                      relativePath samplesRoot result.Job.LogPath
                      message ]
        }
        |> Seq.map (List.map csvEscape >> String.concat ",")

    File.WriteAllLines(path, lines)
    printfn "wrote %s" (relativePath samplesRoot path)

let private gatherExistingLogsInDirectory samplesRoot extraJsonRoots outputDir =
    let graphNames =
        discoverGraphs samplesRoot extraJsonRoots true outputDir (buildDirectory samplesRoot)
        |> Array.map _.Name
        |> Set.ofArray

    if Directory.Exists outputDir then
        Directory.EnumerateFiles(outputDir, "*.out", SearchOption.AllDirectories)
        |> Seq.choose (fun path ->
            let relative = relativePath outputDir path
            let name = relative.Substring(0, relative.Length - Path.GetExtension(relative).Length)
            if graphNames |> Set.contains name then
                Some(name, path)
            else
                None)
        |> Seq.sortBy fst
        |> Seq.map (fun (name, path) ->
            let status, _, _, _, _, logElapsedSeconds, _ = logStats path
            let elapsed =
                match Double.TryParse(logElapsedSeconds, Globalization.NumberStyles.Float, Globalization.CultureInfo.InvariantCulture) with
                | true, seconds -> TimeSpan.FromSeconds seconds
                | _ -> TimeSpan.Zero
            let exitCode = if status = "completed" then 0 else 1

            { Job =
                { Name = name
                  JsonPath = ""
                  WorkingDirectory = ""
                  RunDirectory = ""
                  LogPath = path }
              ExitCode = exitCode
              Elapsed = elapsed })
        |> Seq.toArray
    else
        [||]

let private gatherExistingLogs samplesRoot extraJsonRoots =
    let tmpRoots =
        [ runOutputRoot samplesRoot
          Path.Combine(samplesRoot, "tmp") ]
        |> List.distinct

    seq {
        for tmp in tmpRoots do
            let legacy = Path.Combine(tmp, outputDirectoryName)
            if Directory.Exists legacy then
                yield legacy

            if Directory.Exists tmp then
                yield!
                    Directory.EnumerateDirectories(tmp, outputDirectoryName + "_*", SearchOption.TopDirectoryOnly)
                    |> Seq.collect (fun batch ->
                        Directory.EnumerateDirectories(batch, "repeat_*", SearchOption.TopDirectoryOnly))
    }
    |> Seq.collect (gatherExistingLogsInDirectory samplesRoot extraJsonRoots)
    |> Seq.sortBy _.Job.Name
    |> Seq.toArray

let main (argv: string array) =
    let defaults =
        { SamplesRoot = defaultSamplesRoot ()
          ExtraJsonRoots = []
          IncludeSamples = true
          Jobs = 1
          DebugLevel = 1
          Optimize = false
          SkipBuild = false
          CompileOnly = false
          GatherOnly = false
          Repeat = 1
          RunId = None
          Timeout = Some(TimeSpan.FromMinutes 30.0) }

    match parseArgs defaults (argv |> Array.toList) with
    | Error exitCode -> exitCode
    | Ok options ->
        let samplesRoot = Path.GetFullPath options.SamplesRoot
        let repositoryRoot = repositoryRootFromSamplesRoot samplesRoot

        if options.GatherOnly then
            let results = gatherExistingLogs samplesRoot options.ExtraJsonRoots
            writeCsv samplesRoot (outputDirectory samplesRoot) results
            0
        else
            let runId = options.RunId |> Option.defaultWith timestampRunId
            let batchDir = batchDirectory samplesRoot runId
            let batchBuildDir = batchBuildDirectory samplesRoot runId
            Directory.CreateDirectory batchDir |> ignore
            Directory.CreateDirectory batchBuildDir |> ignore

            let jobs = discoverGraphs samplesRoot options.ExtraJsonRoots options.IncludeSamples batchDir batchBuildDir

            use cancellation = new CancellationTokenSource()

            Console.CancelKeyPress.Add(fun args ->
                args.Cancel <- true
                eprintfn "Stopping running JSON graph job(s)..."
                cancellation.Cancel())

            if jobs.Length = 0 then
                eprintfn "runJson: no JSON graphs found below %s" samplesRoot
                1
            else
                try
                    let mutable failed = false

                    for repeatIndex in 1 .. options.Repeat do
                        cancellation.Token.ThrowIfCancellationRequested()

                        let repeatOutputDir = repeatDirectory batchDir repeatIndex
                        let repeatBuildDir = repeatDirectory batchBuildDir repeatIndex
                        Directory.CreateDirectory repeatOutputDir |> ignore
                        Directory.CreateDirectory repeatBuildDir |> ignore
                        printfn "runJson repeat %d/%d -> %s" repeatIndex options.Repeat (relativePath samplesRoot repeatOutputDir)

                        let repeatJobs = discoverGraphs samplesRoot options.ExtraJsonRoots options.IncludeSamples repeatOutputDir repeatBuildDir

                        let results =
                            runWithParallelism cancellation.Token repositoryRoot options repeatJobs
                            |> _.GetAwaiter().GetResult()

                        writeCsv samplesRoot repeatOutputDir results

                        let failures = results |> Array.filter (fun result -> result.ExitCode <> 0)

                        for failure in failures do
                            failed <- true
                            eprintfn "%s failed with exit code %d; see %s" failure.Job.Name failure.ExitCode (relativePath samplesRoot failure.Job.LogPath)

                    if failed then 1 else 0
                with :? OperationCanceledException ->
                    130
