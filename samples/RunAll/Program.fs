module RunAll

open System
open System.Diagnostics
open System.Globalization
open System.IO
open System.Text.RegularExpressions
open System.Threading
open System.Threading.Tasks

type Options =
    { SamplesRoot: string
      Jobs: int
      DebugLevel: int
      Optimize: bool
      SkipBuild: bool
      GatherOnly: bool
      Timeout: TimeSpan option }

type Sample =
    { Name: string
      Directory: string
      Project: string
      LogPath: string }

type ProcessResult =
    { ExitCode: int
      Elapsed: TimeSpan
      TimedOut: bool }

type RunOutcome =
    { Sample: Sample
      ExitCode: int
      Elapsed: TimeSpan }

let private outputDirectoryName = "runAll"

let private outputDirectory samplesRoot =
    Path.Combine(samplesRoot, "tmp", outputDirectoryName)

let private outputLogPath samplesRoot sampleName =
    Path.Combine(outputDirectory samplesRoot, sampleName + ".out")

let usage () =
    printfn "Usage: ./runAll.sh [-j jobs] [--skip-build] [--debug-level N] [--optimize true|false]"
    printfn ""
    printfn "Runs all sample projects. The default is sequential, which gives cleaner"
    printfn "per-sample timing measurements. Use -j with a value greater than 1 to run"
    printfn "multiple samples in parallel."
    printfn ""
    printfn "Options:"
    printfn "  -j, --jobs N       Run up to N samples at once."
    printfn "  -p, --parallel    Run with one job per logical CPU."
    printfn "  --skip-build      Run samples without first building them."
    printfn $"  --gather-only     Regenerate tmp/{outputDirectoryName}/gather.csv from existing sample logs."
    printfn "  --debug-level N   Pass -d N to each sample. Defaults to 1."
    printfn "  --optimize BOOL   Enable or disable optimizer use. Defaults to false."
    printfn "  --no-optimize     Shortcut for --optimize false."
    printfn "  --timeout N       Stop a build or run after N minutes. Defaults to 30. Use 0 to disable."
    printfn "  -h, --help        Show this help."

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
            eprintfn "runAll: -j/--jobs expects a positive integer"
            Error 2
    | ("-p" | "--parallel") :: rest ->
        parseArgs { options with Jobs = Environment.ProcessorCount } rest
    | "--debug-level" :: value :: rest ->
        match Int32.TryParse value with
        | true, level when level >= 0 -> parseArgs { options with DebugLevel = level } rest
        | _ ->
            eprintfn "runAll: --debug-level expects a non-negative integer"
            Error 2
    | "--optimize" :: value :: rest
    | "--optimizer" :: value :: rest ->
        match Boolean.TryParse value with
        | true, optimize -> parseArgs { options with Optimize = optimize } rest
        | _ ->
            eprintfn "runAll: --optimize expects true or false"
            Error 2
    | ("--no-optimize" | "--no-optimizer") :: rest ->
        parseArgs { options with Optimize = false } rest
    | "--skip-build" :: rest ->
        parseArgs { options with SkipBuild = true } rest
    | "--gather-only" :: rest ->
        parseArgs { options with GatherOnly = true } rest
    | "--timeout" :: value :: rest
    | "--timeout-minutes" :: value :: rest ->
        match Double.TryParse value with
        | true, minutes when minutes = 0.0 -> parseArgs { options with Timeout = None } rest
        | true, minutes when minutes > 0.0 -> parseArgs { options with Timeout = Some(TimeSpan.FromMinutes minutes) } rest
        | _ ->
            eprintfn "runAll: --timeout expects a non-negative number of minutes"
            Error 2
    | "--samples-root" :: value :: rest ->
        parseArgs { options with SamplesRoot = Path.GetFullPath value } rest
    | option :: _ ->
        eprintfn "runAll: unknown option %s" option
        usage ()
        Error 2

let private defaultSamplesRoot () =
    let cwd = Directory.GetCurrentDirectory()
    if String.Equals(Path.GetFileName cwd, "samples", StringComparison.OrdinalIgnoreCase) then
        cwd
    else
        let samples = Path.Combine(cwd, "samples")
        if Directory.Exists samples then samples else cwd

let private relativePath root path =
    Path.GetRelativePath(root, path).Replace(Path.DirectorySeparatorChar, '/')

let private isRunnerProject samplesRoot project =
    let relative = relativePath samplesRoot project
    relative = "RunAll/RunAll.fsproj"
    || relative = "RunJson/RunJson.fsproj"

let private discoverSamples samplesRoot =
    Directory.EnumerateFiles(samplesRoot, "*.fsproj", SearchOption.AllDirectories)
    |> Seq.filter (fun project -> not (isRunnerProject samplesRoot project))
    |> Seq.map (fun project ->
        let dir = Path.GetDirectoryName project
        let name = relativePath samplesRoot dir
        { Name = name
          Directory = dir
          Project = project
          LogPath = outputLogPath samplesRoot name })
    |> Seq.sortBy _.Name
    |> Seq.toArray

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
            | Some timeout ->
                linkedCancellation.CancelAfter timeout
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

let private clearLog (sample: Sample) =
    Directory.CreateDirectory(Path.GetDirectoryName sample.LogPath) |> ignore
    File.WriteAllText(sample.LogPath, "")

let private buildSample (cancellationToken: CancellationToken) timeout (sample: Sample) =
    task {
        clearLog sample
        File.AppendAllText(sample.LogPath, $"== Build {sample.Name} =={Environment.NewLine}")
        printfn "build %s" sample.Name

        let! result =
            runProcessAsync
                cancellationToken
                sample.LogPath
                sample.Directory
                "dotnet"
                [ "build"; sample.Project; "--no-restore"; "--verbosity"; "q"; "--disable-build-servers" ]
                None
                timeout

        File.AppendAllText(sample.LogPath, $"Build finished in {result.Elapsed}.{Environment.NewLine}")
        if result.TimedOut then
            File.AppendAllText(sample.LogPath, $"Build timed out after {result.Elapsed}.{Environment.NewLine}")
        return result.ExitCode
    }

let private runSample (cancellationToken: CancellationToken) timeout debugLevel optimize (sample: Sample) =
    task {
        File.AppendAllText(sample.LogPath, $"{Environment.NewLine}== Run {sample.Name} =={Environment.NewLine}")
        printfn "run %s" sample.Name

        let nativeLibPath = Path.Combine(sample.Directory, "lib")

        let! result =
            runProcessAsync
                cancellationToken
                    sample.LogPath
                    sample.Directory
                    "dotnet"
                    [ "run"; "--no-build"; "--verbosity"; "q"; "--"; "-d"; string debugLevel; "--optimize"; string optimize ]
                    (Some nativeLibPath)
                    timeout

        File.AppendAllText(sample.LogPath, $"Run finished in {result.Elapsed}.{Environment.NewLine}")
        if result.TimedOut then
            File.AppendAllText(sample.LogPath, $"Run timed out after {result.Elapsed}.{Environment.NewLine}")
        return
            { Sample = sample
              ExitCode = result.ExitCode
              Elapsed = result.Elapsed }
    }

let private runWithParallelism (cancellationToken: CancellationToken) jobs timeout debugLevel optimize (samples: Sample array) =
    task {
        use gate = new SemaphoreSlim(jobs)

        let runOne sample =
            task {
                    do! gate.WaitAsync(cancellationToken)

                    try
                        return! runSample cancellationToken timeout debugLevel optimize sample
                    finally
                    gate.Release() |> ignore
            }

        let tasks = samples |> Array.map runOne
        let! results = Task.WhenAll tasks
        return results
    }

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

let private parseStatsFromLog samplesRoot name elapsedSeconds exitCode logPath =
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

    let sinkSummary =
        lines
        |> Array.rev
        |> Array.tryPick (
            tryRegex
                @"estimated peak memory (?<estimatedPeakMemory>\S+) KB / .*actual process RSS peak delta (?<peakMemory>\S+) KB .*actual time (?<actualTime>\S+) (?<actualTimeUnit>ms|s)"
        )

    let oldMemoryLine =
        lines
        |> Array.rev
        |> Array.tryFind (fun line -> line.Contains(" KB / "))

    let token index (line: string) =
        let parts = line.Split([| ' '; '\t' |], StringSplitOptions.RemoveEmptyEntries)
        if index < parts.Length then parts[index] else ""

    let estimatedPeakMemory =
        sinkSummary |> Option.map (groupValue "estimatedPeakMemory") |> Option.defaultValue ""

    let peakMemory =
        sinkSummary
        |> Option.map (groupValue "peakMemory")
        |> Option.orElseWith (fun () -> oldMemoryLine |> Option.map (token 3))
        |> Option.defaultValue ""

    let peakImages =
        match sinkSummary with
        | Some _ -> ""
        | None -> oldMemoryLine |> Option.map (token 7) |> Option.defaultValue ""

    let actualRunSeconds =
        sinkSummary
        |> Option.map (fun m -> secondsFromRunSummary (groupValue "actualTime" m) (groupValue "actualTimeUnit" m))
        |> Option.defaultValue ""

    let elapsedSeconds =
        if String.IsNullOrWhiteSpace elapsedSeconds then
            elapsedSecondsFromLog lines |> Option.defaultValue ""
        else
            elapsedSeconds

    [ name
      status
      estimatedPeakMemory
      peakMemory
      peakImages
      actualRunSeconds
      elapsedSeconds
      string exitCode
      relativePath samplesRoot logPath
      (errorLine |> Option.defaultValue "") ]

let private csvEscape (value: string) =
    if value.Contains(',') || value.Contains('"') || value.Contains('\n') then
        "\"" + value.Replace("\"", "\"\"") + "\""
    else
        value

let private writeGatherCsv samplesRoot rows =
    let csvPath = Path.Combine(outputDirectory samplesRoot, "gather.csv")
    Directory.CreateDirectory(Path.GetDirectoryName csvPath) |> ignore

    let header =
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

    let lines =
        seq {
            yield header
            yield! rows
        }
        |> Seq.map (List.map csvEscape >> String.concat ",")

    File.WriteAllLines(csvPath, lines)
    printfn "wrote %s" (relativePath samplesRoot csvPath)

let private gatherExistingLogs samplesRoot =
    let tmp = outputDirectory samplesRoot

    if Directory.Exists tmp then
        Directory.EnumerateFiles(tmp, "*.out", SearchOption.TopDirectoryOnly)
        |> Seq.filter (fun path -> Path.GetFileName path <> "build.out")
        |> Seq.sort
        |> Seq.map (fun path ->
            let name = Path.GetFileNameWithoutExtension path
            parseStatsFromLog samplesRoot name "" "" path)
        |> Seq.toList
    else
        []

[<EntryPoint>]
let main argv =
    let defaults =
        { SamplesRoot = defaultSamplesRoot ()
          Jobs = 1
          DebugLevel = 1
          Optimize = false
          SkipBuild = false
          GatherOnly = false
          Timeout = Some(TimeSpan.FromMinutes 30.0) }

    match parseArgs defaults (argv |> Array.toList) with
    | Error exitCode -> exitCode
    | Ok options ->
        let samplesRoot = Path.GetFullPath options.SamplesRoot
        let tmp = Path.Combine(samplesRoot, "tmp")
        Directory.CreateDirectory tmp |> ignore

        use cancellation = new CancellationTokenSource()

        Console.CancelKeyPress.Add(fun args ->
            args.Cancel <- true
            eprintfn "Stopping running sample job(s)..."
            cancellation.Cancel())

        let samples = discoverSamples samplesRoot

        if options.GatherOnly then
            writeGatherCsv samplesRoot (gatherExistingLogs samplesRoot)
            0
        elif samples.Length = 0 then
            eprintfn "runAll: no sample projects found below %s" samplesRoot
            1
        else
            try
                let runOutputDir = outputDirectory samplesRoot
                if Directory.Exists runOutputDir then
                    Directory.Delete(runOutputDir, true)

                Directory.CreateDirectory runOutputDir |> ignore

                let builtSamples =
                    if options.SkipBuild then
                        samples
                    else
                        samples
                        |> Array.choose (fun sample ->
                            let exitCode = buildSample cancellation.Token options.Timeout sample |> _.GetAwaiter().GetResult()

                            cancellation.Token.ThrowIfCancellationRequested()

                            if exitCode = 0 then
                                Some sample
                            else
                                eprintfn "%s failed to build; see %s" sample.Name (relativePath samplesRoot sample.LogPath)
                                None)

                cancellation.Token.ThrowIfCancellationRequested()

                let results =
                    runWithParallelism cancellation.Token options.Jobs options.Timeout options.DebugLevel options.Optimize builtSamples
                    |> _.GetAwaiter().GetResult()

                let gatherRows =
                    results
                    |> Array.map (fun outcome ->
                        parseStatsFromLog
                            samplesRoot
                            outcome.Sample.Name
                            (formatElapsedSeconds outcome.Elapsed)
                            outcome.ExitCode
                            outcome.Sample.LogPath)
                    |> Array.toList

                writeGatherCsv samplesRoot gatherRows

                let runFailures =
                    results
                    |> Array.choose (fun outcome ->
                        if outcome.ExitCode = 0 then
                            None
                        else
                            Some(outcome.Sample, outcome.ExitCode))

                for sample, exitCode in runFailures do
                    eprintfn "%s failed with exit code %d; see %s" sample.Name exitCode (relativePath samplesRoot sample.LogPath)

                let buildFailures = samples.Length - builtSamples.Length
                if buildFailures > 0 || runFailures.Length > 0 then 1 else 0
            with :? OperationCanceledException ->
                130
