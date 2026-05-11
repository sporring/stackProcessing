module RunAll

open System
open System.Diagnostics
open System.IO
open System.Text.RegularExpressions
open System.Threading
open System.Threading.Tasks

type Options =
    { SamplesRoot: string
      Jobs: int
      DebugLevel: int
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

let usage () =
    printfn "Usage: ./runAll.sh [-j jobs] [--skip-build] [--debug-level N]"
    printfn ""
    printfn "Runs all sample projects. The default is sequential, which gives cleaner"
    printfn "per-sample timing measurements. Use -j with a value greater than 1 to run"
    printfn "multiple samples in parallel."
    printfn ""
    printfn "Options:"
    printfn "  -j, --jobs N       Run up to N samples at once."
    printfn "  -p, --parallel    Run with one job per logical CPU."
    printfn "  --skip-build      Run samples without first building them."
    printfn "  --gather-only     Regenerate tmp/gather.csv from existing sample logs."
    printfn "  --debug-level N   Pass -d N to each sample. Defaults to 1."
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
          LogPath = Path.Combine(samplesRoot, "tmp", "runAll", name + ".out") })
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
            runProcessAsync cancellationToken sample.LogPath sample.Directory "dotnet" [ "build"; sample.Project; "--verbosity"; "q" ] None timeout

        File.AppendAllText(sample.LogPath, $"Build finished in {result.Elapsed}.{Environment.NewLine}")
        if result.TimedOut then
            File.AppendAllText(sample.LogPath, $"Build timed out after {result.Elapsed}.{Environment.NewLine}")
        return result.ExitCode
    }

let private runSample (cancellationToken: CancellationToken) timeout debugLevel (sample: Sample) =
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
                [ "run"; "--no-build"; "--verbosity"; "q"; "--"; "-d"; string debugLevel ]
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

let private runWithParallelism (cancellationToken: CancellationToken) jobs timeout debugLevel (samples: Sample array) =
    task {
        use gate = new SemaphoreSlim(jobs)

        let runOne sample =
            task {
                do! gate.WaitAsync(cancellationToken)

                try
                    return! runSample cancellationToken timeout debugLevel sample
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

let private parseStatsFromLog name elapsedSeconds exitCode logPath =
    let lines =
        if File.Exists logPath then
            File.ReadAllLines logPath
        else
            [||]

    let sinkSummary =
        lines
        |> Array.rev
        |> Array.tryPick (
            tryRegex
                @"estimated peak memory (?<estimatedPeakMemory>\S+) KB / .*actual process RSS peak delta (?<peakMemory>\S+) KB .*actual time (?<actualTime>\S+) s"
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
        sinkSummary |> Option.map (groupValue "actualTime") |> Option.defaultValue ""

    [ name
      estimatedPeakMemory
      peakMemory
      peakImages
      actualRunSeconds
      elapsedSeconds
      string exitCode ]

let private csvEscape (value: string) =
    if value.Contains(',') || value.Contains('"') || value.Contains('\n') then
        "\"" + value.Replace("\"", "\"\"") + "\""
    else
        value

let private writeGatherCsv samplesRoot rows =
    let csvPath = Path.Combine(samplesRoot, "tmp", "runAll", "gather.csv")
    Directory.CreateDirectory(Path.GetDirectoryName csvPath) |> ignore

    let header =
        [ "name"
          "estimatedPeakMemoryKB"
          "peakMemoryKB"
          "peakImages"
          "actualRunSeconds"
          "processElapsedSeconds"
          "exitCode" ]

    let lines =
        seq {
            yield header
            yield! rows
        }
        |> Seq.map (List.map csvEscape >> String.concat ",")

    File.WriteAllLines(csvPath, lines)
    printfn "wrote %s" (relativePath samplesRoot csvPath)

let private gatherExistingLogs samplesRoot =
    let tmp = Path.Combine(samplesRoot, "tmp", "runAll")

    if Directory.Exists tmp then
        Directory.EnumerateFiles(tmp, "*.out", SearchOption.TopDirectoryOnly)
        |> Seq.filter (fun path -> Path.GetFileName path <> "build.out")
        |> Seq.sort
        |> Seq.map (fun path ->
            let name = Path.GetFileNameWithoutExtension path
            parseStatsFromLog name "" "" path)
        |> Seq.toList
    else
        []

[<EntryPoint>]
let main argv =
    let defaults =
        { SamplesRoot = defaultSamplesRoot ()
          Jobs = 1
          DebugLevel = 1
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
                                eprintfn "%s failed to build; see samples/tmp/runAll/%s.out" sample.Name sample.Name
                                None)

                cancellation.Token.ThrowIfCancellationRequested()

                let results =
                    runWithParallelism cancellation.Token options.Jobs options.Timeout options.DebugLevel builtSamples
                    |> _.GetAwaiter().GetResult()

                let gatherRows =
                    results
                    |> Array.map (fun outcome ->
                        parseStatsFromLog
                            outcome.Sample.Name
                            (sprintf "%.3f" outcome.Elapsed.TotalSeconds)
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
                    eprintfn "%s failed with exit code %d; see samples/tmp/runAll/%s.out" sample.Name exitCode sample.Name

                let buildFailures = samples.Length - builtSamples.Length
                if buildFailures > 0 || runFailures.Length > 0 then 1 else 0
            with :? OperationCanceledException ->
                130
