module RunJson

open System
open System.Diagnostics
open System.IO
open System.Security
open System.Text.RegularExpressions
open System.Threading
open System.Threading.Tasks
open Studio.Compiler
open Studio.Graph

type Options =
    { SamplesRoot: string
      Jobs: int
      DebugLevel: int
      SkipBuild: bool
      CompileOnly: bool
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

let usage () =
    printfn "Usage: ./runJson.sh [-j jobs] [--skip-build] [--compile-only] [--debug-level N]"
    printfn ""
    printfn "Compiles sample Studio JSON graphs to F#, runs them, and stores logs below"
    printfn "samples/tmp/json. The default is sequential."
    printfn ""
    printfn "Options:"
    printfn "  -j, --jobs N       Run up to N generated graphs at once."
    printfn "  -p, --parallel    Run with one job per logical CPU."
    printfn "  --skip-build      Run generated graph projects without building them first."
    printfn "  --compile-only    Generate and build F#, but do not run the programs."
    printfn "  --debug-level N   Accepted for symmetry with runAll; generated graphs carry their saved debug settings."
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
    | "--skip-build" :: rest -> parseArgs { options with SkipBuild = true } rest
    | "--compile-only" :: rest -> parseArgs { options with CompileOnly = true } rest
    | "--timeout" :: value :: rest
    | "--timeout-minutes" :: value :: rest ->
        match Double.TryParse value with
        | true, minutes when minutes = 0.0 -> parseArgs { options with Timeout = None } rest
        | true, minutes when minutes > 0.0 -> parseArgs { options with Timeout = Some(TimeSpan.FromMinutes minutes) } rest
        | _ ->
            eprintfn "runJson: --timeout expects a non-negative number of minutes"
            Error 2
    | "--samples-root" :: value :: rest -> parseArgs { options with SamplesRoot = Path.GetFullPath value } rest
    | option :: _ ->
        eprintfn "runJson: unknown option %s" option
        usage ()
        Error 2

let private relativePath root path =
    Path.GetRelativePath(root, path).Replace(Path.DirectorySeparatorChar, '/')

let private safeName (value: string) =
    Regex.Replace(value.Replace('\\', '/'), @"[^A-Za-z0-9_.-]+", "_")

let private discoverGraphs samplesRoot =
    Directory.EnumerateFiles(samplesRoot, "*.json", SearchOption.AllDirectories)
    |> Seq.filter (fun path ->
        let relative = relativePath samplesRoot path
        not (relative.StartsWith("tmp/", StringComparison.OrdinalIgnoreCase)))
    |> Seq.map (fun path ->
        let relative = relativePath samplesRoot path
        let name = relative.Substring(0, relative.Length - Path.GetExtension(relative).Length)

        { Name = name
          JsonPath = path
          WorkingDirectory = Path.GetDirectoryName path
          RunDirectory = Path.Combine(samplesRoot, "tmp", "json-build", safeName name)
          LogPath = Path.Combine(samplesRoot, "tmp", "json", name + ".out") })
    |> Seq.sortBy _.Name
    |> Seq.toArray

let private xml value =
    SecurityElement.Escape(value)

let private ensureRunProject repositoryRoot job =
    Directory.CreateDirectory job.RunDirectory |> ignore

    let projectPath = Path.Combine(job.RunDirectory, "GraphRun.fsproj")
    let stackProcessingProject = Path.Combine(repositoryRoot, "src", "StackProcessing", "StackProcessing.fsproj")
    let stackProcessingCoreProject = Path.Combine(repositoryRoot, "src", "StackProcessing.Core", "StackProcessing.Core.fsproj")
    let simpleItkManaged = Path.Combine(repositoryRoot, "lib", "SimpleITKCSharpManaged.dll")
    let simpleItkWindowsNative = Path.Combine(repositoryRoot, "lib", "SimpleITKCSharpNative.dll")
    let simpleItkLinuxNative = Path.Combine(repositoryRoot, "lib", "libSimpleITKCSharpNative.so")
    let simpleItkMacNative = Path.Combine(repositoryRoot, "lib", "libSimpleITKCSharpNative.dylib")

    let projectFile =
        $"""<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <Reference Include="SimpleITKCSharp">
      <HintPath>{xml simpleItkManaged}</HintPath>
      <Private>true</Private>
    </Reference>
    <ProjectReference Include="{xml stackProcessingProject}" />
    <ProjectReference Include="{xml stackProcessingCoreProject}" />
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
                    runProcessAsync cancellationToken job.LogPath job.RunDirectory "dotnet" [ "build"; projectPath; "--verbosity"; "q" ] None options.Timeout

                exitCode <- buildResult.ExitCode
                elapsed <- elapsed + buildResult.Elapsed
                File.AppendAllText(job.LogPath, $"Build finished in {buildResult.Elapsed}.{Environment.NewLine}")
                if buildResult.TimedOut then
                    File.AppendAllText(job.LogPath, $"Build timed out after {buildResult.Elapsed}.{Environment.NewLine}")

            if exitCode = 0 && not options.CompileOnly then
                File.AppendAllText(job.LogPath, $"{Environment.NewLine}== Run {job.Name} =={Environment.NewLine}")

                let! runResult =
                    runProcessAsync cancellationToken job.LogPath job.WorkingDirectory "dotnet" [ "run"; "--project"; projectPath; "--no-build"; "--verbosity"; "q" ] (Some(Path.Combine(job.WorkingDirectory, "lib"))) options.Timeout

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

let private writeCsv samplesRoot (results: GraphOutcome array) =
    let path = Path.Combine(samplesRoot, "tmp", "json", "gather.csv")

    let lines =
        seq {
            yield [ "name"; "elapsedSeconds"; "exitCode"; "log" ]

            for result in results do
                yield
                    [ result.Job.Name
                      sprintf "%.3f" result.Elapsed.TotalSeconds
                      string result.ExitCode
                      relativePath samplesRoot result.Job.LogPath ]
        }
        |> Seq.map (List.map csvEscape >> String.concat ",")

    File.WriteAllLines(path, lines)
    printfn "wrote %s" (relativePath samplesRoot path)

[<EntryPoint>]
let main argv =
    let defaults =
        { SamplesRoot = defaultSamplesRoot ()
          Jobs = 1
          DebugLevel = 1
          SkipBuild = false
          CompileOnly = false
          Timeout = Some(TimeSpan.FromMinutes 30.0) }

    match parseArgs defaults (argv |> Array.toList) with
    | Error exitCode -> exitCode
    | Ok options ->
        let samplesRoot = Path.GetFullPath options.SamplesRoot
        let repositoryRoot = Path.GetFullPath(Path.Combine(samplesRoot, ".."))
        let jobs = discoverGraphs samplesRoot

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
                let results =
                    runWithParallelism cancellation.Token repositoryRoot options jobs
                    |> _.GetAwaiter().GetResult()

                writeCsv samplesRoot results

                let failures = results |> Array.filter (fun result -> result.ExitCode <> 0)

                for failure in failures do
                    eprintfn "%s failed with exit code %d; see %s" failure.Job.Name failure.ExitCode (relativePath samplesRoot failure.Job.LogPath)

                if failures.Length > 0 then 1 else 0
            with :? OperationCanceledException ->
                130
