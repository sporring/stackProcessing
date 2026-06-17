# Developer Release Process

## Scope

This note describes the near-term release-candidate process for technically sympathetic testers. The current Studio workflow still treats the generated F# DSL as a fundamental part of execution: Studio generates a graph program, builds it with the .NET SDK, and runs the compiled program. Therefore this is a **developer release**, not yet a one-click end-user installer.

The tester is expected to have:

- .NET 10 SDK installed;
- platform matching supplied native helper libraries;
- permission to run local executables and write temporary output;
- enough disk space for input stacks, temporary outputs, and generated graph builds.

## Release Contents

A developer release should contain:

- the repository source at a tagged commit;
- native helper binaries for the target platform;
- `models/` with the current fitted/default cost model files;
- `samples/` with a few small runnable examples;
- `notes/` or a short release README with the commands below;
- optional tiny TIFF stacks for smoke testing.

Do not ask testers to reconstruct dependencies by hand. The release archive should already include the native helper binaries that match the platform.

## Versioning

Use explicit release-candidate tags:

```bash
git tag v0.1.0-rc1
```

Prefer immutable archives made from the tag, not from a working directory. Keep a short changelog entry that states:

- supported platform;
- .NET SDK version;
- known limitations;
- smoke-test command;
- major benchmark or algorithm changes since the previous RC.

## Pre-Release Checklist

Install dotnet, e.g., on Ubuntu

```bash
sudo apt-get update && sudo apt-get install -y dotnet-sdk-10.0
```

Clone stackProcessing repository

```bash
git clone https://github.com/sporring/stackProcessing.git
```

build and test:

```bash
dotnet restore StackProcessing.sln
dotnet build StackProcessing.sln --configuration Release --nologo
dotnet test --configuration Release --no-build --nologo
```

Check Studio specifically:

```bash
dotnet build src/Studio/Studio.fsproj --configuration Release --nologo
```

Check the Probe/RunSamples path using built DLLs rather than `dotnet run` where timing evidence matters:

```bash
dotnet build src/StackProcessing.Probe/StackProcessing.Probe.fsproj --configuration Release --nologo
dotnet src/StackProcessing.Probe/bin/Release/net10.0/StackProcessing.Probe.dll --help
```

For developer releases, it is acceptable that Studio itself invokes the .NET SDK to compile generated graph programs. That is part of the current design.

## Packaging

Create one archive per platform:

```text
StackProcessing-v0.1.0-rc1-macos-arm64.zip
StackProcessing-v0.1.0-rc1-windows-x64.zip
StackProcessing-v0.1.0-rc1-linux-x64.tar.gz
```

Each archive should contain the repository tree and the matching `lib/` native files. Avoid including large benchmark outputs, raw input/output stacks, or `tmp/`.

Suggested cleanup before creating the archive:

```bash
dotnet clean StackProcessing.sln --configuration Release
rm -rf tmp
find . -type d -name bin -prune -exec rm -rf {} +
find . -type d -name obj -prune -exec rm -rf {} +
```

Then recreate a Release build if you want the archive to include built binaries:

```bash
dotnet build StackProcessing.sln --configuration Release --nologo
```

For developer releases, including built binaries is convenient, but not sufficient by itself because Studio-generated graph programs still require the SDK.

## Tester Setup Instructions

Ask testers to install:

- .NET 10 SDK;
- platform tools needed by the OS to allow local executables;
- enough local disk space for generated data.

Then ask them to unpack the archive and run:

```bash
dotnet build StackProcessing.sln --configuration Release --nologo
dotnet run --project src/Studio/Studio.fsproj --configuration Release
```

For a timing-sensitive command-line run, prefer the built DLL after the initial build:

```bash
dotnet src/StackProcessing.Probe/bin/Release/net10.0/StackProcessing.Probe.dll --help
```

The distinction matters because `dotnet run` goes through the SDK/project runner and can add unpredictable fixed overhead. That overhead is acceptable for launching Studio interactively, but it should not be treated as pipeline execution cost.

## Smoke Test

The release should include one small graph and one tiny TIFF stack. The smoke test should be:

1. Start Studio.
2. Open the included graph.
3. Select the included input stack.
4. Run the graph.
5. Confirm that output files are written.
6. Confirm that the Run panel reports completion, not a build or native-library error.

The smoke graph should use conservative operations:

```text
read -> thresholdRange -> write
```

A second optional smoke graph can exercise a small streaming window:

```text
read -> binaryDilate -> write
```

Keep these graphs tiny. Their job is to detect packaging and platform errors, not to benchmark performance.

## Known Developer-Release Limitations

- Studio requires the .NET SDK because graph execution currently compiles generated F# code.
- The release is platform-specific because native helper libraries are platform-specific.
- macOS users may need to approve the app or executable in system security settings.
- Large TIFF stacks and benchmark outputs can quickly consume disk space.
- Timing results from `dotnet run` should not be used as cost-model evidence.

## Future Directions

The eventual non-technical release should remove the SDK requirement from normal use. Possible paths:

1. **Prebuilt graph runner**

   Ship a self-contained `StackProcessing.Runner` executable that loads Studio graph JSON directly and executes the graph without generating and compiling F# at run time.

2. **Self-contained Studio package**

   Publish Studio per platform with:

   ```bash
   dotnet publish src/Studio/Studio.fsproj -c Release -r <runtime-id> --self-contained true
   ```

   This should bundle the .NET runtime and managed dependencies. Native helper files still need careful platform-specific packaging.

3. **Installer/app bundle**

   For non-technical users, package as:

   - macOS `.app` in a signed/notarized `.dmg`;
   - Windows installer;
   - Linux AppImage, `.deb`, or tarball.

4. **Bundled examples and diagnostics**

   Include a first-run smoke test, example data, and a diagnostics panel that checks .NET/runtime/native-library availability before a user tries a real pipeline.

The main architectural step is the prebuilt graph runner. Until then, developer releases are the honest packaging level.
