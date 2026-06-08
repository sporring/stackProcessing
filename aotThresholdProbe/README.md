# AOT Hello Probe

Tiny standalone probe for testing `PublishTrimmed` and `PublishAot` without the full
StackProcessing, SimpleITK, Zarr, and benchmark dependency graph.

This is intentionally not a StackProcessing pipeline. It is just a publish-mode sanity
check that should be easy to build, run, and compare.

Build and run normally:

```bash
dotnet run -c Release --project aotThresholdProbe/aotThresholdProbe.fsproj
```

Publish trimmed/AOT for the host platform:

```bash
aotThresholdProbe/publish-aot.sh
tmp/publish-aot-hello-probe/aotThresholdProbe
```
