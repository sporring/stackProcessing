# Sample Run Fixes

This file records issues found while running the sample projects and Studio JSON
graphs as user-facing examples.

## RunAll

- Added `--disable-build-servers` and `--no-restore` to child sample builds in
  `RunAll`. Direct builds of the same projects succeeded quickly, while nested
  builds from the runner could otherwise stall and end with `Build FAILED` after
  several minutes and zero compiler errors. For the current sweep, the solution
  is prebuilt and `RunAll` is run with `--skip-build`.
- Trimmed the heavier learning samples to representative ranges so they behave
  like examples rather than stress tests:
  `affineKeypointRegistration`, `fftGaussianCompare`, and `keypoint` now use
  smaller inputs in both their F# scripts and JSON graphs. `fftGaussianCompare`
  now uses a small synthetic noise volume instead of the full sample stack, and
  `chunk` chunks a representative range before reading the chunks back.
- Fixed `RunAll --skip-build` log handling. It now clears sample logs before
  running, instead of appending new evidence to old build/run failures.
- Adjusted samples that accidentally combined incompatible teaching inputs:
  `biasCorrection` and `multiplyMask` now use matching image/mask stacks by
  default, and mask-valued measurements convert the `0/255` rotating-box masks
  to `0/1` before reducers that require binary mask values.
- Made the buffered connected-component edit path tolerate a repeated slice
  index by replacing the buffered slice at that index instead of throwing. This
  keeps binary morphology samples from crashing if an upstream windowed stage
  re-emits a boundary slice.
- Reduced the remaining long-running samples to bounded teaching workloads:
  `keypoint`, `signedDistanceBand`, `structureTensor`, and `serialTransform`
  now operate on representative ranges, `fftGaussianCompare` uses a smaller
  synthetic volume, and `resampleAffineTrilinearSlices` writes/resamples a small
  synthetic chunk volume whose output geometry stays inside the available chunks.
- Made `affineKeypointRegistration` deterministic by feeding two tiny point-set
  streams through `readPointSet`, avoiding empty/noisy keypoint detections in a
  learning sample. `writeMatrix` now accepts a stream of matrices and writes
  indexed CSV files, so affine registration can write both the transform and its
  inverse directly from the DSL/F# pipeline.
- Fixed the direct `resampleAffineTrilinearSlices` sample to pass the same chunk
  size to the resampler that was used by `writeChunks`; the old mismatch made
  the resampler probe a non-existent chunk.

## RunJson

- Applied child-build isolation to `RunJson`, because generated graph projects
  are built in the same nested runner pattern.
- Tightened `RunJson` graph discovery so it ignores generated `bin`, `obj`,
  `RunAll`, `RunJson`, and `tmp` JSON files. Only real sample Studio graphs are
  compiled and run.
- `RunAll` completed cleanly after the fixes above: every sample reported
  `completed` with exit code `0` in `samples/tmp/runAll/gather.csv`.
- Generated graph projects now reference the already-built StackProcessing DLLs
  directly instead of using nested project references. This avoids restore/build
  stalls in generated JSON projects while still exercising the graph compiler.
- Fixed two graph-code generation issues found by JSON compile-only runs:
  histogram equalization now accepts fixed-bin histograms keyed by `float`, and
  `permuteAxes` emits unsigned axis tuples such as `(0u,2u,1u)`.
- Brought stale JSON samples back in line with their current teaching scripts:
  image-stack reads no longer masquerade as volume-file reads, mask reducers get
  `0/1` masks, object repainting uses the full sample image size, and heavier
  JSON examples now use the same bounded ranges or synthetic inputs as their
  `.fs` counterparts.
- Added a tiny shared point-set CSV for the affine-registration graph so it can
  demonstrate point-set registration deterministically without depending on
  keypoint detection finding features in a small image sample.
- Final `RunJson --samples-root samples -j 3 --timeout 5` completed cleanly:
  every JSON graph reported `completed` with exit code `0` in
  `samples/tmp/json/gather.csv`.

## Verification

- `dotnet build --verbosity q --disable-build-servers`
- `dotnet test tests/Studio.Compiler.Tests/Studio.Compiler.Tests.fsproj --verbosity q --disable-build-servers`
- `dotnet test tests/StackProcessing.Tests/StackProcessing.Tests.fsproj --verbosity q --disable-build-servers`
