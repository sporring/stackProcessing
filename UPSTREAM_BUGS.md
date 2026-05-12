# Upstream Bugs To Consider Reporting

This is a running list of issues we have encountered in dependencies while developing StackProcessing. Items here should be verified with small reproductions before reporting upstream.

## Avalonia / NodeEditorAvalonia

- Restoring an object after its size has temporarily been set to zero does not restore its center position correctly.
  - Observed while debugging Studio node/pin layout.
  - The exact owner is still uncertain: it may be Avalonia layout behavior, NodeEditorAvalonia behavior, or an interaction between the two.

## SimpleITK

- `BinaryMorphologicalClosingImageFilter` appears not to expose the background value setter, unlike the other binary morphological operators.
  - This matters because erosion, dilation, and related binary morphology need explicit foreground/background semantics for 0-1 `uint8` images.
  - We should prepare a minimal comparison against erosion/dilation/opening before filing.

- Speckle noise generation appeared to hang when `stddev = 0`.
  - Needs a minimal reproduction that calls the SimpleITK speckle-noise path directly.
  - If confirmed, the expected behavior should probably be either identity output or a clear argument error.

- `DiscreteGaussianImageFilter` may have unexpected behavior in the streaming/boundary setup we need.
  - We saw suspicious smoothing results while using SimpleITK's discrete Gaussian path.
  - StackProcessing currently avoids that path and builds one explicit Gaussian kernel which is reused through the existing convolution/windowing code.
  - Before reporting, we should isolate a minimal comparison between SimpleITK discrete Gaussian filtering and explicit convolution with the same kernel and boundary assumptions.

## ZarrNET

- ZarrNET appears to create debug log files as ordinary filesystem side effects.
  - We observed files named `log.txt` and a Windows-style path fragment `C:\Users\Public\biolog.txt` being created relative to the current directory or application base directory.
  - StackProcessing currently works around this in `StackIO.suppressZarrNetDebugLogging` by deleting those files and setting ZarrNET private static debug counters (`s_writeDebugCount`, `s_readDebugCount`, and local filesystem store `s_debugCount`) high enough to suppress further logging.
  - A minimal upstream report should check whether these logs are intentional, whether logging can be disabled through public API/configuration, and whether Windows absolute paths are being treated as relative filenames on non-Windows platforms.

## Possible But Not Yet Classified

- Avalonia file dialog behavior has intermittently appeared delayed or non-selectable on macOS.
  - This may be application timing, StorageProvider usage, or platform dialog behavior rather than an upstream bug.
  - Keep this here only as a reminder until we have a minimal reproduction.
