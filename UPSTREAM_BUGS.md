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

## Possible But Not Yet Classified

- Avalonia file dialog behavior has intermittently appeared delayed or non-selectable on macOS.
  - This may be application timing, StorageProvider usage, or platform dialog behavior rather than an upstream bug.
  - Keep this here only as a reminder until we have a minimal reproduction.
