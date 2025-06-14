# 3D Stack Image Processing Library

A modular, memory-efficient F# image processing library that wraps SimpleITK for slice-wise 3D image stack processing.

---

## Core Architecture

### `StackPipeline.fs`

Defines the orchestration logic for 3D stack processing using different memory strategies.

#### Memory Profiles

* `Streaming`: Processes each slice independently (low memory usage).
* `Sliding n`: Uses a sliding window of `n` slices.
* `Buffered`: Loads the full stack into memory for batch processing.

> Includes a method to estimate memory usage based on image dimensions.

---

## Data Access and I/O

### `pipelineIO.fs`

Handles input and output of TIFF image stacks from disk.

#### Functions:

* `getDepth`: Returns the number of slices in a directory.
* `getVolumeSize`: Returns the volume dimensions (width, height, depth).
* `readSlices`: Loads image slices.
* `writeSlices`: Writes processed slices to disk.

---

## SimpleITK Processing Wrappers

### `simpleITKWrappers.fs`

Provides thin wrappers over SimpleITK image operations.

#### Key Functions:

* `toVectorUInt32`: Converts an array to a `VectorUInt32` (for pixel access).
* `getVoxelValue`: Reads a voxel intensity from a 3D image.
* Additional helpers for working with SimpleITK pixel/region APIs.

---

## Asynchronous Streaming Utilities

### `AsyncSeqExtensions.fs`

Adds functional utilities to `AsyncSeq` for working with streams of image slices.

#### Examples:

* `map`: Transforms async computations.
* `foldAsync`: Accumulates values from an `AsyncSeq`.

> Powers streaming and sliding-window processing modes efficiently.

---

## Vector Utilities

### `Vector.fs`

Defines a simple generic `Vector<'T>` type for use across modules.

#### Utility:

* `indices`: Returns an array of index positions for vector elements.

Used for:

* Axis labels
* Plotting
* Index management

---

## Example Pipeline Script

### `convolve3d.fs`

An executable entry point that demonstrates applying a processing pipeline.

#### Steps:

1. Define a `convolve3DGaussian` pipeline.
2. Load TIFF slices using `pipelineIO`.
3. Apply the pipeline.
4. Save the processed result.

> Useful as a CLI entry point or testing harness.

---

## Project Setup

### `CoreTypes.fsproj`

* Targets: `.NET 8.0`
* Includes:

  * Reference to `SimpleITKCSharpManaged.dll`
  * Ensures native libraries like `libSimpleITKCSharpNative.dylib` are copied.
  * Compilation includes core modules like `Types.fs` (not included in this snapshot).

---

## Design Highlights

* **Composable Pipelines**: Easily combine multiple operations.
* **Memory Efficient**: Slice-wise or windowed processing modes.
* **Asynchronous**: Backed by `AsyncSeq` for streaming large datasets.
* **SimpleITK Interop**: Uses managed bindings for low-level access.
* **Production Ready**: CLI and modular components for integration and testing.

