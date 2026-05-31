# Convolution Streaming Notes

This note records lessons from the experimental attempt to beat the current
StackProcessing convolution path with a native F#/flat-buffer implementation.
The short conclusion is that dense generic convolution is not a good immediate
target for replacing SimpleITK. The more promising direction is to improve the
central streaming data representation so the optimiser can choose larger
valid-output blocks while still staying memory bounded.

## Current StackProcessing Path

The present StackProcessing convolution path is:

```text
slice stream
-> z-window with halo
-> slab conversion
-> SimpleITK convolution
-> emit selected slices
```

This keeps the implementation simple and delegates dense convolution to ITK's
well-optimised C++ code. The weakness is that the slab depth is usually chosen
from the kernel/window requirement, so SimpleITK is often called on many small
slabs. This repeats filter setup and limits ITK's ability to amortise work over
a larger output region.

The optimiser could improve this without changing the mathematical operation by
choosing a larger z-window:

```text
input block depth = output block depth + z halo
compute output block depth valid slices in one SimpleITK call
emit output block depth slices
```

For example, for a `3x3x3` kernel:

```text
minimal path: input depth 3, emit 1 slice
block path:   input depth 66, emit 64 slices
```

This stays streaming-friendly but gives SimpleITK a larger region per call.

## ITK Lessons

SimpleITK delegates convolution to ITK's `ConvolutionImageFilter`, which builds
a small internal pipeline:

```text
ConvolutionImageFilter
-> flip kernel
-> pad even kernel dimensions if needed
-> ImageKernelOperator
-> NeighborhoodOperatorImageFilter
```

The main work is done by `NeighborhoodOperatorImageFilter`. The important
features are:

- It uses `NeighborhoodAlgorithm::ImageBoundaryFacesCalculator` to split each
  output region into an interior region and boundary faces.
- The interior region avoids boundary-condition work.
- The inner product is a tight coefficient loop over a neighbourhood iterator.
- It uses C++ template-specialised iterators over ITK image buffers.
- It uses ITK dynamic multi-threading over large output regions.
- Thread scheduling happens once per filter execution, not once per emitted
  slice.

This explains why a naive `Parallel.For` inside each emitted 2D slice performed
poorly in the experiments. The granularity is wrong: it creates many small
parallel jobs instead of splitting a larger 3D output region.

## SixLabors Lessons

SixLabors/ImageSharp is primarily a 2D image-processing library, so it is not a
drop-in backend for 3D stack convolution. The useful lessons are design-level:

- Work over contiguous row/span-style buffers.
- Split interior and border handling.
- Keep hot loops over primitive arrays, not tuples, closures, or generic object
  access.
- Make processing regions explicit.
- Avoid accidental allocation in inner loops.
- Use parallelism only when the work units are coarse enough.

These lessons are already compatible with StackProcessing's direction. The
important mapping is:

```text
ImageSharp row/span processing
-> StackProcessing flat 2D slice buffers

ImageSharp rectangle region
-> StackProcessing valid z-range inside a halo window

ImageSharp border handling
-> StackProcessing explicit halo/boundary policy
```

They do not by themselves make a dense F# convolution loop competitive with ITK.

## Native F# Experiment Summary

Several native center-slice convolution variants were tried:

- direct nested loops
- precomputed kernel offsets
- interior fast path plus checked border path
- tap-outer loop order
- row-parallel `Parallel.For`
- column-parallel `Parallel.For`

The small `24^3` tests showed that native code could win for a tiny `3x3x3`
case, but larger or wider kernels favoured SimpleITK. A larger `256^3` test with
an asymmetric `3x3x3` kernel showed the scaling problem clearly:

```text
256^3, 3x3x3

slab/SimpleITK:        about 3 s
native serial:         about 15 s
native parallel rows:  about 92 s
```

A smaller `96^3` row-vs-column sanity check also showed that naive parallelism
inside each emitted slice is not useful:

```text
96^3, 3x3x3

slab/SimpleITK:             about 0.95 s
native serial:              about 1.34 s
native parallel rows:       about 5.36 s
native parallel columns:    about 5.48 s
```

The experiments were removed from the main code after recording these results.

## Basic Improvement Ideas

The most promising basic improvements are not new inner loops, but better
streaming orchestration.

### Larger Valid Blocks

Let the optimiser choose a larger z-window and emit several valid slices per
SimpleITK call. This should reduce repeated SimpleITK setup and give ITK more
work per invocation while staying memory bounded.

### Explicit Valid Region Type

Introduce or enrich a central streaming data type that can represent:

```text
buffered z-range
halo range
valid output range
flat buffer access
slice emission
```

This would make it easier to express "compute only the valid part" without
always collapsing to either a minimal window or a full slab.

### Optimiser-Controlled Throughput vs Memory

The optimiser should eventually choose between:

```text
small window, low memory, more calls
larger block, higher memory, fewer calls
```

This is a natural cost-model decision. It is not pleasant to expose directly in
the user-facing DSL.

### Keep SimpleITK for Dense Generic Convolution

For dense generic kernels, SimpleITK/ITK should remain the default backend for
now. Native code may still become useful for specialised cases, but those should
not be introduced until the central streaming/block representation is clearer.

