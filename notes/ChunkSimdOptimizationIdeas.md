# Chunk SIMD Optimization Ideas

This note summarizes lessons from three SIMD-oriented .NET articles and turns
them into possible StackProcessing/Chunk experiments:

- Marian Todorov, [.NET Core Concepts (SIMD, AVX, Intrinsics)](https://medium.com/@meriffa/net-core-concepts-simd-avx-intrinsics-0e30c845ebca)
- Alexandre Mutel, [10x Performance with SIMD Vectorized Code in C#/.NET](https://xoofx.github.io/blog/2023/07/09/10x-performance-with-simd-in-csharp-dotnet/)
- Bardia Mostafavi, [SIMD, a parallel processing at hardware level in C#](https://dev.to/mstbardia/simd-a-parallel-processing-at-hardware-level-in-c-42p4)

The shared lesson is not "make everything SIMD". The useful pattern is:

1. Work on contiguous spans.
2. Process the largest prefix that is a multiple of the SIMD vector width.
3. Finish the remainder with a scalar tail loop.
4. Benchmark every candidate, because vectorization can lose when access is
   irregular, inputs are small, or extra packing/copying dominates.

Recent StackProcessing work reinforces the same point. The biggest wins in the
TIFF/Zarr path came from keeping data in caller-owned `byte[]` Chunk buffers,
selecting raw/general IO paths outside hot loops, batching writes at the store
boundary, and avoiding unnecessary read-modify-write or per-chunk planning.
Those changes beat any plausible SIMD tweak to the IO path. SIMD is most useful
once the hot loop is already a contiguous memory loop with little abstraction
or allocation overhead.

## Lessons From TIFF/Zarr And Chunk Work

### Remove Abstraction Cost Before Vectorizing

The first `computeStats` cleanup removed per-pixel `box` plus
`Convert.ToDouble` for the common numeric types by dispatching to typed loops.
That is the right first step before trying `Vector<'T>`: a boxed scalar loop is
not a fair scalar baseline, and vectorizing around generic conversion would
hide the real problem.

Rule of thumb:

1. specialize the common pixel types,
2. keep a generic fallback for uncommon types,
3. verify semantics with focused tests,
4. only then compare scalar typed loops against SIMD loops.

### Keep The Byte-Buffer Contract Sacred

The TIFF work showed that raw IO should read and write directly into
`Chunk<'T>.Bytes`/borrowed `ArrayPool<byte>` buffers. A path that decodes into
an intermediate image object, typed array, or unmanaged buffer and then copies
back into Chunk memory gives away the main streaming advantage.

For SIMD experiments, this means:

- prefer kernels over `Span<byte>`/`Span<'T>` views of existing Chunk buffers,
- use `MemoryMarshal.Cast<byte, 'T>` when the representation is already correct,
- avoid allocating temporary typed arrays just to make vectorization easier,
- treat endian swapping, cast/normalize, and threshold as explicit kernels only
  when the input representation really needs transformation.

### Storage Boundaries Often Dominate

The Zarr experiments were a useful negative control. A direct local writer set a
near-ceiling for uncompressed chunk writes, and the remaining gap lived mostly
in `Zarr.NET` local-store planning and file creation, not in StackProcessing's
memory reshaping. SIMD cannot fix per-chunk path resolution, directory
creation, file-open overhead, store abstraction churn, or too-small batches.

Before proposing SIMD for an IO-adjacent benchmark, first ask:

- Are we writing full storage chunks, or causing read-modify-write?
- Is path/key planning happening inside the batch hot loop?
- Are output directories cached or repeatedly created?
- Are borrowed buffers consumed before being returned to the pool?
- Is the batch size large enough to amortize store overhead without excessive
  memory pressure?

### Plan Once, Then Run A Small Hot Loop

The raw/general TIFF split and the Zarr writer changes share the same shape:
inspect metadata, choose a path, and precompute invariants before the per-slice
or per-chunk loop. This is also the right shape for SIMD kernels. Work such as
type dispatch, pixel-size checks, row plans, coefficient selection, and full
chunk byte-count validation should happen once per stage or batch, not once per
row or pixel.

### Direct Benchmarks Are Ceilings, Not Always Designs

The direct-local Zarr writer was valuable because it showed what the filesystem
and payload layout could do. It was not the right production architecture
because Zarr key semantics belong in Zarr.NET. Use these direct experiments as
ceilings:

- direct raw TIFF copy,
- direct Zarr local write,
- no-store TIFF read plus Zarr-shaped split,
- scalar typed loop versus SIMD loop on already-resident chunks.

If the production path is far from the ceiling, look for abstraction, allocation,
layout, or batching costs before reaching for intrinsics.

## Portable Vector Pattern

The simplest managed starting point is `System.Numerics.Vector<'T>`.
`Vector<'T>.Count` gives the runtime vector width in elements. The usual loop
shape is:

```fsharp
let width = Vector<float32>.Count
let vectorEnd = values.Length - values.Length % width

let mutable acc = Vector<float32>.Zero
let mutable i = 0
while i < vectorEnd do
    acc <- acc + Vector<float32>(values.Slice(i, width))
    i <- i + width

let mutable scalar = 0.0f
for lane = 0 to width - 1 do
    scalar <- scalar + acc[lane]

while i < values.Length do
    scalar <- scalar + values[i]
    i <- i + 1
```

For Chunk, this is most relevant when `Chunk.span<'T>` exposes a contiguous
slice or thick chunk. The scalar tail matters for arbitrary image dimensions,
especially when width is not a multiple of the register width.

## Fixed-Width Intrinsics

`Vector128<'T>`, `Vector256<'T>`, and `Vector512<'T>` give more control than
`Vector<'T>`. They can be useful when we want repeatable layout decisions or
when `Vector<'T>` does not generate the code we expect.

The likely progression is:

- First write clear scalar and `Vector<'T>` versions.
- Inspect benchmark results.
- Only then try `Vector256<'T>` or hardware-specific intrinsics for the small
  number of hot loops that remain important.

The Mutel article also points to two low-level managed tricks that may be
relevant after simpler attempts:

- Use `MemoryMarshal.Cast<'T, Vector256<'T>>` where the data can be viewed as
  vector lanes without copying.
- Use `MemoryMarshal.GetReference` plus `Unsafe.Add` to remove bounds checks in
  tight loops while staying GC-friendly, instead of pinning pointers too early.

These should be reserved for small, isolated kernels with tests and benchmarks.

## Span Lessons

Many Chunk functions already naturally want `Span<'T>`/`ReadOnlySpan<'T>`:

- They avoid array slicing allocations.
- They make the contiguous memory contract explicit.
- They are the natural bridge to `Vector<'T>`, `MemoryMarshal.Cast`, and unsafe
  ref-based loops.

Possible cleanup rule: managed Chunk kernels should prefer accepting
`ReadOnlySpan<'T>` and `Span<'T>` internally, even if the public API remains
`Chunk<'T> -> Chunk<'T>`.

This would make it easier to benchmark scalar, portable SIMD, and intrinsic
variants behind the same Chunk-level function.

## Candidate Chunk Experiments

### Reductions

Good first targets because they are linear and contiguous:

- `sum`
- `prod` for float32, if still useful
- `minMax`
- `mean` and variance accumulation
- dense histogram preprocessing where the operation is linear before binning

Current status: `computeStats` now has typed scalar paths for common numeric
pixel types to avoid per-pixel boxing. The next experiment should compare those
typed loops against SIMD implementations for `uint8`, `uint16`, `float32`, and
`float64`.

Expected benefit: modest to good for `float32`, `float64`, `int32`, and
possibly `uint16`. For `uint8`, widening costs may eat some of the gain unless
the implementation accumulates partial sums efficiently. Welford variance is
serially dependent, so a SIMD version will probably use vectorized `sum`,
`sumSq`, `min`, and `max` per chunk, then combine chunk-level statistics with
the existing stable merge formula.

### Pixelwise Math

Likely good SIMD candidates:

- scalar add/subtract/multiply/divide
- image pair add/subtract/multiply
- threshold/comparison to binary mask
- clamp and normalize
- absolute difference
- blend/select-like operations

These are often memory-bandwidth-bound, so the benchmark should report both
runtime and bandwidth-equivalent GB/s.

### Vector Chunks

Vector-valued Chunk images are especially tempting because each pixel has a
small fixed component count:

- vector dot
- vector magnitude
- vector angle
- componentwise multiply/add
- structure tensor per-pixel outer product before smoothing

Two layouts should be compared:

- current interleaved component layout
- component-plane layout, if we ever introduce one for selected operations

Interleaved layout is convenient for pixelwise operations. Component-plane
layout is often better for long linear scans of one component at a time.

### Complex64 Interleaved Helpers

Possible SIMD candidates:

- complex modulus
- complex conjugate
- complex multiply
- spectral kernel multiplication for convolution theorem paths

These work over interleaved real/imaginary float32 pairs. SIMD can help, but the
lane shuffles must be measured carefully.

### Copy and Cast Paths

Candidates:

- `uint8 -> float32`
- `uint16 -> float32`
- `float32 -> uint8` clamp/round
- `float32 -> uint16` clamp/round

These are important because explicit full-volume casting creates memory
pressure. Even when the best convolution path does casts inside C++, fast
managed cast kernels can still matter for IO, display, normalization, and
analysis stages.

The TIFF reader/writer lessons make this category especially important:
cast-on-read and cast-on-write should preserve the Chunk `byte[]` ownership
model. The fast path should read into a pooled Chunk buffer and transform
directly into the destination Chunk buffer, not materialize old image classes or
temporary typed arrays.

### Endian Swap And Format Repair

Endian swapping is a plausible SIMD candidate because it is a regular byte
shuffle over contiguous data. It should live as a selected format-repair kernel,
not as a branch inside the raw TIFF hot loop. The reader should decide once from
metadata whether it is on the raw native-endian path or on a general path that
includes byte swapping.

Likely experiments:

- `uint16` byte swap via scalar, `Vector<byte>` shuffle if available, and
  fixed-width intrinsics,
- `uint32`/`float32` four-byte reversal,
- complex64/complex128 interleaved re/im preservation while swapping each
  scalar lane.

Do not mix this into the sacred raw native-endian path unless the benchmark
proves the branch is free.

## Plane and Slice Structure

StackProcessing's natural unit is often a 2D slice or a small group of slices.
That structure can be exploited even without a specific Plane API.

Possible "plane-aware" experiments:

- Add internal helpers for `forEachRowSpan`, `forEachPlaneSpan`, and
  `forEachThickSliceSpan`.
- Separate kernels that are truly 2D from kernels that need a z halo.
- For row-linear operations, vectorize along x where memory is contiguous.
- For y/z operations, either accept strided access or explicitly tile into
  contiguous buffers only when the tile reuse justifies the copy.
- Compare single-slice, 64-thick, and full-volume chunk variants for the same
  operation.

The warning from the FFT/Zarr work applies here too: subchunking only helps if
the storage layer receives full chunks and avoids rewrite/read-modify-write
cycles.

## Where SIMD Is Unlikely To Help First

Do not start with:

- TIFF/Zarr IO hot paths, unless the measured kernel is a local copy, cast,
  endian-swap, threshold, or buffer-split loop.
- C++-backed convolution, median, FFT, connected components, or morphology
  kernels that already spend most time in native code.
- Random-access histogram bin updates.
- Algorithms dominated by branching or irregular neighborhoods.

For these, better wins are usually layout, chunk size, reduced copying, tighter
window lifetime, or native implementation changes.

The Zarr experiments add one more warning: if the profile shows time in store
planning, directory creation, file opening, task scheduling, or read-modify-write
of partial chunks, SIMD is looking in the wrong layer.

## Benchmark Plan

Add small focused benchmarks before changing user-facing stages:

1. `chunk-simd-reductions`
   - scalar vs `Vector<'T>` vs optional `Vector256<'T>`
   - `sum`, `minMax`, `meanVariance`
   - `uint8`, `uint16`, `float32`
   - include the current typed `computeStats` loop as the scalar baseline

2. `chunk-simd-pixelwise`
   - scalar multiply, add, threshold, clamp
   - measure 256^3, 512^3, and 1024^3 if memory permits

3. `chunk-simd-vector`
   - vector dot/magnitude/angle for 3-component float32 vector chunks
   - compare interleaved and component-plane test layouts

4. `chunk-simd-cast`
   - `uint8 <-> float32`
   - `uint16 <-> float32`
   - include peak memory and allocation counts

5. `chunk-plane-row-kernels`
   - row-major x pass
   - y pass
   - z pass
   - compare direct strided loops against tile-to-contiguous loops

6. `chunk-format-repair`
   - endian swap for `uint16`, `uint32`, `float32`, `float64`
   - complex64 and complex128 interleaved re/im buffers
   - scalar versus SIMD/intrinsics

7. `chunk-storage-boundary-ceilings`
   - direct raw TIFF copy versus production TIFF read/write
   - no-store TIFF read plus Zarr-shaped split
   - direct local Zarr write versus production `Zarr.NET` batch write
   - not a SIMD benchmark, but essential context for deciding whether SIMD is
     the next bottleneck

For each benchmark, record:

- wall time
- internal time
- peak RSS
- Chunk live/peak buffers when relevant
- effective throughput in GB/s
- scalar baseline checksum
- allocation count or allocated bytes for managed kernels
- whether the benchmark is a production path or a direct ceiling experiment

## Practical Rule

Use this order:

1. Remove avoidable copies.
2. Remove avoidable boxing, virtual calls, and per-element generic conversion.
3. Ensure the data is contiguous in the hot loop.
4. Move path/type/metadata decisions outside the hot loop.
5. Use `Span`/`ReadOnlySpan`.
6. Add `Vector<'T>` with scalar tail.
7. Benchmark against a typed scalar baseline and a direct ceiling when possible.
8. Only then try `Vector256<'T>`, `Unsafe.Add`, or hardware-specific intrinsics.

This keeps SIMD as one optimization layer rather than a distraction from the
larger Chunk wins: memory layout, chunk lifetime, streaming boundaries, and
storage alignment.
