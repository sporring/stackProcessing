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

The second `computeStats` cleanup was even more important. The typed loop still
used Welford's online variance update, which is numerically stable but serial:
each pixel paid for a floating division and a mean update. For chunk-local
statistics, a one-pass `sum`/`sumSq`/`min`/`max` loop followed by a final sample
variance calculation is much cheaper and has the same algebraic output contract
for ordinary image data. The existing `addStats` merge still preserves the
streaming reducer shape across chunks.

Focused benchmark, `1024x1024x64`, 67,108,864 values, 7 iterations:

| Type | old `computeStats` | new `computeStats` |
|---|---:|---:|
| `uint8` | ~0.345 s/iter | ~0.069 s/iter |
| `uint16` | ~0.345 s/iter | ~0.069 s/iter |
| `float32` | ~0.353 s/iter | ~0.069 s/iter |
| `float64` | ~0.350 s/iter | ~0.064 s/iter |

This is roughly a 5x production improvement without SIMD. The lesson is useful:
the first optimization question for reductions is not "can this be vectorized?"
but "is the scalar recurrence actually necessary?"

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

Current status:

- `computeStats` has typed scalar paths for common numeric pixel types to avoid
  per-pixel boxing.
- The production paths for `uint8`, `uint16`, and `float64` now use concrete
  `Vector<'T>` implementations behind type dispatch.
- The production `float32` path uses a two-pass vector implementation: first
  vector sum/min/max to compute the mean, then vector sum of squared deviations.
  This is slower than the fastest `sumSq - sum*sum/n` variant, but it avoids the
  large-offset/small-variance cancellation exposed by the adversarial tests.
- A focused benchmark command exists:
  `run-chunk-simd-reductions --pixel-type UInt8|UInt16|Float32|Float64 --shape WxHxD --variant computeStats-current|sum-scalar|moments-scalar|sum-vector|moments-vector|sum-vector-accurate|moments-vector-accurate`.

Focused SIMD benchmark, same `1024x1024x64` shape:

| Variant | `float32` | `float64` |
|---|---:|---:|
| `sum-scalar` | ~0.066 s/iter | ~0.072 s/iter |
| `moments-scalar` | ~0.066 s/iter | ~0.071 s/iter |
| `sum-vector` | ~0.018 s/iter | ~0.037 s/iter |
| `moments-vector` | ~0.018 s/iter | ~0.037 s/iter |
| `sum-vector-accurate` | ~0.017 s/iter | n/a |
| `moments-vector-accurate` | ~0.020 s/iter | n/a |

The `float32` "accurate" variants widen the `float32` lanes to `float` lanes
before accumulating `sum` and `sumSq`. In the focused benchmark, the accurate
moments variant was only modestly slower than the fast float-lane variant and
matched the scalar moments checksum, while the fast float-lane moments checksum
differed slightly. This makes the widened `float32 -> float` variant the more
credible production candidate.

Widened integer SIMD results, same shape:

| Variant | `uint8` | `uint16` |
|---|---:|---:|
| `sum-vector` | ~0.016 s/iter | ~0.030 s/iter |
| `moments-vector` | ~0.020 s/iter | ~0.031 s/iter |

The integer variants widen before accumulation and periodically flush partial
lane sums so the vector accumulators cannot overflow on larger chunks. `uint8`
is a very strong candidate. `uint16` still improves meaningfully, but its
`sumSq` path is heavier because squared `uint16` values need `uint64`
accumulators.

Compared with the old Welford-based `computeStats`, scalar moments are about 5x
faster, `float32` vector moments are about 20x faster, and `float64` vector
moments are about 9.5x faster. Compared with the new production scalar path,
SIMD still offers about 3.8x for `float32` and about 1.9x for `float64`.

The production SIMD decision needs care:

- SIMD changes accumulation order, so exact low-bit agreement is not expected.
- The fastest `float32` vector experiment accumulates in `float32` lanes before
  converting to `float`, which is fast but less accurate than the scalar
  `float` accumulator.
- The widened `float32 -> float` vector variant is a better production
  candidate because it preserves the scalar accumulator type for `sum` and
  `sumSq`.
- Integer SIMD reductions need widening (`uint8`/`uint16` -> larger lanes) to
  avoid overflow. The benchmark now does this, including periodic flushes of
  partial accumulators.

Good next reduction experiments:

1. Decide and document NaN/Inf policy. Finite adversarial cases are now tested,
   but non-finite values should not be silently locked in until the desired
   behavior is explicit.
2. Track effective GB/s and allocated bytes for article-facing reduction runs;
   the focused benchmark now reports both.
3. Consider whether `float64` should stay vectorized in production or keep the
   scalar path for maximum reproducibility. The speedup is smaller, but real.
4. Consider a pairwise/block compensated variant if future data shows
   `float64` or `float32` statistics are still too sensitive to accumulation
   order.

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

Current status:

- `float32` scalar arithmetic stages now dispatch to SIMD-backed stages:
  `addScalar`, `subScalar`, `scalarSub`, `mulScalar`, `divScalar`, and
  `scalarDiv`.
- `float32` pairwise arithmetic stages now dispatch to SIMD-backed stages:
  `add`, `subtract`, `multiply`, and `divide`.
- Existing `float32` unary stages such as `abs`, `sqrt`, `square`,
  `shiftScale`, `clamp`, intensity window, and invert already use
  `Vector<float32>`.
- A focused benchmark command exists:
  `run-chunk-pixelwise-float32 --shape WxHxD --variant scalar-add|vector-add|scalar-mul|vector-mul|scalar-pair-add|vector-pair-add|scalar-pair-mul|vector-pair-mul`.

Focused benchmark, `1024x1024x64`, 67,108,864 values:

| Variant | Scalar | Vector |
|---|---:|---:|
| unary add | ~0.164 s/iter | ~0.149 s/iter |
| unary multiply | ~0.042 s/iter | ~0.022 s/iter |
| pairwise add | ~0.258 s/iter | ~0.131 s/iter |
| pairwise multiply | ~0.229 s/iter | ~0.066 s/iter |

The exact numbers vary with ArrayPool state and memory pressure because these
kernels allocate a new output chunk each iteration. The robust conclusion is
that pairwise float32 arithmetic and multiply-like unary kernels benefit
substantially; simple unary add is mostly memory-bandwidth-bound but not worse.

Good next pixelwise experiments:

1. Add SIMD threshold/comparison to binary mask for `uint8`, `uint16`, and
   `float32`, with output type policy made explicit.
2. Add `absdiff`, `blend/select`, and min/max benchmarks.
3. For integer arithmetic, benchmark before promoting: overflow/clamp semantics
   matter and SIMD may need widening or saturating logic.
4. Separate kernel-only benchmarks from production allocation benchmarks, so we
   can distinguish arithmetic throughput from output-buffer pressure.

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

Current status and caveat:

- The current vector storage is interleaved component-fastest:
  `x0,y0,z0,x1,y1,z1,...`.
- This is good for per-pixel scalar code and compact IO, but awkward for
  `Vector<float32>` reductions such as dot and magnitude because SIMD lanes
  naturally want many `x` values, then many `y` values, then many `z` values.
- Component-plane layout would make dot/magnitude/angle and structure tensor
  outer products easier to vectorize, but it would be a larger representation
  decision.

Recommended next benchmark:

```text
run-chunk-vector-float32
  --shape WxHxD
  --components 3|6
  --layout interleaved|component-plane
  --variant dot|magnitude|angle|component-add|component-mul|structure-tensor
  --iterations N
```

The benchmark should include both a production-shaped interleaved path and a
component-plane ceiling. Only introduce a production component-plane
representation if the ceiling is large enough to justify the extra layout
complexity.

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
   - current status: implemented for current `computeStats`, scalar/vector
     reductions, production typed dispatch for `uint8`, `uint16`, stable
     two-pass `float32`, and `float64`
   - current status: finite adversarial tests cover constants, ramps, random
     tails, and large-offset/small-variance `float32`
   - later: compare portable `Vector<'T>` with `Vector256<'T>` or intrinsics
     only if the portable version remains important and stable

2. `chunk-simd-pixelwise`
   - current status: focused `float32` benchmark covers scalar/vector unary and
     pairwise add/multiply
   - current status: production float32 arithmetic stages dispatch to
     SIMD-backed kernels
   - next: threshold/comparison, absdiff, blend/select, and integer semantics

3. `chunk-simd-vector`
   - next: vector dot/magnitude/angle for 3-component float32 vector chunks
   - next: compare current interleaved layout against component-plane ceiling

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
