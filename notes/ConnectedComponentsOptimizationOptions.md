# Connected Components Optimization Options

This note collects options for improving StackProcessing's streamed connected
component labeling. It is motivated by block-based GPU connected-component
labeling work, in particular the YACCLAB/Block-Based Union Find line cited by
Allegretti, Bolelli, and Grana, and by the PyTorch wrapper repository
`zsef123/Connected_components_PyTorch`, which states that it follows the
YACCLAB block-based union-find algorithm.

The GPU code is not directly reusable in StackProcessing. The useful part is
the algorithmic shape: reduce work by labeling blocks, resolve equivalences on
boundaries, and keep union-find pressure low.

## Current StackProcessing Shape

The current Chunk path lives in
`src/StackProcessing.Core/StackConnectedComponents.fs`.

Important pieces:

- `labelConnectedComponentsSliceSauf3DArrayUf`
  labels one 2D `uint8` slice into a `uint32` label slice.
- It scans every foreground voxel and considers west, north, and previous-slice
  same-position candidates.
- `DenseUInt32UnionFind` stores provisional equivalences.
- `compactConnectedComponentLabelsArrayUf` compacts provisional labels at the
  end of the retained stack/window.
- `connectedComponentsSauf3DUInt8UInt32ParallelCollect`
  windows slices into slabs, labels slabs in parallel, and later stitches
  touching labels across slab boundaries.

This is already a good streaming-friendly baseline, but it still performs
voxel-level tests, voxel-level label writes, and full-boundary stitching.

## What Block-Based Algorithms Suggest

### Label Blocks Instead Of Voxels

The central block-based idea is to treat small groups of pixels as the unit of
local analysis. For 2D images this is commonly a `2x2` block; for streamed 3D
volumes a natural first step is `2x2x1`, keeping the z-streaming contract
unchanged.

Potential wins:

- fewer provisional labels,
- fewer union-find operations,
- fewer neighbor tests,
- better branch locality,
- smaller frontier representation for slab stitching.

Connectivity caveat:

- A single label per nonempty `2x2` block naturally merges diagonal pixels.
- That is compatible with 8-connected 2D and 26-connected 3D style semantics.
- It is not directly compatible with the current 6-connected voxel path.
- For 6-connectivity, a block must be able to hold multiple local subcomponents,
  or it must use a decision table that preserves face-connected structure.

This means the easiest first block benchmark is a 26-connectivity variant. A
6-connectivity block path is possible, but it should be treated as a separate
design rather than a drop-in replacement.

### Precomputed Block Mask Tables

A `2x2` block has only 16 foreground masks. For 6-connected 3D with current and
previous slices, the local decision can still be represented compactly by mask
tables.

Useful tables:

- local components inside a `2x2` mask for 4-connectivity,
- local components inside a `2x2` mask for 8-connectivity,
- west-neighbor contact masks,
- north-neighbor contact masks,
- previous-slice z-contact masks,
- output-pixel-to-local-subcomponent mapping for final label expansion.

The hot loop then becomes:

1. load a block mask,
2. look up local subcomponents,
3. merge only the table-selected neighbor contacts,
4. write one or more block labels.

This should reduce unpredictable branching compared with per-voxel tests.

### Boundary-Only Stitching

The current slab stitch scans the full first/last label slice pair and unions
labels when both sides are nonzero. A block path can expose a smaller frontier:

- boundary block masks,
- boundary block labels,
- optionally per-subcomponent labels for 6-connectivity.

For large slabs, this does not change asymptotic boundary size, but it can cut
the constant factor and avoid reading/writing full label chunks during stitch
planning. It also gives a cleaner place to precompute slab-boundary
equivalences.

### Delay Label Expansion

If the next consumer needs object statistics rather than a label volume, a
block-label representation can be kept internally and expanded only at the
boundary where a label image is emitted.

Possible intermediate representations:

- `uint32` label per block for 26-connectivity,
- `uint32[]` small fixed sublabel payload per block for 6-connectivity,
- run/block descriptors for object streaming.

This is only worth doing if a benchmark shows label-image writes or compaction
dominate. It complicates downstream stages, so it should remain an internal
special path at first.

## Optimization Options

### Option A: Tune The Existing Voxel Path

Low risk, closest to current semantics.

Ideas:

- Split the `previousLabels = None` and `Some` cases into separate specialized
  functions, so the branch is outside the row loop.
- Hoist row-offset calculations and use explicit row spans where possible.
- Add a fast background-skip path using row scans, especially for sparse masks.
- Count foreground density and unions for diagnostics.
- Benchmark `Find`/compression timing: no compression in the labeling pass,
  occasional row compression, or slice-end compression.
- Avoid retaining all label chunks when the stage can stream completed chunks
  safely. This may require frontier tracking.

Pros:

- preserves current 6-connected behavior,
- easiest to validate,
- low implementation risk.

Cons:

- still voxel-level,
- may not close the gap if union-find or label writes dominate.

### Option B: Block-Based 26-Connectivity Prototype

Best first block experiment.

Use `2x2x1` blocks and one provisional label per nonempty block. Neighbor
contacts check west, north, and previous-slice adjacent/touching blocks under
26-connectivity semantics.

Pros:

- simple block representation,
- closest to the cited block-based union-find style,
- likely fewer labels and unions,
- good ceiling for what block-based CCL can buy.

Cons:

- not equivalent to current 6-connected output,
- must be exposed as a different connectivity mode or benchmark-only variant.

### Option C: Block-Based 6-Connectivity Prototype

Use `2x2x1` blocks, but allow multiple local components per block according to
4-connected 2D structure plus z face contacts.

Possible representation:

- a 4-bit foreground mask,
- a table with up to two local subcomponents for 4-connected `2x2`,
- subcomponent labels stored in small fixed arrays or packed lanes,
- expansion table from four voxels to subcomponent id.

Pros:

- can preserve current 6-connected semantics,
- still reduces branch work with lookup tables,
- may reduce union calls for common dense masks.

Cons:

- more complex than 26-connectivity,
- less label-count reduction than one-label-per-block,
- needs careful tests for all block masks and z contacts.

### Option D: Slab Frontier Stitching

Independent of the local labeling algorithm.

Instead of stitching by scanning full boundary label slices, each slab result
could include a compact frontier:

- first and last slice boundary masks,
- local labels or block labels on the boundary,
- dimensions and local label base.

The stitch stage can then union only boundary foreground contacts. For the
current voxel path this may still require scanning `width * height`, but it can
avoid reloading whole label chunks if the frontier is already captured during
labeling. For block paths the frontier is naturally smaller.

### Option E: Native/C++ Hot Loop

Connected components is branch-heavy and union-find-heavy. If managed F# loops
remain expensive after block experiments, a native Chunk kernel may make sense.

Candidate native boundary:

- input: one slab of `uint8` slices,
- output: `uint32` label slices plus compact frontier metadata,
- implementation: C++ block/SAUF/BUF variant,
- StackProcessing keeps ownership of Chunk buffers and staging.

This should come after the managed prototypes have clarified the best
algorithmic shape.

## Benchmark Plan

Add a focused benchmark group, separate from the article-wide TIFF benchmark.

Suggested command:

```text
run-chunk-connected-components-hotloop
  --shape WxHxD
  --pattern sparse|dense|checker|diagonal|tubes|random
  --density P
  --connectivity 6|26
  --variant voxel-sauf|voxel-sauf-tuned|block2x2-26|block2x2-6|block2x2-frontier
  --window-size N
  --workers N
  --iterations N
```

Measure:

- total wall time,
- label time,
- compaction time,
- slab stitch time,
- provisional label count,
- final object count,
- union count,
- `Find` count if cheap to measure,
- foreground voxel count,
- allocated bytes,
- peak Chunk memory,
- output checksum.

Patterns:

- `empty`: all background, validates background skip overhead.
- `full`: all foreground, should be one component and exposes best-case dense
  behavior.
- `checker`: high component count and worst-case label pressure.
- `diagonal`: distinguishes 6-connectivity from 26-connectivity.
- `z-lines`: stresses previous-slice stitching.
- `xy-tubes`: many long components with modest boundary work.
- `random`: density sweep, for example 1%, 5%, 25%, 50%, 75%.

Shapes:

- `256x256x256` for fast iteration,
- `512x512x512` for realistic memory behavior,
- `1024x1024x256` or `1024x1024x1024` only when the smaller runs are stable.

## Correctness Tests

Before performance claims, add small deterministic tests:

- all 16 `2x2` masks for 4- and 8-connectivity,
- pairs of `2x2` masks across west and north boundaries,
- pairs of `2x2` masks across previous-slice z boundaries,
- diagonal-only cases that must differ between 6 and 26 connectivity,
- slab split invariance: results should be equivalent for window sizes
  `1`, `2`, `3`, and full depth,
- worker-count invariance for parallel slab collection.

Label numbers do not need to be identical between algorithms, but component
partitions must be equivalent. Tests should compare partition equivalence, not
raw label ids, except where a compact deterministic labeling order is part of
the contract.

## Recommended Sequence

1. Add instrumentation to the current voxel path:
   provisional labels, unions, compaction time, stitch time.
2. Build the focused hot-loop benchmark.
3. Tune the existing voxel path enough to get a fair scalar baseline.
4. Prototype `block2x2-26` as a benchmark-only ceiling.
5. If `block2x2-26` is promising, design the `block2x2-6` decision-table path.
6. Add frontier metadata to slab results and compare full-boundary stitching
   against frontier stitching.
7. Only after the algorithmic shape is clear, consider moving the hot loop to a
   native kernel.

## Open Questions

- Should `connectedComponents` continue to mean 6-connectivity by default, with
  26-connectivity as an explicit option?
- Should object-statistics consumers be allowed to use a block-label
  representation without expanding a full label volume?
- Is the benchmark dominated by labeling, compaction, stitching, or output
  label writes for the masks used in the article?
- Does the current `UInt32` label type remain enough for worst-case checkerboard
  volumes at target sizes?
- Is deterministic label numbering required, or only deterministic component
  partitions?

