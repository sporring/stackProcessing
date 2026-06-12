# Chunk Payload

The active StackProcessing image payload is `Chunk<'T>`.

This file is kept only so existing links into `notes/Image.md` land on the current
payload model. The active architecture notes are:

- [Concepts.md](Concepts.md)
- [StackProcessing.md](StackProcessing.md)
- [ChunkBackboneStageGaps.md](ChunkBackboneStageGaps.md)

## Chunk Shape

`Chunk<'T>` is an owned byte-backed image block with typed span access and
explicit release semantics:

```fsharp
type Chunk<'T when 'T: equality> =
    { Size: uint64 * uint64 * uint64
      Bytes: byte[]
      ByteLength: int
      Release: unit -> unit
      RefCount: int ref }
```

The active runtime uses chunks for sources, sinks, local image processing,
reducers, vector images, complex images, Studio lowering, Probe, and samples.

## Algorithm Location

- `src/Chunk/Chunk.fs` owns the structural payload: allocation, reference
  counting, shape, typed spans, and indexing helpers.
- `src/Chunk/ChunkFunctions.fs` owns chunk-local image-processing algorithms.
- `src/StackProcessing.Core` lifts Chunk functions into streaming stages,
  sources, sinks, cost models, graph labels, and Studio-facing APIs.

## Representation Choices

- Scalar chunks use one logical component per pixel.
- Vector chunks use a component dimension in the chunk payload.
- Complex64 chunks use interleaved `float32` real/imaginary pairs.
- Preferred storage types are `uint8`, `uint16`, `float32`, and
  complex64-interleaved `float32`.

## Ownership Rule

Stages consume owned chunks and release them at the defined boundary. Retain a
chunk before reusing it. This keeps larger-than-memory workflows predictable and
returns pooled buffers promptly.
