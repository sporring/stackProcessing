
# 🧪 Overview of Stack Processing Framework

## 🧱 Core Concepts

### 📸 `Slice<'T>`
- Represents a 2D image slice with elements of type `'T`
- Building block of 3D stacks
- Usually processed as an `AsyncSeq<Slice<'T>>`

---

### 🔁 `Pipe<'In, 'Out>`
- Represents a transformation from a stream of `'In` to `'Out`
- Conceptually, a stream function: `AsyncSeq<'In> -> AsyncSeq<'Out>`
- Replaces `StackProcessor` in newer versions

```fsharp
type Pipe<'In, 'Out> = {
    Name: string
    Profile: MemoryProfile
    Apply: AsyncSeq<'In> -> AsyncSeq<'Out>
}
```

---

## 🧭 Stream Processing Model

- **Source** → **Pipeline** → **Sink**
- Processing stages are composed using `>>=>` and routed using `Routing` functions

---

## 🔧 Operators & Constructors

| Symbol / Function    | Name                    | Description                                                      |
|----------------------|-------------------------|------------------------------------------------------------------|
| `>>=>`               | `composePipe`           | Compose two `Pipe`s left-to-right                                |
| `source`             | Source constructor      | Turns a file/image input into a stream of `Slice<'T>`            |
| `sink`               | Sink executor           | Runs a pipeline and finalises computation                        |
| `fromMapper`         | `map`                   | Maps values in stream (functor lift)                             |
| `fromReducer`        | `reduce`                | Computes a single scalar from a stream                           |
| `fromConsumer`       | `sink`                  | Consumes a stream for side-effects (e.g., printing)              |
| `join`               | `zipWith`               | Combine two streams pairwise with a function                     |
| `inject` | `withReducer` | Inject scalar into stream pipeline                         |
| `cacheScalar`        | Materialise scalar      | Run a scalar pipe once and reuse its result downstream           |
| `tap`                | Debug / Inspect         | Logs/prints values flowing through a pipe                        |

---

## 📦 Memory Profiles

| Name      | Meaning                                  |
|-----------|------------------------------------------|
| `Streaming` | Processes slices one-by-one             |
| `Sliding n`| Keeps `n` slices in memory (rolling)     |
| `Buffered` | Reads the full input before proceeding   |

---

## 🧭 Pipeline Example

```fsharp
let normalizeWith stats slice =
    subFloat slice stats.Mean
    |> divFloat stats.Std

let readMaker =
    readSlices<float> "image" ".tiff"
    |> source availableMemory width height depth

let statsPipe =
    readMaker >>=> computeStats >>=> cacheScalar "stats"

readMaker
    >>~> (normalizeWith, statsPipe)
    >>=> computeStats
    >>=> print
    |> sink
```

---

## 🧪 Category-Theoretic Notes

- `Pipe<'A, 'B>` is a morphism in a category
- `>>=>` is morphism composition
- `fromMapper` is `fmap`
- `fromReducer` is a monoidal reduction (`fold`)
- `inject` reflects a **Reader monad** pattern
- `cacheScalar` adds **memoisation** to scalar pipelines
