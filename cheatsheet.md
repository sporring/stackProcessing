
# 📘 StackPipeline Cheat Sheet

This cheat sheet provides a quick reference for building stream-based pipelines using `Pipe`, `Slice`, and `source -> pipeline -> sink` architecture.

---

## 🔹 Core Types

### `Slice<'T>`
Represents a 2D slice from a 3D image stack with data of type `'T`.

### `Pipe<'In, 'Out>`
A composable processing unit that transforms `AsyncSeq<'In>` into `AsyncSeq<'Out>`.

---

## 🔹 Profiles

- `Streaming` – handles data slice-by-slice
- `Sliding sz` – maintains a moving window of size `sz`
- `Buffered` – requires full access to the stack before processing

---

## 🔹 Composition Operators

| Operator       | Description                                       |
|----------------|---------------------------------------------------|
| `>>=>`         | Compose two pipes in sequence                     |
| `>>~>`         | Inject scalar into stream (sequentially)          |

---

## 🔹 Builders & Runners

- `source mem w h d` – initialize with memory, width, height, depth
- `sink pipe` – runs and drains the pipe
- `sinkLst [p1; p2; ...]` – run multiple pipelines concurrently

---

## 🔹 Reducers

- `fromReducer name profile reducerFn` – turns an async reducer into a pipe
- `computeStats` – calculates image statistics (mean, std, min, max, etc.)

---

## 🔹 Inject / Join

- `inject f scalar stream` – inject scalar result into stream pipe (parallel)
- `injectAfter f reducer stream` – same, but evaluate scalar first

---

## 🔹 Utilities

| Function        | Description                                   |
|-----------------|-----------------------------------------------|
| `tap label`     | A pipe which logs value in stream with label               |
| `print`         | Outputs elements in the stream                |
| `show`         | Outputs Slice elements to Plotly                |
| `plot`         | Outputs float*float list elements in the Plotly                |
| `ignore`        | Consumes elements without action              |
| `cacheScalar label  | Runs a scalar pipe once and lifts to Pipe     |

---

## 🔹 Example Pattern

```fsharp
let statsMaker = readMaker >>=> computeStats >>=> cacheScalar "stats"
readMaker >>~> (normalizeWith, statsMaker) >>=> computeStats >>=> print |> sink
```

---

## 🔹 Conceptual

- `Pipe` ≈ Arrow (category theory)
- `source -> pipe -> sink` ≈ ETL / stream processing pipeline
- `inject` ≈ map with external environment

