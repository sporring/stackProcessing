
# 📦 Core Concepts in the Streaming Pipeline Framework

This document summarizes the foundational types and functions used in your memory-aware, streaming image processing system.

---

## 🧠 `MemoryProfile`

Describes how much memory an operation consumes and how input/output slices are accessed.

```fsharp
type MemoryProfile =
    | Constant
    | Streaming
    | StreamingConstant
    | SlidingConstant of uint
    | Sliding of uint
    | FullConstant
    | Full
```

| Case                 | Meaning                                                 |
| -------------------- | ------------------------------------------------------- |
| `Constant`           | Stateless; operates slice-by-slice without memory needs |
| `Streaming`          | Operates on a single slice at a time                    |
| `Sliding n`          | Requires a window of `n` stacked slices                 |
| `Full`               | Requires the entire volume in memory                    |
| `*Constant` variants | Produce identical output regardless of input            |

---

## ⚙️ `Pipe<'S, 'T>`

Encapsulates an executable processing step over a stream of slices.

```fsharp
type Pipe<'S, 'T> = {
    Name    : string
    Profile : MemoryProfile
    Apply   : AsyncSeq<'S> -> AsyncSeq<'T>
}
```

| Field     | Description                                         |
| --------- | --------------------------------------------------- |
| `Name`    | Identifier for logs, debugging                      |
| `Profile` | Memory usage strategy (e.g. `Streaming`, `Sliding`) |
| `Apply`   | The actual transformation function                  |

---

## 🔁 `MemoryTransition`

Describes how memory layout changes across pipeline stages.

```fsharp
type MemoryTransition = {
    From  : MemoryProfile
    To    : MemoryProfile
    Check : SliceShape -> bool
}
```

### Constructor

```fsharp
let transition fromProfile toProfile : MemoryTransition
```

| Field   | Description                                          |
| ------- | ---------------------------------------------------- |
| `From`  | Input memory expectation                             |
| `To`    | Output memory layout                                 |
| `Check` | Predicate to validate shape compatibility (optional) |

### 💡 Dual Role of `MemoryTransition`

| Use Case                   | Purpose                                                           |
| -------------------------- | ----------------------------------------------------------------- |
| **Validation**             | Ensure that upstream operations deliver sufficient memory context |
| **Dynamic Implementation** | Guide how to implement an operation based on memory layout        |

Example:

```fsharp
match transition.From with
| Streaming -> buildStreamingConvolution()
| Sliding n when n >= required -> buildWindowedConvolution()
| Full -> buildFullVolumeConvolution()
| _ -> failwith "Unsupported profile"
```

---

## 🔨 `Operation<'S, 'T>`

Wraps a pipe with memory transition metadata to enable validation and planning.

```fsharp
type Operation<'S, 'T> = {
    Name       : string
    Transition : MemoryTransition
    Pipe       : Pipe<'S, 'T>
}
```

| Field        | Description                          |
| ------------ | ------------------------------------ |
| `Name`       | Logical name of the operation        |
| `Transition` | Describes memory layout input/output |
| `Pipe`       | The actual executable transformation |

---

## 🧱 `WindowedProcessor`

Encapsulates a 3D image operation that operates on stacked 2D slices.

```fsharp
type WindowedProcessor<'S, 'T> = {
    Name     : string
    Window   : uint
    Stride   : uint
    Process  : Slice<'S> -> Slice<'T>
}
```

| Field     | Description                                    |
| --------- | ---------------------------------------------- |
| `Window`  | Number of input slices to stack                |
| `Stride`  | Step between windows (overlap control)         |
| `Process` | 3D function applied to the stacked input slice |

Use `fromWindowed` to lift this into a streaming-compatible `Pipe`.

---

## 🧩 Helper Functions

### `fromWindowed`

Wraps a `WindowedProcessor` into a streaming `Pipe`.

```fsharp
val fromWindowed : WindowedProcessor<'S, 'T> -> Pipe<Slice<'S>, Slice<'T>>
```

---

### `liftWindowedOp`

Creates an `Operation` from a 3D `WindowedProcessor`.

```fsharp
val liftWindowedOp :
    name: string ->
    window: uint ->
    stride: uint ->
    f: (Slice<'S> -> Slice<'T>) ->
    Operation<Slice<'S>, Slice<'T>>
```

---

## ✅ Example: Transition-Based Implementation

```fsharp
let buildConvolution (transition: MemoryTransition) =
    match transition.From with
    | Streaming      -> streamingConvolution()
    | Sliding d      -> slidingWindowConvolution d
    | Full           -> fullVolumeConvolution()
    | _              -> failwith "Unsupported profile"
```

---

## 🧪 Validation and Planning

### `validate`

Ensures that memory transitions between operations are compatible.

```fsharp
let validate op1 op2 =
    if op1.Transition.To <> op2.Transition.From then
        failwithf "Memory transition mismatch: %A → %A" op1.Transition.To op2.Transition.From
```

### `plan`

Describes the structure of a composed pipeline for logging/debugging.

```fsharp
let plan ops =
    ops |> List.map (fun op -> $"[{op.Name}]  {op.Transition.From} → {op.Transition.To}")
        |> String.concat "\n"
```
