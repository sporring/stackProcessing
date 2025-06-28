In this context, **module boundaries** refer to how you **organize your F# codebase into conceptual units**, each encapsulating a coherent set of responsibilities.

Since your framework is structured around composable streaming transformations (`Pipe`s), memory profiles, sources/sinks, and utilities like branching or scalar injection — **grouping these into modules** helps with clarity, maintainability, and correct usage.

---

### 🔧 Why Module Boundaries Matter

By putting types and functions into logical **modules**, you:

* **Prevent misuse** (e.g. trying to `>>=>` a `source`).
* Improve **discoverability** and IntelliSense organization.
* Encourage **separation of concerns**.

---

### 🧱 Suggested Module Boundaries for Your System

Here's how your domain breaks down naturally:

#### 1. `Core`

* The **heart** of your system: the composable stream transformers.

```fsharp
module Core =
    type Pipe<'In, 'Out> = ...
    val lift : string -> MemoryProfile -> ('In -> Async<'Out>) -> Pipe<'In, 'Out>
    val (>>=>) : Pipe<'A, 'B> -> Pipe<'B, 'C> -> Pipe<'A, 'C>
    ...
```

#### 2. `Routing`

* Things that **compose or orchestrate** multiple `Pipe`s but are not themselves `Pipe`s.

```fsharp
module Routing =
    val tee : Pipe<'In, 'T> -> Pipe<'In, 'T> * Pipe<'In, 'T>
    val zipWith : ...
    val inject : ...
```

#### 3. `Source`

* Things that **generate** `Pipe<unit, 'T>` from input sources.

```fsharp
module Source =
    val fromTiff : ... -> Pipe<unit, Slice<float>>
    val source : ...
```

#### 4. `Sink`

* Things that **run** a pipe and produce side effects.

```fsharp
module Sink =
    val sink : Pipe<_, unit> -> unit
    val sinkLst : Pipe<_, unit> list -> unit
```

#### 5. `Processing`

* Domain-specific pipelines.

```fsharp
module Processing =
    val computeStats : Pipe<Slice<float>, ImageStats>
    val normalize : ...
```

#### 6. `Memory`

* Definitions and combinators for `MemoryProfile`.

```fsharp
module Memory =
    type MemoryProfile =
        | Streaming
        | Sliding of int
        | Buffered
        | Constant
        | StreamingConstant

    val combineProfile : MemoryProfile -> MemoryProfile -> MemoryProfile
```

---

### 📁 How This Helps

If someone writes:

```fsharp
source >>=> normalize
```

They’ll get a **type error**, but if the modules are clearly separated:

```fsharp
Source.readTiff ...
|> Core.(>>=>) Processing.normalize
```

… it becomes clearer what’s composable and what isn’t.

---

Would you like help reorganizing your code into modules like these? I can also suggest F# `namespace`/`module` layouts and file organization.
