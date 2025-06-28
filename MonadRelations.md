# 🧠 F# Async Data Flow Cheatsheet

We are working with asynchronous computation. 

## 📚 Data Abstraction Layers

### 🔸 1. `'T` – **Concrete Value**

This is your ordinary value. For example:

```fsharp
let x: int = 42
```

No computation or delay involved.

#### What is unit?
`'T = unit` means “no meaningful value” — like void in other languages. It’s used for:

- Indicating side effects,

- Placeholder return values,

- Control flow (e.g. in loops or task pipelines).

---

### 🔸 2. `Async<'T>` – **Single Async Computation**

This represents a **computation** that *will eventually produce* a value of type `'T`.

* Think of it as a task or a promise in other languages (like `Task<'T>` in C# or `Promise<T>` in JS).
* The computation may involve I/O, threading, etc.

Example:

```fsharp
let getIntAsync: Async<int> =
    async { return 42 }
```

To *extract* the value, you need to **run it**:

```fsharp
let value = Async.RunSynchronously getIntAsync  // value = 42
```

#### What is `Async<unit>`?
`Async<unit>` means:

> **An asynchronous operation** that completes in the future and produces *no meaningful result*.

Examples:

```fsharp
async { do! someSideEffect(); return () } : Async<unit>
```


---

### 🔸 3. `AsyncSeq<'T>` – **Sequence of Async Values**

Represents **a sequence of values over time**. This is like a lazy `seq<'T>`, but asynchronous.
This models *both time and structure* — but in practice:

* The *structure* is the sequence (`[]`, `[x]`, `[x; y]`, …)
* The *time* is when each value becomes available
* Each element may be delayed, fetched from disk/network, etc.

So for clarity:

* ✅ `Async<'T>` → *time-only* (delayed single value)
* ✅ `AsyncSeq<'T>` → *structured data arriving over time*


Example:

```fsharp
let nums: AsyncSeq<int> =
    asyncSeq {
        yield 1
        do! Async.Sleep 1000
        yield 2
    }
```

#### What is `AsyncSeq<unit>`?
`AsyncSeq<unit>` means:

> **A sequence of asynchronous steps**, each step producing a `unit`.
> So it represents **a structured stream of side-effects**.

Think of it like a pipeline where each item doesn't carry data — it's just there to represent *a step*, e.g.:

```fsharp
asyncSeq {
    do! someIO ()
    yield ()         // Step 1
    do! someMore ()
    yield ()         // Step 2
}
```

Use Cases for `AsyncSeq<unit>`

This is common when you:

* Process a stream for **side effects only**, not results,
* Discard intermediate values but keep track of structure,
* Want to synchronize execution without emitting values.

#### Example:

In your context:

```fsharp
run pipe : AsyncSeq<unit>
```

This means: “run this pipeline — it has side effects, but no result values.”

---

### 🔸 4. `Pipe<'S, 'T>` – **Stream Transformer Over Structured Async Data**

A `Pipe<'S, 'T>` represents a **stream processor** that transforms a sequence of type `'S` into a sequence of type `'T`, while preserving the asynchronous nature of the data.

It wraps the transformation:

```fsharp
AsyncSeq<'S> -> AsyncSeq<'T>
```

So it's **not just a function** — it’s a *composable building block* that also carries:

* A **name** (for debugging / tracing),
* A **memory profile** (Streaming, Sliding, Buffered, Constant),
* An **execution model** via the `Apply` field.

---

#### 📦 Structure + Time + Context

We can think of a `Pipe<'S, 'T>` as a higher-level abstraction over `AsyncSeq<'S>`:

| Aspect        | Meaning in `Pipe<'S, 'T>`                                                          |
| ------------- | ---------------------------------------------------------------------------------- |
| **Structure** | The transformation preserves or reshapes the stream — map, filter, windowing, etc. |
| **Time**      | Each value in the stream may arrive asynchronously                                 |
| **Context**   | The `Pipe` may manage memory profiles, disk IO, or composition                     |

> `Pipe<'S, 'T>` transforms structured asynchronous data (`AsyncSeq<'S>`) into new structured asynchronous data (`AsyncSeq<'T>`), with added metadata and composition tools.

It’s a **streaming computation**, aware of:

* **Timing** (via async values),
* **Structure** (sequences over time),
* **Resources** (via memory profiles).

#### ✅ What is `Pipe<unit, 'T>`?

This is a pipeline that takes **no input**, and **produces** a stream of `'T`. It represents a **producer**, **generator**, or a **data source**.

It is analogous to:

```fsharp
unit -> AsyncSeq<'T>
```

---

#### ❓ What is `Pipe<'T, unit>`?

This is the **dual**: it takes a stream of `'T` and **produces nothing** — or more precisely, it **consumes** the input.

It corresponds to:

```fsharp
AsyncSeq<'T> -> AsyncSeq<unit>
```

But more semantically:

> 🔸 `Pipe<'T, unit>` is a **Sink** — a consumer.

| Pipe Type        | Interpretation  | Analogous to                                   |
| ---------------- | --------------- | ---------------------------------------------- |
| `Pipe<unit, 'T>` | **Source**      | `unit -> AsyncSeq<'T>`                         |
| `Pipe<'T, unit>` | **Sink**        | `AsyncSeq<'T> -> Async<unit>` (after reducing) |
| `Pipe<'A, 'B>`   | **Transformer** | `AsyncSeq<'A> -> AsyncSeq<'B>`                 |

---

#### 🔁 How is `Pipe<'T, unit>` used?

Often, these are terminal operations in your pipeline:

* `print : Pipe<'T, unit>`
* `ignore : Pipe<'T, unit>`
* `saveToDisk : Pipe<Slice<'T>, unit>`

They're often passed to `sink`, e.g.:

```fsharp
myPipe >=> print |> sink
```

| Type             | Role          | Description                         |
| ---------------- | ------------- | ----------------------------------- |
| `Pipe<unit, 'T>` | **Source**    | Produces data without needing input |
| `Pipe<'T, unit>` | **Sink**      | Consumes data, no further output    |
| `Pipe<'A, 'B>`   | **Processor** | General stream transformer          |

---
## Data Relations
Async<'T> defines a Monad, AsyncSeq<'T> a stream of Monads, and Pipe<'S,'T> a stream transformer or a morphism. This may be called a Kleisli category over morphisms.

### 🔹 `'T -> Async<'T>`

* **Monad's `return` / `pure`**
* Wraps a plain value in the minimal effect
* **No transformation**, just lifting

```fsharp
async.Return x  // 'T -> Async<'T>
```

---

### 🔹 `Async<'T> -> 'T`

* **Reducer / evaluator of async**
* Extracts value from the effect
* Loses laziness or concurrency

```fsharp
Async.RunSynchronously : Async<'T> -> 'T
```

---

### 🔹 `'T -> Async<'U>`

* **Effectful mapper / Kleisli arrow**
* Applies a function *that introduces* asynchronous behavior or IO

This is the **core of "monadic bind"**:

```fsharp
let bind (f: 'T -> Async<'U>) (x: Async<'T>) : Async<'U> =
    async {
        let! t = x
        return! f t
    }
```

🔁 *Think of this as a mapper with side effects or delayed computations.*

---

### 🔹 `Async<'T> -> 'U`

* ❗️This is **not a standard monadic construct**
* Conceptually, this **breaks** the monad
* You're running the async and extracting a value — it's *interpreting* the effect

So:

* It **looks like a reducer** of the monad.
* But **unlike `reduce`**, it typically means you’re collapsing the async structure early.
* It’s generally not composable (not pure) and ties you to evaluation strategies.

**Examples:**

```fsharp
let asyncToValue : Async<int> -> int = Async.RunSynchronously
```

---

### 🧠 What does `Async<'U> -> AsyncSeq<'T>` mean?

It’s a function that:

* **Waits for an asynchronous result** of type `'U`,
* And then **starts emitting a stream** of `'T` values.

---

#### 🧩 Conceptually, it is:

| Perspective           | Interpretation                                                                    |
| --------------------- | --------------------------------------------------------------------------------- |
| **Monad Perspective** | A function *from a single promise to a stream* — it lifts time into structure.    |
| **Producer**          | A stream **generator** based on the result of a one-time async computation.       |
| **Dual to `reduce`**  | If `AsyncSeq<'T> -> Async<'U>` is a *reduction* (fold), then this is *expansion*. |
| **“Unfolding”**       | Like `unfoldAsync` but with the seed coming from an async computation.            |
| **Stream Builder**    | It transforms a future into a source of many values.                              |

---

#### 🧪 Example Use Case

```fsharp
let fetchAndStream (asyncData: Async<string>) : AsyncSeq<char> =
    asyncSeq {
        let! data = asyncData
        for c in data do
            yield c
    }
```

Here, we wait for a string to arrive (an `Async<string>`) and turn it into a stream of characters (`AsyncSeq<char>`).

---

#### 🔖 Possible Names or Roles

Although no single canonical name exists, here are reasonable terms:

* **UnfoldAsync**-style transformer
* **Stream generator**
* **Async-to-stream adapter**
* **Delayed source**
* **AsyncExpander**
* **StreamifyAsync** (in library naming)

---

#### ⛔ Not a Kleisli Arrow

* Kleisli arrows are of the form `'A -> M<'B>` for *some monad* `M`.
* This is **not** such a form because `Async<'U>` is already inside a monad.

---


### 🌀 What About `AsyncSeq<'T> -> Async<'U>`?

This is a **structural reduction**:

#### ✅ It's a **general consumption of the stream**:

* Aggregation (`fold`, `reduce`)
* Summarization (e.g., `mean`, `min`, `max`)
* Exporting data (e.g., writing to disk)
* Side-effectful execution (e.g., `printAsync`)

**Why confusing?**

Because it *looks* like you're mapping (`'T -> 'U`), but you're:

* **consuming many items**
* **producing a single result**
* and doing it **asynchronously**

```fsharp
// E.g., mean over stream
let computeMean (stream: AsyncSeq<float>) : Async<float> = async {
    let! items = stream |> AsyncSeq.toListAsync
    return List.average items
}
```
---

### 💡 Why `'T -> AsyncSeq<'U>` is Special

* It differs from `'T -> 'U` because it **produces multiple results** (zero or more), not just one.
* In monad terms, it's equivalent to `flatMap` or `bind`.
* This enables **dynamic expansion**: one slice in, many slices out.
* Key for operations like `windowing`, `duplication`, or **scatter/gather**.

---

### 🔁 What is `AsyncSeq<'T> -> Pipe<'U, 'V>`?

#### ❓What does this mean?

This would be a function that, **given a stream**, constructs a `Pipe`.

But this **doesn't make sense in general**, because a `Pipe` is **not built from a concrete stream** — it is a **function that transforms a stream**, not one constructed *from* a stream.

#### ✅ Valid Specialized Interpretation

You could imagine something like:

```fsharp
let injectStream (external: AsyncSeq<'T>) : Pipe<unit, 'T>
```

Which **wraps a fixed stream** as a pipeline — effectively a **source**.

This is what your `source` and `lift` do:

* It turns `AsyncSeq<'T>` into a `Pipe<unit, 'T>`

This is equivalent to **embedding a static stream** into a pipeline:

```fsharp
let sourceFromStream name stream =
  { Name = name; Profile = Streaming; Apply = fun _ -> stream }
```

So:

> 🔸 `AsyncSeq<'T> -> Pipe<unit, 'T>`
> is like a **stream source**.

---

### 🔁 And what is the inverse `Pipe<'U, 'V> -> AsyncSeq<'T>`?

#### ❓What does this mean?

You have a stream transformer, and you want to turn it into an **actual stream**. This requires an **input stream**, i.e., `Pipe.Apply : AsyncSeq<'U> -> AsyncSeq<'V>`.

You already have:

```fsharp
let run (p: Pipe<unit, 'T>) : AsyncSeq<'T> =
    p.Apply (AsyncSeq.singleton ())
```

So:

> 🔸 `Pipe<unit, 'T> -> AsyncSeq<'T>`
> is a **pipeline runner** for pipelines that take no stream input.

If `Pipe<'U, 'V>`, then you must supply a stream of `'U` to get a stream of `'V`.

---
### 🧱 Core Relationship: `Pipe<'S,'T>` vs. `AsyncSeq<'T>`

- `AsyncSeq<'T>` is a **value**

       - A lazy, asynchronous sequence of elements of type `'T`
       - It's *data* in an effectful context

- 🔹 `Pipe<'S,'T>` is a **computation**

       - A **transformer** from an `AsyncSeq<'S>` → `AsyncSeq<'T>`
       - Encapsulates:
              - A **name**
              - A **memory profile**
              - An `Apply` function

---

### 🔃 Summary Table

| Signature                                      | Meaning/Role             | Notes                                             |
| ---------------------------------------------- | ------------------------ | ------------------------------------------------- |
| `AsyncSeq<'T> -> Pipe<unit, 'T>`               | Stream → Source          | Wraps a stream as a pipeline                      |
| `Pipe<unit, 'T> -> AsyncSeq<'T>`               | Source → Stream          | Run pipeline with no input                        |
| `Pipe<'U, 'V> -> AsyncSeq<'U> -> AsyncSeq<'V>` | General pipe application | Core of `Pipe.Apply`                              |
| `AsyncSeq<'T> -> Pipe<'U,'V>`                  | ❌ ill-defined in general | Would imply dynamic pipeline creation from stream |


---


### ✅ Conceptual Interpretation

| Layer          | Effect Kind      | Meaning                    |
| -------------- | ---------------- | -------------------------- |
| `'T`           | none             | raw data                   |
| `Async<'T>`    | time             | delayed value (a future)   |
| `AsyncSeq<'T>` | structure        | stream of values over time |
| `Pipe<'S,'T>`                        | Stream-to-stream pipeline | Composition Wrapper     |
| `'T -> Async<'U>`                    | Async computation         | Async Function / Mapper |
| `AsyncSeq<'T> -> Async<'U>`          | Reduce/Fold over stream   | Structural Reduction    |
| `run: Pipe<unit,'T> -> AsyncSeq<'T>` | Execute pipeline          | Realize Stream          |

> So, `AsyncSeq<'T> -> Async<'U>` is a **reduction over structure**, producing a **summary**.


### 🧭 Graphical overview of relations

```
       ┌───────────────┐
       │ Pipe<'In, 'T> │
       └──────┬────────┘
              │  Apply to stream input
              ▼
       ┌──────────────┐
       │ AsyncSeq<'T> │ ← Stream of results
       └──────┬───────┘
              │  Reduce, fold, or head/last
              ▼
       ┌────────────┐
       │ Async<'T>  │ ← A delayed single result
       └──────┬─────┘
              │  Run asynchronously or synchronously
              ▼
       ┌────────────┐
       │     'T     │ ← The actual final value
       └────────────┘
```

```
        ┌──────────────┐
        │      'T      │  ← Plain value (e.g., ImageStats)
        └──────┬───────┘
               │ Async.Return
               ▼
        ┌──────────────┐
        │  Async<'T>   │  ← Single value computed asynchronously
        └──────┬───────┘
               │ AsyncSeq.ofAsync
               ▼
        ┌──────────────────┐
        │ AsyncSeq<'T>     │  ← Stream of values computed asynchronously
        └──────────────────┘
               │ Pipe.Apply or lift
               ▼
        ┌────────────────────┐
        │ Pipe<'In, 'Out>    │  ← Stream-to-stream transformation
        └────────────────────┘
```

---


## 📦 `Pipe<'S,'T>` as a First-Class Computation and more

A `Pipe` is:

```fsharp
type Pipe<'S,'T> =
  {
    Name: string
    Profile: MemoryProfile
    Apply: AsyncSeq<'S> -> AsyncSeq<'T>
  }
```

So it's:

* A **wrapped function** from `AsyncSeq<'S>` to `AsyncSeq<'T>`
* But also **metadata-aware** (`Profile`) and **debuggable** (`Name`)
* It’s your system’s **domain-specific Kleisli arrow**

---

> The pipeline layer **abstracts the streaming context** and **enables composition**, in the same way a monad abstracts and controls effects.

---

## 🛠️ Example

Let’s say you have this:

```fsharp
let normalizeSlice (slice: Slice<float>) : Slice<float> = ...
```

If you want to apply it to every streamed slice, you lift it:

```fsharp
let normalize : Pipe<Slice<float>, Slice<float>> =
  lift "normalize" Streaming (fun s -> async.Return (normalizeSlice s))
```

Then you can compose:

```fsharp
source >=> normalize >=> filter >=> print
```

All without seeing `AsyncSeq` — *it's hidden in the plumbing.*

---

## ✅ Summary

| Concept        | Description                              | Signature / Example                              |
| -------------- | ---------------------------------------- | ------------------------------------------------ |
| `'T`           | A value                                         | `42`, `"hello"`               |
| `Async<'T>`    | A computation that will eventually give `'T`    | `async { return 42 }`         |
| `AsyncSeq<'T>` | A stream of `'T` values, yielded asynchronously | `asyncSeq { yield! [1..10] }` |
| `AsyncSeq<'T>` | Lazy async stream                        | `readSlices<float>` returns this                 |
| `Pipe<'S,'T>`  | Transform from stream to stream          | `map`, `lift`, `reduce`, etc.                    |
| `lift`         | Embed `T -> Async<'U>` into a pipeline   | `lift "name" profile f`                          |
| `run`          | Turn `Pipe<unit,'T>` into `AsyncSeq<'T>` | Used at the start or sink of pipeline            |
| `>=>`         | Compose `Pipe`s                          | Like monadic bind (`Pipe<'A,'B> -> Pipe<'B,'C>`) |


| Step                             | Description                                | Example Functions         |
| -------------------------------- | ------------------------------------------ | ------------------------- |
| `Pipe<'In, 'T>`                  | A reusable pipeline                        | `map`, `reduce`, etc.     |
| `pipe.Apply input`               | Transforms input to a stream               | Internal logic of `Pipe`  |
| `AsyncSeq<'T>`                   | A stream of values (asynchronous iterator) | `printAsync`, `iterAsync` |
| `AsyncSeq.head`, `tryLast`, etc. | Reduces stream to a single result          | Used in `reduce`, `cache` |
| `Async<'T>`                      | Async result waiting to be run             | `Async.RunSynchronously`  |
| `'T`                             | Final value                                | Concrete scalar result    |

---

## 🔹 Monad close terminology

From a **monadic** perspective:

* `Async<'T>` is a monad over time.
* `AsyncSeq<'T>` is a monad over time *and* order — it's like `List` + `Async`.

Both allow you to:

* Lift a pure `'T` into their context (`async.Return`, `AsyncSeq.singleton`)
* Compose computations (`let!`, `bind`, etc.)

Use
* **`reduce`** for `AsyncSeq<'T> -> Async<'R>`
* **`collect`** or **`toList`** for `AsyncSeq<'T> -> Async<'T list>`
* **`aggregate`** if you want something more general


| Concept            | Type Signature                                    | Role / Description                                   |
| ------------------ | ------------------------------------------------- | ---------------------------------------------------- |
| Mapper             | `'T -> 'U`                                        | Basic value-to-value transformation                  |
| Async Mapper       | `'T -> Async<'U>`                                 | Transformation requiring asynchronous context        |
| Stream Mapper      | `AsyncSeq<'T> -> AsyncSeq<'U>`                    | Whole-stream transformation                          |
| FlatMap / Binder   | `'T -> AsyncSeq<'U>`                              | Emits **0 or more** results per input (monadic bind) |
| Reducer            | `AsyncSeq<'T> -> Async<'U>`                       | Collapses a stream into a single result              |
| Batch Reducer      | `AsyncSeq<'T> -> Async<'T list>` or similar       | Gathers stream into a list                           |
| Pipe (stream proc) | `Pipe<'S, 'T>`                                    | Stream transformer, used in composition              |
| Scalar Injector    | `Pipe<'In, 'A> -> Pipe<'In, 'B> -> Pipe<'In, 'C>` | Merges scalar and stream (e.g. via `zipWith`)        |

---

