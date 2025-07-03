Here’s a **clear roadmap** for shifting your public API from:

```fsharp
Pipe<'S,'T> -> Pipe<'T,'U> -> Pipe<'S,'U>
```

…to:

```fsharp
Pipeline<'S,'T> -> Operation<'T,'U> -> Pipeline<'S,'U>
```

while preserving user-facing syntax like:

```fsharp
source memory
>=> convGauss 1.0 None
>=> write "out" ".tif"
|> sink
```

---

## 🧭 Overview: What's Changing?

| Layer           | Before                 | After                                   |
| --------------- | ---------------------- | --------------------------------------- |
| Core functions  | `Pipe<'S,'T>`          | `Operation<'S,'T>`                      |
| User chaining   | `Pipe -> Pipe -> Pipe` | `Pipeline -> Operation -> Pipeline`     |
| Final sink      | `Pipe -> unit`         | `Pipeline -> Pipe -> unit` (via `sink`) |
| Validation      | None / ad hoc          | Automatic (shape + memory)              |
| Memory tracking | Manual or absent       | Built into `Pipeline`                   |

---

## ✅ Step-by-step Transition Plan

---

### 🔹 Step 1: **Update your core operations to return `Operation`**

You already started doing this with `discreteGaussianOp`. Now just:

* Refactor existing `Pipe`-returning functions to return `Operation`
* Or add `…Op` variants (e.g. `convGaussOp`) temporarily

Example:

```fsharp
let convGaussOp sigma =
    discreteGaussianOp "convGauss" sigma None None

// if you still want the raw Pipe:
let convGauss sigma =
    asPipe (convGaussOp sigma)
```

Eventually, you’ll drop the raw `Pipe` version.

---

### 🔹 Step 2: **Use `Operation` everywhere in user-level composition**

Change your composition logic (as you’ve already planned) to:

```fsharp
let (>=>) (pl: Pipeline<'S,'T>) (op: Operation<'T,'U>) : Pipeline<'S,'U> =
    { pl with flow = bindM pl.flow (fun _ -> returnM op) }
```

✅ Now `>=>` is *memory-validated*, shape-aware, and composes semantically.

---

### 🔹 Step 3: **Update public helpers (`source`, `sink`)**

Ensure `source` and `sink` use the new `Pipeline<'S,'T>`:

```fsharp
let source memory = 
    { flow = fun _ _ -> failwith "no shape yet"
      shape = None
      mem = memory }

let sink (pl: Pipeline<'S,'T>) : Pipe<'S,'T> =
    let op, _, _ = pl.flow pl.mem pl.shape
    asPipe op
```

✅ Now the entry and exit points are fully wrapped in `Pipeline`.

---

### 🔹 Step 4: **Drop `Pipe -> Pipe` composition in user-facing API**

Remove or rename your old `>=>`:

```fsharp
// internal use only:
let ( >=>. ) (p1 : Pipe<_,_>) (p2 : Pipe<_,_>) = composePipe p1 p2
```

Don’t expose `>=>` for `Pipe` anymore — just for `Pipeline`.

---

### 🔹 Step 5: **Refactor existing pipelines to use `Operation`**

Before:

```fsharp
let myPipe =
    read "in.tif"
    >=> convGauss 1.0 None
    >=> write "out.tif"
```

After:

```fsharp
let myPipe =
    source 4_000_000_000UL
    >=> convGaussOp 1.0
    >=> writeOp "out.tif"
    |> sink
```

If you want to preserve legacy `convGauss` for now, keep the `asPipe` call there and transition gradually.

---

### 🔹 Step 6: **Optional – Add transitional helpers**

For smoother migration, add a short helper to lift a `Pipe` into an `Operation`:

```fsharp
let inline op name (p: Pipe<'S,'T>) : Operation<'S,'T> =
    { Name = name; Pipe = p; Transition = transition p.Profile p.Profile }
```

Then wrap transitional functions like:

```fsharp
let convGauss sigma = op "convGauss" (convGaussPipe sigma)
```

---

## 🧪 After the transition

You’ll now have:

* All stages expressed as `Operation<'S,'T>`
* A composition model (`Pipeline`) that:

  * checks memory profiles
  * validates shape transitions
  * selects streaming/full profiles dynamically
* The exact same **user-facing syntax** (`>=>`, `|>`, `sink`) with added power

---

## ✅ Final: What’s safe to delete now?

* Old `>=>` for `Pipe`
* `asPipe` from call sites (keep internally for `sink`)
* Any unused `Pipe`-returning versions of ops

---

## ✨ Need a checklist?

Here’s your quick punch list:

☐ Convert all ops to return `Operation<'S,'T>`
☐ Redefine `>=>` for `Pipeline -> Operation -> Pipeline`
☐ Confirm `source`/`sink` use the new `Pipeline`
☐ Update pipelines to use `Operation` stages
☐ Delete or rename the old `Pipe`-level `>=>`
☐ Remove `asPipe` from call sites (only use at `sink`)
☐ Confirm tests and examples pass

