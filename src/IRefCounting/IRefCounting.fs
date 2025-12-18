namespace RefCounting

/// Minimal contract SlimPipeline can rely on
type IRefCounted =
    abstract AddRef  : unit -> unit
    abstract Release : unit -> unit

[<RequireQualifiedAccess>]
module Lifetime =
    let inline addRef  (x: #IRefCounted) = x.AddRef()
    let inline release (x: #IRefCounted) = x.Release()

/// Generic adapter so you can wrap *any* value (e.g., Map) and still satisfy IRefCounted
type Rc<'a>(value: 'a, ?onAddRef: unit -> unit, ?onRelease: unit -> unit) =
    let add = defaultArg onAddRef ignore
    let rel = defaultArg onRelease ignore
    member _.Value = value
    interface IRefCounted with
        member _.AddRef()  = add()
        member _.Release() = rel()

[<RequireQualifiedAccess>]
module Rc =
    /// No-op refcounting (good for immutable values like Map)
    let ofValue (x: 'a) = Rc(x)

    /// Wrap with explicit hooks (use in StackProcessing when lifting Image)
    let withHooks add release (x: 'a) = Rc(x, add, release)

    /// Helpers
    let value (rc: Rc<'a>) = rc.Value
    let map (f: 'a -> 'b) (rc: Rc<'a>) = Rc(f rc.Value)  // default: no-op hooks on new value
