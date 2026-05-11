// Use scalar boxes to compute a value that can feed parameters in larger graphs.
open StackProcessing

[<EntryPoint>]
let main _ =
    let value = System.Math.Sqrt System.Math.PI
    printfn "sqrt(pi) = %.6f" value
    0
