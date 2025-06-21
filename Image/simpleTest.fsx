#r "bin/Debug/net8.0/ImageClass.dll"            
#r "bin/Debug/net8.0/SimpleITKCSharpManaged.dll"
open ImageClass
let I = Image<int>([10u;12u])
let J = Image<int list>([10u;12u])
printfn "%A" I[1,2]
I[1,2] <- 1
printfn "%A" I[1,2]
printfn "%A" J[2,1]
J[1,2] <- [2;1]
printfn "%A" J[1,2];;
let K = Image<int list>([10u;1u;12u]);;
printfn "%A" K
let L = ImageFunctions.squeeze K
printfn "%A" K
printfn "%A" L;;

let O = I + 5
let M = ImageFunctions.concatAlong 0u I O
printfn "%A" M
let N = ImageFunctions.concatAlong 1u I O
printfn "%A" N
let P = ImageFunctions.concatAlong 2u I O
printfn "%A" P
let Q = ImageFunctions.concatAlong 3u I O
printfn "%A" Q
