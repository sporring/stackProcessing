#r "bin/Debug/net8.0/Image.dll"            
#r "bin/Debug/net8.0/SimpleITKCSharpManaged.dll"
open Image
open ImageFunctions
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
printfn "%A" Q;;

let arr = array2D [ [ 10.0f; 20.0f ]; [ 30.0f; 40.0f ] ]
let imgF = Image<float32>.ofArray2D arr
let imgB = imgF.cast<uint8>();;

let img = Image<float>.ofArray2D (array2D [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ])
let ker = Image<float>.ofArray2D (array2D [ [ 0.0; 0.0; 0.0 ]; [ 0.0; 1.0; 0.0 ]; [ 0.0; 0.0; 0.0 ] ])
let result = ImageFunctions.conv img ker;;

let arr = Array2D.init 5 5 (fun m n -> if m=2 && n=2 then 1.0 else 0.0)
let img = Image<float>.ofArray2D arr
let blurred = ImageFunctions.discreteGaussian 1.0 img;;

let arr = Array2D.init 5 5 (fun m n -> if m=2 && n=2 then 1.0 else 0.0)
let img = Image<float>.ofArray2D arr
let blurred = ImageFunctions.recursiveGaussian 1.0 0u img;;

let arr = Array2D.init 5 5 (fun m n -> if m=2 && n=2 then 1.0 else 0.0)
let img = Image<float>.ofArray2D arr
let lap = ImageFunctions.laplacianConvolve 1.0 img;;

let arr = Array2D.init 5 5 (fun m n -> if m=2 && n=2 then 1.0 else 0.0)
let img = Image<float>.ofArray2D arr
let grad = ImageFunctions.gradientConvolve 0u 1u img;;

let img = Image<int>.ofArray2D (array2D [[0;1;1;0]])
let eroded = ImageFunctions.binaryErode 1u 1.0 img;;

let img = Image<int>.ofArray2D (array2D [[0;1;0]])
let dilated = ImageFunctions.binaryDilate 1u 1.0 img;;

let img = Image<int>.ofArray2D (array2D [[0;1;0;1;0]])
let opened = ImageFunctions.binaryOpening 1u 1.0 img;;

let img = Image<int>.ofArray2D (array2D [[1;0;1]])
let closed = ImageFunctions.binaryClosing 1u 1.0 img;;

let img = Image<int>.ofArray2D (array2D [[1;1;1]; [1;0;1]; [1;1;1]])
let filled = ImageFunctions.binaryFillHoles 1.0 img;;

let img = Image<int>.ofArray2D (array2D [[1;0;1]])
let cc = ImageFunctions.connectedComponents img;;

let img = Image<int>.ofArray2D (array2D [[1;0;1]])
let cc = ImageFunctions.connectedComponents img
let relabeled = ImageFunctions.relabelComponents 2u cc;;

let img = Image<int>.ofArray2D (array2D [[0;1;0]])
let dmap = ImageFunctions.signedDistanceMap img;;

let img = Image<int>.ofArray2D (array2D [[0;1;0;1;0]])
let ws = ImageFunctions.watershed 0.0 img;;
