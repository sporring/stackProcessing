module Slice
open Image
open ImageFunctions
open System.IO
open Plotly.NET

/// <summary>
/// Represents a slice of a stack of 2d images. 
/// </summary>
type Slice<'T when 'T: equality> = {
    Index: uint
    Image: Image<'T>
}

let create<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) (idx: uint) : Slice<'T> =
    {Index= idx; Image=if depth > 1u then Image<'T>([width;height;depth]) else Image<'T>([width;height]) }

let liftUnary (f: Image<'T> -> Image<'T>) : Slice<'T> -> Slice<'T> =
    fun s -> { s with Image = f s.Image }

let liftUnary1 (f: 'a -> Image<'T> -> Image<'T>) : 'a -> Slice<'T> -> Slice<'T> =
    fun a s -> { s with Image = f a s.Image }

let liftUnary2 (f: 'a -> 'b -> Image<'T> -> Image<'T>) : 'a -> 'b -> Slice<'T> -> Slice<'T> =
    fun a b s -> { s with Image = f a b s.Image }

let liftBinary (f: Image<'T> -> Image<'T> -> Image<'T>) : Slice<'T> -> Slice<'T> -> Slice<'T> =
    fun s1 s2 -> { s1 with Image = f s1.Image s2.Image }

let liftBinary1 (f: 'a  -> Image<'T> -> Image<'T> -> Image<'T>) : 'a -> Slice<'T> -> Slice<'T> -> Slice<'T> =
    fun a s1 s2 -> { s1 with Image = f a s1.Image s2.Image }

let liftBinary2 (f: 'a -> 'b  -> Image<'T> -> Image<'T> -> Image<'T>) : 'a -> 'b -> Slice<'T> -> Slice<'T> -> Slice<'T> =
    fun a b s1 s2 -> { s1 with Image = f a b s1.Image s2.Image }

let liftBinary3 (f: 'a -> 'b -> 'c -> Image<'T> -> Image<'T> -> Image<'T>) : 'a -> 'b -> 'c -> Slice<'T> -> Slice<'T> -> Slice<'T> =
    fun a b c s1 s2 -> { s1 with Image = f a b c s1.Image s2.Image }

let liftBinaryOp (f: Image<'T> * Image<'T> -> Image<'T>) : Slice<'T> * Slice<'T> -> Slice<'T> =
    fun (s1, s2) -> { s1 with Image = f (s1.Image, s2.Image) }

let liftBinaryOpInt (f: Image<int> * int -> Image<int>) : Slice<int> * int -> Slice<int> =
    fun (s1, s2) -> { s1 with Image = f (s1.Image, s2) }
let liftBinaryOpUInt8 (f: Image<uint8> * uint8 -> Image<uint8>) : Slice<uint8> * uint8 -> Slice<uint8> =
    fun (s1, s2) -> { s1 with Image = f (s1.Image, s2) }
let liftBinaryOpFloat (f: Image<float> * float -> Image<float>) : Slice<float> * float -> Slice<float> =
    fun (s1, s2) -> { s1 with Image = f (s1.Image, s2) }

let liftBinaryCmp (f: Image<'T> * Image<'T> -> bool) : Slice<'T> * Slice<'T> -> bool =
    fun (s1,s2) -> f (s1.Image, s2.Image)

let abs<'T when 'T: equality> (s: Slice<'T>) = liftUnary ImageFunctions.abs s
let log<'T when 'T: equality> (s: Slice<'T>) = liftUnary ImageFunctions.log s
let log10<'T when 'T: equality> (s: Slice<'T>) = liftUnary ImageFunctions.log10 s
let exp<'T when 'T: equality> (s: Slice<'T>) = liftUnary ImageFunctions.exp s
let sqrt<'T when 'T: equality> (s: Slice<'T>) = liftUnary ImageFunctions.sqrt s
let square<'T when 'T: equality> (s: Slice<'T>) = liftUnary ImageFunctions.square s
let sin<'T when 'T: equality> (s: Slice<'T>) = liftUnary ImageFunctions.sin s
let cos<'T when 'T: equality> (s: Slice<'T>) = liftUnary ImageFunctions.cos s
let tan<'T when 'T: equality> (s: Slice<'T>) = liftUnary ImageFunctions.tan s
let asin<'T when 'T: equality> (s: Slice<'T>) = liftUnary ImageFunctions.asin s
let acos<'T when 'T: equality> (s: Slice<'T>) = liftUnary ImageFunctions.acos s
let atan<'T when 'T: equality> (s: Slice<'T>) = liftUnary ImageFunctions.atan s
let round<'T when 'T: equality> (s: Slice<'T>) = liftUnary ImageFunctions.round s

let convolve a b c (s: Slice<'T>) (t: Slice<'T>) = liftBinary3 convolve a b c s t
let conv (s: Slice<'T>) (t: Slice<'T>) = liftBinary conv s t
let discreteGaussian a (s: Slice<'T>) = liftUnary1 discreteGaussian a s
let recursiveGaussian a b (s: Slice<'T>) = liftUnary2 recursiveGaussian a b s
let laplacianConvolve a (s: Slice<'T>) = liftUnary1 laplacianConvolve a s
let gradientConvolve a b (s: Slice<'T>) = liftUnary2 gradientConvolve 
let binaryErode a b (s: Slice<'T>) = liftUnary2 binaryErode a b s
let binaryDilate a b (s: Slice<'T>) = liftUnary2 binaryDilate a b s
let binaryOpening a b (s: Slice<'T>) = liftUnary2 binaryOpening a b s
let binaryClosing a b (s: Slice<'T>) = liftUnary2 binaryClosing a b s
let binaryFillHoles a (s: Slice<'T>) = liftUnary1 binaryFillHoles a s

let squeeze (s: Slice<'T>) = liftUnary squeeze s
let concatAlong a (s: Slice<'T>) (t: Slice<'T>) = liftBinary1 concatAlong a s t

let connectedComponents (s: Slice<'T>) = liftUnary connectedComponents s
let relabelComponents a (s: Slice<'T>) = liftUnary1 relabelComponents a s
let watershed a (s: Slice<'T>) = liftUnary1 watershed a s
let otsuThreshold (s: Slice<'T>) = liftUnary otsuThreshold s
let otsuMultiThreshold a (s: Slice<'T>) = liftUnary1 otsuMultiThreshold a s
let momentsThreshold (s: Slice<'T>) = liftUnary momentsThreshold s

let signedDistanceMap (inside: uint8) (outside: uint8) (img: Slice<uint8>) : Slice<float> =
    {Index = img.Index; Image = ImageFunctions.signedDistanceMap inside outside img.Image}
let generateCoordinateAxis (axis: int) (size: int list) : Slice<uint32> =
    {Index = 0u; Image = ImageFunctions.generateCoordinateAxis (axis: int) (size: int list)}
let unique (img: Slice<'T>) : 'T list when 'T : comparison =
    ImageFunctions.unique img.Image
let labelShapeStatistics (img: Slice<'T>) : Map<int64, ImageFunctions.LabelShapeStatistics> =
    ImageFunctions.labelShapeStatistics img.Image
let computeStats (img: Slice<'T>) : ImageStats =
    ImageFunctions.computeStats img.Image
let histogram (img: Slice<'T>) : Map<'T, uint64> =
    ImageFunctions.histogram img.Image
let addHistogram (h1: Map<'T, uint64>) (h2: Map<'T, uint64>): Map<'T, uint64> =
    ImageFunctions.addHistogram h1 h2
let map2pairs (map: Map<'T, 'S>): ('T * 'S) list =
    ImageFunctions.map2pairs map
let inline pairs2floats<^T, ^S when ^T : (static member op_Explicit : ^T -> float)
                                 and ^S : (static member op_Explicit : ^S -> float)>
                                 (pairs: (^T * ^S) list) : (float * float) list =
    ImageFunctions.pairs2floats pairs
let inline pairs2int<^T, ^S when ^T : (static member op_Explicit : ^T -> int)
                                 and ^S : (static member op_Explicit : ^S -> int)>
                                 (pairs: (^T * ^S) list) : (int * int) list =
    ImageFunctions.pairs2int pairs

let swap f a b = f b a
let add (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.(+) (a,b) // types are a nuissance, for overload with constants, we need one variant per type, sigh
let addInt (a: Slice<int>) (b: int) = liftBinaryOpInt Image<int>.(+) (a,b)
let addUInt8 (a: Slice<uint8>) (b: uint8) = liftBinaryOpUInt8 Image<uint8>.(+) (a,b)
let addFloat (a: Slice<float>) (b: float) = liftBinaryOpFloat Image<float>.(+) (a,b)
let sub (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.(-) (a,b)
let subInt (a: Slice<int>) (b: int) = liftBinaryOpInt Image<int>.(-) (a,b)
let subFloat (a: Slice<float>) (b: float) = liftBinaryOpFloat Image<float>.(-) (a,b)
let mul (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.(*) (a,b)
let mulInt (a: Slice<int>) (b: int) = liftBinaryOpInt Image<int>.(*) (a,b)
let mulUInt8 (a: Slice<uint8>) (b: uint8) = liftBinaryOpUInt8 Image<uint8>.(*) (a,b)
let mulFloat (a: Slice<float>) (b: float) = liftBinaryOpFloat Image<float>.(*) (a,b)
let div (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.(/) (a,b)
let divInt (a: Slice<int>) (b: int) = liftBinaryOpInt Image<int>.(/) (a,b)
let divFloat (a: Slice<float>) (b: float) = liftBinaryOpFloat Image<float>.(/) (a,b)
let modulus (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.(%) (a,b)
let pow (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.Pow (a,b)
let isGreaterEqual (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.isGreaterEqual (a,b)
let gte (a: Slice<'T>) (b: Slice<'T>) = liftBinaryCmp Image<'T>.gte (a,b)
let isGreater (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.isGreater (a,b)
let ge (a: Slice<'T>) (b: Slice<'T>) = liftBinaryCmp Image<'T>.gt (a,b)
let isEqual (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.isEqual (a,b)
let eq (a: Slice<'T>) (b: Slice<'T>) = liftBinaryCmp Image<'T>.eq (a,b)
let isNotEqual (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.isNotEqual (a,b)
let neq (a: Slice<'T>) (b: Slice<'T>) = liftBinaryCmp Image<'T>.neq (a,b)
let isLessThanEqual (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.isLessThanEqual (a,b)
let lte (a: Slice<'T>) (b: Slice<'T>) = liftBinaryCmp Image<'T>.lte (a,b)
let isLessThan (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.isLessThan (a,b)
let lt (a: Slice<'T>) (b: Slice<'T>) = liftBinaryCmp Image<'T>.lt (a,b)
let sAnd (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.(&&&) (a,b) // which I could think of better names...
let sOr (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.(|||) (a,b)
let sXor (a: Slice<'T>) (b: Slice<'T>) = liftBinaryOp Image<'T>.(^^^) (a,b)
let sNot (s: Slice<'T>) = liftUnary Image<'T>.(~~~) s

let addNormalNoise a b (s: Slice<'T>) = liftUnary2 ImageFunctions.addNormalNoise a b s
let threshold a b (s: Slice<'T>) = liftUnary2 ImageFunctions.threshold a b s

let stack (slices: Slice<'T> list) : Slice<'T> =
    match slices with
        elm :: rst ->
            let images = slices |> List.map (fun s -> s.Image)
            {elm with Image = ImageFunctions.stack images }
        | _ -> failwith "Can't stack an empty list"

let extractSlice a (s: Slice<'T>) = liftUnary1 ImageFunctions.extractSlice a s


// IO stuff
let getDepth (inputDir: string) (suffix: string) : uint =
    let files = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    files.Length |> uint

type FileInfo = ImageFunctions.FileInfo
let getFileInfo(fname: string): FileInfo = getFileInfo(fname)

let getVolumeSize (inputDir: string) (suffix: string): uint * uint * uint =
    let files = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    if files.Length = 0 then
        failwith $"No {suffix} files found in directory: {inputDir}"
    let fi = getFileInfo(files[0]);
    let sz = fi.size |> List.map uint
    (sz[0], sz[1], uint files.Length)

let readSlice<'T when 'T: equality> (idx: uint) (filename: string) : Slice<'T> =
    {Index = idx; Image = Image<'T>.ofFile(filename)}

let readRandomSlice<'T when 'T: equality> (inputDir: string) (suffix: string): Slice<'T> =
    let filename = Directory.GetFiles(inputDir, "*"+suffix) |> Array.randomChoices 1 |> Array.tryHead
    match filename with
        Some fn -> readSlice<'T> 0u fn
        | _ -> failwith "No files in directory" // Here it would be better with a Slice structure with error handling

let writeSlice (filename: string) (slice: Slice<'T>) : unit =
    slice.Image.toFile(filename)

let inline plotSlice (slice: Slice<^T>) : unit  when ^T: (static member op_Explicit: ^T -> float) =
    let image = slice.Image
    let width = image.GetWidth() |> int
    let height = image.GetHeight() |> int
    let data =
        Seq.init height (fun y ->
            Seq.init width (fun x ->
                image[x, y] |> float))
    data |> Chart.Heatmap |> Chart.show

let plotList (vec: float list) =
    let idx = [1..vec.Length-1]
    Chart.Line(idx, vec) |> Chart.show
