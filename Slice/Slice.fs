module Slice
open Image
open ImageFunctions
open System.IO

/// <summary>
/// Represents a slice of a stack of 2d images. 
/// </summary>
type Slice<'T when 'T: equality> = {
    Index: uint
    Image: Image<'T>
}
with 
    member this.EstimateUsage() = this.Image.memoryEstimate()

let create<'T when 'T: equality> (width: uint) (height: uint) (depth: uint) (idx: uint) : Slice<'T> =
    {Index= idx; Image=if depth > 1u then Image<'T>([width;height;depth]) else Image<'T>([width;height]) }

let GetDepth (s: Slice<'T>) = s.Image.GetDepth()
let GetDimensions (s: Slice<'T>) = s.Image.GetDimensions()
let GetHeight (s: Slice<'T>) = s.Image.GetHeight()
let GetWidth (s: Slice<'T>) = s.Image.GetWidth()
let GetSize (s: Slice<'T>) = s.Image.GetSize()
let ToString (s: Slice<'T>) = s.Image.ToString()
let toArray2D (s: Slice<'T>) = s.Image.toArray2D()
let toArray3D (s: Slice<'T>) = s.Image.toArray3D()
let toArray4D (s: Slice<'T>) = s.Image.toArray4D()
let toImage (s: Slice<'T>) = s.Image
let toSimpleITK (s: Slice<'T>) = s.Image.toSimpleITK()
let cast<'T when 'T: equality> (s: Slice<_>) = {Index = s.Index; Image=s.Image.cast<'T>()}
let castUInt8ToFloat (s: Slice<uint8>) : Slice<float> = {Index = s.Index; Image=s.Image.castUInt8ToFloat()} // slice with does not work, since this sets the type
let castFloatToUInt8 (s: Slice<float>) : Slice<uint8> = {Index = s.Index; Image=s.Image.castFloatToUInt8()} // slice with does not work, since this sets the type

let toFloat (value: obj) =
    match value with
    | :? float   as f -> f
    | :? float32 as f -> float f
    | :? int     as i -> float i
    | :? byte    as b -> float b
    | :? int64   as l -> float l
    | _ -> failwithf "Cannot convert value of type %s to float" (value.GetType().FullName)

let toSeqSeq (s: Slice<'T>): seq<seq<float>> =
    let width = s |> GetWidth |> int
    let height = s |> GetHeight |> int
    Seq.init height (fun y ->
        Seq.init width (fun x ->
            s.Image[x,y] |> box |> toFloat))

let private liftSource (f: unit -> Image<'T>) : unit -> Slice<'T> =
    fun () -> {Index = 0u; Image = f () }

let private liftSource1 (f: 'a -> Image<'T>) : 'a -> Slice<'T> =
    fun a -> { Index = 0u;  Image = f a }

let private liftSource2 (f: 'a -> 'b -> Image<'T>) : 'a -> 'b-> Slice<'T> =
    fun a b -> { Index = 0u;  Image = f a b }

let private liftUnary (f: Image<'T> -> Image<'T>) : Slice<'T> -> Slice<'T> =
    fun s -> { s with Image = f s.Image }

let private liftUnary1 (f: 'a -> Image<'T> -> Image<'T>) : 'a -> Slice<'T> -> Slice<'T> =
    fun a s -> { s with Image = f a s.Image }

let private liftUnary2 (f: 'a -> 'b -> Image<'T> -> Image<'T>) : 'a -> 'b -> Slice<'T> -> Slice<'T> =
    fun a b s -> { s with Image = f a b s.Image }

let private liftUnary3 (f: 'a -> 'b -> 'c -> Image<'T> -> Image<'T>) : 'a -> 'b -> 'c -> Slice<'T> -> Slice<'T> =
    fun a b c s -> { s with Image = f a b c s.Image }

let private liftUnary4 (f: 'a -> 'b -> 'c -> 'd -> Image<'T> -> Image<'T>) : 'a -> 'b -> 'c -> 'd -> Slice<'T> -> Slice<'T> =
    fun a b c d s -> { s with Image = f a b c d s.Image }

let private liftBinary (f: Image<'T> -> Image<'T> -> Image<'T>) : Slice<'T> -> Slice<'T> -> Slice<'T> =
    fun s1 s2 -> { s1 with Image = f s1.Image s2.Image }

let private liftBinary1 (f: 'a  -> Image<'T> -> Image<'T> -> Image<'T>) : 'a -> Slice<'T> -> Slice<'T> -> Slice<'T> =
    fun a s1 s2 -> { s1 with Image = f a s1.Image s2.Image }

let private liftBinary2 (f: 'a -> 'b  -> Image<'T> -> Image<'T> -> Image<'T>) : 'a -> 'b -> Slice<'T> -> Slice<'T> -> Slice<'T> =
    fun a b s1 s2 -> { s1 with Image = f a b s1.Image s2.Image }

let private liftBinary3 (f: 'a -> 'b -> 'c -> Image<'T> -> Image<'T> -> Image<'T>) : 'a -> 'b -> 'c -> Slice<'T> -> Slice<'T> -> Slice<'T> =
    fun a b c s1 s2 -> { s1 with Image = f a b c s1.Image s2.Image }

let private liftBinaryOp (f: Image<'T> * Image<'T> -> Image<'T>) : Slice<'T> * Slice<'T> -> Slice<'T> =
    fun (s1, s2) -> { s1 with Image = f (s1.Image, s2.Image) }

let private liftBinaryOpInt (f: Image<int> * int -> Image<int>) : Slice<int> * int -> Slice<int> =
    fun (s1, s2) -> { s1 with Image = f (s1.Image, s2) }
let private liftBinaryOpUInt8 (f: Image<uint8> * uint8 -> Image<uint8>) : Slice<uint8> * uint8 -> Slice<uint8> =
    fun (s1, s2) -> { s1 with Image = f (s1.Image, s2) }
let private liftBinaryOpFloat (f: Image<float> * float -> Image<float>) : Slice<float> * float -> Slice<float> =
    fun (s1, s2) -> { s1 with Image = f (s1.Image, s2) }

let private liftBinaryCmp (f: Image<'T> * Image<'T> -> bool) : Slice<'T> * Slice<'T> -> bool =
    fun (s1,s2) -> f (s1.Image, s2.Image)

let absSlice<'T when 'T: equality> (s: Slice<'T>) = liftUnary absImage s
let logSlice<'T when 'T: equality> (s: Slice<'T>) = liftUnary logImage s
let log10Slice<'T when 'T: equality> (s: Slice<'T>) = liftUnary log10Image s
let expSlice<'T when 'T: equality> (s: Slice<'T>) = liftUnary expImage s
let sqrtSlice<'T when 'T: equality> (s: Slice<'T>) = liftUnary sqrtImage s
let squareSlice<'T when 'T: equality> (s: Slice<'T>) = liftUnary squareImage s
let sinSlice<'T when 'T: equality> (s: Slice<'T>) = liftUnary sinImage s
let cosSlice<'T when 'T: equality> (s: Slice<'T>) = liftUnary cosImage s
let tanSlice<'T when 'T: equality> (s: Slice<'T>) = liftUnary tanImage s
let asinSlice<'T when 'T: equality> (s: Slice<'T>) = liftUnary asinImage s
let acosSlice<'T when 'T: equality> (s: Slice<'T>) = liftUnary acosImage s
let atanSlice<'T when 'T: equality> (s: Slice<'T>) = liftUnary atanImage s
let roundSlice<'T when 'T: equality> (s: Slice<'T>) = liftUnary roundImage s

type BoundaryCondition = ImageFunctions.BoundaryCondition
type OutputRegionMode = ImageFunctions.OutputRegionMode

let convolve a (s: Slice<'T>) (t: Slice<'T>) = liftBinary1 convolve a s t
let conv (s: Slice<'T>) (t: Slice<'T>) = liftBinary conv s t
let discreteGaussian a b c (s: Slice<'T>) = liftUnary3 discreteGaussian a b c s

let gauss (sigma: float) (kernelSize: uint option) : Slice<float> = liftSource2 gauss sigma kernelSize
let finiteDiffFilter1D (order: uint) : Slice<float> = liftSource1 finiteDiffFilter1D order
let finiteDiffFilter2D (direction: uint) (order: uint) : Slice<float> = liftSource2 finiteDiffFilter2D direction order
let finiteDiffFilter3D (direction: uint) (order: uint) : Slice<float> = liftSource2 finiteDiffFilter3D direction order
let finiteDiffFilter4D (direction: uint) (order: uint) : Slice<float> = liftSource2 finiteDiffFilter4D direction order

let gradientConvolve a b (s: Slice<'T>) = liftUnary2 gradientConvolve 
let binaryErode a (s: Slice<uint8>) = liftUnary1 binaryErode a s
let binaryDilate a (s: Slice<uint8>) = liftUnary1 binaryDilate a s
let binaryOpening a (s: Slice<uint8>) = liftUnary1 binaryOpening a s
let binaryClosing a (s: Slice<uint8>) = liftUnary1 binaryClosing a s
let binaryFillHoles (s: Slice<uint8>) = liftUnary binaryFillHoles s

let squeeze (s: Slice<'T>) = liftUnary squeeze s
let concatAlong a (s: Slice<'T>) (t: Slice<'T>) = liftBinary1 concatAlong a s t
let constantPad2D<'T when 'T : equality> a b c (s: Slice<'T>): Slice<'T> = liftUnary3 constantPad2D a b c s

let connectedComponents (s: Slice<uint8>) : Slice<uint64>= 
    {Index = s.Index;  Image = (connectedComponents s.Image) }
let relabelComponents a (s: Slice<uint64>) = liftUnary1 relabelComponents a s
let watershed a (s: Slice<'T>) = liftUnary1 watershed a s
let otsuThreshold (s: Slice<'T>) : Slice<'T> = liftUnary otsuThreshold s
let otsuMultiThreshold a (s: Slice<'T>) = liftUnary1 otsuMultiThreshold a s
let momentsThreshold (s: Slice<'T>) = liftUnary momentsThreshold s

let signedDistanceMap (inside: uint8) (outside: uint8) (s: Slice<uint8>) : Slice<float> =
    {Index = s.Index; Image = ImageFunctions.signedDistanceMap inside outside s.Image}
let generateCoordinateAxis (axis: int) (size: int list) : Slice<uint32> =
    {Index = 0u; Image = ImageFunctions.generateCoordinateAxis (axis: int) (size: int list)}
let unique (s: Slice<'T>) : 'T list when 'T : comparison =
    ImageFunctions.unique s.Image
let labelShapeStatistics (s: Slice<'T>) : Map<int64, ImageFunctions.LabelShapeStatistics> =
    ImageFunctions.labelShapeStatistics s.Image
type ImageStats = ImageFunctions.ImageStats
let computeStats (s: Slice<'T>) : ImageStats =
    ImageFunctions.computeStats s.Image
let addComputeStats (s1: ImageStats) (s2: ImageStats): ImageStats =
    ImageFunctions.addComputeStats s1 s2
let histogram (s: Slice<'T>) : Map<'T, uint64> =
    ImageFunctions.histogram s.Image
let addHistogram (h1: Map<'T, uint64>) (h2: Map<'T, uint64>): Map<'T, uint64> =
    ImageFunctions.addHistogram h1 h2
let map2pairs (map: Map<'T, 'S>): ('T * 'S) list =
    ImageFunctions.map2pairs map
let inline pairs2floats<^T, ^S when ^T : (static member op_Explicit : ^T -> float)
                                 and ^S : (static member op_Explicit : ^S -> float)>
                                 (pairs: (^T * ^S) list) : (float * float) list =
    ImageFunctions.pairs2floats pairs
let inline pairs2ints<^T, ^S when ^T : (static member op_Explicit : ^T -> int)
                                 and ^S : (static member op_Explicit : ^S -> int)>
                                 (pairs: (^T * ^S) list) : (int * int) list =
    ImageFunctions.pairs2ints pairs

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

let stack (sLst: Slice<'T> list) : Slice<'T> =
    match sLst with
        elm :: rst ->
            let imgLst = sLst |> List.map (fun s -> s.Image)
            {elm with Image = ImageFunctions.stack imgLst }
        | _ -> failwith "Can't stack an empty list"

let extractSlice a (s: Slice<'T>) : Slice<'T> = liftUnary1 ImageFunctions.extractSlice a s

let unstack (s: Slice<'T>): Slice<'T> list =
    let dim = s |> GetDimensions
    if dim < 2u then
        failwith $"Cannot unstack a {dim}-dimensional image along the 3rd axis"
    let imgLst =
        if dim = 2u then
            [s.Image]
        else
            ImageFunctions.unstack s.Image
    let idxLst = List.mapi (fun i _ -> uint i) imgLst
    let baseIndex = s.Index
    List.zip idxLst imgLst |> List.map (fun (i,im) -> {Index = baseIndex + i; Image = im})

// IO stuff
type FileInfo = ImageFunctions.FileInfo
let getFileInfo (filename: string) : FileInfo = ImageFunctions.getFileInfo filename

let readSlice<'T when 'T: equality> (idx: uint) (filename: string) : Slice<'T> =
    {Index = idx; Image = Image<'T>.ofFile(filename)}

let writeSlice (filename: string) (s: Slice<'T>) : unit =
    s.Image.toFile(filename)

let getStackDepth (inputDir: string) (suffix: string) : uint =
    let files = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    files.Length |> uint

let getStackInfo (inputDir: string) (suffix: string): FileInfo =
    let files = Directory.GetFiles(inputDir, "*"+suffix) |> Array.sort
    let depth = files.Length |> uint64
    if depth = 0uL then
        failwith $"No {suffix} files found in directory: {inputDir}"
    let fi = ImageFunctions.getFileInfo(files[0])
    {fi with dimensions = fi.dimensions+1u; size = fi.size @ [depth]}

let getStackSize (inputDir: string) (suffix: string): uint64 list =
    let fi = getStackInfo inputDir suffix
    fi.size

let getStackWidth (inputDir: string) (suffix: string): uint64 =
    let fi = getStackInfo inputDir suffix
    fi.size[0]

let getStackHeight (inputDir: string) (suffix: string): uint64 =
    let fi = getStackInfo inputDir suffix
    fi.size[1]
