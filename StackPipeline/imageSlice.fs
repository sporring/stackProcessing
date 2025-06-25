module Slice
open Image
open ImageFunctions

/// <summary>
/// Represents a slice of a stack of 2d images. 
/// </summary>
type Slice<'T> = {
    Index: uint
    Image: Image<'T>
}

let liftUnary (f: Image<'T> -> Image<'T>) : Slice<'T> -> Slice<'T> =
    fun s -> { s with Image = f s.Image }

let liftUnary1 (f: 'a -> Image<'T> -> Image<'T>) : 'a -> Slice<'T> -> Slice<'T> =
    fun a s -> { s with Image = f a s.Image }

let liftUnary2 (f: 'a -> 'b -> Image<'T> -> Image<'T>) : 'a -> 'b -> Slice<'T> -> Slice<'T> =
    fun a b s -> { s with Image = f a b s.Image }

let liftBinary (f: Image<'T> -> Image<'T> -> Image<'T>) : Slice<'T> -> Slice<'T> -> Slice<'T> =
    fun s1 s2 -> { s1 with Image = f s1.Image s2.Image }

let liftBinary2 (f: 'a -> 'b  -> Image<'T> -> Image<'T> -> Image<'T>) : 'a -> 'b -> Slice<'T> -> Slice<'T> -> Slice<'T> =
    fun a b s1 s2 -> { s1 with Image = f a b s1.Image s2.Image }

let liftBinary3 (f: 'a -> 'b -> 'c -> Image<'T> -> Image<'T> -> Image<'T>) : 'a -> 'b -> 'c -> Slice<'T> -> Slice<'T> -> Slice<'T> =
    fun a b c s1 s2 -> { s1 with Image = f a b c s1.Image s2.Image }

let abs = liftUnary ImageFunctions.abs
let log = liftUnary ImageFunctions.log
let log10 = liftUnary ImageFunctions.log10
let exp = liftUnary ImageFunctions.exp
let sqrt = liftUnary ImageFunctions.sqrt
let square = liftUnary ImageFunctions.square
let sin = liftUnary ImageFunctions.sin
let cos = liftUnary ImageFunctions.cos
let tan = liftUnary ImageFunctions.tan
let asin = liftUnary ImageFunctions.asin
let acos = liftUnary ImageFunctions.acos
let atan = liftUnary ImageFunctions.atan
let round = liftUnary ImageFunctions.round

let convolve = liftBinary3 convolve
let conv = liftBinary conv 
let discreteGaussian = liftUnary1 discreteGaussian
let recursiveGaussian = liftUnary2 recursiveGaussian
let laplacianConvolve = liftUnary1 laplacianConvolve
let gradientConvolve = liftUnary2 gradientConvolve 
let binaryErode = liftUnary2 binaryErode
let binaryDilate = liftUnary2 binaryDilate
let binaryOpening = liftUnary2 binaryOpening
let binaryClosing = liftUnary2 binaryClosing
let binaryFillHoles = liftUnary1 binaryFillHoles

let squeeze = liftUnary squeeze
let concatAlong = liftBinary1 concatAlong
let computeStats (img: Image<'T>) : ImageStats =

let connectedComponents = liftUnary connectedComponents
let relabelComponents = liftUnary1 relabelComponents
let signedDistanceMap = liftUnary2 signedDistanceMap
let watershed = liftUnary1 watershed
let otsuThreshold = liftUnary otsuThreshold
let otsuMultiThreshold = liftUnary1 otsuMultiThreshold
let momentsThreshold = liftUnary momentsThreshold

let generateCoordinateAxis (axis: int) (size: int list) : Slice<uint32> =
    {Index = 0; Image = ImageFunctions.generateCoordinateAxis (axis: int) (size: int list)}
let unique (img: Slice<'T>) : 'T list when 'T : comparison =
    ImageFunctions.unique img.Image
let labelShapeStatistics (img: Slice<'T>) : Map<int64, ImageFunctions.LabelShapeStatistics> =
    ImageFunctions.labelShapeStatistics img.Image
let histogram (img: Image<'T>) : Map<'T, uint64> =
    ImageFunctions.histogram img

let addConst = liftUnary1 (+)
let add = liftBinary (+)
let subConst = liftUnary1 (-) // Not symmetric, partner missing.
let sub = liftBinary (-)
let mulConst = liftUnary1 (*)
let mul = liftBinary (*)
let divConst = liftUnary1 (/) // Not symmetric, partner missing.
let div = liftBinary (/)
let modConst = liftUnary1 (%) // Not symmetric, partner missing.
let mod = liftBinary (%)
let powConst = liftUnary1 ImageFunctions.Pow // Not symmetric, partner missing.
let pow = liftBinary ImageFunctions.Pow
let gteConst = liftUnary1 (>=)
let gte = liftBinary (>=)
let gtConst = liftUnary1 (>)
let ge = liftBinary (>)
let eqConst = liftUnary1 (=)
let eq = liftBinary (=)
let neqConst = liftUnary (<>)
let ne = liftBinary (<>)
let ltConst = liftUnary1 (<)
let lt = liftBinary (<)
let lteConst = liftUnary1 (<=)
let lt = liftBinary (<=)
let andConst = liftUnary1 (&&&)
let and = liftBinary (&&&)
let orConst = liftUnary (|||)
let or = liftBinary (|||)
let xorConst = liftUnary1(^^^)
let xor = liftBinary (^^^)
let not = liftUnary (~~~) // Probably not a good idea...

let addNormalNoise = liftUnary2 ImageFunctions.addNormalNoise
let threshold = liftUnary2 ImageFunctions.threshold

let stack (slices: Slice<'T> list) : Slice<'T> =
    match slices with
        elm :: rst ->
            let images = slices |> List.map (fun s -> s.Image)
            {elm with Image = ImageFunctions.stack images }
        | _ -> failwith "Can't stack an empty list"

let extractSlice = liftUnary1 ImageFunctions.extractSlice
