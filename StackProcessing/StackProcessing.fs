module StackProcessing

open SlimPipeline // Core processing model
open System.IO

type Stage<'S,'T> = SlimPipeline.Stage<'S,'T>
type Profile = SlimPipeline.Profile
type ProfileTransition = SlimPipeline.ProfileTransition
//type Slice<'S when 'S: equality> = Slice.Slice<'S>
type Image<'S when 'S: equality> = Image.Image<'S>

let getMem () =
    System.GC.Collect()
    System.GC.WaitForPendingFinalizers()
    System.GC.Collect()

let incIfImage x =
    match box x with
    | :? Image<uint8> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int8> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<uint16> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int16> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<uint> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<uint64> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int64> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<float32> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<float> as im -> im.incRefCount()   // or img.incNRefCount(1) if it takes an int
    | _ -> ()
    x
let incRef () =
    Stage.map "incRefCountOp" incIfImage id id
let decIfImage x =
    match box x with
    | :? Image<uint8> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int8> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<uint16> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int16> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<uint> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<uint64> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<int64> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<float32> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | :? Image<float> as im -> im.decRefCount()   // or img.incNRefCount(1) if it takes an int
    | _ -> ()
    x
let decRef () =
    Stage.map "decRefCountOp" decIfImage id id
let releaseAfter (f: Image<'S>->'T) (I:Image<'S>) = 
    let v = f I
    I.decRefCount()
    v
let releaseAfter2 (f: Image<'S>->Image<'S>->'T) (I:Image<'S>) (J:Image<'S>) = 
    let v = f I J
    decIfImage I |> ignore
    decIfImage J |> ignore
    v
(*
let releaseNAfter (n: int) (f: Image<'S> list->'T list) (sLst: Image<'S> list) : 'T list =
    let tLst = f sLst;
    sLst |> List.take (int n) |> List.map (decIfImage >> ignore) |> ignore
    tLst 
*)
let (>=>) = Plan.(>=>)
let (-->) = Stage.(-->)
let source = Plan.source 
let debug availableMemory = 
    Image.Image<_>.setDebug true; 
    Plan.debug availableMemory
 
let zip = Plan.zip

(*
let inline isExactlyImage<'T> () =
    let t = typeof<'T>
    t.IsGenericType && t.GetGenericTypeDefinition() = typedefof<Image<_>>
*)
let promoteStreamingToWindow 
    (name: string)
    (winSz: uint)
    (pad: uint)
    (stride: uint)
    (emitStart: uint)
    (emitCount: uint)
    (stage: Stage<'T,'S>)
    : Stage<'T, 'S> = // Does not change shape
        let zeroMaker i = id
        (Stage.window $"{name}:window" winSz pad zeroMaker stride) 
        --> (Stage.map $"{name}:skip and take" (fun lst ->
                let result = lst |> List.skip (int stride) |> List.take 1
                printfn $"disposing of {stride} initial images"
                lst |> List.take (int stride) |> List.map decIfImage |> ignore
                result
            ) id id )
        --> Stage.flatten $"{name}:flatten"
        --> stage

let (>=>>) (pl: Plan<'In, 'S>) (stage1: Stage<'S, 'U>, stage2: Stage<'S, 'V>) : Plan<'In, 'U * 'V> = 
    let stream2Window winSz pad stride stg = 
        stg |> promoteStreamingToWindow "makeWindow" winSz pad stride 0u 1u

    let stg1,stg2 =
        match stage1.Transition.From, stage2.Transition.From with
        | Streaming, Streaming -> stage1, stage2
        | Window (a1,b1,c1,d1,e1), Window (a2,b2,c2,d2,e2) when a1=a2 && b1=b2 && c1=c2 && d1=d2 && e1=e2 -> stage1, stage2
        | Streaming, Window (winSz, stride, pad, emitStart, emitCount) -> 
            printfn "left is promoted"
            stream2Window winSz pad stride stage1, stage2 
        | Window (winSz, stride, pad, emitStart, emitCount), Streaming -> 
            printfn "right is promoted"
            stage1, stream2Window winSz pad stride stage2
        | _,_ -> failwith $"[>=>>] does not know how to combine the stage-profiles: {stage1.Transition.From} vs {stage2.Transition.From}"

    Plan.(>=>>) (pl >=> incRef ()) (stg1, stg2)

let (>>=>) = Plan.(>>=>)
let (>>=>>) = Plan.(>>=>>)
let zeroMaker (index:int) (ex:Image<'S>) : Image<'S> = new Image<'S>(ex.GetSize(), 1u, "padding", index)
let window windowSize pad stride= Stage.window "window" windowSize pad zeroMaker stride
let flatten () = Stage.flatten "flatten"
let map f = Stage.map "map" f id id
let sink (pl: Plan<unit,unit>) : unit = 
    Plan.sink pl
let sinkList (plLst: Plan<unit,unit> list) : unit = Plan.sinkList plLst
//let combineIgnore = Plan.combineIgnore
let drainSingle pl = Plan.drainSingle "drainSingle" pl
let drainList pl = Plan.drainList "drainList" pl
let drainLast pl = Plan.drainLast "drainLast" pl
//let tap str = incRefCountOp () --> (Stage.tap str)
let tap = Stage.tap
//let tap str = Stage.tap str --> incRef()// tap and tapIt neither realeases after nor increases number of references
let tapIt = Stage.tapIt
let ignoreSingles () : Stage<Image<_>,unit> = Stage.ignore (decIfImage>>ignore)
let ignorePairs () : Stage<_,unit> = Stage.ignorePairs<_,unit> ((decIfImage>>ignore),(decIfImage>>ignore))
let idOp<'T> = Stage.idOp<'T>

let liftUnary name  = Stage.liftReleaseUnary name ignore
let liftUnaryReleaseAfter 
    (name: string)
    (f: Image<'S> -> Image<'T>)
    (memoryNeed: MemoryNeed)
    (lengthTransformation: LengthTransformation) = 
    Stage.liftReleaseUnary name (decIfImage>>ignore) f memoryNeed lengthTransformation

let getBytesPerComponent<'T> = (typeof<'T> |> Image.getBytesPerComponent |> uint64)

type System.String with // From https://stackoverflow.com/questions/1936767/f-case-insensitive-string-compare
    member s1.icompare(s2: string) =
        System.String.Equals(s1, s2, System.StringComparison.CurrentCultureIgnoreCase)

let write (outputDir: string) (suffix: string) : Stage<Image<'T>, unit> =
    let t = typeof<'T>
    if (suffix.icompare ".tif" || suffix.icompare ".tiff") 
        && not (t = typeof<uint8> || t = typeof<int8> || t = typeof<uint16> || t = typeof<int16> || t = typeof<float32>) then
        failwith $"[write] tiff images only supports (u)int8, (u)int16 and float32 but was {t.Name}" 
    if not (Directory.Exists(outputDir)) then
        Directory.CreateDirectory(outputDir) |> ignore
    let consumer (debug: bool) (idx: int) (image:Image<'T>) =
        let fileName = Path.Combine(outputDir, sprintf "image_%03d%s" idx suffix)
        if debug then printfn "[write] Saved image %d to %s as %s" idx fileName (typeof<'T>.Name) 
        image.toFile(fileName)
        image.decRefCount()
    let memoryNeed = id
    Stage.consumeWith $"write \"{outputDir}/*{suffix}\"" consumer memoryNeed

let show (plt: Image<'T> -> unit) : Stage<Image<'T>, unit> =
    let consumer (debug: bool) (idx: int) (image:Image<'T>) =
        if debug then printfn "[show] Showing image %d" idx
        let width = image.GetWidth() |> int
        let height = image.GetHeight() |> int
        plt image
        image.decRefCount()
    let memoryNeed = id
    Stage.consumeWith "show" consumer memoryNeed

let plot (plt: (float list)->(float list)->unit) : Stage<(float * float) list, unit> = // better be (float*float) list
    let consumer (debug: bool) (idx: int) (points: (float*float) list) =
        if debug then printfn $"[plot] Plotting {points.Length} 2D points"
        let x,y = points |> List.unzip
        plt x y
    let memoryNeed = id
    Stage.consumeWith "plot" consumer memoryNeed

let print () : Stage<'T, unit> =
    let consumer (debug: bool) (idx: int) (elm: 'T) =
        if debug then printfn "[print]"
        printfn "%d -> %A" idx elm
    let memoryNeed = id
    Stage.consumeWith "print" consumer memoryNeed

(*
let liftImageSource (name: string) (img: Image<'T>) : Pipe<unit, Image<'T>> =
    {
        Name = name
        Profile = Streaming
        Apply = fun _ -> img |> Image.unstack |> AsyncSeq.ofSeq
    }

let axisSource
    (axis: int) 
    (size: int list)
    (pl : Plan<unit, unit>) 
    : Plan<unit, Image<uint>> =
    let img = Image.generateCoordinateAxis axis size
    let sz = Image.GetSize img
    let shapeUpdate = fun (s: Shape) -> s
    let op : Stage<unit, Image<uint>> =
        {
            Name = "axisSource"
            Pipe = img |> liftImageSource "axisSource"
            Transition = ProfileTransition.create Unit Streaming
            ShapeUpdate = shapeUpdate
        }
    let width, height, depth = sz[0], sz[1], sz[2]
    let context = ShapeContext.create (fun _ -> width*height |> uint64) (fun _ -> depth)
    {
        flow = Flow.returnM op
        mem = pl.memAvail
        shape = Slice (width,height) |> Some
        context = context
        debug = pl.debug
    }
*)

/// Pixel type casting
let cast<'S,'T when 'S: equality and 'T: equality> = 
    let name = sprintf "cast(%s->%s)" typeof<'S>.Name typeof<'T>.Name
    let f (I: Image<'S>) = 
        let result = I.castTo<'T> ()
        I.decRefCount()
        result
    Stage.cast<Image<'S>,Image<'T>> name f id

/// Basic arithmetic
let liftRelease2 f I J = releaseAfter2 (fun a b -> f a b) I J

let memNeeded<'T> nTimes nElems = 3UL*nElems*nTimes*getBytesPerComponent<'T> // We need input, output, and potentially a cast in between
let add (image: Image<'T>) = 
    liftUnaryReleaseAfter "add" ((+) image) id id
let addPair I J = liftRelease2 ( + ) I J
let inline scalarAddImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) = 
    liftUnaryReleaseAfter "scalarAddImage" (fun (s:Image<^T>)->ImageFunctions.scalarAddImage<^T> i s) id id
let inline imageAddScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "imageAddScalar" (fun (s:Image<^T>)->ImageFunctions.imageAddScalar<^T> s i) id id

let sub (image: Image<'T>) = 
    liftUnaryReleaseAfter "sub" ((-) image) id id

let subPair I J = liftRelease2 ( - ) I J
let inline scalarSubImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "scalarSubImage" (fun (s:Image<^T>)->ImageFunctions.scalarSubImage<^T> i s) id id
let inline imageSubScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "imageSubScalar" (fun (s:Image<^T>)->ImageFunctions.imageSubScalar<^T> s i) id id

let mul (image: Image<'T>) = liftUnaryReleaseAfter "mul" (( * ) image) id id
let mulPair I J = liftRelease2 ( * ) I J
let inline scalarMulImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "scalarMulImage" (fun (s:Image<^T>)->ImageFunctions.scalarMulImage<^T> i s) id id
let inline imageMulScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "imageMulScalar" (fun (s:Image<^T>)->ImageFunctions.imageMulScalar<^T> s i) id id

let div (image: Image<'T>) = liftUnaryReleaseAfter "div" ((/) image) id id

let divPair I J = liftRelease2 ( / ) I J
let inline scalarDivImage<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "scalarDivImage" (fun (s:Image<^T>)->ImageFunctions.scalarDivImage<^T> i s) id id
let inline imageDivScalar<^T when ^T: equality and ^T: (static member op_Explicit: ^T -> float)> (i: ^T) =
    liftUnaryReleaseAfter "imageDivScalar" (fun (s:Image<^T>)->ImageFunctions.imageDivScalar<^T> s i) id id

let maxOfPair I J = liftRelease2 Image.maximumImage I J

let minOfPair I J = liftRelease2 Image.minimumImage I J
let getMinMax I = releaseAfter Image.getMinMax I;

let failTypeMismatch<'T> name lst =
    let t = typeof<'T>
    if lst |> List.exists ((=) t) |> not then
        let names = List.map (fun (t: System.Type) -> t.Name) lst
        failwith $"[{name}] wrong type. Type {t} must be one of {names}"

/// Simple functions
let private floatNintTypes = [typeof<float>;typeof<float32>;typeof<int>]
let private floatTypes = [typeof<float>;typeof<float32>]
let abs<'T when 'T: equality> : Stage<Image<'T>,Image<'T>> = 
    failTypeMismatch<'T> "abs" floatNintTypes
    liftUnaryReleaseAfter "abs"    ImageFunctions.absImage id id
let acos<'T when 'T: equality>     : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "acos" floatTypes
    liftUnaryReleaseAfter "acos"   ImageFunctions.acosImage id id
let asin<'T when 'T: equality>     : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "asin" floatTypes
    liftUnaryReleaseAfter "asin"   ImageFunctions.asinImage id id
let atan<'T when 'T: equality>     : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "atan" floatTypes
    liftUnaryReleaseAfter "atan"   ImageFunctions.atanImage id id
let cos<'T when 'T: equality>     : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "cos" floatTypes
    liftUnaryReleaseAfter "cos"    ImageFunctions.cosImage id id
let sin<'T when 'T: equality>     : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "sin" floatTypes
    liftUnaryReleaseAfter "sin"    ImageFunctions.sinImage id id
let tan<'T when 'T: equality>     : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "tan" floatTypes
    liftUnaryReleaseAfter "tan"    ImageFunctions.tanImage id id
let exp<'T when 'T: equality>     : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "exp" floatTypes
    liftUnaryReleaseAfter "exp"    ImageFunctions.expImage id id
let log10<'T when 'T: equality>     : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "log10" floatTypes
    liftUnaryReleaseAfter "log10"  ImageFunctions.log10Image id id
let log<'T when 'T: equality>     : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "log" floatTypes
    liftUnaryReleaseAfter "log"    ImageFunctions.logImage id id
let round<'T when 'T: equality>     : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "round" floatTypes
    liftUnaryReleaseAfter "round"  ImageFunctions.roundImage id id
let sqrt<'T when 'T: equality>     : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "sqrt" floatNintTypes
    liftUnaryReleaseAfter "sqrt"   ImageFunctions.sqrtImage id id
let square<'T when 'T: equality>     : Stage<Image<'T>,Image<'T>> =      
    failTypeMismatch<'T> "square" floatNintTypes
    liftUnaryReleaseAfter "square" ImageFunctions.squareImage id id

//let histogram<'T when 'T: comparison> = histogramOp<'T> "histogram"
let imageHistogram () =
    Stage.map<Image<'T>,Map<'T,uint64>> "histogram:map" (releaseAfter ImageFunctions.histogram) id id// Assumed max for uint8, can be done better

let imageHistogramFold () =
    Stage.fold<Map<'T,uint64>, Map<'T,uint64>> "histogram:fold" ImageFunctions.addHistogram (Map.empty<'T, uint64>) id id

let histogram () =
    imageHistogram () --> imageHistogramFold ()

let inline map2pairs< ^T, ^S when ^T: comparison and ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = 
    let map2pairs (map: Map<'T, 'S>): ('T * 'S) list =
        map |> Map.toList
    liftUnary "map2pairs" map2pairs id id
let inline pairs2floats< ^T, ^S when ^T : (static member op_Explicit : ^T -> float) and  ^S : (static member op_Explicit : ^S -> float) > = 
    let pairs2floats (pairs: (^T * ^S) list) : (float * float) list =
        pairs |> List.map (fun (k, v) -> (float k, float v)) 
    liftUnary "pairs2floats" pairs2floats id id
let inline pairs2ints< ^T, ^S when ^T : (static member op_Explicit : ^T -> int) and  ^S : (static member op_Explicit : ^S -> int) > = 
    let pairs2ints (pairs: (^T * ^S) list) : (int * int) list =
        pairs |> List.map (fun (k, v) -> (int k, int v)) 
    liftUnary "pairs2ints" pairs2ints id id

type ImageStats = ImageFunctions.ImageStats
let imageComputeStats () =
    Stage.map<Image<'T>,ImageStats> "computeStats:map" (releaseAfter ImageFunctions.computeStats) id id

let imageComputeStatsFold () =
    let zeroStats: ImageStats = { 
        NumPixels = 0u
        Mean = 0.0
        Std = 0.0
        Min = infinity
        Max = -infinity
        Sum = 0.0
        Var = 0.0
    }
    Stage.fold<ImageStats, ImageStats> "computeStats:fold" ImageFunctions.addComputeStats zeroStats id id

let computeStats () =
    imageComputeStats () --> imageComputeStatsFold ()

////////////////////////////////////////////////
/// Convolution like operators

// Chained type definitions do expose the originals
open type ImageFunctions.OutputRegionMode
open type ImageFunctions.BoundaryCondition

let stackFUnstack f (images : Image<'T> list) =
    let stck = images |> ImageFunctions.stack 
    stck |> releaseAfter (f >> ImageFunctions.unstack)

let skipNTakeM (n: uint) (m: uint) (lst: 'a list) : 'a list =
    let m = uint lst.Length - 2u*n;
    if m = 0u then []
    else lst |> List.skip (int n) |> List.take (int m) // This needs releaseAfter!!!

let stackFUnstackTrim trim (f: Image<'T>->Image<'S>) (images : Image<'T> list) =
    //printfn $"stackFUnstackTrim: stacking"
    let stck = images |> ImageFunctions.stack 
    //printfn $"stackFUnstackTrim: applying f and ustacking"
    //let result = stck |> (f >> ImageFunctions.unstack >> skipNTakeM trim m)
    let result = 
        //printfn $"applying function to stack"
        let volRes = f stck
        stck.decRefCount()
        //printfn $"unstacking function result"
//        let imageLst = ImageFunctions.unstack volRes
        let m = uint images.Length - 2u*trim // last stack may be smaller if stride > 1
        let imageLst = ImageFunctions.unstackSkipNTakeM trim m volRes
        volRes.decRefCount()
        imageLst
        //printfn $"skipntakem"
        //let r = skipNTakeM trim m imageLst
        //imageLst |> List.iteri (fun i I -> if i < (int trim) || i >= (int (trim + m)) then I.decRefCount())
        //printfn $"result ready"
        //r
    //printfn $"stackFUnstackTrim: returning result"
    result

let volFctToLstFctReleaseAfter (f:Image<'S>->Image<'T>) pad stride images =
    let stack = ImageFunctions.stack images 
    images |> List.take (int stride) |> List.iter (fun I -> I.decRefCount())
    let vol = f stack
    stack.decRefCount ()
    let result = ImageFunctions.unstackSkipNTakeM pad stride vol
    vol.decRefCount ()
    result


let discreteGaussianOp (name:string) (sigma:float) (outputRegionMode: ImageFunctions.OutputRegionMode option) (boundaryCondition: ImageFunctions.BoundaryCondition option) (winSz: uint option): Stage<Image<float>, Image<float>> =
    let roundFloatToUint v = uint (v+0.5)

    let ksz = 4.0 * sigma + 1.0 |> roundFloatToUint
    let win = Option.defaultValue ksz winSz |> max ksz // max should be found by memory availability
    let stride = win - ksz + 1u
    let pad = 
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> ksz/2u //floor
    // length-2*pad = n*stride+windowSize
    // stride = windowSize-2*pad
    // => n = (windowSize-length-2*pad)/(2*pad-windowSize)
    // e.g., integer solutions for 
    // windowSize = 1, 6, 15, or 26, pad = 2, length = 22, => n = 21, 10, 1, or 0
    printfn $"discreteGaussianOp: sigma {sigma}, ksz {ksz}, win {win}, stride {stride}, pad {pad}"
    let f = volFctToLstFctReleaseAfter (ImageFunctions.discreteGaussian 3u sigma (ksz |> Some) outputRegionMode boundaryCondition) pad stride
    let memoryNeed nPixels = (2UL*nPixels*(uint64 win) + (uint64 ksz))*(typeof<float> |> Image.getBytesPerComponent |> uint64)
    let lengthTransformation nElems = 
        match outputRegionMode with
            | Some Valid -> nElems - 2UL * uint64 pad
            |_ -> nElems
    let stg = Stage.map name f memoryNeed lengthTransformation // wrong for Valid, where the sequences becomes shorter
    (window win pad stride) --> stg --> flatten ()

let discreteGaussian = discreteGaussianOp "discreteGaussian"
let convGauss sigma = discreteGaussianOp "convGauss" sigma None None None

// stride calculation example
// ker = 3, win = 7
// Image position:  2 1 0 1 2 3 4 5 6 7 8 9 
// First window         * * * * * * *
// Kern position1   * * *            
//                    * * *         
//                      * * * √        
//                        * * * √      
//                          * * * √   
//                            * * * √    
//                              * * * √   
//                                * * *
//                                  * * *
//                                    * * *
// Next window                    * * * * * * *
// Kern                       * * *
//                              * * *         
//                                * * * √  
//                                  * * * √   
//                                    * * * √
//.                                     * * * √
//                                        * * * √
//                                          * * *
//                                            * * *

let createPadding name pad : Stage<unit,Image<'S>>=
    let transition = ProfileTransition.create Streaming Streaming
    let memoryNeed nPixels = nPixels*(typeof<'S> |> Image.getBytesPerComponent |> uint64)
    let lengthTransformation nElems = nElems + (uint64 pad)
    let zeroMaker i = Image<'S>([0u;0u],1u,"Padding",i)
    Stage.init "padding" pad zeroMaker transition memoryNeed lengthTransformation

let convolveOp (name: string) (kernel: Image<'T>) (outputRegionMode: ImageFunctions.OutputRegionMode option) (bc: ImageFunctions.BoundaryCondition option) (winSz: uint option): Stage<Image<'T>, Image<'T>> =
    let windowFromKernel (k: Image<'T>) : uint =
        max 1u (k.GetDepth())
    let ksz = windowFromKernel kernel
    let win = Option.defaultValue ksz winSz |> min ksz
    let stride = win-ksz+1u
    let pad = 
        match outputRegionMode with
            | Some Valid -> 0u
            | _ -> ksz/2u //floor
    let f = volFctToLstFctReleaseAfter (fun image3D -> ImageFunctions.convolve outputRegionMode bc image3D kernel) pad stride
    let memoryNeed nPixels = (2UL*nPixels*(uint64 win) + (uint64 ksz))*(typeof<'T> |> Image.getBytesPerComponent |> uint64)
    let lengthTransformation nElems = nElems - 2UL*(uint64 pad) 
    let stg = Stage.map name f memoryNeed lengthTransformation
    let padding = createPadding "padding" pad 
    (Stage.prepend "prepend" padding) --> (Stage.append "append" padding) --> (window win 0u stride) --> stg --> flatten ()

let convolve kernel outputRegionMode boundaryCondition winSz = convolveOp "convolve" kernel outputRegionMode boundaryCondition winSz
let conv kernel = convolveOp "conv" kernel None None None

let finiteDiff (sigma: float) (direction: uint) (order: uint) =
    let kernel = ImageFunctions.finiteDiffFilter3D sigma direction order
    convolveOp "finiteDiff" kernel None None None

// these only works on uint8
let private makeMorphOp (name:string) (radius:uint) (winSz: uint option) (core: uint -> Image<'T> -> Image<'T>) : Stage<Image<'T>,Image<'T>> when 'T: equality =
    let ksz   = 2u * radius + 1u
    let pad = ksz/2u
    let win = Option.defaultValue ksz winSz |> min ksz
    let stride = win - ksz + 1u

    let f = volFctToLstFctReleaseAfter (core radius) pad stride
    let stg = Stage.map name f id id
    (window win pad stride) --> stg --> flatten ()


let erode radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryErode
let dilate radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryDilate
let opening radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryOpening
let closing radius = makeMorphOp "binaryErode"  radius None ImageFunctions.binaryClosing

/// Full stack operators
let binaryFillHoles (winSz: uint)= 
    let pad, stride = 0u, winSz
    let f = volFctToLstFctReleaseAfter (ImageFunctions.binaryFillHoles) pad stride
    let stg = Stage.map "fillHoles" f id id
    (window winSz pad stride) --> stg --> flatten ()


let connectedComponents (winSz: uint) = 
    let pad, stride = 0u, winSz
    let f = volFctToLstFctReleaseAfter (ImageFunctions.connectedComponents) pad stride
    let btUint8 = typeof<uint8>|>Image.getBytesPerComponent |> uint64
    let btUint64 = typeof<uint64> |> Image.getBytesPerComponent |> uint64
    // Sliding window applying f cost assuming f has no extra costs:
    //   window takes winSz Image<uint8>:            nPixels*winSz*1
    //   produces a stack of equal size in <uint8>:  2*nPixels*winSz*1
    //   releases stride Image<uint8>:               nPixels*(2*winSz*1-*stride*1)
    //   produces a stack of equal size in <uint64>: nPixels*(winSz*(2*1+8)-stride*1)
    //   releases <uint8> stack:                     nPixels*(winSz*(1+8)-stride*1)
    //   produces a stride list of Image<uint64>:    nPixels*(winSz*(1+8)+stride*(8-1))
    //   releases <uint64> stack:                    nPixels*(winSz*1+stride*(8-1))
    let memoryNeed nPixels = 
        let bt8 = typeof<uint8>|>Image.getBytesPerComponent |> uint64
        let bt64 = typeof<uint64> |> Image.getBytesPerComponent |> uint64
        let wsz = uint64 winSz
        let str = uint64 stride
        max (nPixels*(wsz*(2UL*bt8+bt64)-str*bt8)) (nPixels*(wsz*(bt8+bt64)+str*(bt64-bt8)))
    let lengthTransformation = id
    let stg = Stage.map "connectedComponents" f memoryNeed lengthTransformation
    (window winSz pad stride) --> stg --> flatten ()

let relabelComponents a (winSz: uint) = 
    let pad, stride = 0u, winSz
    let f = volFctToLstFctReleaseAfter (ImageFunctions.relabelComponents a) pad stride
    let stg = Stage.map "relabelComponents" f id id
    (window winSz pad stride) --> stg --> flatten ()

let watershed a (winSz: uint) =
    let pad, stride = 0u, winSz
    let f = volFctToLstFctReleaseAfter (ImageFunctions.watershed a) pad stride
    let stg = Stage.map "watershed" f id id
    (window winSz pad stride) --> stg --> flatten ()
let signedDistanceMap (winSz: uint) =
    let pad, stride = 0u, winSz
    let f = volFctToLstFctReleaseAfter (ImageFunctions.signedDistanceMap 0uy 1uy) pad stride
    let stg = Stage.map "signedDistanceMap" f id id
    (window winSz pad stride) --> stg --> flatten ()
let otsuThreshold (winSz: uint) =
    let pad, stride = 0u, winSz
    let f = volFctToLstFctReleaseAfter (ImageFunctions.otsuThreshold) pad stride
    let stg = Stage.map "otsuThreshold" f id id
    (window winSz pad stride) --> stg --> flatten ()

let momentsThreshold (winSz: uint) =
    let pad, stride = 0u, winSz
    let f = volFctToLstFctReleaseAfter (ImageFunctions.momentsThreshold) pad stride
    let stg = Stage.map "momentsThreshold" f id id
    (window winSz pad stride) --> stg --> flatten ()

let threshold a b = liftUnaryReleaseAfter "threshold" (ImageFunctions.threshold a b) id id

let addNormalNoise a b = liftUnaryReleaseAfter "addNormalNoise" (ImageFunctions.addNormalNoise a b) id id

let ImageConstantPad<'T when 'T : equality> (padLower : uint list) (padUpper : uint list) (c : double) =
    liftUnaryReleaseAfter "constantPad2D" (ImageFunctions.constantPad2D padLower padUpper c) id id // Check that constantPad2D makes a new image!!!

// Not Pipes nor Operators
type FileInfo = ImageFunctions.FileInfo
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

let getStackSize (inputDir: string) (suffix: string): uint*uint*uint =
    let fi = getStackInfo inputDir suffix 
    (uint fi.size[0],uint fi.size[1],uint fi.size[2])

let getStackWidth (inputDir: string) (suffix: string): uint64 =
    let fi = getStackInfo inputDir suffix
    fi.size[0]

let getStackHeight (inputDir: string) (suffix: string): uint64 =
    let fi = getStackInfo inputDir suffix
    //printfn "%A" fi
    fi.size[1]

let zero<'T when 'T: equality> 
    (width: uint) 
    (height: uint) 
    (depth: uint) 
    (pl : Plan<unit, unit>) 
    : Plan<unit, Image<'T>> =
    // width, heigth, depth should be replaced with shape and shapeUpdate, and mapper should be deferred to outside Core!!!
    if pl.debug then printfn $"[zero] {width}x{height}x{depth}"
    let mapper (i: int) : Image<'T> = 
        let image = new Image<'T>([width; height], 1u,$"zero[{i}]", i)
        if pl.debug then printfn "[zero] Created image %A" i
        image
    let transition = ProfileTransition.create Unit Streaming
    let memPeak = Image<'T>.memoryEstimate width height 1u
    let memoryNeed = fun _ -> memPeak
    let lengthTransformation = id
    let stage = Stage.init "create" depth mapper transition memoryNeed lengthTransformation |> Some
    //let flow = Flow.returnM stage
    let nElems = (uint64 width) * (uint64 height)
    let memPeak = Image<'T>.memoryEstimate width height 1u
    Plan.create stage pl.memAvail memPeak nElems (uint64 depth)  pl.debug

let createByEuler2DTransform<'T when 'T: equality> 
    (img: Image<'T>) 
    (depth: uint) 
    (transform: uint -> (float*float*float) * (float*float))
    (pl : Plan<unit, unit>) 
    : Plan<unit, Image<'T>> =
    // width, heigth, depth should be replaced with shape and shapeUpdate, and mapper should be deferred to outside Core!!!
    let width= img.GetWidth()
    let height = img.GetHeight()
    if pl.debug then printfn $"[createByTranslation] {width}x{height}x{depth}"
    let mapper (i: int) : Image<'T> =
        let rot, trans = transform (uint i)
        let image = ImageFunctions.euler2DTransform img rot trans
        if pl.debug then printfn "[createByTranslation] Created image %A" i
        image
    let transition = ProfileTransition.create Unit Streaming
    let memPeak = Image<'T>.memoryEstimate width height 1u
    let memoryNeed = fun _ -> memPeak
    let lengthTransformation = id
    let stage = Stage.init "create" depth mapper transition memoryNeed lengthTransformation |> Some
    //let flow = Flow.returnM stage
    let nElems = (uint64 width) * (uint64 height)
    let memPeak = Image<'T>.memoryEstimate width height 1u
    Plan.create stage pl.memAvail memPeak nElems (uint64 depth)  pl.debug

let readFilteredOp<'T when 'T: equality> (name:string) (inputDir : string) (suffix : string) (filter: string[]->string[]) (pl : Plan<unit, unit>) : Plan<unit, Image<'T>> =
    // much should be deferred to outside Core!!!
    if pl.debug then printfn $"[{name} cast to {typeof<'T>.Name}]"
    let (width,height,depth) = getStackSize inputDir suffix
    let filenames = Directory.GetFiles(inputDir, "*"+suffix) |> filter
    let depth = uint filenames.Length
    let mapper (i: int) : Image<'T> = 
        let fileName = filenames[i]; 
        if pl.debug then printfn "[%s] Reading image %A from %s as %s" name i fileName (typeof<'T>.Name)
        let image = Image<'T>.ofFile (fileName, fileName, i)
        image
    let transition = ProfileTransition.create Unit Streaming
    let memPeak = Image<'T>.memoryEstimate width height 1u
    let memoryNeed = fun _ -> memPeak
    let lengthTransformation = id
    let stage = Stage.init $"{name}" (uint depth) mapper transition memoryNeed lengthTransformation |> Some
    //let flow = Flow.returnM stage
    let memPerElem = (uint64 width)*(uint64 height)*getBytesPerComponent<'T>
    let length = (uint64 depth)
    Plan.create stage pl.memAvail memPeak memPerElem length  pl.debug

let readFiltered<'T when 'T: equality> (inputDir : string) (suffix : string)  (filter: string[]->string[]) (pl : Plan<unit, unit>) : Plan<unit, Image<'T>> =
    readFilteredOp<'T> $"readFiltered" inputDir suffix filter pl

let read<'T when 'T: equality> (inputDir : string) (suffix : string) (pl : Plan<unit, unit>) : Plan<unit, Image<'T>> =
    readFilteredOp<'T> $"read" inputDir suffix Array.sort pl

let readRandom<'T when 'T: equality> (count: uint) (inputDir : string) (suffix : string) (pl : Plan<unit, unit>) : Plan<unit, Image<'T>> =
    readFilteredOp<'T> $"readRandom" inputDir suffix (Array.randomChoices (int count)) pl

let empty (pl : Plan<unit, unit>) : Plan<unit, unit> =
    let stage = "empty" |> Stage.empty |> Some
    Plan.create stage pl.memAvail 0UL 0UL 0UL  pl.debug
