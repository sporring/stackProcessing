module Tests.PooledImageBackend

open System
open System.IO
open Expecto
open Image

let private withEnv name value f =
    let old = Environment.GetEnvironmentVariable(name)
    try
        Environment.SetEnvironmentVariable(name, value)
        f()
    finally
        Environment.SetEnvironmentVariable(name, old)

let private withPooledBackend poison f =
    withEnv "STACKPROCESSING_IMAGE_BACKEND" "arraypool" (fun () ->
        withEnv "STACKPROCESSING_ARRAYPOOL_POISON_ON_RETURN" (if poison then "1" else null) f)

let private withTempTiff name (body: string -> unit) =
    let path = Path.Combine(Path.GetTempPath(), $"{name}-{Guid.NewGuid():N}.tiff")
    try
        body path
    finally
        if File.Exists path then
            File.Delete path

let private withTempImage extension name (body: string -> unit) =
    let path = Path.Combine(Path.GetTempPath(), $"{name}-{Guid.NewGuid():N}.{extension}")
    try
        body path
    finally
        if File.Exists path then
            File.Delete path

let private makeUInt8AwkwardImage () =
    Array2D.init 17 13 (fun x y -> uint8 ((x * 17 + y * 31) % 251))

let private makeUInt16CastImage () =
    Array2D.init 7 5 (fun x y -> uint16 ((x * 9 + y * 11) % 200))

let private makeFloat32AwkwardImage () =
    Array2D.init 17 13 (fun x y -> float32 x * 0.25f + float32 y * 0.5f + 4.0f)

let private expectedThreshold lower (values: uint8[,]) =
    Array2D.init
        (values.GetLength 0)
        (values.GetLength 1)
        (fun x y -> if values[x, y] >= lower then 1uy else 0uy)

let private expectFloat32ArrayClose (actual: float32[,]) (expected: float32[,]) message =
    Expect.equal (actual.GetLength 0, actual.GetLength 1) (expected.GetLength 0, expected.GetLength 1) $"{message}: shape"
    for x in 0 .. actual.GetLength 0 - 1 do
        for y in 0 .. actual.GetLength 1 - 1 do
            Expect.floatClose Accuracy.high (float actual[x, y]) (float expected[x, y]) $"{message}: [{x},{y}]"

let private expectBalancedCounters message =
    let counters = getPooledImageDebugCounters()
    Expect.equal counters.Live 0L $"{message}: all pooled buffers should be returned."
    Expect.equal counters.Returns counters.Rents $"{message}: pooled rents and returns should balance."

[<Tests>]
let pooledImageBackendTests =
    testSequenced <| testList "Pooled Image backend" [
        testCase "ImageSharp reads non-TIFF UInt8 slices into pooled storage" <| fun _ ->
            withTempImage "png" "pooled-imagesharp-png" (fun path ->
                let values = makeUInt8AwkwardImage()
                let source = Image<uint8>.ofArray2D(values, "pooled-png-source")
                source.toFile(path)
                source.decRefCount()

                withPooledBackend true (fun () ->
                    resetPooledImageDebugCounters()
                    let read = ImageIO.readSimpleItkSlice<uint8> path 2 17u 13u 0 "pooled-png-read" 0

                    Expect.equal (read.GetFacts().Backend) "ArrayPool" "PNG read should use ImageSharp pooled storage."
                    Expect.equal (read.toArray2D()) values "ImageSharp pooled read should preserve UInt8 PNG pixels."

                    read.decRefCount()
                    let counters = getPooledImageDebugCounters()
                    Expect.equal counters.Rents 1L "ImageSharp read should rent one pooled buffer."
                    expectBalancedCounters "ImageSharp PNG read"))

        testCase "unsupported non-TIFF formats fall back to SimpleITK under pooled mode" <| fun _ ->
            withTempImage "mha" "pooled-imagesharp-fallback" (fun path ->
                let values = makeUInt8AwkwardImage()
                let source = Image<uint8>.ofArray2D(values, "pooled-mha-source")
                source.toFile(path)
                source.decRefCount()

                withPooledBackend true (fun () ->
                    resetPooledImageDebugCounters()
                    let read = ImageIO.readSimpleItkSlice<uint8> path 2 17u 13u 0 "pooled-mha-read" 0

                    Expect.notEqual (read.GetFacts().Backend) "ArrayPool" "MHA should remain on the SimpleITK fallback path."
                    Expect.equal (read.toArray2D()) values "SimpleITK fallback should preserve UInt8 MHA pixels."

                    read.decRefCount()
                    let counters = getPooledImageDebugCounters()
                    Expect.equal counters.Rents 0L "unsupported non-TIFF fallback should not rent a pooled buffer."
                    expectBalancedCounters "ImageSharp unsupported fallback"))

        testCase "copy and threshold preserve awkward TIFF slices and balance pooled buffers" <| fun _ ->
            withTempTiff "pooled-copy-threshold" (fun path ->
                let values = makeUInt8AwkwardImage()
                let source = Image<uint8>.ofArray2D(values, "pooled-source")
                ImageIO.writeTiffSliceFile path source
                source.decRefCount()

                withPooledBackend true (fun () ->
                    resetPooledImageDebugCounters()
                    let input = ImageIO.readTiffSliceFile<uint8> path 0L
                    let copied = input.copy("pooled-copy")
                    let thresholded = ImageFunctions.threshold 128.0 infinity input

                    Expect.equal (input.GetFacts().Backend) "ArrayPool" "TIFF read should opt into pooled storage."
                    Expect.equal (copied.GetFacts().Backend) "ArrayPool" "copy should stay in pooled storage."
                    Expect.equal (thresholded.GetFacts().Backend) "ArrayPool" "threshold should stay in pooled storage."

                    let copiedValues = copied.toArray2D()
                    let thresholdValues = thresholded.toArray2D()
                    let inputValues = input.toArray2D()

                    Expect.equal inputValues values "pooled TIFF read should preserve awkward-sized UInt8 pixels."
                    Expect.equal copiedValues values "pooled copy should preserve UInt8 pixels."
                    Expect.equal thresholdValues (expectedThreshold 128uy values) "pooled threshold should match expected binary mask."

                    input.decRefCount()
                    copied.decRefCount()
                    thresholded.decRefCount()

                    let counters = getPooledImageDebugCounters()
                    Expect.equal counters.Rents 3L "read, copy, and threshold should each own one pooled buffer."
                    Expect.equal counters.PeakLive 3L "all three pooled buffers should be live before conversion to arrays."
                    expectBalancedCounters "copy/threshold"))

        testCase "scalar arithmetic stays pooled for one-dimensional ArrayPool storage" <| fun _ ->
            withTempTiff "pooled-scalar-arithmetic" (fun path ->
                let values = makeFloat32AwkwardImage()
                let source = Image<float32>.ofArray2D(values, "pooled-arithmetic-source")
                ImageIO.writeTiffSliceFile path source
                source.decRefCount()

                withPooledBackend true (fun () ->
                    resetPooledImageDebugCounters()
                    let input = ImageIO.readTiffSliceFile<float32> path 0L
                    let added = ImageFunctions.imageAddScalar input 2.0f
                    let subtracted = ImageFunctions.scalarSubImage 20.0f input
                    let multiplied = ImageFunctions.imageMulScalar input 3.0f
                    let divided = ImageFunctions.imageDivScalar input 2.0f

                    Expect.equal (input.GetFacts().Backend) "ArrayPool" "input should be pooled."
                    Expect.equal (added.GetFacts().Backend) "ArrayPool" "add scalar should stay pooled."
                    Expect.equal (subtracted.GetFacts().Backend) "ArrayPool" "scalar minus image should stay pooled."
                    Expect.equal (multiplied.GetFacts().Backend) "ArrayPool" "multiply scalar should stay pooled."
                    Expect.equal (divided.GetFacts().Backend) "ArrayPool" "divide scalar should stay pooled."

                    expectFloat32ArrayClose (added.toArray2D()) (values |> Array2D.map (fun v -> v + 2.0f)) "pooled add scalar"
                    expectFloat32ArrayClose (subtracted.toArray2D()) (values |> Array2D.map (fun v -> 20.0f - v)) "pooled scalar sub image"
                    expectFloat32ArrayClose (multiplied.toArray2D()) (values |> Array2D.map (fun v -> v * 3.0f)) "pooled mul scalar"
                    expectFloat32ArrayClose (divided.toArray2D()) (values |> Array2D.map (fun v -> v / 2.0f)) "pooled div scalar"

                    input.decRefCount()
                    added.decRefCount()
                    subtracted.decRefCount()
                    multiplied.decRefCount()
                    divided.decRefCount()
                    expectBalancedCounters "scalar arithmetic"))

        testCase "comparison and logical operators stay pooled for matching pooled images" <| fun _ ->
            withTempTiff "pooled-comparison-a" (fun pathA ->
                withTempTiff "pooled-comparison-b" (fun pathB ->
                    let aValues = Array2D.init 17 13 (fun x y -> uint8 ((x + y) % 4))
                    let bValues = Array2D.init 17 13 (fun x y -> uint8 ((x * 2 + y) % 4))
                    let sourceA = Image<uint8>.ofArray2D(aValues, "pooled-comparison-a")
                    let sourceB = Image<uint8>.ofArray2D(bValues, "pooled-comparison-b")
                    ImageIO.writeTiffSliceFile pathA sourceA
                    ImageIO.writeTiffSliceFile pathB sourceB
                    sourceA.decRefCount()
                    sourceB.decRefCount()

                    withPooledBackend true (fun () ->
                        resetPooledImageDebugCounters()
                        let a = ImageIO.readTiffSliceFile<uint8> pathA 0L
                        let b = ImageIO.readTiffSliceFile<uint8> pathB 0L
                        let eq = ImageFunctions.equalImage a b
                        let gt = ImageFunctions.greaterImage a b
                        let ored = ImageFunctions.orImage eq gt
                        let inverted = ImageFunctions.notImage ored

                        Expect.equal (eq.GetFacts().Backend) "ArrayPool" "equalImage should stay pooled."
                        Expect.equal (gt.GetFacts().Backend) "ArrayPool" "greaterImage should stay pooled."
                        Expect.equal (ored.GetFacts().Backend) "ArrayPool" "orImage should stay pooled."
                        Expect.equal (inverted.GetFacts().Backend) "ArrayPool" "notImage should stay pooled."

                        let expectedEq = Array2D.init 17 13 (fun x y -> if aValues[x, y] = bValues[x, y] then 1uy else 0uy)
                        let expectedGt = Array2D.init 17 13 (fun x y -> if aValues[x, y] > bValues[x, y] then 1uy else 0uy)
                        let expectedOr = Array2D.init 17 13 (fun x y -> if expectedEq[x, y] <> 0uy || expectedGt[x, y] <> 0uy then 1uy else 0uy)
                        let expectedNot = expectedOr |> Array2D.map (fun v -> if v = 0uy then 1uy else 0uy)

                        Expect.equal (eq.toArray2D()) expectedEq "pooled equalImage should match expected mask."
                        Expect.equal (gt.toArray2D()) expectedGt "pooled greaterImage should match expected mask."
                        Expect.equal (ored.toArray2D()) expectedOr "pooled orImage should match expected mask."
                        Expect.equal (inverted.toArray2D()) expectedNot "pooled notImage should match expected mask."

                        a.decRefCount()
                        b.decRefCount()
                        eq.decRefCount()
                        gt.decRefCount()
                        ored.decRefCount()
                        inverted.decRefCount()
                        expectBalancedCounters "comparison/logical")))

        testCase "reference counts delay pooled return until the last release" <| fun _ ->
            withTempTiff "pooled-ref-count" (fun path ->
                let values = makeUInt8AwkwardImage()
                let source = Image<uint8>.ofArray2D(values)
                ImageIO.writeTiffSliceFile path source
                source.decRefCount()

                withPooledBackend true (fun () ->
                    resetPooledImageDebugCounters()
                    let input = ImageIO.readTiffSliceFile<uint8> path 0L
                    input.incRefCount()
                    input.decRefCount()

                    let afterFirstRelease = getPooledImageDebugCounters()
                    Expect.equal afterFirstRelease.Rents 1L "read should rent one pooled buffer."
                    Expect.equal afterFirstRelease.Returns 0L "first release should not return while another reference is live."
                    Expect.equal afterFirstRelease.Live 1L "the pooled buffer should remain live after the first release."

                    input.decRefCount()
                    expectBalancedCounters "reference counted image"))

        testCase "toSimpleITK handoff returns the pooled buffer once and preserves pixels" <| fun _ ->
            withTempTiff "pooled-sitk-handoff" (fun path ->
                let values = makeUInt8AwkwardImage()
                let source = Image<uint8>.ofArray2D(values)
                ImageIO.writeTiffSliceFile path source
                source.decRefCount()

                withPooledBackend true (fun () ->
                    resetPooledImageDebugCounters()
                    let input = ImageIO.readTiffSliceFile<uint8> path 0L
                    let sitk = input.toSimpleITK()
                    let afterHandoff = getPooledImageDebugCounters()

                    Expect.equal afterHandoff.Rents 1L "read should rent one pooled buffer."
                    Expect.equal afterHandoff.Returns 1L "toSimpleITK should return the pooled buffer during handoff."
                    Expect.equal afterHandoff.Live 0L "after handoff, no pooled buffer should remain live."
                    Expect.equal (sitk.GetSize() |> Image.InternalHelpers.fromVectorUInt32) [ 17u; 13u ] "SimpleITK handoff should preserve size."
                    Expect.equal (input.toArray2D()) values "Image should remain readable after SimpleITK handoff."

                    input.decRefCount()
                    expectBalancedCounters "SimpleITK handoff"))

        testCase "mismatched TIFF layout falls back to SimpleITK instead of pooled byte reinterpretation" <| fun _ ->
            withTempTiff "pooled-layout-mismatch" (fun path ->
                let values = makeUInt16CastImage()
                let source = Image<uint16>.ofArray2D(values)
                ImageIO.writeTiffSliceFile path source
                source.decRefCount()

                withPooledBackend true (fun () ->
                    resetPooledImageDebugCounters()
                    let readAsUInt8 = ImageIO.readTiffSliceFile<uint8> path 0L
                    let counters = getPooledImageDebugCounters()

                    Expect.equal counters.Rents 0L "layout mismatch should not adopt pooled typed storage."
                    Expect.notEqual (readAsUInt8.GetFacts().Backend) "ArrayPool" "layout mismatch should fall back to SimpleITK/cast."
                    let expected = values |> Array2D.map uint8
                    Expect.equal (readAsUInt8.toArray2D()) expected "fallback cast should preserve small UInt16 values as UInt8."

                    readAsUInt8.decRefCount()
                    expectBalancedCounters "layout mismatch"))
    ]
