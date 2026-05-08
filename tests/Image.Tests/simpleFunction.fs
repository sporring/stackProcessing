module Tests.simpleFunction

open Expecto
open itk.simple
open Image

[<Tests>]
let simpleFunctionTests =
  testList "Simple arithmetic function tests" [

  testCase "sum for uint8 2x2 image" <| fun _ ->
    let arr = array2D [ [1uy; 2uy]; [3uy; 4uy] ]
    let img = Image<uint8>.ofArray2D arr
    Expect.equal (img |> ImageFunctions.sum) 10uy "Sum of 1+2+3+4 uint8"
  testCase "sum for int8 2x2 image" <| fun _ ->
    let arr = array2D [ [1y; 2y]; [3y; 4y] ]
    let img = Image<int8>.ofArray2D arr
    Expect.equal (img |> ImageFunctions.sum) 10y "Sum of 1+2+3+4 int8"
  testCase "sum for uint16 2x2 image" <| fun _ ->
    let arr = array2D [ [1us; 2us]; [3us; 4us] ]
    let img = Image<uint16>.ofArray2D arr
    Expect.equal (img |> ImageFunctions.sum) 10us "Sum of 1+2+3+4 uint16"
  testCase "sum for int16 2x2 image" <| fun _ ->
    let arr = array2D [ [1s; 2s]; [3s; 4s] ]
    let img = Image<int16>.ofArray2D arr
    Expect.equal (img |> ImageFunctions.sum) 10s "Sum of 1+2+3+4 int16"
  testCase "sum for uint 2x2 image" <| fun _ ->
    let arr = array2D [ [1u; 2u]; [3u; 4u] ]
    let img = Image<uint>.ofArray2D arr
    Expect.equal (img |> ImageFunctions.sum) 10u "Sum of 1+2+3+4 uint"
  testCase "sum for int 2x2 image" <| fun _ ->
    let arr = array2D [ [1; 2]; [3; 4] ]
    let img = Image<int>.ofArray2D arr
    Expect.equal (img |> ImageFunctions.sum) 10 "Sum of 1+2+3+4 int"
  testCase "sum for uint64 2x2 image" <| fun _ ->
    let arr = array2D [ [1uL; 2uL]; [3uL; 4uL] ]
    let img = Image<uint64>.ofArray2D arr
    Expect.equal (img |> ImageFunctions.sum) 10uL "Sum of 1+2+3+4 uint64"
  testCase "sum for int64 2x2 image" <| fun _ ->
    let arr = array2D [ [1L; 2L]; [3L; 4L] ]
    let img = Image<int64>.ofArray2D arr
    Expect.equal (img |> ImageFunctions.sum) 10L "Sum of 1+2+3+4 int64"
  testCase "sum for float32 2x2 image" <| fun _ ->
    let arr = array2D [ [1.0f; 2.0f]; [3.0f; 4.0f] ]
    let img = Image<float32>.ofArray2D arr
    Expect.floatClose Accuracy.veryHigh (img |> ImageFunctions.sum|>float) 10.0 "Sum of float32"
  testCase "sum for float 2x2 image" <| fun _ ->
    let arr = array2D [ [1.0; 2.0]; [3.0; 4.0] ]
    let img = Image<float>.ofArray2D arr
    Expect.floatClose Accuracy.veryHigh (img |> ImageFunctions.sum|>float) 10.0 "Sum of floats"

  testCase "stack joins matching 2D slices into a 3D volume" <| fun _ ->
    let first = Image<uint8>.ofArray2D (array2D [ [1uy; 2uy]; [3uy; 4uy] ])
    let second = Image<uint8>.ofArray2D (array2D [ [5uy; 6uy]; [7uy; 8uy] ])
    let stacked = ImageFunctions.stack [first; second]
    Expect.equal (stacked.GetSize()) [2u; 2u; 2u] "Two 2D slices should become one 3D volume."
    Expect.equal stacked.[0,0,0] 1uy "First slice should be at z=0."
    Expect.equal stacked.[1,1,1] 8uy "Second slice should be at z=1."

  testCase "stack rejects empty and mismatched image lists" <| fun _ ->
    let twoByTwo = Image<uint8>.ofArray2D (array2D [ [1uy; 2uy]; [3uy; 4uy] ])
    let twoByThree = Image<uint8>.ofArray2D (array2D [ [1uy; 2uy; 3uy]; [4uy; 5uy; 6uy] ])

    Expect.throws (fun () -> ImageFunctions.stack ([]: Image<uint8> list) |> ignore) "Empty stacks should fail clearly."
    Expect.throws (fun () -> ImageFunctions.stack [twoByTwo; twoByThree] |> ignore) "2D slices must have identical sizes."

  testCase "convolve rejects kernels larger than the image" <| fun _ ->
    let img = Image<float>.ofArray2D (array2D [ [1.0; 2.0]; [3.0; 4.0] ])
    let kernel =
      Image<float>.ofArray2D (
        array2D [
          [1.0; 1.0; 1.0]
          [1.0; 1.0; 1.0]
          [1.0; 1.0; 1.0] ])

    Expect.throws (fun () -> ImageFunctions.convolve (Some ImageFunctions.Valid) None img kernel |> ignore) "Kernels larger than the image should be rejected before calling SimpleITK."

  testCase "convolve Same preserves 3D image size in every dimension" <| fun _ ->
    let img = Image<float>.ofArray3D (Array3D.init 5 6 7 (fun x y z -> float (x + y + z)))
    let kernel = Image<float>.ofArray3D (Array3D.create 3 3 3 (1.0 / 27.0))

    let defaultSame = ImageFunctions.convolve None None img kernel
    let explicitSame = ImageFunctions.convolve (Some ImageFunctions.Same) None img kernel
    let convDefault = ImageFunctions.conv img kernel

    Expect.equal (defaultSame.GetSize()) [5u; 6u; 7u] "Default convolve should preserve x, y, and z."
    Expect.equal (explicitSame.GetSize()) [5u; 6u; 7u] "Explicit Same convolve should preserve x, y, and z."
    Expect.equal (convDefault.GetSize()) [5u; 6u; 7u] "conv should use Same-sized output."

    defaultSame.decRefCount()
    explicitSame.decRefCount()
    convDefault.decRefCount()
    kernel.decRefCount()
    img.decRefCount()

  testCase "convolve Valid trims 3D image size by kernel size minus one" <| fun _ ->
    let img = Image<float>.ofArray3D (Array3D.init 5 6 7 (fun x y z -> float (x + y + z)))
    let kernel = Image<float>.ofArray3D (Array3D.create 3 4 5 (1.0 / 60.0))

    let valid = ImageFunctions.convolve (Some ImageFunctions.Valid) None img kernel

    Expect.equal (valid.GetSize()) [3u; 3u; 3u] "Valid convolve should reduce each dimension by kernelSize - 1."

    valid.decRefCount()
    kernel.decRefCount()
    img.decRefCount()

  testCase "finiteDiffFilter3D creates directional stencil kernels" <| fun _ ->
    let x = ImageFunctions.finiteDiffFilter3D 0u 1u
    let z = ImageFunctions.finiteDiffFilter3D 2u 2u

    Expect.equal (x.GetSize()) [3u; 1u; 1u] "X first derivative should be a 3x1x1 stencil."
    Expect.equal (z.GetSize()) [1u; 1u; 3u] "Z second derivative should be a 1x1x3 stencil."
    Expect.floatClose Accuracy.high x.[0,0,0] 0.5 "First derivative stencil should follow the existing positive-to-negative convention."
    Expect.floatClose Accuracy.high x.[2,0,0] -0.5 "First derivative stencil should end negative."
    Expect.floatClose Accuracy.high z.[0,0,1] -2.0 "Second derivative center should be -2."

  ]
