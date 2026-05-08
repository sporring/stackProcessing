module Tests.vector

open Expecto
open Image   // open the namespace that defines Image<'T>

[<Tests>]
let vectorSuite =
  testList "Vector image support" [

    testCase "ofArray3DVector roundtrip" <| fun _ ->
      let arr = Array3D.init 2 2 3 (fun x y k -> x + 10*y + 100*k)
      let img = Image<int list>.ofArray3DVector arr
      Expect.equal img.[1,0] [1; 101; 201] "Vector value at (1,0) mismatch"
      Expect.equal (img.GetNumberOfComponentsPerPixel()) 3u "The vector component count should come from the array's third dimension."
      let arr2 = Image<int>.toArray3DVector img
      Expect.equal arr2 arr "Roundtrip array mismatch"

    testCase "single component vector images still use a legal SimpleITK vector component count" <| fun _ ->
      let arr = Array3D.init 2 2 1 (fun x y k -> x + 10*y + 100*k)
      let img = Image<int list>.ofArray3DVector arr
      Expect.equal (img.GetNumberOfComponentsPerPixel()) 2u "SimpleITK vector images require at least two components internally."
      Expect.equal img.[1,1] [11; 0] "The requested component should be preserved and the spare component should be zero."
  ]
