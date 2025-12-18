module Tests.comparable

open Expecto
open Image

[<Tests>]
let getHashCodeTests =
  testList "Image.GetHashCode and IComparable" [

    // Hash codes can have collisions, so we're only testing for equality
    testCase "Hash codes match for equal 2D images" <| fun _ ->
      let a = Image<uint8>.ofArray2D (array2D [ [1uy; 2uy]; [3uy; 4uy] ])
      let b = Image<uint8>.ofArray2D (array2D [ [1uy; 2uy]; [3uy; 4uy] ])
      Expect.equal (a.GetHashCode()) (b.GetHashCode()) "Hash codes should be equal"

    testCase "Hash codes match for equal 3D images" <| fun _ ->
      let a = Image<uint8>.ofArray3D (Array3D.init 2 2 1 (fun x y z -> byte (x + y + z)))
      let b = Image<uint8>.ofArray3D (Array3D.init 2 2 1 (fun x y z -> byte (x + y + z)))
      Expect.equal (a.GetHashCode()) (b.GetHashCode()) "3D image hash codes should match"

    testCase "Hash codes differ for different 4D images" <| fun _ ->
      let arr1 = Array4D.init 1 1 1 2 (fun _ _ _ i -> byte (i + 1))
      let arr2 = Array4D.init 1 1 1 2 (fun _ _ _ i -> byte (i + 1))
      let a = Image<uint8>.ofArray4D arr1
      let b = Image<uint8>.ofArray4D arr2
      Expect.equal (a.GetHashCode()) (b.GetHashCode()) "4D image hash codes should differ"

    testCase "IComparable compares identical images as equal" <| fun _ ->
      let a = Image<int8>.ofArray2D (array2D [ [1y; 2y] ])
      let b = Image<int8>.ofArray2D (array2D [ [1y; 2y] ])
      let cmp = compare a b
      Expect.equal cmp 0 "Comparable equal images should return 0"

    testCase "IComparable orders different images correctly" <| fun _ ->
      let a = Image<int8>.ofArray2D (array2D [ [1y; 2y] ])
      let b = Image<int8>.ofArray2D (array2D [ [2y; 3y] ])
      let cmp = compare a b
      Expect.isLessThan cmp 0 "Smaller image should compare less than larger"
  ]