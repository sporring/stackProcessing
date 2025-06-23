// Tests.fs
module Tests

open Expecto
open ImageClass // adjust to match actual namespace

[<Tests>]
let imageClassTests =
  testList "ImageClass Tests" [
    testCase "Dummy test" <| fun _ ->
      Expect.equal 1 1 "Sanity check"
  ]

[<EntryPoint>]
let main args =
  runTestsWithArgs defaultConfig args imageClassTests
