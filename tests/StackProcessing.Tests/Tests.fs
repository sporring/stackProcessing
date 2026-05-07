module Tests.entrypoint

open Expecto

[<Tests>]
let allTests =
    testSequenced
    <| testList "All Tests" [
        Tests.StackProcessingSupportTests.stackProcessingSupportSuite
        Tests.StackProcessingCorrectnessTests.stackProcessingCorrectnessSuite
    ]

[<EntryPoint>]
let main argv =
    runTestsWithCLIArgs [] argv allTests
