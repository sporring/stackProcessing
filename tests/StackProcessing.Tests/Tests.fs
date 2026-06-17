module Tests.entrypoint

open Expecto

[<Tests>]
let allTests =
    testSequenced
    <| testList "All Tests" [
        Tests.ChunkTests.chunkSuite
    ]

[<EntryPoint>]
let main argv =
    runTestsWithCLIArgs [] argv allTests
