module Tests.entrypoint

open Expecto


[<Tests>]
let allTests = 
  testList "All Tests" [
  ]


[<EntryPoint>]
let main argv =
    printfn "Running tests!"   
    runTestsWithCLIArgs [] argv allTests
