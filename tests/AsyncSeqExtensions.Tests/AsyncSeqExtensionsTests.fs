module Tests.AsyncSeqExtensionsTests

open Expecto
open FSharp.Control
open AsyncSeqExtensions

let private toList seq =
    seq |> AsyncSeq.toListAsync |> Async.RunSynchronously

let private ofList values =
    values |> AsyncSeq.ofSeq

[<Tests>]
let asyncSeqExtensionsSuite =
    testList "AsyncSeqExtensions" [
        testCase "ofAsync yields one value" <| fun _ ->
            let actual = async { return 42 } |> ofAsync |> toList
            Expect.equal actual [42] "ofAsync should yield exactly the async result."

        testCase "zipConcurrent zips until the shorter sequence ends" <| fun _ ->
            let left = ofList [1; 2; 3]
            let right = ofList ["a"; "b"]
            let actual = zipConcurrent left right |> toList
            Expect.equal actual [(1, "a"); (2, "b")] "zipConcurrent should stop at the shorter stream."

        testCase "windowedWithPad creates padded sliding windows" <| fun _ ->
            let zeroMaker i _ = i
            let actual =
                ofList [10; 20; 30]
                |> windowedWithPad 3u 1u 1u 1u zeroMaker
                |> toList

            Expect.equal actual [[-1; 10; 20]; [10; 20; 30]; [20; 30; 3]; [30; 3]] "Padded windows should include pre/post padding and the final remainder window."

        testCase "windowedWithPad rejects zero window size" <| fun _ ->
            Expect.throws (fun () ->
                ofList [1]
                |> windowedWithPad 0u 1u 0u 0u (fun _ x -> x)
                |> toList
                |> ignore) "windowSize must be positive."

        testCase "windowedWithPad rejects zero stride" <| fun _ ->
            Expect.throws (fun () ->
                ofList [1]
                |> windowedWithPad 1u 0u 0u 0u (fun _ x -> x)
                |> toList
                |> ignore) "stride must be positive."
    ]
