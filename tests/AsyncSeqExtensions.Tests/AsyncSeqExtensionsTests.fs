module Tests.AsyncSeqExtensionsTests

open Expecto
open FSharp.Control
open AsyncSeqExtensions

let private toList seq =
    seq |> AsyncSeq.toListAsync |> Async.RunSynchronously

let private ofList values =
    values |> AsyncSeq.ofSeq

let private moveNext (enum: System.Collections.Generic.IAsyncEnumerator<'T>) =
    enum.MoveNextAsync().AsTask()
    |> Async.AwaitTask
    |> Async.RunSynchronously

let private dispose (enum: System.Collections.Generic.IAsyncEnumerator<'T>) =
    enum.DisposeAsync().AsTask()
    |> Async.AwaitTask
    |> Async.RunSynchronously

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

        testCase "zipConcurrent pulls one item from each side per emitted pair" <| fun _ ->
            let leftPulls = ref 0
            let rightPulls = ref 0
            let counted counter (values: 'T list) =
                asyncSeq {
                    for value in values do
                        counter := !counter + 1
                        yield value
                }

            let left = counted leftPulls [1; 2; 3]
            let right = counted rightPulls ["a"; "b"; "c"]
            let enum = (zipConcurrent left right).GetAsyncEnumerator()

            try
                Expect.isTrue (moveNext enum) "The first pair should be available."
                Expect.equal enum.Current (1, "a") "The first pair should combine the first values."
                Expect.equal !leftPulls 1 "The left side should only have supplied one item."
                Expect.equal !rightPulls 1 "The right side should only have supplied one item."
            finally
                dispose enum

        testCase "zipConcurrent disposes both input enumerators when consumer stops early" <| fun _ ->
            let mutable leftDisposed = false
            let mutable rightDisposed = false
            let source markDisposed =
                asyncSeq {
                    try
                        yield 1
                        yield 2
                    finally
                        markDisposed()
                }

            let enum =
                (zipConcurrent (source (fun () -> leftDisposed <- true)) (source (fun () -> rightDisposed <- true)))
                    .GetAsyncEnumerator()

            Expect.isTrue (moveNext enum) "The first pair should be available."
            dispose enum

            Expect.isTrue leftDisposed "Disposing the zipped stream should dispose the left input."
            Expect.isTrue rightDisposed "Disposing the zipped stream should dispose the right input."

        testCase "windowedWithPad creates padded sliding windows" <| fun _ ->
            let zeroMaker i _ = i
            let actual =
                ofList [10; 20; 30]
                |> windowedWithPad 3u 1u 1u 1u zeroMaker
                |> toList

            Expect.equal actual [[-1; 10; 20]; [10; 20; 30]; [20; 30; 3]; [30; 3]] "Padded windows should include pre/post padding and the final remainder window."

        testCase "windowedWithPad has bounded read-ahead for first window" <| fun _ ->
            let mutable pulls = 0
            let source =
                asyncSeq {
                    for value in [1..10] do
                        pulls <- pulls + 1
                        yield value
                }

            let enum =
                (source
                 |> windowedWithPad 3u 1u 0u 0u (fun _ x -> x))
                    .GetAsyncEnumerator()

            try
                Expect.isTrue (moveNext enum) "The first window should be available."
                Expect.equal enum.Current [1; 2; 3] "The first window should contain the first three values."
                Expect.isLessThanOrEqual pulls 4 "The first window should need only the window plus one look-ahead item."
            finally
                dispose enum

        testCase "windowedWithPad disposes input enumerator when consumer stops early" <| fun _ ->
            let mutable disposed = false
            let source =
                asyncSeq {
                    try
                        yield 1
                        yield 2
                        yield 3
                    finally
                        disposed <- true
                }

            let enum =
                (source
                 |> windowedWithPad 2u 1u 0u 0u (fun _ x -> x))
                    .GetAsyncEnumerator()

            Expect.isTrue (moveNext enum) "The first window should be available."
            dispose enum

            Expect.isTrue disposed "Disposing the windowed stream should dispose the input sequence."

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
