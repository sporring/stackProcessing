module Tests.SlimPipelineTests

open Expecto
open SlimPipeline

let private source availableMemory = Plan.source availableMemory

let private initInts name count =
    Stage.init name count id (ProfileTransition.create Unit Streaming) (fun _ -> 0UL) (fun _ -> uint64 count)

let private mapInts name f =
    Stage.map<int, int> name (fun _ x -> f x) (fun _ -> 0UL) id

let private apply stage plan =
    Plan.composePlan ">=>" plan stage

let private planFrom stage length =
    Plan.create (Some stage) 1024UL 0UL 0UL length false

[<Tests>]
let singleOrPairSuite =
    testList "SingleOrPair" [
        testCase "map preserves shape" <| fun _ ->
            Expect.equal (SingleOrPair.map ((+) 1UL) (Single 2UL)) (Single 3UL) "Single value should be mapped."
            Expect.equal (SingleOrPair.map ((+) 1UL) (Pair(2UL, 3UL))) (Pair(3UL, 4UL)) "Pair values should both be mapped."

        testCase "add combines singles and pairs" <| fun _ ->
            Expect.equal (SingleOrPair.add (Single 2UL) (Single 3UL)) (Single 5UL) "Singles should add to Single."
            Expect.equal (SingleOrPair.add (Single 2UL) (Pair(3UL, 4UL))) (Pair(5UL, 6UL)) "Single should broadcast over Pair."
            Expect.equal (SingleOrPair.add (Pair(1UL, 2UL)) (Pair(3UL, 4UL))) (Pair(4UL, 6UL)) "Pairs should add pointwise."
    ]

[<Tests>]
let profileSuite =
    testList "Profile" [
        testCase "estimateUsage reflects profile shape" <| fun _ ->
            Expect.equal (Profile.estimateUsage Unit 10UL) 0UL "Unit should not consume element memory."
            Expect.equal (Profile.estimateUsage Constant 10UL) 10UL "Constant should consume one element."
            Expect.equal (Profile.estimateUsage Streaming 10UL) 10UL "Streaming should consume one element."
            Expect.equal (Profile.estimateUsage (Window(3u, 1u, 0u, 0u, 1u)) 10UL) 30UL "Window should scale by window size."

        testCase "combine chooses the more demanding profile" <| fun _ ->
            Expect.equal (Profile.combine Unit Streaming) Streaming "Streaming should dominate Unit."
            Expect.equal (Profile.combine Constant Streaming) Streaming "Streaming should dominate Constant."
            Expect.equal (Profile.combine (Window(3u, 2u, 1u, 0u, 1u)) Streaming) (Window(3u, 2u, 1u, 0u, 1u)) "Window should dominate Streaming."
    ]

[<Tests>]
let resourceOpsSuite =
    testList "ResourceOps" [
        testCase "ResourceOps.none is inert" <| fun _ ->
            ResourceOps.retainAndReturn ResourceOps.none 1 |> Expect.equal <| 1 <| "retainAndReturn should return the value."
            ResourceOps.release ResourceOps.none 1
            Expect.equal (ResourceOps.memoryOf ResourceOps.none 1) None "No memory estimate should be available."

        testCase "ResourceOps forwards retain release and memory callbacks" <| fun _ ->
            let mutable retained = 0
            let mutable released = 0
            let ops =
                { Retain = fun _ -> retained <- retained + 1
                  Release = fun _ -> released <- released + 1
                  MemoryOf = fun x -> Some(uint64 x) }

            let value = ResourceOps.retainAndReturn ops 7
            ResourceOps.release ops value
            Expect.equal retained 1 "Retain should be called once."
            Expect.equal released 1 "Release should be called once."
            Expect.equal (ResourceOps.memoryOf ops value) (Some 7UL) "Memory callback should be forwarded."
    ]

[<Tests>]
let planSuite =
    testList "Plan and Stage" [
        testCase "source init map drainList executes a simple stream" <| fun _ ->
            let actual =
                source 1024UL
                |> fun _ -> planFrom (initInts "numbers" 4u) 4UL
                |> apply (mapInts "double" ((*) 2))
                |> Plan.drainList "numbers"

            Expect.equal actual [0; 2; 4; 6] "Pipeline should initialize and map stream values."

        testCase "drainSingle accepts exactly one value" <| fun _ ->
            let actual =
                source 1024UL
                |> fun _ -> planFrom (initInts "one" 1u) 1UL
                |> Plan.drainSingle "one"

            Expect.equal actual 0 "Single-item pipeline should drain to the one value."

        testCase "drainSingle rejects multiple values" <| fun _ ->
            Expect.throws (fun () ->
                source 1024UL
                |> fun _ -> planFrom (initInts "two" 2u) 2UL
                |> Plan.drainSingle "two"
                |> ignore) "drainSingle should reject multi-item streams."

        testCase "plan graph is built while composing DSL" <| fun _ ->
            let plan =
                source 1024UL
                |> fun _ -> planFrom (initInts "numbers" 3u) 3UL
                |> apply (mapInts "inc" ((+) 1))

            let graph = Plan.graph plan
            let names = graph.Nodes |> List.map _.Name
            Expect.containsAll names ["numbers"; "inc"] "Graph should include composed stage nodes."
            Expect.isGreaterThanOrEqual graph.Edges.Length 1 "Graph should connect composed stage nodes."
    ]
