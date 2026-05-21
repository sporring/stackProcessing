module Tests.SlimPipelineTests

open System
open System.IO
open Expecto
open FSharp.Control
open SlimPipeline
open StackProcessingCost

let private source availableMemory = Plan.source availableMemory

let private initInts name count =
    Stage.init name count id (ProfileTransition.create Unit Streaming) (fun _ -> 0UL) id

let private mapInts name f =
    Stage.map<int, int> name (fun _ x -> f x) (fun _ -> 0UL) id

let private costedMap name memoryPeak costUnits calibrationKey =
    let stage = Stage.map<int, int> name (fun _ x -> x) (fun _ -> memoryPeak) id
    let memoryModel = StageMemoryModel.fromSinglePeak Map (fun _ -> memoryPeak)
    let timeCostModel = StageTimeCostModel.cpu Map calibrationKey (fun _ -> costUnits)
    let costModel = StageCostModel.create memoryModel timeCostModel
    { stage with
        MemoryNeed = StageCostModel.memoryNeed costModel
        MemoryModel = memoryModel
        CostModel = costModel }

let private costedWindowLikeMap name windowSize memoryPeak costUnits calibrationKey =
    let stage = Stage.map<int, int> name (fun _ x -> x) (fun _ -> memoryPeak) id
    let memoryModel = StageMemoryModel.windowLike windowSize windowSize 0u
    let timeCostModel = StageTimeCostModel.cpu (Windowed(windowSize, windowSize, 0u)) calibrationKey (fun _ -> costUnits)
    let costModel = StageCostModel.create memoryModel timeCostModel
    { stage with
        MemoryNeed = StageCostModel.memoryNeed costModel
        MemoryModel = memoryModel
        CostModel = costModel }

let private costScore estimate =
    estimate.CpuCostUnits
    + estimate.NativeCostUnits
    + float estimate.IoReadBytes
    + float estimate.IoWriteBytes
    + float estimate.IoReadOps
    + float estimate.IoWriteOps

let private optimizerCandidate kind windowSize name stage =
    let cost = StageCostModel.estimate stage.CostModel (Single 1UL)
    { Name = name
      Payload = stage
      Kind = kind
      SemanticsPreserving = true
      EstimatedMemoryBytes = cost.Memory.Peak
      EstimatedTimeMilliseconds = StageTimeCalibration.estimateMilliseconds cost.Time
      EstimatedCostScore = costScore cost.Time
      WindowSize = windowSize
      Explanation = "" }

let private candidate name memoryPeak costUnits =
    let stage = costedMap name memoryPeak costUnits None
    optimizerCandidate OtherExecutionChoice None name stage

let private windowCandidate name windowSize memoryPeak costUnits =
    let stage = costedWindowLikeMap name windowSize memoryPeak costUnits None
    optimizerCandidate WindowSize (Some windowSize) name stage

let private apply stage plan =
    Plan.composePlan ">=>" plan stage

let private planFrom stage length =
    Plan.create (Some stage) 1024UL 0UL 0UL length false

let private asyncSeqToList seq =
    seq |> AsyncSeq.toListAsync |> Async.RunSynchronously

let private captureStdout f =
    let original = Console.Out
    use writer = new StringWriter()
    Console.SetOut writer
    try
        f ()
        writer.ToString()
    finally
        Console.SetOut original

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
let sliceDomainSuite =
    testList "SliceDomain" [
        testCase "preserve keeps stream length" <| fun _ ->
            Expect.equal (SliceCardinality.length 10UL SliceCardinality.preserve) (Some 10UL) "Preserve should keep the input length."

        testCase "trim shortens both ends" <| fun _ ->
            let cardinality = Domain(SliceDomain.trim 2u 3u)
            Expect.equal (SliceCardinality.length 10UL cardinality) (Some 5UL) "Trim should subtract before and after from the stream length."

        testCase "expand grows both ends" <| fun _ ->
            let cardinality = Domain(SliceDomain.expand 2u 3u)
            Expect.equal (SliceCardinality.length 10UL cardinality) (Some 15UL) "Expand should add before and after to the stream length."

        testCase "skip then take composes as a bounded domain" <| fun _ ->
            let composed = SliceDomain.compose (SliceDomain.skip 2u) (SliceDomain.take 4UL)
            Expect.equal composed { StartOffset = 2L; End = CountFromStart 4UL } "Skip followed by take should remember the shifted start and fixed count."
            Expect.equal (SliceDomain.length 10UL composed) (Some 4UL) "Composed skip/take should expose the taken count."

        testCase "trim then skip preserves logical coordinate offset" <| fun _ ->
            let trim = SliceDomain.trim 2u 2u
            let composed = SliceDomain.compose trim (SliceDomain.skip 1u)
            Expect.equal composed { StartOffset = 3L; End = RelativeToInputEnd -2L } "Trim then skip should advance the start without moving the end."
            Expect.equal (SliceDomain.length 10UL composed) (Some 5UL) "Composed trim/skip should calculate length from the logical domain."

        testCase "expand then trim captures same-style even kernel convention" <| fun _ ->
            let sameEven =
                SliceDomain.compose
                    (SliceDomain.expand 4u 4u)
                    (SliceDomain.trim 4u 3u)

            Expect.equal sameEven { StartOffset = 0L; End = RelativeToInputEnd 1L } "Even-kernel Same with symmetric padding should expose the current +1 end convention."
            Expect.equal (SliceDomain.length 10UL sameEven) (Some 11UL) "Even-kernel Same should grow by one under the current padding convention."
    ]

[<Tests>]
let costModelSuite =
    testList "Stage cost model" [
        testCase "memory model mapLike accounts for input output and work buffers" <| fun _ ->
            let model = StageMemoryModel.mapLike ((*) 2UL) ((*) 3UL)
            let estimate = model.Estimate (Single 10UL)

            Expect.equal estimate.InputLive 10UL "Map stages keep the input live while computing."
            Expect.equal estimate.OutputLive 20UL "Output bytes should be estimated from the supplied function."
            Expect.equal estimate.WorkLive 30UL "Temporary work bytes should be estimated from the supplied function."
            Expect.equal estimate.Peak 60UL "Peak should include input, output, work, and retained bytes."
            Expect.equal (StageMemoryModel.memoryNeed model (Single 10UL)) (Single 60UL) "Legacy memory need should be derived from the peak."

        testCase "time cost estimates add cpu native and io components" <| fun _ ->
            let left = StageTimeCostEstimate.create 1.0 2.0 3UL 4UL 5UL 6UL (Some "left")
            let right = StageTimeCostEstimate.create 10.0 20.0 30UL 40UL 50UL 60UL None
            let combined = StageTimeCostEstimate.add left right

            Expect.equal combined.CpuCostUnits 11.0 "CPU units should add."
            Expect.equal combined.NativeCostUnits 22.0 "Native units should add."
            Expect.equal combined.IoReadBytes 33UL "Read bytes should add."
            Expect.equal combined.IoWriteBytes 44UL "Write bytes should add."
            Expect.equal combined.IoReadOps 55UL "Read ops should add."
            Expect.equal combined.IoWriteOps 66UL "Write ops should add."
            Expect.equal combined.CalibrationKey None "Combined time costs from multiple stages should not pretend to have a single calibration key."

        testCase "calibration json is optional and supports camelCase coefficients" <| fun _ ->
            StageTimeCalibration.clear()
            let path = Path.Combine(Path.GetTempPath(), $"slimpipeline-cost-{System.Guid.NewGuid():N}.json")
            File.WriteAllText(path, """{"calibrations":{"stage":{"cpuMillisecondsPerUnit":2.0,"ioReadMillisecondsPerByte":0.5,"ioWriteMillisecondsPerOp":3.0}}}""")

            try
                Expect.isTrue (StageTimeCalibration.loadJson path) "Calibration JSON should load when the file exists and has a calibrations object."
                let estimate = StageTimeCostEstimate.create 4.0 0.0 10UL 0UL 0UL 2UL (Some "stage")
                Expect.equal (StageTimeCalibration.estimateMilliseconds estimate) (Some 19.0) "Registered coefficients should estimate elapsed milliseconds."
                Expect.isFalse (StageTimeCalibration.loadJson (path + ".missing")) "Missing calibration files should be ignored cleanly."
            finally
                StageTimeCalibration.clear()
                if File.Exists path then File.Delete path

        testCase "source peek stores source metadata for optimizers" <| fun _ ->
            let peek = SourcePeek.create "read" 8UL (Some 12UL) (Map.ofList ["width", "64"; "height", "32"])
            let plan = Plan.source 1024UL |> Plan.withSourcePeek peek

            Expect.equal plan.sourcePeek (Some peek) "Plan should retain source metadata for later optimization."
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
let debugLevelSuite =
    testList "DebugLevel" [
        testCase "level 3 includes level 1 and 2 behaviour plus RSS measurements" <| fun _ ->
            let debugPlan = Plan.debug 3u true 1024UL
            Expect.equal debugPlan.debugLevel 3u "The plan should store the requested debug level."

            let plan =
                captureStdout (fun () ->
                    debugPlan |> Plan.sink)

            Expect.stringContains plan "Optimization accepted" "Debug level 3 should keep level 2 optimization summaries."
            Expect.stringContains plan "Measured peak delta" "Debug level 3 should still include measured RSS output."

        testCase "debug stores optimizer control independently of debug output" <| fun _ ->
            let plan = Plan.debug 1u false 1024UL

            Expect.isTrue plan.debug "Debug output should still be enabled."
            Expect.equal plan.debugLevel 1u "The requested debug level should be preserved."
            Expect.isFalse plan.optimize "Optimizer control should be stored independently."
    ]

[<Tests>]
let asyncStreamSemanticsSuite =
    testList "Async stream semantics" [
        testCase "map2Sync shares upstream stream without duplicating pulls" <| fun _ ->
            let pulls = ref 0
            let source =
                asyncSeq {
                    for value in [1; 2; 3] do
                        pulls := !pulls + 1
                        yield value
                }

            let left = Stage.map<int, int> "left" (fun _ x -> x + 10) (fun _ -> 0UL) id
            let right = Stage.map<int, int> "right" (fun _ x -> x * 2) (fun _ -> 0UL) id
            let combined = Stage.map2Sync "combine" false (fun a b -> a, b) left right (fun _ -> Single 0UL) id

            let actual = (combined.Build()).Apply false source |> asyncSeqToList

            Expect.equal actual [(11, 2); (12, 4); (13, 6)] "map2Sync should combine corresponding branch results."
            Expect.equal !pulls 3 "The shared upstream should be pulled once per input element, not once per branch."

        testCase "window then flatten preserves singleton-window stream order" <| fun _ ->
            let source = [1; 2; 3; 4] |> AsyncSeq.ofSeq
            let window = Stage.window "window" 1u 0u (fun _ x -> x) 1u
            let flatten = Stage.flattenWindow "flatten"
            let stage = Stage.compose window flatten

            let actual = (stage.Build()).Apply false source |> asyncSeqToList

            Expect.equal actual [1; 2; 3; 4] "Singleton windows should flatten back to the original stream."

        testCase "windowed stage with padding exposes expected padded windows" <| fun _ ->
            let source = [10; 20; 30] |> AsyncSeq.ofSeq
            let window = Stage.window "window" 3u 1u (fun i _ -> i) 1u

            let actual =
                (window.Build()).Apply false source
                |> AsyncSeq.map _.Items
                |> asyncSeqToList

            Expect.equal actual [[-1; 10; 20]; [10; 20; 30]; [20; 30; 3]; [30; 3]] "Pipe.window should preserve AsyncSeqExtensions padding semantics."
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

[<Tests>]
let optimizerSuite =
    testList "Optimizer" [
        testCase "chooses lowest cost candidate within memory limit" <| fun _ ->
            let result =
                [ candidate "fast-too-large" 200UL 1.0
                  candidate "balanced" 80UL 2.0
                  candidate "small-slow" 10UL 10.0 ]
                |> Optimizer.choose 100UL

            Expect.equal (result.Selected |> Option.map _.Name) (Some "balanced") "Optimizer should reject the oversized candidate and choose the lowest-cost accepted one."
            Expect.equal (result.Decisions |> List.filter _.Accepted |> List.map _.CandidateName) ["balanced"; "small-slow"] "Only candidates under the memory ceiling should be accepted."

        testCase "returns no selection when all candidates exceed memory limit" <| fun _ ->
            let result =
                [ candidate "large" 200UL 1.0
                  candidate "larger" 300UL 0.5 ]
                |> Optimizer.choose 100UL

            Expect.isNone result.Selected "No candidate should be selected if none fit the memory ceiling."
            Expect.isTrue (result.Decisions |> List.forall (fun decision -> not decision.Accepted)) "Every decision should be rejected."

        testCase "uses calibrated milliseconds before raw cost score when available" <| fun _ ->
            StageTimeCalibration.clear()
            StageTimeCalibration.register "slow-calibrated" { StageTimeCoefficients.zero with CpuMillisecondsPerUnit = 10.0 }
            StageTimeCalibration.register "fast-calibrated" { StageTimeCoefficients.zero with CpuMillisecondsPerUnit = 0.1 }

            let candidates =
                [ optimizerCandidate OtherExecutionChoice None "slow-calibrated" (costedMap "slow-calibrated" 10UL 1.0 (Some "slow-calibrated"))
                  optimizerCandidate OtherExecutionChoice None "fast-calibrated" (costedMap "fast-calibrated" 10UL 10.0 (Some "fast-calibrated")) ]

            let result = Optimizer.choose 100UL candidates
            StageTimeCalibration.clear()

            Expect.equal (result.Selected |> Option.map _.Name) (Some "fast-calibrated") "Calibrated elapsed time should beat raw cost units."

        testCase "prefers larger accepted window when costs are nearly tied" <| fun _ ->
            let result =
                [ windowCandidate "small-window" 3u 10UL 100.0
                  windowCandidate "large-window" 9u 10UL 102.0 ]
                |> Optimizer.choose 100UL

            Expect.equal (result.Selected |> Option.map _.Name) (Some "large-window") "The optimizer should prefer larger windows when the estimated cost is within the tie tolerance."

        testCase "keeps clearly cheaper smaller window when larger window is much more expensive" <| fun _ ->
            let result =
                [ windowCandidate "small-window" 3u 10UL 100.0
                  windowCandidate "large-window" 9u 10UL 130.0 ]
                |> Optimizer.choose 100UL

            Expect.equal (result.Selected |> Option.map _.Name) (Some "small-window") "Window preference should not override a clearly cheaper candidate."
    ]
