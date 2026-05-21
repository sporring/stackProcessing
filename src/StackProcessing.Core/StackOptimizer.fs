module StackOptimizer

open SlimPipeline

let private costScore estimate =
    estimate.CpuCostUnits
    + estimate.NativeCostUnits
    + float estimate.IoReadBytes
    + float estimate.IoWriteBytes
    + float estimate.IoReadOps
    + float estimate.IoWriteOps

let candidateFromStage name kind semanticsPreserving inputShape (stage: Stage<'S,'T>) explanation : StackProcessingCost.OptimizationCandidate<Stage<'S,'T>> =
    let cost = StageCostModel.estimate stage.CostModel inputShape

    let windowSize =
        match stage.CostModel.Memory.Evaluation with
        | Windowed(windowSize, _, _) -> Some windowSize
        | _ -> None

    { Name = name
      Payload = stage
      Kind = kind
      SemanticsPreserving = semanticsPreserving
      EstimatedMemoryBytes = cost.Memory.Peak
      EstimatedTimeMilliseconds = StageTimeCalibration.estimateMilliseconds cost.Time
      EstimatedCostScore = costScore cost.Time
      WindowSize = windowSize
      Explanation = explanation }

let windowSizeCandidate name inputShape stage explanation =
    candidateFromStage name StackProcessingCost.WindowSize true inputShape stage explanation

let materializationCandidate name inputShape stage explanation =
    candidateFromStage name StackProcessingCost.MaterializationPoint true inputShape stage explanation

let chooseStageWithPolicy policy availableMemory inputShape candidates =
    candidates
    |> List.map (fun (name, kind, semanticsPreserving, stage, explanation) ->
        candidateFromStage name kind semanticsPreserving inputShape stage explanation)
    |> StackProcessingCost.Optimizer.chooseWithPolicy policy availableMemory

let chooseStage availableMemory inputShape candidates =
    chooseStageWithPolicy StackProcessingCost.Optimizer.defaultPolicy availableMemory inputShape candidates

let chooseStageOrThrow availableMemory inputShape candidates =
    let result = chooseStage availableMemory inputShape candidates
    match result.Selected with
    | Some candidate -> candidate.Payload, result
    | None -> failwith $"No optimization candidate fits within {availableMemory} B."
