module StackProcessingCost

open System.IO
open SlimPipeline

type ImageStageShape =
    { InputImages: uint64
      OutputImages: uint64
      WorkImages: uint64
      RetainedImages: uint64 }

module ImageStageShape =
    let mapLike =
        { InputImages = 1UL
          OutputImages = 1UL
          WorkImages = 0UL
          RetainedImages = 0UL }

    let iterLike =
        { InputImages = 1UL
          OutputImages = 0UL
          WorkImages = 0UL
          RetainedImages = 0UL }

    let windowLike windowSize =
        { InputImages = uint64 windowSize
          OutputImages = uint64 windowSize
          WorkImages = 0UL
          RetainedImages = 0UL }

let inputVoxels input =
    input |> SingleOrPair.sum |> SingleOrPair.fst

let pixelTypeName<'T> =
    typeof<'T>.Name

let imageBytes<'T> (nVoxels: uint64) =
    Image.ImageFacts.memoryBytesForType<'T> nVoxels 1u

let sliceBytes<'T> width height =
    Image.ImageFacts.sliceBytesForType<'T> width height

let imageStageMemory<'T> evaluation shape =
    { Evaluation = evaluation
      Estimate =
        fun input ->
            let bytes = inputVoxels input |> imageBytes<'T>
            StageMemoryEstimate.create
                (bytes * shape.InputImages)
                (bytes * shape.OutputImages)
                (bytes * shape.WorkImages)
                (bytes * shape.RetainedImages) }

let imageMapCost<'T> calibrationKey costUnits =
    StageCostModel.create
        (imageStageMemory<'T> Map ImageStageShape.mapLike)
        (StageTimeCostModel.native Map (Some calibrationKey) costUnits)

let imageIoCost<'T> kind evaluation calibrationKey bytes ops : StageTimeCostModel =
    match kind with
    | "read" -> StageTimeCostModel.ioRead evaluation (Some calibrationKey) bytes ops
    | "write" -> StageTimeCostModel.ioWrite evaluation (Some calibrationKey) bytes ops
    | _ -> StageTimeCostModel.zero evaluation

let withCostModel costModel stage =
    { stage with
        CostModel = costModel
        MemoryModel = costModel.Memory
        MemoryNeed = StageCostModel.memoryNeed costModel }

let tryLoadTimeCalibration path =
    if File.Exists path then
        StageTimeCalibration.loadJson path
    else
        false

let tryLoadFirstTimeCalibration paths =
    paths |> List.tryFind tryLoadTimeCalibration |> Option.isSome
