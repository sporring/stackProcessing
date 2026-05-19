module StackProcessingCost

open System
open System.Globalization
open System.IO
open System.Text.Json
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

type CostCoefficient =
    { Measurement: string
      FeatureKey: string
      Coefficient: float
      SupportCount: int
      RowCount: int
      Rmse: float
      R2: float
      Solver: string }

type CostModel =
    { SchemaVersion: int
      Name: string
      CreatedUtc: DateTimeOffset
      Assumptions: string array
      Coefficients: CostCoefficient array }

module CostModel =
    let private jsonOptions =
        JsonSerializerOptions(WriteIndented = true)

    let defaultModel =
        { SchemaVersion = 1
          Name = "StackProcessing default cost model"
          CreatedUtc = DateTimeOffset.UnixEpoch
          Assumptions =
            [| "No calibrated coefficients are available."
               "Runtime falls back to structural memory estimates and uncalibrated time scores." |]
          Coefficients = Array.empty }

    let loadOrDefault path =
        if String.IsNullOrWhiteSpace path || not (File.Exists path) then
            defaultModel
        else
            let json = File.ReadAllText path
            JsonSerializer.Deserialize<CostModel>(json, jsonOptions)
            |> Option.ofObj
            |> Option.defaultValue defaultModel

    let tryFind measurement featureKey model =
        model.Coefficients
        |> Array.tryFind (fun coefficient ->
            String.Equals(coefficient.Measurement, measurement, StringComparison.OrdinalIgnoreCase)
            && String.Equals(coefficient.FeatureKey, featureKey, StringComparison.Ordinal))

    let estimate measurement featureValues model =
        featureValues
        |> Seq.sumBy (fun (featureKey, featureValue) ->
            match tryFind measurement featureKey model with
            | Some coefficient -> coefficient.Coefficient * featureValue
            | None -> 0.0)

    let save (path: string) model =
        let directory = Path.GetDirectoryName path
        if not (String.IsNullOrWhiteSpace directory) then
            Directory.CreateDirectory directory |> ignore

        File.WriteAllText(path, JsonSerializer.Serialize(model, jsonOptions))

module Fitting =
    type EvidenceRow =
        { RowId: string
          Measurement: string
          Value: float
          SourcePath: string
          FeatureKey: string
          FeatureValue: float }

    let private invariant (value: float) =
        Convert.ToString(value, CultureInfo.InvariantCulture)

    let private csvEscape (value: string) =
        if isNull value then
            ""
        elif value.Contains(",") || value.Contains("\"") || value.Contains("\n") || value.Contains("\r") then
            "\"" + value.Replace("\"", "\"\"") + "\""
        else
            value

    let writeEvidenceCsv (path: string) (rows: EvidenceRow seq) =
        let directory = Path.GetDirectoryName path
        if not (String.IsNullOrWhiteSpace directory) then
            Directory.CreateDirectory directory |> ignore

        use writer = new StreamWriter(path)
        writer.WriteLine("rowId,measurement,value,sourcePath,featureKey,featureValue")

        for row in rows do
            [ row.RowId
              row.Measurement
              invariant row.Value
              row.SourcePath
              row.FeatureKey
              invariant row.FeatureValue ]
            |> List.map csvEscape
            |> String.concat ","
            |> writer.WriteLine

    let writeCoefficientModel path name assumptions (coefficients: CostCoefficient seq) =
        { SchemaVersion = 1
          Name = name
          CreatedUtc = DateTimeOffset.UtcNow
          Assumptions = assumptions |> Seq.toArray
          Coefficients = coefficients |> Seq.toArray }
        |> CostModel.save path

type CostDiscrepancyPolicy =
    { Enabled: bool
      TimeRatio: float
      MemoryRatio: float
      MinimumTimeMs: float
      MinimumMemoryBytes: uint64 }

module CostDiscrepancyPolicy =
    let disabled =
        { Enabled = false
          TimeRatio = 4.0
          MemoryRatio = 4.0
          MinimumTimeMs = 100.0
          MinimumMemoryBytes = 64UL * 1024UL * 1024UL }

    let defaultEnabled =
        { disabled with Enabled = true }
