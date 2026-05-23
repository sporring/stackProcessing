module StackProcessingCost

open System
open System.Globalization
open System.IO
open System.Text
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
    let t = typeof<'T>
    if t = typeof<float> then "Float64"
    elif t = typeof<float32> then "Float32"
    elif t = typeof<uint8> then "UInt8"
    elif t = typeof<int8> then "Int8"
    elif t = typeof<uint16> then "UInt16"
    elif t = typeof<int16> then "Int16"
    elif t = typeof<uint32> || t = typeof<uint> then "UInt32"
    elif t = typeof<int32> || t = typeof<int> then "Int32"
    elif t = typeof<uint64> then "UInt64"
    elif t = typeof<int64> then "Int64"
    else t.Name

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

type OptimizationKind =
    | WindowSize
    | MaterializationPoint
    | OtherExecutionChoice

type OptimizationCandidate<'T> =
    { Name: string
      Payload: 'T
      Kind: OptimizationKind
      SemanticsPreserving: bool
      EstimatedMemoryBytes: uint64
      EstimatedTimeMilliseconds: float option
      EstimatedCostScore: float
      WindowSize: uint option
      Explanation: string }

type OptimizationDecision =
    { CandidateName: string
      Kind: OptimizationKind
      Accepted: bool
      EstimatedMemoryBytes: uint64
      EstimatedTimeMilliseconds: float option
      EstimatedCostScore: float
      Reason: string }

type OptimizationResult<'T> =
    { Selected: OptimizationCandidate<'T> option
      Decisions: OptimizationDecision list }

type OptimizationPolicy =
    { RelativeTieTolerance: float
      PreferLargerWindowWhenTied: bool
      AllowMaterialization: bool }

module Optimizer =
    let defaultPolicy =
        { RelativeTieTolerance = 0.05
          PreferLargerWindowWhenTied = true
          AllowMaterialization = true }

    module Candidate =
        let executionChoice name kind payload estimatedMemoryBytes estimatedTimeMilliseconds estimatedCostScore windowSize explanation =
            { Name = name
              Payload = payload
              Kind = kind
              SemanticsPreserving = true
              EstimatedMemoryBytes = estimatedMemoryBytes
              EstimatedTimeMilliseconds = estimatedTimeMilliseconds
              EstimatedCostScore = estimatedCostScore
              WindowSize = windowSize
              Explanation = explanation }

        let windowSize name payload estimatedMemoryBytes estimatedTimeMilliseconds estimatedCostScore windowSize explanation =
            executionChoice name WindowSize payload estimatedMemoryBytes estimatedTimeMilliseconds estimatedCostScore (Some windowSize) explanation

        let materializationPoint name payload estimatedMemoryBytes estimatedTimeMilliseconds estimatedCostScore explanation =
            executionChoice name MaterializationPoint payload estimatedMemoryBytes estimatedTimeMilliseconds estimatedCostScore None explanation

    let private nearlyEqual tolerance left right =
        let scale = max 1.0 (max (abs left) (abs right))
        abs (left - right) <= scale * tolerance

    let private windowPreference (candidate: OptimizationCandidate<'T>) =
        candidate.WindowSize |> Option.defaultValue 0u

    let private compareWindowPreference leftCandidate rightCandidate =
        compare (windowPreference rightCandidate) (windowPreference leftCandidate)

    let private compareAccepted (_leftCandidate, leftDecision) (_rightCandidate, rightDecision) =
        match leftDecision.EstimatedTimeMilliseconds, rightDecision.EstimatedTimeMilliseconds with
        | Some leftMs, Some rightMs -> compare leftMs rightMs
        | Some _, None -> -1
        | None, Some _ -> 1
        | None, None -> compare leftDecision.EstimatedCostScore rightDecision.EstimatedCostScore

    let private compareAcceptedWithWindowPreference policy (leftCandidate, leftDecision) (rightCandidate, rightDecision) =
        let tiedByTime leftMs rightMs =
            policy.PreferLargerWindowWhenTied && nearlyEqual policy.RelativeTieTolerance leftMs rightMs

        let tiedByScore leftScore rightScore =
            policy.PreferLargerWindowWhenTied && nearlyEqual policy.RelativeTieTolerance leftScore rightScore

        match leftDecision.EstimatedTimeMilliseconds, rightDecision.EstimatedTimeMilliseconds with
        | Some leftMs, Some rightMs when tiedByTime leftMs rightMs ->
            let preference = compareWindowPreference leftCandidate rightCandidate
            if preference <> 0 then preference else compare leftMs rightMs
        | Some _, Some _ ->
            compareAccepted (leftCandidate, leftDecision) (rightCandidate, rightDecision)
        | Some _, None -> -1
        | None, Some _ -> 1
        | None, None when tiedByScore leftDecision.EstimatedCostScore rightDecision.EstimatedCostScore ->
            let preference = compareWindowPreference leftCandidate rightCandidate
            if preference <> 0 then preference else compare leftDecision.EstimatedCostScore rightDecision.EstimatedCostScore
        | None, None ->
            compare leftDecision.EstimatedCostScore rightDecision.EstimatedCostScore

    let chooseWithPolicy policy (availableMemory: uint64) (candidates: OptimizationCandidate<'T> list) : OptimizationResult<'T> =
        let evaluated =
            candidates
            |> List.map (fun candidate ->
                let accepted, reason =
                    if not candidate.SemanticsPreserving then
                        false, "Rejected: candidate can change the pipeline result."
                    elif candidate.Kind = MaterializationPoint && not policy.AllowMaterialization then
                        false, "Rejected: materialization candidates are disabled by policy."
                    elif candidate.EstimatedMemoryBytes > availableMemory then
                        false, $"Rejected: estimated memory {candidate.EstimatedMemoryBytes} B exceeds available memory {availableMemory} B."
                    else
                        match candidate.EstimatedTimeMilliseconds with
                        | Some ms -> true, $"Accepted: estimated {ms:g} ms within {availableMemory} B."
                        | None -> true, $"Accepted: estimated cost score {candidate.EstimatedCostScore:g} within {availableMemory} B."

                candidate,
                { CandidateName = candidate.Name
                  Kind = candidate.Kind
                  Accepted = accepted
                  EstimatedMemoryBytes = candidate.EstimatedMemoryBytes
                  EstimatedTimeMilliseconds = candidate.EstimatedTimeMilliseconds
                  EstimatedCostScore = candidate.EstimatedCostScore
                  Reason = reason })

        let selected =
            evaluated
            |> List.filter (fun (_, decision) -> decision.Accepted)
            |> List.sortWith (compareAcceptedWithWindowPreference policy)
            |> List.tryHead
            |> Option.map fst

        { Selected = selected
          Decisions = evaluated |> List.map snd }

    let choose availableMemory candidates =
        chooseWithPolicy defaultPolicy availableMemory candidates

    let chooseOrThrow availableMemory candidates =
        let result = choose availableMemory candidates
        match result.Selected with
        | Some candidate -> candidate.Payload, result
        | None -> failwith $"No optimization candidate fits within {availableMemory} B."

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
          FeatureValue: float
          Operator: string
          PixelType: string option
          Width: uint64 option
          Height: uint64 option
          Depth: uint64 option
          Voxels: uint64 option
          SlicePixels: uint64 option
          SliceBytes: uint64 option
          VolumeBytes: uint64 option
          WindowSize: float option
          Radius: float option
          KernelSize: float option
          Sigma: float option }

    let private invariant (value: float) =
        Convert.ToString(value, CultureInfo.InvariantCulture)

    let private medianFloat (values: float list) =
        match values |> List.sort with
        | [] -> 0.0
        | sorted ->
            let n = sorted.Length
            if n % 2 = 1 then
                sorted[n / 2]
            else
                0.5 * (sorted[n / 2 - 1] + sorted[n / 2])

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
        writer.WriteLine("rowId,measurement,value,sourcePath,featureKey,featureValue,operator,pixelType,width,height,depth,voxels,slicePixels,sliceBytes,volumeBytes,windowSize,radius,kernelSize,sigma")

        let uintOption value =
            value |> Option.map string |> Option.defaultValue ""

        let floatOption value =
            value |> Option.map invariant |> Option.defaultValue ""

        for row in rows do
            [ row.RowId
              row.Measurement
              invariant row.Value
              row.SourcePath
              row.FeatureKey
              invariant row.FeatureValue
              row.Operator
              row.PixelType |> Option.defaultValue ""
              uintOption row.Width
              uintOption row.Height
              uintOption row.Depth
              uintOption row.Voxels
              uintOption row.SlicePixels
              uintOption row.SliceBytes
              uintOption row.VolumeBytes
              floatOption row.WindowSize
              floatOption row.Radius
              floatOption row.KernelSize
              floatOption row.Sigma ]
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

    type OperatorTermCoefficient =
        { Measurement: string
          TermKey: string
          Operator: string
          PixelType: string
          Term: string
          Coefficient: float
          SupportCount: int
          RowCount: int
          ColumnCount: int
          Rmse: float
          R2: float
          Solver: string }

    type OperatorTermPrediction =
        { RowId: string
          Measurement: string
          Actual: float
          Predicted: float
          Residual: float }

    type OperatorTermFit =
        { Measurement: string
          Coefficients: OperatorTermCoefficient list
          Predictions: OperatorTermPrediction list }

    type OperatorCostModel =
        { SchemaVersion: int
          Name: string
          CreatedUtc: DateTimeOffset
          Assumptions: string array
          Coefficients: OperatorTermCoefficient array }

    module OperatorCostModel =
        let private jsonOptions =
            JsonSerializerOptions(WriteIndented = true)

        let empty =
            { SchemaVersion = 1
              Name = "StackProcessing empty operator cost model"
              CreatedUtc = DateTimeOffset.UnixEpoch
              Assumptions = [| "No fitted operator-term coefficients are available." |]
              Coefficients = Array.empty }

        let save (path: string) model =
            let directory = Path.GetDirectoryName path
            if not (String.IsNullOrWhiteSpace directory) then
                Directory.CreateDirectory directory |> ignore

            File.WriteAllText(path, JsonSerializer.Serialize(model, jsonOptions))

        let loadOrDefault (path: string) =
            if String.IsNullOrWhiteSpace path || not (File.Exists path) then
                empty
            else
                try
                    let json = File.ReadAllText path
                    JsonSerializer.Deserialize<OperatorCostModel>(json, jsonOptions)
                    |> Option.ofObj
                    |> Option.defaultValue empty
                with _ ->
                    empty

    type OperatorEstimateContext =
        { Operator: string
          PixelType: string option
          Voxels: uint64 option
          VolumeBytes: uint64 option
          WindowSize: float option
          Radius: float option
          KernelSize: float option
          Sigma: float option }

    module OperatorEstimateContext =
        let create operator pixelType voxels volumeBytes windowSize radius kernelSize sigma =
            { Operator = operator
              PixelType = pixelType
              Voxels = voxels
              VolumeBytes = volumeBytes
              WindowSize = windowSize
              Radius = radius
              KernelSize = kernelSize
              Sigma = sigma }

    module OperatorCostRuntime =
        let private calibratedMillisecondsKey = "__stackprocessing_operator_cost_model_milliseconds"
        let mutable private activeModel = OperatorCostModel.empty
        let mutable private activePath: string option = None
        let mutable private loadedDefault = false

        let private registerMillisecondsCalibration () =
            StageTimeCalibration.register
                calibratedMillisecondsKey
                { StageTimeCoefficients.zero with NativeMillisecondsPerUnit = 1.0 }

        let private userModelPath () =
            let home =
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)

            if String.IsNullOrWhiteSpace home then
                ""
            else
                Path.Combine(home, ".stackprocessing", "cost", "stackprocessing.operator-cost.json")

        let private tryFindRepositoryModelPath name =
            let rec loop (directory: DirectoryInfo) =
                let candidate = Path.Combine(directory.FullName, "models", name, "stackprocessing.operator-cost.json")

                if File.Exists candidate then
                    Some candidate
                elif isNull directory.Parent then
                    None
                else
                    loop directory.Parent

            loop (DirectoryInfo(Directory.GetCurrentDirectory()))

        let private repositoryModelPath name =
            tryFindRepositoryModelPath name
            |> Option.defaultValue (Path.Combine(Directory.GetCurrentDirectory(), "models", name, "stackprocessing.operator-cost.json"))

        let private tryFindRelativePathFromParents (path: string) =
            if Path.IsPathRooted path then
                None
            else
                let rec loop (directory: DirectoryInfo) =
                    let candidate = Path.Combine(directory.FullName, path)

                    if File.Exists candidate then
                        Some candidate
                    elif isNull directory.Parent then
                        None
                    else
                        loop directory.Parent

                loop (DirectoryInfo(Directory.GetCurrentDirectory()))

        let private resolveModelPath (path: string) =
            if String.IsNullOrWhiteSpace path then
                path
            elif File.Exists path then
                path
            else
                tryFindRelativePathFromParents path
                |> Option.defaultValue path

        let defaultSearchPaths () =
            [ Environment.GetEnvironmentVariable("STACKPROCESSING_COST_MODEL")
              userModelPath ()
              repositoryModelPath "fitted"
              repositoryModelPath "default" ]
            |> List.filter (String.IsNullOrWhiteSpace >> not)

        let load path =
            let resolvedPath = resolveModelPath path
            activeModel <- OperatorCostModel.loadOrDefault resolvedPath
            activePath <- if File.Exists resolvedPath then Some(Path.GetFullPath resolvedPath) else None
            loadedDefault <- true
            registerMillisecondsCalibration ()
            activeModel

        let loadFirstAvailable paths =
            match paths |> List.tryFind File.Exists with
            | Some path -> load path
            | None ->
                activeModel <- OperatorCostModel.empty
                activePath <- None
                loadedDefault <- true
                registerMillisecondsCalibration ()
                activeModel

        let ensureLoaded () =
            if not loadedDefault then
                loadFirstAvailable (defaultSearchPaths ()) |> ignore

        let currentPath () =
            ensureLoaded ()
            activePath

        let currentModel () =
            ensureLoaded ()
            activeModel

        let private termKey (operator: string) (pixelType: string) (term: string) =
            $"operator={operator}|pixelType={pixelType}|term={term}"

        let private coefficient (measurement: string) key =
            currentModel().Coefficients
            |> Array.tryFind (fun coefficient ->
                String.Equals(coefficient.Measurement, measurement, StringComparison.OrdinalIgnoreCase)
                && String.Equals(coefficient.TermKey, key, StringComparison.Ordinal))
            |> Option.map _.Coefficient
            |> Option.defaultValue 0.0

        let private hasCoefficient (measurement: string) key =
            currentModel().Coefficients
            |> Array.exists (fun coefficient ->
                String.Equals(coefficient.Measurement, measurement, StringComparison.OrdinalIgnoreCase)
                && String.Equals(coefficient.TermKey, key, StringComparison.Ordinal))

        let private isMemoryMeasurement (measurement: string) =
            measurement.Contains("memory", StringComparison.OrdinalIgnoreCase)
            || measurement.Contains("rss", StringComparison.OrdinalIgnoreCase)
            || measurement.Contains("bytes", StringComparison.OrdinalIgnoreCase)
            || measurement.Contains("kb", StringComparison.OrdinalIgnoreCase)

        let estimate (measurement: string) (context: OperatorEstimateContext) =
            ensureLoaded ()
            let pixelType = context.PixelType |> Option.defaultValue ""
            let add term value =
                coefficient measurement (termKey context.Operator pixelType term) * value

            let hasAnyCoefficient =
                [ "constant"
                  "operationCount"
                  "dataMB"
                  "windowDataMB"
                  "radius2DataMB"
                  "radiusDataMB"
                  "radius3DataMB"
                  "kernelDiameterDataMB"
                  "kernelVoxelsDataMB"
                  "dataMBLog2Voxels"
                  "voxelsMLog2Voxels"
                  "voxelsM"
                  "windowVoxelsM"
                  "radius2VoxelsM"
                  "volumeMB"
                  "windowVolumeMB"
                  "radius2VolumeMB" ]
                |> List.exists (fun term -> hasCoefficient measurement (termKey context.Operator pixelType term))

            let constant = add "constant" 1.0

            let scaled =
                if isMemoryMeasurement measurement then
                    let volumeMB =
                        context.VolumeBytes
                        |> Option.map (fun bytes -> float bytes / 1.0e6)
                        |> Option.defaultValue 0.0

                    let dataMB = volumeMB

                    let windowVolumeMB =
                        match context.WindowSize with
                        | Some windowSize -> windowSize * volumeMB
                        | None -> 0.0

                    let radius2VolumeMB =
                        match context.Radius with
                        | Some radius -> radius * radius * volumeMB
                        | None -> 0.0

                    let radiusVolumeMB =
                        match context.Radius with
                        | Some radius -> radius * volumeMB
                        | None -> 0.0

                    let radius3VolumeMB =
                        match context.Radius with
                        | Some radius -> radius * radius * radius * volumeMB
                        | None -> 0.0

                    let kernelDiameter =
                        context.KernelSize
                        |> Option.orElseWith (fun () ->
                            context.Radius |> Option.map (fun radius -> 2.0 * radius + 1.0))
                        |> Option.orElseWith (fun () ->
                            context.Sigma |> Option.map (fun sigma -> 2.0 * Math.Ceiling(2.0 * sigma) + 1.0))
                        |> Option.defaultValue 0.0

                    let logVoxels =
                        context.Voxels
                        |> Option.filter (fun voxels -> voxels > 1UL)
                        |> Option.map (fun voxels -> Math.Log(float voxels, 2.0))
                        |> Option.defaultValue 0.0

                    add "operationCount" 1.0
                    + add "dataMB" dataMB
                    + add "windowDataMB" windowVolumeMB
                    + add "radius2DataMB" radius2VolumeMB
                    + add "radiusDataMB" radiusVolumeMB
                    + add "radius3DataMB" radius3VolumeMB
                    + add "kernelDiameterDataMB" (kernelDiameter * dataMB)
                    + add "kernelVoxelsDataMB" (kernelDiameter * kernelDiameter * kernelDiameter * dataMB)
                    + add "dataMBLog2Voxels" (dataMB * logVoxels)
                    + add "volumeMB" volumeMB
                    + add "windowVolumeMB" windowVolumeMB
                    + add "radius2VolumeMB" radius2VolumeMB
                else
                    let voxelsM =
                        context.Voxels
                        |> Option.map (fun voxels -> float voxels / 1.0e6)
                        |> Option.defaultValue 0.0

                    let dataMB =
                        context.VolumeBytes
                        |> Option.map (fun bytes -> float bytes / 1.0e6)
                        |> Option.defaultValue 0.0

                    let windowVoxelsM =
                        match context.WindowSize with
                        | Some windowSize -> windowSize * voxelsM
                        | None -> 0.0

                    let radius2VoxelsM =
                        match context.Radius with
                        | Some radius -> radius * radius * voxelsM
                        | None -> 0.0

                    let windowDataMB =
                        match context.WindowSize with
                        | Some windowSize -> windowSize * dataMB
                        | None -> 0.0

                    let radius2DataMB =
                        match context.Radius with
                        | Some radius -> radius * radius * dataMB
                        | None -> 0.0

                    let radiusDataMB =
                        match context.Radius with
                        | Some radius -> radius * dataMB
                        | None -> 0.0

                    let radius3DataMB =
                        match context.Radius with
                        | Some radius -> radius * radius * radius * dataMB
                        | None -> 0.0

                    let kernelDiameter =
                        context.KernelSize
                        |> Option.orElseWith (fun () ->
                            context.Radius |> Option.map (fun radius -> 2.0 * radius + 1.0))
                        |> Option.orElseWith (fun () ->
                            context.Sigma |> Option.map (fun sigma -> 2.0 * Math.Ceiling(2.0 * sigma) + 1.0))
                        |> Option.defaultValue 0.0

                    let logVoxels =
                        context.Voxels
                        |> Option.filter (fun voxels -> voxels > 1UL)
                        |> Option.map (fun voxels -> Math.Log(float voxels, 2.0))
                        |> Option.defaultValue 0.0

                    add "operationCount" 1.0
                    + add "dataMB" dataMB
                    + add "windowDataMB" windowDataMB
                    + add "radius2DataMB" radius2DataMB
                    + add "radiusDataMB" radiusDataMB
                    + add "radius3DataMB" radius3DataMB
                    + add "kernelDiameterDataMB" (kernelDiameter * dataMB)
                    + add "kernelVoxelsDataMB" (kernelDiameter * kernelDiameter * kernelDiameter * dataMB)
                    + add "dataMBLog2Voxels" (dataMB * logVoxels)
                    + add "voxelsMLog2Voxels" (voxelsM * logVoxels)
                    + add "voxelsM" voxelsM
                    + add "windowVoxelsM" windowVoxelsM
                    + add "radius2VoxelsM" radius2VoxelsM

            let estimate = constant + scaled
            if hasAnyCoefficient then Some(max 0.0 estimate) else None

        let timeCostModel evaluation context fallback =
            { Kind = Native
              Evaluation = evaluation
              Estimate =
                fun input ->
                    let estimateContext = context input
                    let tags =
                        [ yield "operator", estimateContext.Operator
                          match estimateContext.PixelType with
                          | Some value -> yield "pixelType", value
                          | None -> ()
                          match estimateContext.Voxels with
                          | Some value -> yield "voxels", string value
                          | None -> ()
                          match estimateContext.WindowSize with
                          | Some value -> yield "windowSize", Convert.ToString(value, CultureInfo.InvariantCulture)
                          | None -> ()
                          match estimateContext.Radius with
                          | Some value -> yield "radius", Convert.ToString(value, CultureInfo.InvariantCulture)
                          | None -> ()
                          match estimateContext.KernelSize with
                          | Some value -> yield "kernelSize", Convert.ToString(value, CultureInfo.InvariantCulture)
                          | None -> ()
                          match estimateContext.Sigma with
                          | Some value -> yield "sigma", Convert.ToString(value, CultureInfo.InvariantCulture)
                          | None -> () ]

                    match estimate "elapsedMilliseconds" estimateContext with
                    | Some milliseconds ->
                        StageTimeCostEstimate.create 0.0 milliseconds 0UL 0UL 0UL 0UL (Some calibratedMillisecondsKey)
                        |> StageTimeCostEstimate.withTags tags
                    | None -> fallback input |> StageTimeCostEstimate.withTags tags }

    let private parseCsvLine (line: string) =
        let values = ResizeArray<string>()
        let current = StringBuilder()
        let mutable quoted = false
        let mutable i = 0

        while i < line.Length do
            match line[i] with
            | '"' when quoted && i + 1 < line.Length && line[i + 1] = '"' ->
                current.Append('"') |> ignore
                i <- i + 2
            | '"' ->
                quoted <- not quoted
                i <- i + 1
            | ',' when not quoted ->
                values.Add(current.ToString())
                current.Clear() |> ignore
                i <- i + 1
            | ch ->
                current.Append(ch) |> ignore
                i <- i + 1

        values.Add(current.ToString())
        values |> Seq.toList

    let private readCsvMaps path =
        if not (File.Exists path) then
            []
        else
            match File.ReadLines path |> Seq.toList with
            | [] -> []
            | headerLine :: rows ->
                let header = parseCsvLine headerLine
                rows
                |> List.choose (fun line ->
                    if String.IsNullOrWhiteSpace line then
                        None
                    else
                        let values = parseCsvLine line
                        Some(
                            header
                            |> List.mapi (fun index name ->
                                let value = if index < values.Length then values[index] else ""
                                name, value)
                            |> Map.ofList))

    let private field name (row: Map<string, string>) =
        row |> Map.tryFind name |> Option.defaultValue ""

    let private tryFloat (value: string) =
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, parsed -> Some parsed
        | _ -> None

    let private tryUInt64 (value: string) =
        match UInt64.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture) with
        | true, parsed -> Some parsed
        | _ -> None

    let private optionFloat name row =
        field name row |> tryFloat

    let private optionUInt64 name row =
        field name row |> tryUInt64

    let readEvidenceCsv path =
        readCsvMaps path
        |> List.choose (fun row ->
            match tryFloat (field "value" row), tryFloat (field "featureValue" row) with
            | Some value, Some featureValue ->
                Some
                    { RowId = field "rowId" row
                      Measurement = field "measurement" row
                      Value = value
                      SourcePath = field "sourcePath" row
                      FeatureKey = field "featureKey" row
                      FeatureValue = featureValue
                      Operator = field "operator" row
                      PixelType =
                        match field "pixelType" row with
                        | "" -> None
                        | value -> Some value
                      Width = optionUInt64 "width" row
                      Height = optionUInt64 "height" row
                      Depth = optionUInt64 "depth" row
                      Voxels = optionUInt64 "voxels" row
                      SlicePixels = optionUInt64 "slicePixels" row
                      SliceBytes = optionUInt64 "sliceBytes" row
                      VolumeBytes = optionUInt64 "volumeBytes" row
                      WindowSize = optionFloat "windowSize" row
                      Radius = optionFloat "radius" row
                      KernelSize = optionFloat "kernelSize" row
                      Sigma = optionFloat "sigma" row }
            | _ -> None)

    let private termKey operator pixelType term =
        $"operator={operator}|pixelType={pixelType}|term={term}"

    let private addTerm terms key value =
        if Double.IsFinite value && value <> 0.0 then
            (key, value) :: terms
        else
            terms

    let private isMemoryMeasurement (measurement: string) =
        measurement.Contains("memory", StringComparison.OrdinalIgnoreCase)
        || measurement.Contains("rss", StringComparison.OrdinalIgnoreCase)
        || measurement.Contains("bytes", StringComparison.OrdinalIgnoreCase)
        || measurement.Contains("kb", StringComparison.OrdinalIgnoreCase)

    let private log2 (value: float) =
        Math.Log(value, 2.0)

    let private kernelDiameter (row: EvidenceRow) =
        row.KernelSize
        |> Option.orElseWith (fun () ->
            row.Radius |> Option.map (fun radius -> 2.0 * radius + 1.0))
        |> Option.orElseWith (fun () ->
            row.Sigma |> Option.map (fun sigma -> 2.0 * Math.Ceiling(2.0 * sigma) + 1.0))

    let private addSizeAndKernelTerms scale operator pixelType (row: EvidenceRow) terms =
        let mutable terms = terms

        match row.VolumeBytes, row.Voxels with
        | Some volumeBytes, Some voxels when voxels > 1UL ->
            let dataMB = scale * float volumeBytes / 1.0e6
            let voxelsM = scale * float voxels / 1.0e6
            let logVoxels = log2 (float voxels)
            terms <- addTerm terms (termKey operator pixelType "dataMBLog2Voxels") (dataMB * logVoxels)
            terms <- addTerm terms (termKey operator pixelType "voxelsMLog2Voxels") (voxelsM * logVoxels)
        | _ -> ()

        match kernelDiameter row, row.VolumeBytes with
        | Some diameter, Some volumeBytes ->
            let dataMB = scale * float volumeBytes / 1.0e6
            terms <- addTerm terms (termKey operator pixelType "kernelDiameterDataMB") (diameter * dataMB)
            terms <- addTerm terms (termKey operator pixelType "kernelVoxelsDataMB") (diameter * diameter * diameter * dataMB)
        | _ -> ()

        match row.Radius, row.VolumeBytes with
        | Some radius, Some volumeBytes ->
            let dataMB = scale * float volumeBytes / 1.0e6
            terms <- addTerm terms (termKey operator pixelType "radiusDataMB") (radius * dataMB)
            terms <- addTerm terms (termKey operator pixelType "radius3DataMB") (radius * radius * radius * dataMB)
        | _ -> ()

        terms

    let private evidenceTerms measurement (row: EvidenceRow) =
        let operator =
            if String.IsNullOrWhiteSpace row.Operator then row.FeatureKey else row.Operator

        if String.Equals(operator, "Ignore", StringComparison.OrdinalIgnoreCase) then
            []
        else
            let pixelType = row.PixelType |> Option.defaultValue ""
            let scale = row.FeatureValue
            let mutable terms = []

            if String.Equals(operator, "intercept", StringComparison.OrdinalIgnoreCase) then
                terms <- addTerm terms (termKey operator pixelType "constant") scale
            else
                let operationCount =
                    row.Depth
                    |> Option.map (fun depth -> scale * float depth)
                    |> Option.defaultValue scale

                terms <- addTerm terms (termKey operator pixelType "operationCount") operationCount

                match row.WindowSize, row.Radius, row.VolumeBytes with
                | None, None, Some value ->
                    let dataMB = scale * float value / 1.0e6
                    terms <- addTerm terms (termKey operator pixelType "dataMB") dataMB
                | _ -> ()

            if isMemoryMeasurement measurement then
                match row.WindowSize, row.VolumeBytes with
                | Some windowSize, Some volumeBytes ->
                    terms <- addTerm terms (termKey operator pixelType "windowDataMB") (scale * windowSize * float volumeBytes / 1.0e6)
                | _ -> ()

                match row.Radius, row.VolumeBytes with
                | Some radius, Some volumeBytes ->
                    terms <- addTerm terms (termKey operator pixelType "radius2DataMB") (scale * radius * radius * float volumeBytes / 1.0e6)
                | _ -> ()

                terms <- addSizeAndKernelTerms scale operator pixelType row terms
            else
                match row.WindowSize, row.VolumeBytes with
                | Some windowSize, Some volumeBytes ->
                    terms <- addTerm terms (termKey operator pixelType "windowDataMB") (scale * windowSize * float volumeBytes / 1.0e6)
                | _ -> ()

                match row.Radius, row.VolumeBytes with
                | Some radius, Some volumeBytes ->
                    terms <- addTerm terms (termKey operator pixelType "radius2DataMB") (scale * radius * radius * float volumeBytes / 1.0e6)
                | _ -> ()

                terms <- addSizeAndKernelTerms scale operator pixelType row terms

            terms

    let private termParts (key: string) =
        key.Split('|')
        |> Array.fold (fun state part ->
            let index = part.IndexOf('=')
            if index <= 0 then
                state
            else
                state |> Map.add (part.Substring(0, index)) (part.Substring(index + 1))) Map.empty<string, string>

    let private aggregateEvidenceRows (evidence: EvidenceRow seq) =
        let runTerms =
            evidence
            |> Seq.groupBy (fun row -> row.Measurement, row.RowId, row.SourcePath)
            |> Seq.choose (fun ((measurement, rowId, _), rows) ->
                let rows = rows |> Seq.toList

                match rows with
                | [] -> None
                | first :: _ ->
                    let terms =
                        rows
                        |> List.collect (evidenceTerms measurement)
                        |> List.groupBy fst
                        |> List.map (fun (key, values) -> key, values |> List.sumBy snd)
                        |> List.filter (fun (_, value) -> value <> 0.0)

                    Some(measurement, rowId, first.Value, terms))
            |> Seq.toList

        runTerms
        |> List.groupBy (fun (measurement, rowId, _, _) -> measurement, rowId)
        |> List.choose (fun ((measurement, rowId), rows) ->
            match rows with
            | [] -> None
            | _ ->
                let value =
                    rows
                    |> List.map (fun (_, _, value, _) -> value)
                    |> medianFloat

                let terms =
                    rows
                    |> List.map (fun (_, _, _, terms) -> terms)
                    |> List.maxBy List.length

                Some(measurement, rowId, value, terms))

    let private predictionRowsFromCoefficients (coefficients: OperatorTermCoefficient seq) (evidence: EvidenceRow seq) =
        let coefficientByTerm =
            coefficients
            |> Seq.map (fun coefficient -> coefficient.TermKey, coefficient.Coefficient)
            |> Map.ofSeq

        evidence
        |> aggregateEvidenceRows
        |> Seq.map (fun (measurement, rowId, actual, terms) ->
                let predicted =
                    terms
                    |> List.sumBy (fun (key, value) ->
                        coefficientByTerm
                        |> Map.tryFind key
                        |> Option.map (fun coefficient -> coefficient * value)
                        |> Option.defaultValue 0.0)

                { RowId = rowId
                  Measurement = measurement
                  Actual = actual
                  Predicted = predicted
                  Residual = actual - predicted })

    let private rmseAndR2 (actual: float[]) (predicted: float[]) =
        let mean = actual |> Array.average
        let residuals = Array.map2 (fun y yHat -> y - yHat) actual predicted
        let sse = residuals |> Array.sumBy (fun value -> value * value)
        let sst = actual |> Array.sumBy (fun value -> let d = value - mean in d * d)
        let rmse = sqrt (sse / float actual.Length)
        let r2 = if sst > 0.0 then 1.0 - sse / sst else 1.0
        rmse, r2, residuals

    let fitOperatorTerms ridge minSupport (evidence: EvidenceRow seq) =
        let rowTerms =
            evidence
            |> aggregateEvidenceRows

        rowTerms
        |> List.groupBy (fun (measurement, _, _, _) -> measurement)
        |> List.choose (fun (measurement, measurementRows) ->
            let support =
                measurementRows
                |> List.collect (fun (_, _, _, terms) -> terms |> List.map fst |> List.distinct)
                |> List.countBy id
                |> Map.ofList

            let columns =
                support
                |> Map.toList
                |> List.filter (fun (_, count) -> count >= minSupport)
                |> List.map fst
                |> List.sort

            if measurementRows.IsEmpty || columns.IsEmpty then
                None
            else
                let columnIndex =
                    columns |> List.mapi (fun index key -> key, index) |> Map.ofList

                let a = Array2D.zeroCreate<float> measurementRows.Length columns.Length
                let y = Array.zeroCreate<float> measurementRows.Length

                measurementRows
                |> List.iteri (fun rowIndex (_, _, value, terms) ->
                    y[rowIndex] <- value

                    for key, value in terms do
                        match columnIndex |> Map.tryFind key with
                        | Some col -> a[rowIndex, col] <- value
                        | None -> ())

                let coefficients =
                    TinyLinAlg.Dense.nonNegativeLeastSquares ridge 20000 1e-10 a y

                let predicted = TinyLinAlg.Dense.predict a coefficients
                let rmse, r2, residuals = rmseAndR2 y predicted

                let coefficientRows =
                    columns
                    |> List.mapi (fun index key ->
                        let parts = termParts key
                        { Measurement = measurement
                          TermKey = key
                          Operator = parts |> Map.tryFind "operator" |> Option.defaultValue ""
                          PixelType = parts |> Map.tryFind "pixelType" |> Option.defaultValue ""
                          Term = parts |> Map.tryFind "term" |> Option.defaultValue ""
                          Coefficient = coefficients[index]
                          SupportCount = support[key]
                          RowCount = measurementRows.Length
                          ColumnCount = columns.Length
                          Rmse = rmse
                          R2 = r2
                          Solver = "nonNegativeLeastSquares" })

                let predictionRows =
                    measurementRows
                    |> List.mapi (fun index (_, rowId, actual, _) ->
                        { RowId = rowId
                          Measurement = measurement
                          Actual = actual
                          Predicted = predicted[index]
                          Residual = residuals[index] })

                Some
                    { Measurement = measurement
                      Coefficients = coefficientRows
                      Predictions = predictionRows })

    let writeOperatorTermCoefficientsCsv (path: string) (fits: OperatorTermFit seq) =
        use writer = new StreamWriter(path)
        writer.WriteLine("measurement,termKey,operator,pixelType,term,coefficient,supportCount,rowCount,columnCount,rmse,r2,solver")

        for fit in fits do
            for row in fit.Coefficients do
                [ row.Measurement
                  row.TermKey
                  row.Operator
                  row.PixelType
                  row.Term
                  invariant row.Coefficient
                  string row.SupportCount
                  string row.RowCount
                  string row.ColumnCount
                  invariant row.Rmse
                  invariant row.R2
                  row.Solver ]
                |> List.map csvEscape
                |> String.concat ","
                |> writer.WriteLine

    let writeOperatorTermPredictionsCsv (path: string) (fits: OperatorTermFit seq) =
        use writer = new StreamWriter(path)
        writer.WriteLine("rowId,measurement,actual,predicted,residual")

        for fit in fits do
            for row in fit.Predictions do
                [ row.RowId
                  row.Measurement
                  invariant row.Actual
                  invariant row.Predicted
                  invariant row.Residual ]
                |> List.map csvEscape
                |> String.concat ","
                |> writer.WriteLine

    let writeOperatorTermDiscrepanciesCsv (path: string) (minimumActual: float) (ratioThreshold: float) (fits: OperatorTermFit seq) =
        use writer = new StreamWriter(path)
        writer.WriteLine("rowId,measurement,actual,predicted,residual,absoluteResidual,ratio,flagged")

        for fit in fits do
            for row in fit.Predictions do
                let absoluteResidual = abs row.Residual
                let ratio =
                    if row.Actual > 0.0 && row.Predicted > 0.0 then
                        max (row.Actual / row.Predicted) (row.Predicted / row.Actual)
                    elif row.Actual = 0.0 && row.Predicted = 0.0 then
                        1.0
                    else
                        Double.PositiveInfinity

                let flagged =
                    row.Actual >= minimumActual
                    && (ratio >= ratioThreshold || absoluteResidual >= minimumActual * (ratioThreshold - 1.0))

                [ row.RowId
                  row.Measurement
                  invariant row.Actual
                  invariant row.Predicted
                  invariant row.Residual
                  invariant absoluteResidual
                  invariant ratio
                  string flagged ]
                |> List.map csvEscape
                |> String.concat ","
                |> writer.WriteLine

    let writeOperatorTermDiagnosticsCsv (path: string) (fits: OperatorTermFit seq) =
        use writer = new StreamWriter(path)
        writer.WriteLine("measurement,rowCount,columnCount,rmse,r2")

        for fit in fits do
            match fit.Coefficients with
            | first :: _ ->
                [ fit.Measurement
                  string first.RowCount
                  string first.ColumnCount
                  invariant first.Rmse
                  invariant first.R2 ]
                |> List.map csvEscape
                |> String.concat ","
                |> writer.WriteLine
            | [] -> ()

    let writeOperatorCostModel path name assumptions (fits: OperatorTermFit seq) =
        { SchemaVersion = 1
          Name = name
          CreatedUtc = DateTimeOffset.UtcNow
          Assumptions = assumptions |> Seq.toArray
          Coefficients =
            fits
            |> Seq.collect _.Coefficients
            |> Seq.toArray }
        |> OperatorCostModel.save path

    let predictWithOperatorCostModel (model: OperatorCostModel) (evidence: EvidenceRow seq) =
        predictionRowsFromCoefficients model.Coefficients evidence
        |> Seq.toList

    let fitOperatorTermsFromCsv evidencePath ridge minSupport outputDirectory modelPath =
        let evidence = readEvidenceCsv evidencePath
        let fits = fitOperatorTerms ridge minSupport evidence
        Directory.CreateDirectory outputDirectory |> ignore
        writeOperatorTermCoefficientsCsv (Path.Combine(outputDirectory, "operatorModelCoefficients.csv")) fits
        writeOperatorTermPredictionsCsv (Path.Combine(outputDirectory, "operatorModelPredictions.csv")) fits
        writeOperatorTermDiscrepanciesCsv (Path.Combine(outputDirectory, "operatorModelDiscrepancies.csv")) 100.0 4.0 fits
        writeOperatorTermDiagnosticsCsv (Path.Combine(outputDirectory, "operatorModelDiagnostics.csv")) fits
        writeOperatorCostModel
            modelPath
            "StackProcessing fitted operator cost model"
            [ "Fitted from Probe costEvidence.csv."
              "Streaming operators use operationCount plus dataMB-derived terms so whole-volume probe evidence can be applied per emitted slice."
              "Windowed and radius-dependent operators add windowDataMB, radius/kernel terms, and n log n terms when the graph exposes those variables."
              "The intercept feature is the only fitted whole-graph constant."
              "Ignore sinks are treated as zero-cost evidence terms." ]
            fits
        fits

let operatorImageTimeCost<'T> operator evaluation windowSize radius kernelSize sigma fallback : StageTimeCostModel =
    let context input =
        let voxels = inputVoxels input
        Fitting.OperatorEstimateContext.create
            operator
            (Some(pixelTypeName<'T>))
            (Some voxels)
            (Some(imageBytes<'T> voxels))
            windowSize
            radius
            kernelSize
            sigma

    Fitting.OperatorCostRuntime.timeCostModel evaluation context fallback

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
