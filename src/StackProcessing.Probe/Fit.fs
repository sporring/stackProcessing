module ProbeFit

open System
open System.Globalization
open System.IO
open System.Text.Json
open System.Text.RegularExpressions
open StackProcessingCost

type Options =
    { MeasurementStorePath: string
      OutputDirectory: string
      ModelOutputPath: string
      Ridge: float
      MinSupport: int
      FixedThrough: string option
      Selector: ProbeSelection.EvidenceSelector }

let private usage () =
    printfn "Usage: dotnet src/StackProcessing.Probe/bin/Debug/net10.0/StackProcessing.Probe.dll fit [options]"
    printfn ""
    printfn "Fits an operator cost model from the durable measurement store without recollecting data."
    printfn ""
    printfn "Options:"
    printfn "  --measurement-store PATH  Measurement JSONL store. Defaults to measurements/stackprocessing-probe.jsonl."
    printfn "  --output PATH             Fit diagnostics directory. Defaults to tmp/fit."
    printfn "  --model-output PATH       Fitted model path. Defaults to models/fitted/stackprocessing.operator-cost.json."
    printfn "  --family LIST             Families to fit, e.g. empty,io,io-cast."
    printfn "  --up-to FAMILY            Fit ladder families through FAMILY."
    printfn "  --fixed-through FAMILY    Fit FAMILY and earlier first, then hold those coefficients fixed."
    printfn "  --member LIST             Restrict to operator/member ids."
    printfn "  --shape WxHxD             Restrict to one probed image shape."
    printfn "  --shapes LIST             Restrict to comma-separated probed image shapes."
    printfn "  --ridge VALUE             Non-negative ridge value. Defaults to 1e-8."
    printfn "  --min-support N           Minimum term support. Defaults to 3."

let private repositoryRoot () =
    let cwd = Directory.GetCurrentDirectory()
    if File.Exists(Path.Combine(cwd, "StackProcessing.sln")) then cwd else cwd

let private defaultOptions () =
    let root = repositoryRoot ()
    { MeasurementStorePath = Path.Combine(root, "measurements", "stackprocessing-probe.jsonl")
      OutputDirectory = Path.Combine(root, "tmp", "fit")
      ModelOutputPath = Path.Combine(root, "models", "fitted", "stackprocessing.operator-cost.json")
      Ridge = 1e-8
      MinSupport = 3
      FixedThrough = None
      Selector = { Families = [ "all" ]; Members = []; UpTo = None; Shapes = [] } }

let rec private parseArgs options args =
    match args with
    | [] -> Ok options
    | ("-h" | "--help") :: _ ->
        usage ()
        Error 0
    | "--measurement-store" :: value :: rest ->
        parseArgs { options with MeasurementStorePath = Path.GetFullPath value } rest
    | "--output" :: value :: rest ->
        parseArgs { options with OutputDirectory = Path.GetFullPath value } rest
    | "--model-output" :: value :: rest ->
        parseArgs { options with ModelOutputPath = Path.GetFullPath value } rest
    | "--family" :: value :: rest
    | "--families" :: value :: rest ->
        match ProbeSelection.parseFamilies value with
        | Some families -> parseArgs { options with Selector = { options.Selector with Families = families; UpTo = None } } rest
        | None ->
            eprintfn "fit: --family expects empty,io,io-cast,singleton,window-slab,neighbourhood,geometry,fourier,keypoints,dependency,reducers, or all"
            Error 2
    | "--up-to" :: value :: rest
    | "--max-step" :: value :: rest ->
        match ProbeSelection.normalizeFamily value with
        | Some family -> parseArgs { options with Selector = { options.Selector with UpTo = Some family } } rest
        | None ->
            eprintfn "fit: unknown ladder family '%s'" value
            Error 2
    | "--fixed-through" :: value :: rest
    | "--fixed-up-to" :: value :: rest
    | "--up-to-fixed" :: value :: rest ->
        match ProbeSelection.normalizeFamily value with
        | Some "all" ->
            eprintfn "fit: --fixed-through expects a concrete ladder family, not all"
            Error 2
        | Some family -> parseArgs { options with FixedThrough = Some family } rest
        | None ->
            eprintfn "fit: unknown fixed ladder family '%s'" value
            Error 2
    | "--member" :: value :: rest
    | "--members" :: value :: rest
    | "--operator" :: value :: rest
    | "--operators" :: value :: rest ->
        parseArgs { options with Selector = { options.Selector with Members = options.Selector.Members @ ProbeSelection.splitCsvList value } } rest
    | "--shape" :: value :: rest ->
        match ProbeSelection.normalizeShape value with
        | Some shape -> parseArgs { options with Selector = { options.Selector with Shapes = [ shape ] } } rest
        | None ->
            eprintfn "fit: --shape expects WxHxD, for example 512x512x128"
            Error 2
    | "--shapes" :: value :: rest ->
        match ProbeSelection.parseShapes value with
        | Some shapes -> parseArgs { options with Selector = { options.Selector with Shapes = shapes } } rest
        | None ->
            eprintfn "fit: --shapes expects comma-separated shapes, for example 256x256x256,512x512x128,1024x1024x64"
            Error 2
    | "--ridge" :: value :: rest ->
        match Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, ridge when ridge >= 0.0 -> parseArgs { options with Ridge = ridge } rest
        | _ ->
            eprintfn "fit: --ridge expects a non-negative floating-point value"
            Error 2
    | "--min-support" :: value :: rest ->
        match Int32.TryParse value with
        | true, minSupport when minSupport > 0 -> parseArgs { options with MinSupport = minSupport } rest
        | _ ->
            eprintfn "fit: --min-support expects a positive integer"
            Error 2
    | option :: _ ->
        eprintfn "fit: unknown option %s" option
        usage ()
        Error 2

let private jsonOptions =
    let options = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
    options

let readMeasurementStore path =
    if not (File.Exists path) then
        []
    else
        File.ReadLines path
        |> Seq.choose (fun line ->
            if String.IsNullOrWhiteSpace line then
                None
            else
                try
                    JsonSerializer.Deserialize<ProbeAnalysis.StoredMeasurementRecord>(line, jsonOptions) |> Some
                with _ ->
                    None)
        |> Seq.toList

let private tryParseUInt64Text (value: string) =
    let m = Regex.Match(value, @"^\s*(\d+)")
    if m.Success then
        match UInt64.TryParse(m.Groups[1].Value, NumberStyles.Integer, CultureInfo.InvariantCulture) with
        | true, parsed -> Some parsed
        | _ -> None
    else
        None

let private tryParseFloatText (value: string) =
    let m = Regex.Match(value, @"^\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
    if m.Success then
        match Double.TryParse(m.Value, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, parsed -> Some parsed
        | _ -> None
    else
        None

let rec private normalizePixelType (value: string) =
    match value with
    | null -> None
    | text ->
        let trimmed = text.Trim()
        let lower = trimmed.ToLowerInvariant()
        let compoundMatch = Regex.Match(trimmed, @"^(.+?)To(.+)$", RegexOptions.IgnoreCase)

        if compoundMatch.Success then
            match normalizePixelType compoundMatch.Groups[1].Value, normalizePixelType compoundMatch.Groups[2].Value with
            | Some sourceType, Some targetType -> Some $"{sourceType}To{targetType}"
            | _ -> Some trimmed
        else
        match lower with
        | "" -> None
        | "byte" | "uint8" -> Some "UInt8"
        | "sbyte" | "int8" -> Some "Int8"
        | "uint16" -> Some "UInt16"
        | "int16" -> Some "Int16"
        | "uint32" -> Some "UInt32"
        | "int32" -> Some "Int32"
        | "uint64" -> Some "UInt64"
        | "int64" -> Some "Int64"
        | "single" | "float32" -> Some "Float32"
        | "double" | "float" | "float64" -> Some "Float64"
        | "complex" -> Some "Complex"
        | other -> Some other

let private pixelTypeBytes (pixelType: string) =
    let baseType =
        let trimmed = pixelType.Trim()
        let compoundMatch = Regex.Match(trimmed, @"^(.+?)To(.+)$", RegexOptions.IgnoreCase)

        if compoundMatch.Success then
            compoundMatch.Groups[2].Value
        else
            trimmed.Split('.', StringSplitOptions.RemoveEmptyEntries)
            |> Array.tryHead
            |> Option.defaultValue trimmed
        |> fun text -> text.Trim().ToLowerInvariant()

    match baseType with
    | "byte" | "uint8" | "sbyte" | "int8" -> Some 1UL
    | "uint16" | "int16" -> Some 2UL
    | "uint32" | "int32" | "single" | "float32" -> Some 4UL
    | "uint64" | "int64" | "double" | "float" | "float64" -> Some 8UL
    | "complex" -> Some 16UL
    | _ -> None

let private splitPixelTypeFormat (pixelType: string) =
    let parts =
        pixelType.Split('.', StringSplitOptions.RemoveEmptyEntries)
        |> Array.map (fun part -> part.Trim())
        |> Array.filter (fun part -> not (String.IsNullOrWhiteSpace part))

    match parts with
    | [| baseType; format |] -> normalizePixelType baseType, Some(format.ToLowerInvariant())
    | [| baseType |] -> normalizePixelType baseType, None
    | _ -> normalizePixelType pixelType, None

let private formatOperator operator format =
    match format with
    | Some format when
        (String.Equals(operator, "Read", StringComparison.OrdinalIgnoreCase)
         || String.Equals(operator, "ReadCast", StringComparison.OrdinalIgnoreCase)
         || String.Equals(operator, "Write", StringComparison.OrdinalIgnoreCase)) ->
        $"{operator}.{format}"
    | _ -> operator

let private formatFromSuffixText (suffix: string) =
    if String.IsNullOrWhiteSpace suffix then
        None
    else
        let format = suffix.Trim().TrimStart('.').ToLowerInvariant()
        if String.IsNullOrWhiteSpace format then None else Some format

type private FeatureMetadata =
    { Operator: string
      Parameters: Map<string, string> }

let private metadataFormat (metadata: FeatureMetadata) =
    metadata.Parameters
    |> Map.tryFind "suffix"
    |> Option.bind formatFromSuffixText

let private parameterPixelType key (metadata: FeatureMetadata) =
    metadata.Parameters
    |> Map.tryFind key
    |> Option.bind normalizePixelType

let private castPairPixelType (metadata: FeatureMetadata) =
    match parameterPixelType "sourceType" metadata, parameterPixelType "targetType" metadata with
    | Some sourceType, Some targetType -> Some $"{sourceType}To{targetType}"
    | _ -> None

let private specializeOperator (metadata: FeatureMetadata) =
    match metadata.Operator.Trim(), metadata.Parameters |> Map.tryFind "function", metadata.Parameters |> Map.tryFind "operation" with
    | operator, Some functionName, _ when String.Equals(operator, "UnaryImageFunction", StringComparison.OrdinalIgnoreCase) ->
        $"{operator}.{functionName.Trim().ToLowerInvariant()}"
    | operator, _, Some operation when String.Equals(operator, "ImageOpScalar", StringComparison.OrdinalIgnoreCase) ->
        let operationName =
            match operation.Trim() with
            | "+" -> "add"
            | "-" -> "subtract"
            | "*" -> "multiply"
            | "/" -> "divide"
            | other -> other.ToLowerInvariant()

        $"{operator}.{operationName}"
    | operator, _, _ -> formatOperator operator (metadataFormat metadata)

let private operatorCoveredByRuntime (runtimeOperators: Set<string>) (operator: string) =
    let normalized = operator.Trim().ToLowerInvariant()
    runtimeOperators.Contains normalized
    || runtimeOperators |> Seq.exists (fun runtimeOperator -> runtimeOperator.StartsWith(normalized + ".", StringComparison.OrdinalIgnoreCase))

type private EvidenceContext =
    { PixelType: string option
      Width: uint64 option
      Height: uint64 option
      Depth: uint64 option }

let private parseFeatureMetadata (feature: string) =
    let parts = feature.Split(':', StringSplitOptions.RemoveEmptyEntries)
    let operator = if parts.Length = 0 then feature else parts[0]
    let parameters =
        parts
        |> Seq.skip 1
        |> Seq.choose (fun part ->
            let index = part.IndexOf('=')
            if index <= 0 then None else Some(part.Substring(0, index), part.Substring(index + 1)))
        |> Map.ofSeq
    { Operator = operator; Parameters = parameters }

let private parameterUInt64 key metadata =
    metadata.Parameters |> Map.tryFind key |> Option.bind tryParseUInt64Text

let private parameterFloat key metadata =
    metadata.Parameters |> Map.tryFind key |> Option.bind tryParseFloatText

let private featureSize metadata =
    match parameterUInt64 "width" metadata, parameterUInt64 "height" metadata, parameterUInt64 "depth" metadata with
    | Some width, Some height, Some depth -> Some(width, height, depth)
    | _ -> None

let private contextFromInputJson (record: ProbeAnalysis.StoredMeasurementRecord) =
    let mutable pixelType = None
    let mutable width = None
    let mutable height = None
    let mutable depth = None
    try
        use doc = JsonDocument.Parse(record.InputDescriptionJson)
        let root = doc.RootElement
        let tryString (name: string) =
            match root.TryGetProperty(name) with
            | true, property when property.ValueKind = JsonValueKind.String -> Some(property.GetString())
            | _ -> None
        pixelType <- tryString "pixelType" |> Option.bind normalizePixelType
        width <- tryString "width" |> Option.bind tryParseUInt64Text
        height <- tryString "height" |> Option.bind tryParseUInt64Text
        depth <- tryString "depth" |> Option.bind tryParseUInt64Text
    with _ ->
        ()
    { PixelType = pixelType; Width = width; Height = height; Depth = depth }

type private ReadCastCase =
    { SourceType: string
      TargetType: string
      SourceFormat: string option
      Mode: string }

let private graphNameFromRowId (rowId: string) =
    let normalized = rowId.Replace('\\', '/')
    let slash = normalized.LastIndexOf('/')
    if slash >= 0 then normalized.Substring(slash + 1) else normalized

let private tryReadCastCase (rowId: string) =
    let name = graphNameFromRowId rowId
    let parts = name.Split('-', StringSplitOptions.RemoveEmptyEntries) |> Array.toList

    match parts with
    | _bottomup :: _index :: "read" :: sourceType :: targetType :: "implicit" :: _ ->
        match normalizePixelType sourceType, normalizePixelType targetType with
        | Some sourceType, Some targetType ->
            Some { SourceType = sourceType; TargetType = targetType; SourceFormat = None; Mode = "Implicit" }
        | _ -> None
    | _bottomup :: _index :: "read" :: sourceType :: targetType :: format :: "implicit" :: _ ->
        match normalizePixelType sourceType, normalizePixelType targetType with
        | Some sourceType, Some targetType ->
            Some { SourceType = sourceType; TargetType = targetType; SourceFormat = Some format; Mode = "Implicit" }
        | _ -> None
    | _bottomup :: _index :: "read" :: sourceType :: targetType :: "explicit" :: "cast" :: _ ->
        match normalizePixelType sourceType, normalizePixelType targetType with
        | Some sourceType, Some targetType ->
            Some { SourceType = sourceType; TargetType = targetType; SourceFormat = None; Mode = "Explicit" }
        | _ -> None
    | _bottomup :: _index :: "read" :: sourceType :: targetType :: format :: "explicit" :: "cast" :: _ ->
        match normalizePixelType sourceType, normalizePixelType targetType with
        | Some sourceType, Some targetType ->
            Some { SourceType = sourceType; TargetType = targetType; SourceFormat = Some format; Mode = "Explicit" }
        | _ -> None
    | _ -> None

let private tag key (term: ProbeAnalysis.StoredRuntimeCostTerm) =
    term.Tags |> Array.tryFind (fun tag -> tag.Key = key) |> Option.map _.Value

let private runtimeTermEvidenceRow (record: ProbeAnalysis.StoredMeasurementRecord) (featureMetadata: FeatureMetadata list) (term: ProbeAnalysis.StoredRuntimeCostTerm) : Fitting.EvidenceRow option =
    let operator =
        tag "operator" term
        |> Option.orElseWith (fun () -> if String.IsNullOrWhiteSpace term.StageName then None else Some term.StageName)

    operator
    |> Option.map (fun operator ->
        let metadata =
            featureMetadata
            |> List.tryFind (fun metadata -> String.Equals(metadata.Operator, operator, StringComparison.OrdinalIgnoreCase))
        let operator =
            metadata
            |> Option.map specializeOperator
            |> Option.defaultValue operator
        let rawPixelType = tag "pixelType" term
        let pixelType, format =
            match rawPixelType with
            | Some value -> splitPixelTypeFormat value
            | None -> None, None
        let operator = formatOperator operator format
        let voxels = tag "voxels" term |> Option.bind tryParseUInt64Text
        let bytesPerPixel = pixelType |> Option.bind pixelTypeBytes
        let volumeBytes =
            match voxels, bytesPerPixel with
            | Some voxels, Some bytesPerPixel -> Some(voxels * bytesPerPixel)
            | _ -> None
        let featureKey =
            let tags =
                term.Tags
                |> Array.map (fun tag -> tag.Key, tag.Value)
                |> Array.sortBy fst
                |> Array.map (fun (key, value) -> key + "=" + value)
            if tags.Length = 0 then operator else operator + ":" + String.concat ":" tags
        { RowId = record.RowId
          Measurement = record.Measurement
          Value = record.Value
          SourcePath = record.SourcePath
          FeatureKey = featureKey
          FeatureValue = float term.Multiplicity
          Operator = operator
          PixelType = pixelType
          Width = None
          Height = None
          Depth = None
          Voxels = voxels
          SlicePixels = None
          SliceBytes = None
          VolumeBytes = volumeBytes
          WindowSize = tag "windowSize" term |> Option.bind tryParseFloatText
          Radius = tag "radius" term |> Option.bind tryParseFloatText
          KernelSize = tag "kernelSize" term |> Option.orElseWith (fun () -> tag "minimumWindowSize" term) |> Option.bind tryParseFloatText
          Sigma = tag "sigma" term |> Option.bind tryParseFloatText })

let private featureEvidenceRow (context: EvidenceContext) (record: ProbeAnalysis.StoredMeasurementRecord) (feature: ProbeAnalysis.StoredFloatValue) : Fitting.EvidenceRow =
    let metadata = parseFeatureMetadata feature.Key
    let operator = specializeOperator metadata
    let featurePixelType =
        metadata.Parameters
        |> Map.tryFind "type"
        |> Option.bind normalizePixelType
        |> Option.orElseWith (fun () ->
            if String.Equals(metadata.Operator, "Cast", StringComparison.OrdinalIgnoreCase) then
                castPairPixelType metadata
            else
                None)
        |> Option.orElseWith (fun () ->
            match metadata.Operator.Trim().ToLowerInvariant() with
            | "coordinatex"
            | "coordinatey"
            | "coordinatez" -> Some "Float64"
            | _ -> None)
    let width, height, depth =
        match context.Width, context.Height, context.Depth with
        | Some width, Some height, Some depth -> Some width, Some height, Some depth
        | _ ->
            match featureSize metadata with
            | Some(width, height, depth) -> Some width, Some height, Some depth
            | None -> context.Width, context.Height, context.Depth
    let pixelType = featurePixelType |> Option.orElse context.PixelType
    let slicePixels =
        match width, height with
        | Some width, Some height -> Some(width * height)
        | _ -> None
    let voxels =
        match slicePixels, depth with
        | Some slicePixels, Some depth -> Some(slicePixels * depth)
        | _ -> None
    let bytesPerPixel = pixelType |> Option.bind pixelTypeBytes
    let sliceBytes =
        match slicePixels, bytesPerPixel with
        | Some slicePixels, Some bytesPerPixel -> Some(slicePixels * bytesPerPixel)
        | _ -> None
    let volumeBytes =
        match voxels, bytesPerPixel with
        | Some voxels, Some bytesPerPixel -> Some(voxels * bytesPerPixel)
        | _ -> None
    let sigma = parameterFloat "sigma" metadata
    let gaussianKernelSizeFromSigma = sigma |> Option.map (fun sigma -> 2.0 * Math.Ceiling(2.0 * sigma) + 1.0)
    { RowId = record.RowId
      Measurement = record.Measurement
      Value = record.Value
      SourcePath = record.SourcePath
      FeatureKey = feature.Key
      FeatureValue = feature.Value
      Operator = operator
      PixelType = pixelType
      Width = width
      Height = height
      Depth = depth
      Voxels = voxels
      SlicePixels = slicePixels
      SliceBytes = sliceBytes
      VolumeBytes = volumeBytes
      WindowSize = parameterFloat "windowSize" metadata
      Radius = parameterFloat "radius" metadata
      KernelSize =
        parameterFloat "kernelSize" metadata
        |> Option.orElseWith (fun () -> parameterFloat "minimumWindowSize" metadata)
        |> Option.orElse gaussianKernelSizeFromSigma
      Sigma = sigma }

let private imageGeometry context pixelType =
    let slicePixels =
        match context.Width, context.Height with
        | Some width, Some height -> Some(width * height)
        | _ -> None

    let voxels =
        match slicePixels, context.Depth with
        | Some slicePixels, Some depth -> Some(slicePixels * depth)
        | _ -> None

    let bytesPerPixel = pixelTypeBytes pixelType
    let sliceBytes =
        match slicePixels, bytesPerPixel with
        | Some slicePixels, Some bytesPerPixel -> Some(slicePixels * bytesPerPixel)
        | _ -> None

    let volumeBytes =
        match voxels, bytesPerPixel with
        | Some voxels, Some bytesPerPixel -> Some(voxels * bytesPerPixel)
        | _ -> None

    slicePixels, voxels, sliceBytes, volumeBytes

let private readCastEvidenceRows (context: EvidenceContext) (record: ProbeAnalysis.StoredMeasurementRecord) (case: ReadCastCase) : Fitting.EvidenceRow list =
    let sourceSlicePixels, sourceVoxels, sourceSliceBytes, sourceVolumeBytes =
        imageGeometry context case.SourceType

    let targetSlicePixels, targetVoxels, targetSliceBytes, targetVolumeBytes =
        imageGeometry context case.TargetType

    let castPixelType = $"{case.SourceType}To{case.TargetType}"
    let readOperator = formatOperator "Read" case.SourceFormat
    let readCastOperator = formatOperator "ReadCast" case.SourceFormat
    let readFormat = case.SourceFormat |> Option.defaultValue ""

    if String.Equals(case.Mode, "Implicit", StringComparison.OrdinalIgnoreCase) then
        [ { RowId = record.RowId
            Measurement = record.Measurement
            Value = record.Value
            SourcePath = record.SourcePath
            FeatureKey = $"ReadCast:sourceType={case.SourceType}:targetType={case.TargetType}:format={readFormat}:mode={case.Mode}"
            FeatureValue = 1.0
            Operator = readCastOperator
            PixelType = Some castPixelType
            Width = context.Width
            Height = context.Height
            Depth = context.Depth
            Voxels = targetVoxels
            SlicePixels = targetSlicePixels
            SliceBytes = targetSliceBytes
            VolumeBytes = targetVolumeBytes
            WindowSize = None
            Radius = None
            KernelSize = None
            Sigma = None } ]
    else
        [ { RowId = record.RowId
            Measurement = record.Measurement
            Value = record.Value
            SourcePath = record.SourcePath
            FeatureKey = $"Read:sourceType={case.SourceType}:format={readFormat}:readCastMode={case.Mode}"
            FeatureValue = 1.0
            Operator = readOperator
            PixelType = Some case.SourceType
            Width = context.Width
            Height = context.Height
            Depth = context.Depth
            Voxels = sourceVoxels
            SlicePixels = sourceSlicePixels
            SliceBytes = sourceSliceBytes
            VolumeBytes = sourceVolumeBytes
            WindowSize = None
            Radius = None
            KernelSize = None
            Sigma = None }
          { RowId = record.RowId
            Measurement = record.Measurement
            Value = record.Value
            SourcePath = record.SourcePath
            FeatureKey = $"Cast:sourceType={case.SourceType}:targetType={case.TargetType}:mode={case.Mode}"
            FeatureValue = 1.0
            Operator = "Cast"
            PixelType = Some castPixelType
            Width = context.Width
            Height = context.Height
            Depth = context.Depth
            Voxels = targetVoxels
            SlicePixels = targetSlicePixels
            SliceBytes = targetSliceBytes
            VolumeBytes = targetVolumeBytes
            WindowSize = None
            Radius = None
            KernelSize = None
            Sigma = None } ]

let membersOfRecord (record: ProbeAnalysis.StoredMeasurementRecord) =
    seq {
        for term in record.RuntimeCostTerms do
            match tag "operator" term with
            | Some operator -> yield operator
            | None when not (String.IsNullOrWhiteSpace term.StageName) -> yield term.StageName
            | _ -> ()
        for feature in record.Features do
            let metadata = parseFeatureMetadata feature.Key
            if not (String.Equals(metadata.Operator, "intercept", StringComparison.OrdinalIgnoreCase)) then
                yield metadata.Operator
    }
    |> Seq.distinct
    |> Seq.toList

let selectorMatchesRecord selector (record: ProbeAnalysis.StoredMeasurementRecord) =
    let familyOk =
        ProbeSelection.familyForRowId record.RowId
        |> Option.map (ProbeSelection.selectorMatchesFamily selector)
        |> Option.defaultValue true
    familyOk
    && ProbeSelection.selectorMatchesMembers selector (membersOfRecord record)
    && ProbeSelection.selectorMatchesShape selector record.RowId

let evidenceRowsFromRecord (record: ProbeAnalysis.StoredMeasurementRecord) =
    let featureMetadata =
        record.Features
        |> Array.toList
        |> List.map (fun feature -> parseFeatureMetadata feature.Key)

    match record.RuntimeCostTerms |> Array.toList with
    | _ :: _ ->
        let readCastCase = tryReadCastCase record.RowId

        let runtimeRows =
            record.RuntimeCostTerms
            |> Array.choose (fun term ->
                match readCastCase, tag "operator" term |> Option.map (fun value -> value.Trim().ToLowerInvariant()) with
                | Some _, Some operator when operator = "cast" || operator = "read" || operator.StartsWith("read.") -> None
                | _ -> runtimeTermEvidenceRow record featureMetadata term)
            |> Array.toList

        let runtimeOperators =
            runtimeRows
            |> List.map (fun row -> row.Operator.Trim().ToLowerInvariant())
            |> Set.ofList

        let context = contextFromInputJson record
        let readCastRows =
            match readCastCase with
            | Some case -> readCastEvidenceRows context record case
            | None -> []

        let supplementalRows =
            record.Features
            |> Array.toList
            |> List.choose (fun feature ->
                let metadata = parseFeatureMetadata feature.Key
                let operator = metadata.Operator.Trim().ToLowerInvariant()
                if operator = "ignore" then
                    None
                elif readCastCase.IsSome && (operator = "read" || operator = "cast") then
                    None
                elif operator = "intercept" || not (operatorCoveredByRuntime runtimeOperators metadata.Operator) then
                    Some(featureEvidenceRow context record feature)
                else
                    None)

        runtimeRows @ readCastRows @ supplementalRows
    | [] ->
        let context = contextFromInputJson record
        record.Features
        |> Array.toList
        |> List.map (featureEvidenceRow context record)

let private median values =
    let sorted = values |> List.sort
    let n = sorted.Length

    if n = 0 then
        0.0
    elif n % 2 = 1 then
        sorted[n / 2]
    else
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])

let private aggregateEvidenceRows (rows: Fitting.EvidenceRow list) =
    rows
    |> List.groupBy (fun row ->
        row.RowId,
        row.Measurement,
        row.FeatureKey,
        row.FeatureValue,
        row.Operator,
        row.PixelType,
        row.Width,
        row.Height,
        row.Depth,
        row.Voxels,
        row.SlicePixels,
        row.SliceBytes,
        row.VolumeBytes,
        row.WindowSize,
        row.Radius,
        row.KernelSize,
        row.Sigma)
    |> List.map (fun (_, group) ->
        let first = group.Head
        { first with
            Value = group |> List.map _.Value |> median
            SourcePath = group |> List.map _.SourcePath |> List.tryHead |> Option.defaultValue first.SourcePath })

let fit options =
    let allRecords = readMeasurementStore options.MeasurementStorePath

    let records =
        allRecords
        |> List.filter (selectorMatchesRecord options.Selector)

    let evidence =
        records
        |> List.collect evidenceRowsFromRecord
        |> aggregateEvidenceRows

    let fixedCoefficients =
        match options.FixedThrough with
        | None -> []
        | Some family ->
            let fixedSelector : ProbeSelection.EvidenceSelector =
                { Families = [ "all" ]
                  Members = []
                  UpTo = Some family
                  Shapes = options.Selector.Shapes }

            let fixedEvidence =
                allRecords
                |> List.filter (selectorMatchesRecord fixedSelector)
                |> List.collect evidenceRowsFromRecord
                |> aggregateEvidenceRows

            let fixedFits =
                Fitting.fitOperatorTerms options.Ridge options.MinSupport fixedEvidence

            fixedFits |> List.collect _.Coefficients

    Directory.CreateDirectory options.OutputDirectory |> ignore
    Fitting.writeEvidenceCsv (Path.Combine(options.OutputDirectory, "costEvidence.csv")) evidence
    let fits =
        match fixedCoefficients with
        | [] -> Fitting.fitOperatorTerms options.Ridge options.MinSupport evidence
        | fixedCoefficients -> Fitting.fitOperatorTermsWithFixed options.Ridge options.MinSupport fixedCoefficients evidence

    Fitting.writeOperatorTermCoefficientsCsv (Path.Combine(options.OutputDirectory, "operatorModelCoefficients.csv")) fits
    Fitting.writeOperatorTermPredictionsCsv (Path.Combine(options.OutputDirectory, "operatorModelPredictions.csv")) fits
    Fitting.writeOperatorTermDiscrepanciesCsv (Path.Combine(options.OutputDirectory, "operatorModelDiscrepancies.csv")) 100.0 4.0 fits
    Fitting.writeOperatorTermDiagnosticsCsv (Path.Combine(options.OutputDirectory, "operatorModelDiagnostics.csv")) fits
    Fitting.writeOperatorCostModel
        options.ModelOutputPath
        "StackProcessing fitted operator cost model"
        [ "Fitted from the durable Probe measurement store."
          "Fit selection can be restricted by ladder family/member."
          "Repeated collection is collapsed to median evidence per graph, measurement, and feature before fitting."
          match options.FixedThrough with
          | Some family -> $"Coefficients through ladder family '{family}' were fitted first and held fixed while fitting the selected scope."
          | None -> "No ladder coefficients were held fixed during fitting."
          "Streaming operators use operationCount plus dataMB-derived terms so whole-volume probe evidence can be applied per emitted slice."
          "Read-cast probes decompose read<T> and read<diskT> followed by cast<diskT,T> into Read(sourceType) plus Cast(sourceType->targetType) evidence."
          "Windowed and radius-dependent operators add windowDataMB, radius/kernel terms, and n log n terms when the graph exposes those variables."
          "The intercept feature is the only fitted whole-graph constant."
          "Ignore sinks are treated as zero-cost evidence terms." ]
        fits
    records.Length, evidence.Length, fits.Length

let main argv =
    match parseArgs (defaultOptions ()) (Array.toList argv) with
    | Error exitCode -> exitCode
    | Ok options ->
        try
            let records, evidence, fits = fit options
            printfn "fit selected %d measurement record(s), %d evidence row(s), %d measurement fit(s)." records evidence fits
            printfn "wrote fitted model to %s" options.ModelOutputPath
            0
        with ex ->
            eprintfn "fit failed: %s" ex.Message
            1
