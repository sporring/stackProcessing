namespace Studio.Compiler

open System
open System.Collections.Generic
open System.Globalization
open System.Text
open System.Text.RegularExpressions
open Studio.Graph

module PipelineCodeGenerator =
    type private ParameterExpression =
        { Value: string
          IsLinked: bool }

    type private NamedBinding =
        { Name: string
          Dependencies: Set<string>
          Text: string }

    type private TerminalExpression =
        { Dependencies: Set<string>
          Text: string }

    let private savedParamValue key (node: SavedNode) =
        node.Parameters
        |> Seq.tryFind (fun p -> p.Key = key)
        |> Option.map _.Value
        |> Option.defaultValue ""

    let private quote (value: string) =
        "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\""

    let private hasSuffix suffixes (value: string) =
        suffixes
        |> List.exists (fun suffix -> value.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))

    let private tryParseFloat (value: string) =
        match Double.TryParse(value.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, parsed -> Some parsed
        | false, _ -> None

    let private normalizeFloatLiteral (value: string) =
        let trimmed = value.Trim()
        let lower = trimmed.ToLowerInvariant()

        if String.IsNullOrWhiteSpace trimmed
           || lower = "nan"
           || lower = "infinity"
           || lower = "-infinity" then
            trimmed
        elif trimmed.EndsWith(".", StringComparison.Ordinal) then
            trimmed + "0"
        elif trimmed.Contains(".", StringComparison.Ordinal)
             || trimmed.Contains("e", StringComparison.OrdinalIgnoreCase) then
            trimmed
        else
            $"{trimmed}.0"

    let private float64Literal (value: string) =
        normalizeFloatLiteral value

    let private uintTuple3Literal (value: string) =
        let trimmed = value.Trim()
        let m = Regex.Match(trimmed, @"^\(?\s*(\d+)u?\s*,\s*(\d+)u?\s*,\s*(\d+)u?\s*\)?$")

        if m.Success then
            $"({m.Groups[1].Value}u,{m.Groups[2].Value}u,{m.Groups[3].Value}u)"
        else
            trimmed

    let private intArray3Literal (value: string) =
        let trimmed = value.Trim()
        let m = Regex.Match(trimmed, @"^\(?\s*(\d+)u?\s*,\s*(\d+)u?\s*,\s*(\d+)u?\s*\)?$")

        if m.Success then
            $"[| {m.Groups[1].Value}; {m.Groups[2].Value}; {m.Groups[3].Value} |]"
        else
            trimmed

    let private integerLiteralOrCast castName suffix (value: string) =
        let trimmed = value.Trim()

        if hasSuffix [ suffix ] trimmed then
            trimmed
        else
            match tryParseFloat trimmed with
            | Some parsed when Double.IsFinite parsed && parsed = Math.Truncate parsed ->
                let integerText = parsed.ToString("0", CultureInfo.InvariantCulture)
                $"{integerText}{suffix}"
            | _ ->
                $"{castName} {normalizeFloatLiteral trimmed}"

    let private standardNumericConstant (value: string) =
        match value.Trim().ToLowerInvariant() with
        | "e" -> Some "System.Math.E"
        | "pi" -> Some "System.Math.PI"
        | _ -> None

    let private numericLiteral numericType (value: string) =
        let trimmed = value.Trim()

        match standardNumericConstant trimmed with
        | Some constant ->
            match numericType with
            | UInt8 -> $"uint8 {constant}"
            | Int8 -> $"int8 {constant}"
            | UInt16 -> $"uint16 {constant}"
            | Int16 -> $"int16 {constant}"
            | UInt32 -> $"uint32 {constant}"
            | Int32 -> $"int {constant}"
            | UInt64 -> $"uint64 {constant}"
            | Int64 -> $"int64 {constant}"
            | Float32 -> $"float32 {constant}"
            | Float64
            | Number -> constant
            | Complex64 -> $"Image.ComplexFloat32(float32 {constant}, 0.0f)"
            | Complex -> $"System.Numerics.Complex({constant}, 0.0)"
        | None ->
            match numericType with
            | UInt8 ->
                integerLiteralOrCast "uint8" "uy" trimmed
            | Int8 ->
                integerLiteralOrCast "int8" "y" trimmed
            | UInt16 ->
                integerLiteralOrCast "uint16" "us" trimmed
            | Int16 ->
                integerLiteralOrCast "int16" "s" trimmed
            | UInt32 ->
                integerLiteralOrCast "uint32" "u" trimmed
            | Int32 ->
                match tryParseFloat trimmed with
                | Some parsed when Double.IsFinite parsed && parsed = Math.Truncate parsed ->
                    parsed.ToString("0", CultureInfo.InvariantCulture)
                | _ ->
                    $"int {normalizeFloatLiteral trimmed}"
            | UInt64 ->
                integerLiteralOrCast "uint64" "UL" trimmed
            | Int64 ->
                integerLiteralOrCast "int64" "L" trimmed
            | Float32 ->
                if hasSuffix [ "f" ] trimmed then trimmed else $"{float64Literal trimmed}f"
            | Float64
            | Number ->
                float64Literal trimmed
            | Complex64 ->
                if trimmed.StartsWith("Image.ComplexFloat32", StringComparison.Ordinal)
                   || trimmed.StartsWith("ComplexFloat32", StringComparison.Ordinal) then
                    trimmed
                else
                    let realLiteral = if hasSuffix [ "f" ] trimmed then trimmed else $"{float64Literal trimmed}f"
                    $"Image.ComplexFloat32({realLiteral}, 0.0f)"
            | Complex ->
                if trimmed.StartsWith("System.Numerics.Complex", StringComparison.Ordinal)
                   || trimmed.StartsWith("Complex", StringComparison.Ordinal) then
                    trimmed
                else
                    $"System.Numerics.Complex({float64Literal trimmed}, 0.0)"

    let private numericLiteralForPixelType pixelType value =
        match pixelType with
        | "uint8" -> numericLiteral UInt8 value
        | "int8" -> numericLiteral Int8 value
        | "uint16" -> numericLiteral UInt16 value
        | "int16" -> numericLiteral Int16 value
        | "uint32" -> numericLiteral UInt32 value
        | "int32" -> numericLiteral Int32 value
        | "uint64" -> numericLiteral UInt64 value
        | "int64" -> numericLiteral Int64 value
        | "float32" -> numericLiteral Float32 value
        | "float" -> numericLiteral Float64 value
        | "Image.ComplexFloat32" -> numericLiteral Complex64 value
        | "System.Numerics.Complex" -> numericLiteral Complex value
        | _ -> value

    let private literalValue basicType value =
        match basicType with
        | BasicType.Numeric numericType -> numericLiteral numericType value
        | BasicType.Bool -> value.Trim().ToLowerInvariant()
        | BasicType.String -> quote value
        | BasicType.Map -> value.Trim()
        | BasicType.Unit -> "()"

    let private scalarValueLiteral (node: SavedNode) =
        let value = savedParamValue "value" node

        let scalarType =
            savedParamValue "type" node
            |> BasicType.tryParse

        match scalarType with
        | Some basicType -> literalValue basicType value
        | _ -> value

    let private optionUInt (value: string) =
        let trimmed = value.Trim()
        if System.String.IsNullOrWhiteSpace trimmed || System.String.Equals(trimmed, "None", StringComparison.OrdinalIgnoreCase) then
            "None"
        elif trimmed.StartsWith("Some", StringComparison.Ordinal) then
            trimmed
        else
            $"(Some {trimmed.TrimEnd('u', 'U')}u)"

    let private optionInt (value: string) =
        let trimmed = value.Trim()
        if System.String.IsNullOrWhiteSpace trimmed || System.String.Equals(trimmed, "None", StringComparison.OrdinalIgnoreCase) then
            "None"
        elif trimmed.StartsWith("Some", StringComparison.Ordinal) then
            trimmed
        else
            $"(Some {trimmed})"

    let private optionQualified moduleName (value: string) =
        let trimmed = value.Trim()

        if System.String.IsNullOrWhiteSpace trimmed || System.String.Equals(trimmed, "None", StringComparison.OrdinalIgnoreCase) then
            "None"
        else
            let inner =
                if trimmed.StartsWith("(Some ", StringComparison.Ordinal) && trimmed.EndsWith(")", StringComparison.Ordinal) then
                    trimmed.Substring(6, trimmed.Length - 7).Trim()
                elif trimmed.StartsWith("Some ", StringComparison.Ordinal) then
                    trimmed.Substring(5).Trim()
                else
                    trimmed

            let qualified =
                if inner.Contains(".", StringComparison.Ordinal) then inner else $"{moduleName}.{inner}"

            $"(Some {qualified})"

    let private pixelTypeNameFromSuffix suffix =
        match suffix with
        | "UInt8" -> "uint8"
        | "Int8" -> "int8"
        | "UInt16" -> "uint16"
        | "Int16" -> "int16"
        | "UInt32" -> "uint32"
        | "Int32" -> "int32"
        | "UInt64" -> "uint64"
        | "Int64" -> "int64"
        | "Float32" -> "float32"
        | "Float64" -> "float"
        | "Complex64" | "ComplexFloat32" -> "Image.ComplexFloat32"
        | "Complex" | "Complex128" | "ComplexFloat64" -> "System.Numerics.Complex"
        | _ -> suffix

    let private pixelTypeNameFromParameter key defaultType (node: SavedNode) =
        let configuredType = savedParamValue key node

        if String.IsNullOrWhiteSpace configuredType then
            pixelTypeNameFromSuffix defaultType
        else
            pixelTypeNameFromSuffix configuredType

    let private uint64Literal (value: string) =
        let trimmed = value.Trim()

        if trimmed.EndsWith("UL", StringComparison.OrdinalIgnoreCase) then
            trimmed
        else
            $"{trimmed}UL"

    let private sourcePrefix availableMemory line =
        $"debug 1u (optimizerEnabled ()) {uint64Literal availableMemory}{Environment.NewLine}{line}"

    let private imageOpImageFunctionName (node: SavedNode) =
        match savedParamValue "operation" node with
        | "+" -> "addPair"
        | "-" -> "subPair"
        | "/" -> "divPair"
        | "max" -> "maxOfPair"
        | "min" -> "minOfPair"
        | _ -> "mulPair"

    let private pairStageFunctionName (node: SavedNode) =
        match node.FunctionId with
        | "ImageOpImage" -> Some(imageOpImageFunctionName node)
        | "ComplexFromReIm" -> Some "toComplex64"
        | "ComplexPolar" -> Some "polarToComplex64"
        | "ToVectorImage" -> Some "toVectorImage<float>"
        | "AppendVectorElement" -> Some "appendVectorElement<float>"
        | "VectorDot" -> Some "vectorDot"
        | "VectorCross3D" -> Some "vectorCross3D"
        | "AffineRegistration" ->
            let maxIterations = savedParamValue "maxIterations" node |> numericLiteral Int32
            let initialLinearStep = savedParamValue "initialLinearStep" node |> numericLiteral Float64
            let initialTranslationStep = savedParamValue "initialTranslationStep" node |> numericLiteral Float64
            let minStep = savedParamValue "minStep" node |> numericLiteral Float64
            let stepShrink = savedParamValue "stepShrink" node |> numericLiteral Float64
            Some $"affineRegistrationMatrices {{ defaultAffineRegistrationOptions with MaxIterations = {maxIterations}; InitialLinearStep = {initialLinearStep}; InitialTranslationStep = {initialTranslationStep}; MinStep = {minStep}; StepShrink = {stepShrink} }}"
        | "ImageComparison" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let comparison =
                match savedParamValue "operation" node with
                | "=" | "==" | "equal" -> "equal"
                | "<>" | "!=" | "notEqual" -> "notEqual"
                | ">=" | "greaterEqual" -> "greaterEqual"
                | "<" | "less" -> "less"
                | "<=" | "lessEqual" -> "lessEqual"
                | _ -> "greater"
            Some $"{comparison}<{pixelType}>"
        | "MaskLogic" ->
            let logic =
                match savedParamValue "operation" node with
                | "or" | "||" -> "maskOr"
                | "xor" | "^" -> "maskXor"
                | _ -> "maskAnd"
            Some logic
        | _ -> None

    let private pairCompositionOperator (node: SavedNode) =
        match node.FunctionId with
        | "ImageOpImage" -> ">=>"
        | _ -> ">=>"

    let private safeIdentifier (value: string) =
        let chars =
            value
            |> Seq.map (fun c -> if Char.IsLetterOrDigit c then c else '_')
            |> Seq.toArray

        new String(chars)

    let private scalarImageFunctionName (node: SavedNode) =
        let operation =
            match savedParamValue "operation" node with
            | "+"
            | "-"
            | "*"
            | "/" as operation -> operation
            | _ -> "*"

        match node.FunctionId, operation with
        | "ImageOpScalar", "+" -> Some "addScalar"
        | "ImageOpScalar", "-" -> Some "subScalar"
        | "ImageOpScalar", "*" -> Some "mulScalar"
        | "ImageOpScalar", "/" -> Some "divScalar"
        | "ScalarOpImage", "+" -> Some "scalarAdd"
        | "ScalarOpImage", "-" -> Some "scalarSub"
        | "ScalarOpImage", "*" -> Some "scalarMul"
        | "ScalarOpImage", "/" -> Some "scalarDiv"
        | _ -> None

    let private isScalarImageFunction functionId =
        functionId = "ImageOpScalar" || functionId = "ScalarOpImage"

    let private unaryImageFunctionNames =
        [ "abs"; "acos"; "asin"; "atan"; "cos"; "sin"; "tan"; "exp"; "log10"; "log"; "round"; "sqrt"; "square" ]

    let private unaryImageFunctionName (node: SavedNode) =
        let configuredFunction = savedParamValue "function" node

        if unaryImageFunctionNames |> List.contains configuredFunction then
            configuredFunction
        else
            "sqrt"

    let private unaryImageStageFunctionName (node: SavedNode) =
        match unaryImageFunctionName node with
        | "abs" -> "absFloat32"
        | "sqrt" -> "sqrtFloat32"
        | "square" -> "squareFloat32"
        | name -> name

    let private scalarFunctionExpression functionName argument =
        match functionName with
        | "abs" -> $"(System.Math.Abs {argument})"
        | "acos" -> $"(System.Math.Acos {argument})"
        | "asin" -> $"(System.Math.Asin {argument})"
        | "atan" -> $"(System.Math.Atan {argument})"
        | "cos" -> $"(System.Math.Cos {argument})"
        | "sin" -> $"(System.Math.Sin {argument})"
        | "tan" -> $"(System.Math.Tan {argument})"
        | "exp" -> $"(System.Math.Exp {argument})"
        | "log10" -> $"(System.Math.Log10 {argument})"
        | "log" -> $"(System.Math.Log {argument})"
        | "round" -> $"(System.Math.Round {argument})"
        | "square" -> $"({argument} * {argument})"
        | _ -> $"(System.Math.Sqrt {argument})"

    let private comparisonStageFunctionName (node: SavedNode) =
        match savedParamValue "operation" node with
        | "=" | "==" | "equal" -> "equal"
        | "<>" | "!=" | "notEqual" -> "notEqual"
        | ">=" | "greaterEqual" -> "greaterEqual"
        | "<" | "less" -> "less"
        | "<=" | "lessEqual" -> "lessEqual"
        | _ -> "greater"

    let private maskLogicStageFunctionName (node: SavedNode) =
        match savedParamValue "operation" node with
        | "or" | "||" -> "maskOr"
        | "xor" | "^" -> "maskXor"
        | _ -> "maskAnd"

    let private scalarTypeName (node: SavedNode) =
        match node.FunctionId with
        | "ScalarFunction" -> "Float64"
        | "RandomRigidTransform" -> "Affine"
        | "OtsuThresholdFromHistogram"
        | "MomentsThresholdFromHistogram" -> "Float64"
        | _ -> savedParamValue "type" node

    let private scalarParameter key value =
        { Key = key
          Value = value
          UseInput = false }

    let private replaceFileDirectorySelectors (graph: SavedGraph) =
        let nodes =
            graph.Nodes
            |> Array.map (fun node ->
                if node.FunctionId = "FileDirectory" then
                    { node with
                        FunctionId = "Scalar"
                        Parameters =
                            [| scalarParameter "type" "String"
                               scalarParameter "value" (savedParamValue "value" node) |] }
                else
                    node)

        { graph with Nodes = nodes }

    let private scalarNames (scalarNodes: SavedNode array) =
        scalarNodes
        |> Array.groupBy scalarTypeName
        |> Array.collect (fun (typeName, nodes) ->
            nodes
            |> Array.mapi (fun index node -> node.Id, $"{typeName}{index}"))
        |> Map.ofArray

    let private computeStatsFieldName portIndex =
        [| "NumPixels"; "Mean"; "Std"; "Min"; "Max"; "Sum"; "Var" |]
        |> Array.tryItem portIndex

    let private objectSizeStatsFieldName portIndex =
        [| "Count"; "Mean"; "Variance"; "Minimum"; "Maximum" |]
        |> Array.tryItem portIndex

    let private isSingleValueReducerNode (node: SavedNode) =
        node.FunctionId = "SurfaceArea"
        || node.FunctionId = "Volume"
        || node.FunctionId = "PointPairDistances"
        || node.FunctionId = "FitBiasModel"
        || node.FunctionId = "FitBiasModelMasked"

    let private imageInfoFieldExpression bindingName portIndex =
        [| $"{bindingName}.format"
           $"{bindingName}.dimensions"
           $"{bindingName}.size"
           $"{bindingName}.componentType"
           $"{bindingName}.numberOfComponents"
           $"{bindingName}.chunks"
           $"{bindingName}.chunks[0]"
           $"{bindingName}.chunks[1]"
           $"{bindingName}.chunks[2]"
           $"{bindingName}.size[0]"
           $"{bindingName}.size[1]"
           $"{bindingName}.size[2]" |]
        |> Array.tryItem portIndex

    let private chunkInfoFieldExpression bindingName portIndex =
        [| $"{bindingName}.chunks"
           $"{bindingName}.size"
           $"{bindingName}.topLeftInfo.componentType"
           $"{bindingName}.topLeftInfo.numberOfComponents"
           $"{bindingName}.chunks[0]"
           $"{bindingName}.chunks[1]"
           $"{bindingName}.chunks[2]"
           $"{bindingName}.size[0]"
           $"{bindingName}.size[1]"
           $"{bindingName}.size[2]" |]
        |> Array.tryItem portIndex

    let private isHistogramDataNode (node: SavedNode) =
        node.FunctionId = "Histogram"
        || node.FunctionId = "ImHistogramData"
        || node.FunctionId = "EstimateHistogram"

    let private isQuantilesNode (node: SavedNode) =
        node.FunctionId = "Quantiles"

    let private isSerialVolumeGeometryNode (node: SavedNode) =
        node.FunctionId = "SerialEstBoundingBox"

    let private isExpandNode (node: SavedNode) =
        node.FunctionId = "Expand"

    let private isWriteImageInfoNode (node: SavedNode) =
        node.FunctionId = "Write"

    let private isWriteChunkInfoNode (node: SavedNode) =
        node.FunctionId = "WriteChunks"

    let private readFormat (node: SavedNode) =
        savedParamValue "format" node

    let private isReadImageInfoNode (node: SavedNode) =
        match node.FunctionId with
        | "Read"
        | "ReadRandom"
        | "ReadRange" -> true
        | _ -> false

    let private isImageInfoNode (node: SavedNode) =
        node.FunctionId = "GetZarrInfo" || node.FunctionId = "GetNexusInfo"

    let private isChunkInfoNode (node: SavedNode) =
        node.FunctionId = "GetChunkInfo"

    let private isReadChunkInfoNode (node: SavedNode) =
        false

    let private quantileFieldExpression bindingName portIndex =
        if portIndex >= 0 && portIndex < 5 then Some $"{bindingName}[{portIndex}]" else None

    let private histogramFieldExpression (nodesById: Map<string, SavedNode>) nodeId bindingName portIndex =
        match nodesById |> Map.tryFind nodeId with
        | Some node when node.FunctionId = "EstimateHistogram" ->
            match portIndex with
            | 0 -> Some $"{bindingName}.Histogram"
            | 1 -> Some $"{bindingName}.Samples"
            | 2 -> Some $"{bindingName}.CdfHalfWidth"
            | 3 -> Some $"{bindingName}.HoldoutMaxCdfDelta"
            | _ -> None
        | _ ->
            if portIndex = 0 then Some bindingName else None

    let private parameterExpression (graph: SavedGraph) (nodesById: Map<string, SavedNode>) (scalarNamesByNodeId: Map<string, string>) (statsNamesByNodeId: Map<string, string>) (translationTableNamesByNodeId: Map<string, string>) (histogramNamesByNodeId: Map<string, string>) (quantileNamesByNodeId: Map<string, string>) (imageInfoNamesByNodeId: Map<string, string>) (chunkInfoNamesByNodeId: Map<string, string>) (serialGeometryNamesByNodeId: Map<string, string>) (node: SavedNode) parameterIndex key =
        let linkedExpression =
            graph.Edges
            |> Seq.tryFind (fun edge ->
                edge.ToNode = node.Id
                && edge.ToKind = "parameterInput"
                && edge.ToPort = parameterIndex)
            |> Option.bind (fun edge ->
                match edge.FromKind with
                | "scalarOutput" ->
                    scalarNamesByNodeId |> Map.tryFind edge.FromNode
                | "reducerOutput" ->
                    match statsNamesByNodeId |> Map.tryFind edge.FromNode with
                    | Some statsName ->
                        match nodesById |> Map.tryFind edge.FromNode with
                        | Some reducerNode when isSingleValueReducerNode reducerNode && edge.FromPort = 0 -> Some statsName
                        | Some reducerNode when reducerNode.FunctionId = "ComputeStats" && edge.FromPort = 0 ->
                            Some statsName
                        | Some reducerNode when reducerNode.FunctionId = "ObjectSizeStats" && edge.FromPort = 0 ->
                            Some statsName
                        | Some reducerNode when reducerNode.FunctionId = "Expand" ->
                            graph.Edges
                            |> Array.tryFind (fun inputEdge ->
                                inputEdge.ToNode = reducerNode.Id
                                && inputEdge.ToKind <> "parameterInput"
                                && inputEdge.ToPort = 0)
                            |> Option.bind (fun inputEdge -> nodesById |> Map.tryFind inputEdge.FromNode)
                            |> Option.bind (fun sourceNode ->
                                if sourceNode.FunctionId = "ObjectSizeStats" then
                                    objectSizeStatsFieldName edge.FromPort
                                else
                                    computeStatsFieldName edge.FromPort)
                            |> Option.map (fun fieldName -> $"{statsName}.{fieldName}")
                        | _ -> None
                    | _ ->
                        match translationTableNamesByNodeId |> Map.tryFind edge.FromNode with
                        | Some name -> Some name
                        | None ->
                            match histogramNamesByNodeId |> Map.tryFind edge.FromNode with
                            | Some name -> histogramFieldExpression nodesById edge.FromNode name edge.FromPort
                            | None ->
                                match quantileNamesByNodeId |> Map.tryFind edge.FromNode with
                                | Some name -> quantileFieldExpression name edge.FromPort
                                | None ->
                                    imageInfoNamesByNodeId
                                    |> Map.tryFind edge.FromNode
                                    |> Option.bind (fun name ->
                                        match nodesById |> Map.tryFind edge.FromNode with
                                        | Some sourceNode when sourceNode.FunctionId = "Expand" ->
                                            imageInfoFieldExpression name edge.FromPort
                                        | Some _ when edge.FromPort = 0 ->
                                            Some name
                                        | _ ->
                                            None)
                                    |> Option.orElseWith (fun () ->
                                        chunkInfoNamesByNodeId
                                        |> Map.tryFind edge.FromNode
                                        |> Option.bind (fun name ->
                                            match nodesById |> Map.tryFind edge.FromNode with
                                            | Some sourceNode when sourceNode.FunctionId = "Expand" ->
                                                chunkInfoFieldExpression name edge.FromPort
                                            | Some sourceNode when isChunkInfoNode sourceNode ->
                                                chunkInfoFieldExpression name edge.FromPort
                                            | Some _ when edge.FromPort = 0 ->
                                                Some name
                                            | _ ->
                                                None))
                                    |> Option.orElseWith (fun () -> serialGeometryNamesByNodeId |> Map.tryFind edge.FromNode)
                | "output" ->
                    imageInfoNamesByNodeId
                    |> Map.tryFind edge.FromNode
                    |> Option.orElseWith (fun () -> chunkInfoNamesByNodeId |> Map.tryFind edge.FromNode)
                    |> Option.orElseWith (fun () -> serialGeometryNamesByNodeId |> Map.tryFind edge.FromNode)
                | _ ->
                    None)

        match linkedExpression with
        | Some expression ->
            { Value = expression
              IsLinked = true }
        | None ->
            { Value = savedParamValue key node
              IsLinked = false }

    let private scalarBinding (graph: SavedGraph) (nodesById: Map<string, SavedNode>) (scalarNamesByNodeId: Map<string, string>) (statsNamesByNodeId: Map<string, string>) (translationTableNamesByNodeId: Map<string, string>) (histogramNamesByNodeId: Map<string, string>) (quantileNamesByNodeId: Map<string, string>) (imageInfoNamesByNodeId: Map<string, string>) (chunkInfoNamesByNodeId: Map<string, string>) (serialGeometryNamesByNodeId: Map<string, string>) (node: SavedNode) =
        let name = scalarNamesByNodeId |> Map.find node.Id

        let value =
            match node.FunctionId with
            | "ScalarOp" ->
                let parameterExpression key =
                    node.Parameters
                    |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
                    |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node index key)
                    |> Option.defaultValue { Value = ""; IsLinked = false }

                let operation =
                    match savedParamValue "operation" node with
                    | "+"
                    | "-"
                    | "*"
                    | "/" as operation -> operation
                    | _ -> "*"

                let scalarType =
                    savedParamValue "type" node
                    |> BasicType.tryParse
                    |> Option.defaultValue (BasicType.Numeric Float64)

                let typedParameter key =
                    let expression = parameterExpression key
                    if expression.IsLinked then expression.Value else literalValue scalarType expression.Value

                let left = typedParameter "a"
                let right = typedParameter "b"
                $"({left} {operation} {right})"
            | "ScalarFunction" ->
                let parameterExpression =
                    node.Parameters
                    |> Seq.tryFindIndex (fun parameter -> parameter.Key = "a")
                    |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node index "a")
                    |> Option.defaultValue { Value = ""; IsLinked = false }

                let argument =
                    if parameterExpression.IsLinked then
                        parameterExpression.Value
                    else
                        literalValue (BasicType.Numeric Float64) parameterExpression.Value

                scalarFunctionExpression (unaryImageFunctionName node) argument
            | "RandomRigidTransform" ->
                let parameterExpression key =
                    node.Parameters
                    |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
                    |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node index key)
                    |> Option.defaultValue { Value = ""; IsLinked = false }

                let seed = (parameterExpression "seed").Value
                let width = (parameterExpression "width").Value
                let height = (parameterExpression "height").Value
                let depth = (parameterExpression "depth").Value
                let maxTranslation = (parameterExpression "maxTranslation").Value
                $"randomRigidTransform (int {seed}) (uint {width}) (uint {height}) (uint {depth}) (float {maxTranslation})"
            | "OtsuThresholdFromHistogram"
            | "MomentsThresholdFromHistogram" ->
                let functionName =
                    if node.FunctionId = "OtsuThresholdFromHistogram" then
                        "otsuThresholdFromHistogram"
                    else
                        "momentsThresholdFromHistogram"

                let histogram =
                    node.Parameters
                    |> Seq.tryFindIndex (fun parameter -> parameter.Key = "histogram")
                    |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node index "histogram")
                    |> Option.defaultValue { Value = savedParamValue "histogram" node; IsLinked = false }

                $"{functionName} {histogram.Value}"
            | _ ->
                scalarValueLiteral node

        name, $"let {name} = {value}"

    let private chunkInfoBinding (graph: SavedGraph) (nodesById: Map<string, SavedNode>) (scalarNamesByNodeId: Map<string, string>) (statsNamesByNodeId: Map<string, string>) (translationTableNamesByNodeId: Map<string, string>) (histogramNamesByNodeId: Map<string, string>) (quantileNamesByNodeId: Map<string, string>) (imageInfoNamesByNodeId: Map<string, string>) (chunkInfoNamesByNodeId: Map<string, string>) (serialGeometryNamesByNodeId: Map<string, string>) (node: SavedNode) =
        let parameterExpression key =
            node.Parameters
            |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
            |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node index key)
            |> Option.defaultValue { Value = ""; IsLinked = false }

        let stringArgument key =
            let expression = parameterExpression key
            if expression.IsLinked then expression.Value else quote expression.Value

        let name = chunkInfoNamesByNodeId |> Map.find node.Id
        let input = stringArgument "input"

        let suffix = stringArgument "suffix"
        name, $"let {name} = getChunkInfo {input} {suffix}"

    let private savedElementLine (graph: SavedGraph) (nodesById: Map<string, SavedNode>) (scalarNamesByNodeId: Map<string, string>) (statsNamesByNodeId: Map<string, string>) (translationTableNamesByNodeId: Map<string, string>) (histogramNamesByNodeId: Map<string, string>) (quantileNamesByNodeId: Map<string, string>) (imageInfoNamesByNodeId: Map<string, string>) (chunkInfoNamesByNodeId: Map<string, string>) (serialGeometryNamesByNodeId: Map<string, string>) (node: SavedNode) =
        let parameterExpressionForNode (targetNode: SavedNode) key =
            targetNode.Parameters
            |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
            |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId targetNode index key)
            |> Option.defaultValue { Value = ""; IsLinked = false }

        let parameterExpression key =
            parameterExpressionForNode node key

        let dynamicParameterTypeForNode (targetNode: SavedNode) key =
            let configuredNumericType () =
                savedParamValue "type" targetNode
                |> NumericType.tryParse
                |> Option.map BasicType.Numeric

            match targetNode.FunctionId, key with
            | "ScalarOp", ("a" | "b") ->
                configuredNumericType ()
            | "ScalarFunction", "a" ->
                Some(BasicType.Numeric Float64)
            | id, "value" when isScalarImageFunction id ->
                configuredNumericType ()
            | _ ->
                BuiltInCatalog.tryFind targetNode.FunctionId
                |> Option.bind (fun definition ->
                    definition.Parameters
                    |> Seq.tryFind (fun parameter -> parameter.Key = key)
                    |> Option.map _.Type)

        let parameterValueForNode targetNode key =
            let expression = parameterExpressionForNode targetNode key

            match dynamicParameterTypeForNode targetNode key with
            | Some(BasicType.Numeric Float32) when expression.IsLinked ->
                $"float32 ({expression.Value})"
            | Some(BasicType.Numeric numericType) when not expression.IsLinked ->
                numericLiteral numericType expression.Value
            | Some BasicType.Bool when not expression.IsLinked ->
                expression.Value.Trim().ToLowerInvariant()
            | Some BasicType.Unit when not expression.IsLinked ->
                "()"
            | Some BasicType.Map when not expression.IsLinked ->
                expression.Value
            | Some BasicType.String when not expression.IsLinked ->
                expression.Value
            | _ ->
                expression.Value

        let parameterValue key =
            parameterValueForNode node key

        let parameterValueOrDefault key fallback =
            let value = parameterValue key
            if String.IsNullOrWhiteSpace value then fallback else value

        let isNoneValue (value: string) =
            let trimmed = value.Trim()
            String.IsNullOrWhiteSpace trimmed
            || String.Equals(trimmed, "None", StringComparison.OrdinalIgnoreCase)

        let uintWindowOrDefault defaultValue value =
            if isNoneValue value then
                defaultValue
            else
                let trimmed = value.Trim()
                let digits = trimmed.TrimEnd('u', 'U')

                if digits.Length > 0 && digits |> Seq.forall Char.IsDigit then
                    $"{digits}u"
                else
                    trimmed

        let rec inferredImagePixelType visited (targetNode: SavedNode) inputPort =
            if visited |> Set.contains targetNode.Id then
                None
            else
                let visited = visited |> Set.add targetNode.Id

                graph.Edges
                |> Array.tryFind (fun edge ->
                    edge.ToNode = targetNode.Id
                    && edge.ToKind <> "parameterInput"
                    && edge.ToPort = inputPort)
                |> Option.bind (fun edge ->
                    nodesById
                    |> Map.tryFind edge.FromNode
                    |> Option.bind (fun sourceNode ->
                        match sourceNode.FunctionId, edge.FromPort with
                        | "SerialEstTrans", 0 ->
                            inferredImagePixelType visited sourceNode 0
                        | "Cast", 0 ->
                            Some(pixelTypeNameFromParameter "targetType" "Float32" sourceNode)
                        | _ ->
                            let configured = savedParamValue "type" sourceNode
                            if String.IsNullOrWhiteSpace configured then None else Some(pixelTypeNameFromSuffix configured)))

        let rec inferredVectorPixelType visited (targetNode: SavedNode) inputPort =
            if visited |> Set.contains targetNode.Id then
                None
            else
                let visited = visited |> Set.add targetNode.Id

                graph.Edges
                |> Array.tryFind (fun edge ->
                    edge.ToNode = targetNode.Id
                    && edge.ToKind <> "parameterInput"
                    && edge.ToPort = inputPort)
                |> Option.bind (fun edge ->
                    nodesById
                    |> Map.tryFind edge.FromNode
                    |> Option.bind (fun sourceNode ->
                        match sourceNode.FunctionId, edge.FromPort with
                        | "Gradient", 0 -> Some "float32"
                        | "StructureTensor", 0 -> Some "float32"
                        | "PCA", port when port >= 0 && port <= 8 -> Some "float32"
                        | "VectorRange", 0
                        | "AppendVectorElement", 0
                        | "VectorMapElements", 0 ->
                            inferredVectorPixelType visited sourceNode 0
                        | "ToVectorImage", 0
                        | "ColorToVector3", 0
                        | "VectorCross3D", 0 -> Some "float"
                        | _ -> None))

        let pipelineVectorPixelType defaultType =
            inferredVectorPixelType Set.empty node 0
            |> Option.defaultValue defaultType

        let inputSourcePortType inputPort =
            graph.Edges
            |> Array.tryFind (fun edge ->
                edge.ToNode = node.Id
                && edge.ToKind <> "parameterInput"
                && edge.ToPort = inputPort)
            |> Option.bind (fun edge ->
                nodesById
                |> Map.tryFind edge.FromNode
                |> Option.bind (fun sourceNode ->
                    BuiltInCatalog.tryFind sourceNode.FunctionId
                    |> Option.bind (fun definition ->
                        definition.Outputs
                        |> List.tryItem edge.FromPort
                        |> Option.map _.Type)))

        let pipelinePixelType defaultType =
            inferredImagePixelType Set.empty node 0
            |> Option.defaultValue (pixelTypeNameFromParameter "type" defaultType node)

        let quotedParameter key =
            let expression = parameterExpression key
            if expression.IsLinked then expression.Value else quote expression.Value

        let quotedParameterOrDefault key fallback =
            let expression = parameterExpression key

            if expression.IsLinked then
                expression.Value
            else
                let value =
                    if String.IsNullOrWhiteSpace expression.Value then fallback else expression.Value

                quote value

        let stringParameter key =
            let expression = parameterExpression key
            if expression.IsLinked then $"(string {expression.Value})" else quote expression.Value

        let printInputNameForNode (printNode: SavedNode) key index =
            let configuredName =
                printNode.Parameters
                |> Seq.tryFind (fun parameter -> parameter.Key = key)
                |> Option.map _.Value
                |> Option.map _.Trim()
                |> Option.filter (String.IsNullOrWhiteSpace >> not)

            configuredName
            |> Option.orElseWith (fun () ->
                graph.Edges
                |> Seq.tryFind (fun edge ->
                    edge.ToNode = printNode.Id
                    && edge.ToKind = "parameterInput"
                    && edge.ToPort = index)
                |> Option.bind (fun edge ->
                    match edge.FromKind with
                    | "reducerOutput" ->
                        nodesById
                        |> Map.tryFind edge.FromNode
                        |> Option.bind (fun sourceNode ->
                            if sourceNode.FunctionId = "Expand" then
                                graph.Edges
                                |> Array.tryFind (fun inputEdge ->
                                    inputEdge.ToNode = sourceNode.Id
                                    && inputEdge.ToKind <> "parameterInput"
                                    && inputEdge.ToPort = 0)
                                |> Option.bind (fun inputEdge -> nodesById |> Map.tryFind inputEdge.FromNode)
                                |> Option.map (fun recordSource ->
                                    if recordSource.FunctionId = "ComputeStats" then
                                        PortType.Custom "ImageStats"
                                    elif recordSource.FunctionId = "ObjectSizeStats" then
                                        PortType.Custom "ObjectSizeStats"
                                    elif isImageInfoNode recordSource || isWriteImageInfoNode recordSource || isReadImageInfoNode recordSource then
                                        PortType.Custom "ImageInfo"
                                    elif isChunkInfoNode recordSource || isReadChunkInfoNode recordSource then
                                        PortType.Custom "ChunkInfo"
                                    else
                                        PortType.Custom "Record")
                                |> Option.bind (fun recordType ->
                                    BuiltInCatalog.expandOutputsFor recordType
                                    |> List.tryItem edge.FromPort
                                    |> Option.map _.Name)
                            else
                                BuiltInCatalog.tryFind sourceNode.FunctionId
                                |> Option.bind (fun definition ->
                                    definition.Outputs
                                    |> List.tryItem edge.FromPort
                                    |> Option.map _.Name))
                    | "scalarOutput" ->
                        nodesById
                        |> Map.tryFind edge.FromNode
                        |> Option.map (fun sourceNode ->
                                if sourceNode.FunctionId = "Tap" then
                                    "I"
                                elif sourceNode.FunctionId = "Scalar" then
                                    savedParamValue "type" sourceNode
                                else
                                    sourceNode.FunctionId)
                    | _ ->
                        None))
            |> Option.map (fun name ->
                let index = name.IndexOf(':')

                if index > 0 then
                    name.Substring(0, index).Trim()
                else
                    name.Trim())
            |> Option.filter (String.IsNullOrWhiteSpace >> not)
            |> Option.defaultValue key

        let printInputName key index =
            printInputNameForNode node key index

        let interpolatedStringExpression (format: string) (inputs: (string * string * string) list) =
            let byName =
                inputs
                |> List.collect (fun (key, name, expression) -> [ key, expression; name, expression ])
                |> Map.ofList

            let allowedNames =
                byName
                |> Map.keys
                |> Set.ofSeq

            let invalidFormat reason =
                failwith $"Invalid Print format: {reason}"

            let validateFormat () =
                if format.Contains("\"", StringComparison.Ordinal) then
                    invalidFormat "double quotes are not allowed."

                let mutable offset = 0

                while offset < format.Length do
                    match format[offset] with
                    | '{' ->
                        let closeIndex = format.IndexOf('}', offset + 1)

                        if closeIndex < 0 then
                            invalidFormat "each '{' must close a placeholder."

                        let name = format.Substring(offset + 1, closeIndex - offset - 1).Trim()

                        if String.IsNullOrWhiteSpace name then
                            invalidFormat "empty placeholders are not allowed."

                        if name.Contains("{", StringComparison.Ordinal) then
                            invalidFormat "nested placeholders are not allowed."

                        if not (allowedNames |> Set.contains name) then
                            let allowed =
                                allowedNames
                                |> Seq.sort
                                |> String.concat ", "

                            invalidFormat $"'{name}' is not a linked Print input. Allowed placeholders are: {allowed}."

                        offset <- closeIndex + 1
                    | '}' ->
                        invalidFormat "each '}' must belong to a placeholder."
                    | _ ->
                        offset <- offset + 1

            validateFormat()

            let escapeText (value: string) =
                value
                    .Replace("\\\\", "\u0000")
                    .Replace("\\n", "\n")
                    .Replace("\\r", "\r")
                    .Replace("\\t", "\t")
                    .Replace("\\\"", "\"")
                    .Replace("\u0000", "\\")
                    .Replace("\\", "\\\\")
                    .Replace("\"", "\\\"")
                    .Replace("{", "{{")
                    .Replace("}", "}}")

            let builder = StringBuilder()
            let mutable offset = 0

            for m in Regex.Matches(format, @"\{([^{}]+)\}") do
                if m.Index > offset then
                    builder.Append(escapeText (format.Substring(offset, m.Index - offset))) |> ignore

                let name = m.Groups[1].Value.Trim()

                match byName |> Map.tryFind name with
                | Some expression ->
                    builder.Append("{").Append(expression).Append("}") |> ignore
                | None -> invalidFormat $"'{name}' is not a linked Print input."

                offset <- m.Index + m.Length

            if offset < format.Length then
                builder.Append(escapeText (format.Substring offset)) |> ignore

            "$\"" + builder.ToString() + "\""

        let histogramStageExpression () =
            let hasParameter key =
                node.Parameters
                |> Seq.exists (fun parameter -> parameter.Key = key)

            if [ "firstLeftEdge"; "lastLeftEdge"; "bins" ] |> List.forall hasParameter then
                let firstLeftEdge = parameterValue "firstLeftEdge"
                let lastLeftEdge = parameterValue "lastLeftEdge"
                let bins = parameterValue "bins"
                let pixelType = pipelinePixelType "Float32"
                $">=> imageHistogramFixedBins<{pixelType}> ({firstLeftEdge}) ({lastLeftEdge}) ({bins})"
            else
                let pixelType = pipelinePixelType "Float32"
                $">=> imageHistogram<{pixelType}> ()"

        match node.FunctionId with
        | "Empty" ->
            let availableMemory = parameterValue "availableMemory"
            "|> empty" |> sourcePrefix availableMemory
        | "Zero" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            $"|> zero<{pixelType}> {width} {height} {depth}" |> sourcePrefix availableMemory
        | "PolygonMask" ->
            let availableMemory = parameterValue "availableMemory"
            let width = parameterValue "width"
            let height = parameterValue "height"
            let polygon = parameterValue "polygon"
            let suffix = safeIdentifier node.Id
            let maskName = $"polygonMask_{suffix}"

            String.concat Environment.NewLine
                [ $"let {maskName} = polygonMask {width} {height} {polygon}"
                  $"debug 1u (optimizerEnabled ()) {uint64Literal availableMemory}"
                  $"|> repeat {maskName} 1u" ]
        | "CoordinateX" ->
            let availableMemory = parameterValue "availableMemory"
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            $"|> coordinateX<float32> {width} {height} {depth}" |> sourcePrefix availableMemory
        | "CoordinateY" ->
            let availableMemory = parameterValue "availableMemory"
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            $"|> coordinateY<float32> {width} {height} {depth}" |> sourcePrefix availableMemory
        | "CoordinateZ" ->
            let availableMemory = parameterValue "availableMemory"
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            $"|> coordinateZ<float32> {width} {height} {depth}" |> sourcePrefix availableMemory
        | "NormalNoise" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            let mean = parameterValue "mean"
            let std = parameterValue "std"
            $"|> zero<{pixelType}> {width} {height} {depth}{Environment.NewLine}>=> addNormalNoise<{pixelType}> {mean} {std}" |> sourcePrefix availableMemory
        | "SaltAndPepperNoise" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            let probability = parameterValue "probability"
            $"|> zero<{pixelType}> {width} {height} {depth}{Environment.NewLine}>=> addSaltAndPepperNoise<{pixelType}> {probability}" |> sourcePrefix availableMemory
        | "ShotNoise" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            let scale = parameterValue "scale"
            $"|> zero<{pixelType}> {width} {height} {depth}{Environment.NewLine}>=> addShotNoise<{pixelType}> {scale}" |> sourcePrefix availableMemory
        | "SpeckleNoise" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            let std = parameterValue "std"
            $"|> speckleNoise<{pixelType}> {width} {height} {depth} {std}" |> sourcePrefix availableMemory
        | "CreateByEuler2DTransform" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "UInt8" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            let polygon = parameterValue "polygon"
            let transform = savedParamValue "transform" node
            let suffix = safeIdentifier node.Id
            let maskName = $"eulerMask_{suffix}"
            let transformName = $"eulerTransform_{suffix}"

            String.concat Environment.NewLine
                [ $"let {maskName} = polygonMask {width} {height} {polygon}"
                  $"let {transformName} = euler2DTransformPath {width} {height} {depth} {quote transform}"
                  $"debug 1u (optimizerEnabled ()) {uint64Literal availableMemory}"
                  $"|> repeat {maskName} 1u"
                  $">=> cast<uint8,{pixelType}>"
                  $">=> createByEuler2DTransform {depth} {transformName}" ]
        | "ReadRandom" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let format = savedParamValue "format" node
            let depth = parameterValue "depth"
            let input = quotedParameter "input"
            let suffix = quotedParameterOrDefault "suffix" ".tiff"
            match format with
            | "Volume file" ->
                let volumeInput = $"(volumeFilePath {input} {suffix})"
                $"|> readVolumeRandom<{pixelType}> {depth} {volumeInput}" |> sourcePrefix availableMemory
            | "OME-Zarr" ->
                let multiscaleIndex = parameterValue "multiscaleIndex"
                let datasetIndex = parameterValue "datasetIndex"
                let timepoint = parameterValue "timepoint"
                let channel = parameterValue "channel"
                let maxParallelChunks = parameterValue "maxParallelChunks"
                $"|> readZarrRandom<{pixelType}> {depth} {input} {multiscaleIndex} {datasetIndex} {timepoint} {channel} {maxParallelChunks}" |> sourcePrefix availableMemory
            | "NeXus/HDF5" ->
                let datasetPath = quotedParameter "datasetPath"
                let frameAxis = parameterValue "frameAxis"
                let yAxis = parameterValue "yAxis"
                let xAxis = parameterValue "xAxis"
                $"|> readNexusRandom<{pixelType}> {depth} {input} {datasetPath} {frameAxis} {yAxis} {xAxis}" |> sourcePrefix availableMemory
            | _ ->
                $"|> readRandom<{pixelType}> {depth} {input} {suffix}" |> sourcePrefix availableMemory
        | "EstimateHistogram" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let slices = parameterValue "slices"
            let input = quotedParameter "input"
            let suffix = quotedParameter "suffix"
            $"|> readRandom<{pixelType}> {slices} {input} {suffix}{Environment.NewLine}>=> imageHistogram<{pixelType}> ()" |> sourcePrefix availableMemory
        | "ReadRange" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let format = savedParamValue "format" node
            let first = parameterValue "first"
            let step = parameterValue "step"
            let last = parameterValue "last"
            let input = quotedParameter "input"
            let suffix = quotedParameterOrDefault "suffix" ".tiff"
            match format with
            | "Volume file" ->
                let volumeInput = $"(volumeFilePath {input} {suffix})"
                $"|> readVolumeRange<{pixelType}> {first} {step} {last} {volumeInput}" |> sourcePrefix availableMemory
            | "OME-Zarr" ->
                let multiscaleIndex = parameterValue "multiscaleIndex"
                let datasetIndex = parameterValue "datasetIndex"
                let timepoint = parameterValue "timepoint"
                let channel = parameterValue "channel"
                let maxParallelChunks = parameterValue "maxParallelChunks"
                $"|> readZarrRange<{pixelType}> {first} {step} {last} {input} {multiscaleIndex} {datasetIndex} {timepoint} {channel} {maxParallelChunks}" |> sourcePrefix availableMemory
            | "NeXus/HDF5" ->
                let datasetPath = quotedParameter "datasetPath"
                let frameAxis = parameterValue "frameAxis"
                let yAxis = parameterValue "yAxis"
                let xAxis = parameterValue "xAxis"
                $"|> readNexusRange<{pixelType}> {first} {step} {last} {input} {datasetPath} {frameAxis} {yAxis} {xAxis}" |> sourcePrefix availableMemory
            | _ ->
                $"|> readRange<{pixelType}> {first} {step} {last} {input} {suffix}" |> sourcePrefix availableMemory
        | "ReadPointSet" ->
            let availableMemory = parameterValue "availableMemory"
            let input = quotedParameter "input"
            $"|> readPointSet {input}" |> sourcePrefix availableMemory
        | "Read" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let format = savedParamValue "format" node
            let input = quotedParameter "input"
            let suffix = quotedParameterOrDefault "suffix" ".tiff"
            let thickDepth = parameterValue "thickDepth"
            let multiscaleIndex = parameterValue "multiscaleIndex"
            let datasetIndex = parameterValue "datasetIndex"
            let timepoint = parameterValue "timepoint"
            let channel = parameterValue "channel"
            let maxParallelChunks = parameterValue "maxParallelChunks"
            let datasetPath = quotedParameter "datasetPath"
            let frameAxis = parameterValue "frameAxis"
            let yAxis = parameterValue "yAxis"
            let xAxis = parameterValue "xAxis"
            match format with
            | "Volume file" ->
                let volumeInput = $"(volumeFilePath {input} {suffix})"
                $"|> readVolume<{pixelType}> {volumeInput}" |> sourcePrefix availableMemory
            | "OME-Zarr" ->
                $"|> readZarrThick<{pixelType}> 0u System.UInt32.MaxValue {thickDepth} {input} {multiscaleIndex} {datasetIndex} {timepoint} {channel} {maxParallelChunks}" |> sourcePrefix availableMemory
            | "NeXus/HDF5" ->
                $"|> readNexusSlab<{pixelType}> {input} {datasetPath} {frameAxis} {yAxis} {xAxis}" |> sourcePrefix availableMemory
            | _ ->
                $"|> read<{pixelType}> {input} {suffix}" |> sourcePrefix availableMemory
        | "Resize" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            let interpolation = quotedParameter "interpolation"
            $"|> resize<{pixelType}> {width} {height} {depth} {interpolation}"
        | "Resample" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let factorX = parameterValue "factorX"
            let factorY = parameterValue "factorY"
            let factorZ = parameterValue "factorZ"
            let interpolation = quotedParameter "interpolation"
            $"|> resample<{pixelType}> {factorX} {factorY} {factorZ} {interpolation}"
        | "Repeat" ->
            let depth = parameterValue "depth"
            $">=> repeatStage {depth}"
        | "WriteChunks" ->
            let output = quotedParameter "output"
            let suffix = quotedParameter "suffix"
            let chunkX = parameterValue "chunkX"
            let chunkY = parameterValue "chunkY"
            let chunkZ = parameterValue "chunkZ"
            $">=> writeChunks {output} {suffix} {chunkX} {chunkY} {chunkZ}"
        | "Write" ->
            let format = savedParamValue "format" node
            let output = quotedParameter "output"
            let suffix = quotedParameterOrDefault "suffix" ".tiff"
            match format with
            | "Volume file" ->
                $">=> writeVolume (volumeFilePath {output} {suffix})"
            | "OME-Zarr" ->
                let name = quotedParameter "name"
                let depth = parameterValue "depth"
                let chunkX = parameterValue "chunkX"
                let chunkY = parameterValue "chunkY"
                let chunkZ = parameterValue "chunkZ"
                let physicalSizeX = parameterValue "physicalSizeX"
                let physicalSizeY = parameterValue "physicalSizeY"
                let physicalSizeZ = parameterValue "physicalSizeZ"
                let maxConcurrentWrites = parameterValue "maxConcurrentWrites"
                $">=> writeZarrThick {output} {name} {depth} {chunkX} {chunkY} {chunkZ} {physicalSizeX} {physicalSizeY} {physicalSizeZ} {maxConcurrentWrites}"
            | "NeXus/HDF5" ->
                let datasetPath = quotedParameter "datasetPath"
                let depth = parameterValue "depth"
                let chunkX = parameterValue "chunkX"
                let chunkY = parameterValue "chunkY"
                let chunkZ = parameterValue "chunkZ"
                let frameAxis = parameterValue "frameAxis"
                let yAxis = parameterValue "yAxis"
                let xAxis = parameterValue "xAxis"
                $">=> writeNexus {output} {datasetPath} {depth} {chunkX} {chunkY} {chunkZ} {frameAxis} {yAxis} {xAxis}"
            | _ ->
                match inputSourcePortType 0 with
                | Some(PortType.Custom "ColorImage") -> $">=> writeColor {output} {suffix}"
                | _ -> $">=> write {output} {suffix}"
        | "WriteMesh" ->
            let output = quotedParameter "output"
            let format = quotedParameter "format"
            $">=> writeMesh {output} {format}"
        | "WritePointSet" ->
            let output = quotedParameter "output"
            let suffix = quotedParameter "suffix"
            $">=> writePointSet {output} {suffix}"
        | "WriteMatrix" ->
            let output = quotedParameter "output"
            let suffix = quotedParameter "suffix"
            $">=> writeMatrix {output} {suffix}"
        | "WriteCSV" ->
            let output = quotedParameter "output"
            match savedParamValue "dataKind" node with
            | "Matrix" -> $">=> writeCSVMatrix {output}"
            | "Histogram" -> $">=> histogramCounts >=> writeCSVHistogram {output}"
            | _ -> $">=> writeCSVPointSet {output}"
        | "WriteThrough" ->
            let output = quotedParameter "output"
            let suffix = quotedParameter "suffix"
            $">=> writeThrough {output} {suffix}"
        | "Ignore" ->
            ""
        | "Tap" ->
            let tapPrintNode =
                graph.Edges
                |> Array.choose (fun edge ->
                    if edge.FromNode = node.Id
                       && edge.FromKind = "scalarOutput"
                       && edge.ToKind = "parameterInput" then
                        nodesById |> Map.tryFind edge.ToNode
                    else
                        None)
                |> Array.filter (fun candidate -> candidate.FunctionId = "Print")
                |> Array.distinctBy _.Id
                |> Array.tryHead

            match tapPrintNode with
            | Some printNode ->
                let tapInputName =
                    printNode.Parameters
                    |> Seq.mapi (fun index parameter -> index, parameter)
                    |> Seq.tryPick (fun (parameterIndex, parameter) ->
                        if parameter.Key.StartsWith("input", StringComparison.Ordinal)
                           && parameter.UseInput
                           && graph.Edges
                              |> Array.exists (fun edge ->
                                  edge.ToNode = printNode.Id
                                  && edge.ToKind = "parameterInput"
                                  && edge.ToPort = parameterIndex
                                  && edge.FromNode = node.Id
                                  && edge.FromKind = "scalarOutput") then
                            Some(printInputNameForNode printNode parameter.Key parameterIndex)
                        else
                            None)
                    |> Option.defaultValue "I"

                let tapVariable = safeIdentifier tapInputName

                let inputs =
                    printNode.Parameters
                    |> Seq.mapi (fun index parameter -> index, parameter)
                    |> Seq.choose (fun (parameterIndex, parameter) ->
                        if parameter.Key.StartsWith("input", StringComparison.Ordinal) && parameter.UseInput then
                            let expression =
                                graph.Edges
                                |> Array.tryFind (fun edge ->
                                    edge.ToNode = printNode.Id
                                    && edge.ToKind = "parameterInput"
                                    && edge.ToPort = parameterIndex)
                                |> Option.bind (fun edge ->
                                    if edge.FromNode = node.Id && edge.FromKind = "scalarOutput" then
                                        Some tapVariable
                                    else
                                        Some(parameterValueForNode printNode parameter.Key))
                                |> Option.defaultValue (parameterValueForNode printNode parameter.Key)

                            Some(parameter.Key, printInputNameForNode printNode parameter.Key parameterIndex, expression)
                        else
                            None)
                    |> Seq.toList

                let format = interpolatedStringExpression (savedParamValue "format" printNode) inputs
                $">=> tapIt (fun {tapVariable} -> {format})"
            | None ->
                let label = stringParameter "label"
                $">=> tap {label}"
        | "Print" ->
            let inputs =
                node.Parameters
                |> Seq.mapi (fun index parameter -> index, parameter)
                |> Seq.choose (fun (parameterIndex, parameter) ->
                    if parameter.Key.StartsWith("input", StringComparison.Ordinal) && parameter.UseInput then
                        Some(parameter.Key, printInputName parameter.Key parameterIndex, parameterValue parameter.Key)
                    else
                        None)
                |> Seq.toList


            let format = interpolatedStringExpression (savedParamValue "format" node) inputs

            $"printfn {format}"
        | "ImHistogram" ->
            let histogramStage = histogramStageExpression ()
            let title = quotedParameter "title"
            let xAxis = quotedParameter "xAxis"
            let yAxis = quotedParameter "yAxis"
            $"{histogramStage} >=> histogram2pairs --> pairs2floats --> plot (showChartXYWithLabels \"Column\" {title} {xAxis} {yAxis})"
        | "ImHistogramData" ->
            histogramStageExpression ()
        | "Histogram" ->
            let binWidth = parameterValue "binWidth"
            $">=> histogram {binWidth}"
        | "Chart" ->
            let inputValue = parameterValue "input"
            let values =
                match graph.Edges |> Array.tryFind (fun edge -> edge.ToNode = node.Id && edge.ToKind = "parameterInput" && edge.ToPort = 1) with
                | Some edge ->
                    match nodesById |> Map.tryFind edge.FromNode with
                    | Some source when source.FunctionId = "EstimateHistogram" ->
                        $"{inputValue}.Counts"
                    | Some source when isHistogramDataNode source ->
                        $"{inputValue}.Counts"
                    | _ ->
                        inputValue
                | None ->
                    inputValue
            let kind = savedParamValue "kind" node
            let title = quotedParameter "title"
            let xAxis = quotedParameter "xAxis"
            let yAxis = quotedParameter "yAxis"
            $"showChartWithLabels {quote kind} {title} {xAxis} {yAxis} {values}"
        | "ShowImage" ->
            let pixelType = pipelinePixelType "Float32"
            let colorMap = quotedParameter "colorMap"
            let title = quotedParameter "title"
            let xAxis = quotedParameter "xAxis"
            let yAxis = quotedParameter "yAxis"
            $">=> show (showChunkWithLabels<{pixelType}> {colorMap} {title} {xAxis} {yAxis})"
        | "SumProjection" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let functionName = savedParamValue "function" node
            $">=> sumProjection<{pixelType}> {quote functionName}"
        | "UnaryImageFunction" ->
            $">=> {unaryImageStageFunctionName node}"
        | "ComplexRe" ->
            ">=> complex64Real"
        | "ComplexIm" ->
            ">=> complex64Imag"
        | "ComplexModulus" ->
            ">=> complex64Modulus"
        | "ComplexArg" ->
            ">=> complex64Argument"
        | "ComplexConjugate" ->
            ">=> complex64Conjugate"
        | "FFT" ->
            ">=> fft"
        | "InvFFT" ->
            ">=> invFft"
        | "ShiftFFT" ->
            ">=> fftShift3D"
        | "VectorElement" ->
            let componentId = parameterValue "component"
            let pixelType = pipelineVectorPixelType "float"
            $">=> vectorElement<{pixelType}> {componentId}"
        | "VectorRange" ->
            let firstComponent = parameterValue "firstComponent"
            let componentCount = parameterValue "componentCount"
            let pixelType = pipelineVectorPixelType "float"
            $">=> vectorRange<{pixelType}> {firstComponent} {componentCount}"
        | "Vector3ToColor" ->
            let inputMinimum = parameterValue "inputMinimum"
            let inputMaximum = parameterValue "inputMaximum"
            match pipelineVectorPixelType "float" with
            | "float32" -> $">=> vector3ToColorFloat32 (float32 ({inputMinimum})) (float32 ({inputMaximum}))"
            | _ -> $">=> vector3ToColor {inputMinimum} {inputMaximum}"
        | "ColorToVector3" ->
            let outputMinimum = parameterValue "outputMinimum"
            let outputMaximum = parameterValue "outputMaximum"
            $">=> colorToVector3 {outputMinimum} {outputMaximum}"
        | "VectorMapElements" ->
            let functionName = quotedParameter "function"
            $">=> vectorMapElements {functionName}"
        | "VectorAngleTo" ->
            let x = parameterValue "x"
            let y = parameterValue "y"
            let z = parameterValue "z"
            $">=> vectorAngleTo [ {x}; {y}; {z} ]"
        | "Gradient" ->
            let sigma = parameterValueOrDefault "sigma" "1.0"
            let radius = parameterValueOrDefault "radius" "7"
            let workers = parameterValueOrDefault "workers" "4"
            $">=> gradientVector {sigma} {radius} {workers}"
        | "StructureTensor" ->
            let sigma = parameterValueOrDefault "sigma" "1.0"
            let radius = parameterValueOrDefault "radius" "7"
            let rho = parameterValueOrDefault "rho" "2.0"
            let rhoRadius = parameterValueOrDefault "rhoRadius" "7"
            let workers = parameterValueOrDefault "workers" "4"
            $">=> structureTensor {sigma} {radius} {rho} {rhoRadius} {workers}"
        | "PCA" ->
            let components = parameterValue "components"
            $">=> pcaFloat32 {components}"
        | id when isScalarImageFunction id ->
            let value = parameterValue "value"
            $">=> {scalarImageFunctionName node |> Option.get} ({value})"
        | id when pairStageFunctionName node |> Option.isSome ->
            $"{pairCompositionOperator node} {pairStageFunctionName node |> Option.get}"
        | "SmoothWGauss" ->
            let sigma = parameterValue "sigma"
            let pixelType = pipelinePixelType "Float32"
            let radius =
                match parameterValue "windowSize" |> optionUInt with
                | "None" -> $"(int (System.Math.Ceiling(3.0 * ({sigma}))))"
                | someWindow -> $"(int ((Option.get {someWindow} - 1u) / 2u))"
            $">=> gaussianFilter<{pixelType}> {sigma} {radius} 1"
        | "Convolve" ->
            let kernel = parameterValue "kernel"
            let pixelType = pipelinePixelType "Float32"
            let workers =
                match parameterValue "windowSize" |> optionUInt with
                | "None" -> "4"
                | someWindow -> $"int (Option.get {someWindow})"
            $">=> convolveFixedKernel<{pixelType}> ({kernel}) {workers}"
        | "FiniteDiff" ->
            let direction = parameterValue "direction"
            let order = parameterValue "order"
            let pixelType = pipelinePixelType "Float32"
            match direction.Trim().TrimEnd('u', 'U').ToLowerInvariant() with
            | "0" | "x" -> $">=> finiteDiffX<{pixelType}> {order} 1"
            | "1" | "y" -> $">=> finiteDiffY<{pixelType}> {order} 1"
            | "2" | "z" -> $">=> finiteDiffZ<{pixelType}> {order} 1"
            | _ -> $">=> finiteDiffX<{pixelType}> {order} 1"
        | "Clamp" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let lower = parameterValue "lower"
            let upper = parameterValue "upper"
            $">=> clamp<{pixelType}> {lower} {upper}"
        | "ShiftScale" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let shift = parameterValue "shift"
            let scale = parameterValue "scale"
            $">=> shiftScale<{pixelType}> {shift} {scale}"
        | "IntensityStretch" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let inputMinimum = parameterValue "inputMinimum"
            let inputMaximum = parameterValue "inputMaximum"
            let outputMinimum = parameterValue "outputMinimum"
            let outputMaximum = parameterValue "outputMaximum"
            $">=> intensityWindow<{pixelType}> {inputMinimum} {inputMaximum} {outputMinimum} {outputMaximum}"
        | "HistogramEqualization" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let histogram = parameterValue "histogram"
            $">=> histogramEqualization<{pixelType}> ({histogram} :> obj)"
        | "CreatePadding" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let beforeX = parameterValue "beforeX"
            let afterX = parameterValue "afterX"
            let beforeY = parameterValue "beforeY"
            let afterY = parameterValue "afterY"
            let beforeZ = parameterValue "beforeZ"
            let afterZ = parameterValue "afterZ"
            let value = parameterValue "value" |> numericLiteralForPixelType pixelType
            $">=> pad<{pixelType}> {beforeX} {afterX} {beforeY} {afterY} {beforeZ} {afterZ} {value}"
        | "Crop" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let beforeX = parameterValue "beforeX"
            let afterX = parameterValue "afterX"
            let beforeY = parameterValue "beforeY"
            let afterY = parameterValue "afterY"
            let beforeZ = parameterValue "beforeZ"
            let afterZ = parameterValue "afterZ"
            $">=> crop<{pixelType}> {beforeX} {afterX} {beforeY} {afterY} {beforeZ} {afterZ}"
        | "SmoothWMedian" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let radius = parameterValue "radius"
            let workers =
                match parameterValue "windowSize" |> optionUInt with
                | "None" -> "4"
                | someWindow -> $"int (Option.get {someWindow})"
            let radius = $"(int ({radius}))"
            match pixelType with
            | "uint8" -> $">=> medianUInt8 {radius} {workers}"
            | "uint16" -> $">=> medianUInt16 {radius} {workers}"
            | "int" | "int32" -> $">=> medianInt32 {radius} {workers}"
            | _ -> $">=> medianFloat32 {radius} {workers}"
        | "SmoothWBilateral" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let domainSigma = parameterValue "domainSigma"
            let rangeSigma = parameterValue "rangeSigma"
            let windowSize = parameterValue "windowSize" |> optionUInt
            $">=> smoothWBilateral<{pixelType}> {domainSigma} {rangeSigma} {windowSize}"
        | "GradientMagnitude" ->
            let sigma = "1.0"
            $">=> gradientMagnitude {sigma} 3 1"
        | "SobelEdge" ->
            $">=> sobelMagnitude 1"
        | "Laplacian" ->
            let sigma = "1.0"
            $">=> laplacian {sigma} 3 1"
        | "GrayscaleErode" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let radius = parameterValue "radius"
            let windowSize = parameterValue "windowSize" |> optionUInt
            $">=> grayscaleErode<{pixelType}> {radius} {windowSize}"
        | "GrayscaleDilate" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let radius = parameterValue "radius"
            let windowSize = parameterValue "windowSize" |> optionUInt
            $">=> grayscaleDilate<{pixelType}> {radius} {windowSize}"
        | "GrayscaleOpening" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let radius = parameterValue "radius"
            let windowSize = parameterValue "windowSize" |> optionUInt
            $">=> grayscaleOpening<{pixelType}> {radius} {windowSize}"
        | "GrayscaleClosing" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let radius = parameterValue "radius"
            let windowSize = parameterValue "windowSize" |> optionUInt
            $">=> grayscaleClosing<{pixelType}> {radius} {windowSize}"
        | "WhiteTopHat" ->
            let radius = parameterValue "radius"
            let windowSize = savedParamValue "windowSize" node
            if String.Equals(windowSize.Trim(), "None", StringComparison.OrdinalIgnoreCase) then
                $">=> binaryWhiteTopHat {radius}"
            else
                let windowSize = numericLiteral Int32 windowSize
                $">=> binaryWhiteTopHatWindowed {radius} {windowSize}"
        | "BlackTopHat" ->
            let radius = parameterValue "radius"
            let windowSize = savedParamValue "windowSize" node
            if String.Equals(windowSize.Trim(), "None", StringComparison.OrdinalIgnoreCase) then
                $">=> binaryBlackTopHat {radius}"
            else
                let windowSize = numericLiteral Int32 windowSize
                $">=> binaryBlackTopHatWindowed {radius} {windowSize}"
        | "MorphologicalGradient" ->
            let radius = parameterValue "radius"
            let windowSize = savedParamValue "windowSize" node
            if String.Equals(windowSize.Trim(), "None", StringComparison.OrdinalIgnoreCase) then
                $">=> binaryGradient {radius}"
            else
                let windowSize = numericLiteral Int32 windowSize
                $">=> binaryGradientWindowed {radius} {windowSize}"
        | "ImageComparison" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            $">=> {comparisonStageFunctionName node}<{pixelType}>"
        | "MaskLogic" ->
            $">=> {maskLogicStageFunctionName node}"
        | "MaskNot" ->
            ">=> maskNot"
        | "BinaryContour" ->
            let fullyConnected = parameterValue "fullyConnected"
            let windowSize = savedParamValue "windowSize" node
            if String.Equals(windowSize.Trim(), "None", StringComparison.OrdinalIgnoreCase) then
                $">=> binaryContour {fullyConnected}"
            else
                let windowSize = numericLiteral Int32 windowSize
                $">=> binaryContourWindowed {fullyConnected} {windowSize}"
        | "BinaryMedian" ->
            let radius = parameterValue "radius"
            let workers =
                match parameterValue "windowSize" |> optionUInt with
                | "None" -> "4"
                | someWindow -> $"int (Option.get {someWindow})"
            $">=> medianUInt8 (int ({radius})) {workers}"
        | "RemoveSmallObjects" ->
            let maximumVolume = parameterValue "maximumVolume"
            let connectivity = parameterValue "connectivity"
            $">=> removeSmallObjects {maximumVolume} ObjectConnectivity.{connectivity}"
        | "FillSmallHoles" ->
            let maximumVolume = parameterValue "maximumVolume"
            let connectivity = parameterValue "connectivity"
            $">=> fillSmallHoles {maximumVolume} ObjectConnectivity.{connectivity}"
        | "LabelContour" ->
            let pixelType = pixelTypeNameFromParameter "type" "UInt64" node
            let fullyConnected = parameterValue "fullyConnected"
            let windowSize = parameterValue "windowSize" |> optionUInt
            $">=> labelContour<{pixelType}> {fullyConnected} {windowSize}"
        | "ChangeLabel" ->
            let pixelType = pixelTypeNameFromParameter "type" "UInt64" node
            let fromLabel = parameterValue "fromLabel"
            let toLabel = parameterValue "toLabel"
            $">=> changeLabel<{pixelType}> {fromLabel} {toLabel}"
        | "ComputeStats" ->
            $">=> computeStats ()"
        | "SurfaceArea" ->
            let xUnit = parameterValue "xUnit"
            let yUnit = parameterValue "yUnit"
            let zUnit = parameterValue "zUnit"
            $">=> surfaceArea {xUnit} {yUnit} {zUnit}"
        | "Volume" ->
            let xUnit = parameterValue "xUnit"
            let yUnit = parameterValue "yUnit"
            let zUnit = parameterValue "zUnit"
            $">=> volume {xUnit} {yUnit} {zUnit}"
        | "FitBiasModel" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let order = parameterValue "order"
            let depth = parameterValue "depth"
            $">=> fitBiasModel<{pixelType}> {order} {depth}"
        | "FitBiasModelMasked" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let order = parameterValue "order"
            let depth = parameterValue "depth"
            $">=> fitBiasModelMasked<{pixelType}> {order} {depth}"
        | "CorrectBias" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let model = parameterValue "model"
            $">=> correctBias<{pixelType}> {model}"
        | "CorrectBiasMasked" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let model = parameterValue "model"
            $">=> correctBiasMasked<{pixelType}> {model}"
        | "SerialPolynomialBiasCorrect" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let order = parameterValue "order"
            $">=> serialPolynomialBiasCorrect<{pixelType}> {order}"
        | "SerialEstTrans" ->
            let pixelType = pipelinePixelType "Float32"
            let searchRadius = parameterValue "searchRadius"
            let method = quotedParameter "method"
            let scale = parameterValue "scale"
            let pixelFraction = parameterValue "pixelFraction"
            $">=> serialEstTrans<{pixelType}> {searchRadius} {method} {scale} {pixelFraction}"
        | "SerialApplyTrans" ->
            let pixelType = pipelinePixelType "Float32"
            let background = parameterValue "background" |> numericLiteralForPixelType pixelType
            let geometryExpression = parameterExpression "geometry"
            let geometry =
                if geometryExpression.IsLinked then
                    $"(Some {geometryExpression.Value})"
                elif String.IsNullOrWhiteSpace geometryExpression.Value then
                    "None"
                else
                    geometryExpression.Value
            $">=> serialApplyTrans<{pixelType}> {background} {geometry}"
        | "SerialEstBoundingBox" ->
            let pixelType = pipelinePixelType "Float32"
            $">=> serialEstBoundingBox<{pixelType}>"
        | "SerialApplyManifestInBoundingBox" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let manifest = parameterValue "manifest"
            let background = parameterValue "background"
            $">=> serialApplyManifestInBoundingBox<{pixelType}> {manifest} {background}"
        | "PointPairDistances" ->
            let xUnit = parameterValue "xUnit"
            let yUnit = parameterValue "yUnit"
            let zUnit = parameterValue "zUnit"
            $">=> pointPairDistances {xUnit} {yUnit} {zUnit}"
        | "AddNormalNoise" ->
            let pixelType = pipelinePixelType "Float32"
            let mean = parameterValue "mean"
            let std = parameterValue "std"
            $">=> addNormalNoise<{pixelType}> {mean} {std}"
        | "AddSaltAndPepperNoise" ->
            let pixelType = pipelinePixelType "Float32"
            let probability = parameterValue "probability"
            $">=> addSaltAndPepperNoise<{pixelType}> {probability}"
        | "AddShotNoise" ->
            let pixelType = pipelinePixelType "Float32"
            let scale = parameterValue "scale"
            $">=> addShotNoise<{pixelType}> {scale}"
        | "AddSpeckleNoise" ->
            let std = parameterValue "std"
            $">=> addSpeckleNoise {std}"
        | "Threshold" ->
            let pixelType = pipelinePixelType "Float32"
            let lower = parameterValue "lower"
            let upper = parameterValue "upper"
            $">=> thresholdRange<{pixelType}> {lower} {upper}"
        | "Erode" ->
            let radius = parameterValue "radius"
            $">=> binaryErode {radius}"
        | "Dilate" ->
            let radius = parameterValue "radius"
            $">=> binaryDilate {radius}"
        | "DilateZonohedral" ->
            let radius = parameterValue "radius"
            $">=> binaryDilate {radius}"
        | "ErodeZonohedral" ->
            let radius = parameterValue "radius"
            $">=> binaryErode {radius}"
        | "Opening" ->
            let radius = parameterValue "radius"
            $">=> binaryOpening {radius}"
        | "OpeningZonohedral" ->
            let radius = parameterValue "radius"
            $">=> binaryOpening {radius}"
        | "Closing" ->
            let radius = parameterValue "radius"
            $">=> binaryClosing {radius}"
        | "ClosingZonohedral" ->
            let radius = parameterValue "radius"
            $">=> binaryClosing {radius}"
        | "ConnectedComponents" ->
            let windowSize = savedParamValue "windowSize" node
            if String.Equals(windowSize.Trim(), "None", StringComparison.OrdinalIgnoreCase) then
                ">=> connectedComponentsUInt32 ()"
            else
                let windowSize = numericLiteral Int32 windowSize
                $">=> connectedComponentsUInt32Windowed {windowSize} System.Environment.ProcessorCount"
        | "MarchingCubes" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let surfaceValue = parameterValue "surfaceValue"
            $">=> marchingCubes<{pixelType}> {surfaceValue}"
        | "DogKeypoints" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let sigma0 = parameterValue "sigma0"
            let scaleFactor = parameterValue "scaleFactor"
            let scaleLevels = parameterValue "scaleLevels"
            let contrastThreshold = parameterValue "contrastThreshold"
            let stride = parameterValue "stride"
            $">=> dogKeypoints<{pixelType}> {sigma0} {scaleFactor} {scaleLevels} {contrastThreshold} {stride}"
        | "SiftKeypoints" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let sigma0 = parameterValue "sigma0"
            let scaleFactor = parameterValue "scaleFactor"
            let scaleLevels = parameterValue "scaleLevels"
            let contrastThreshold = parameterValue "contrastThreshold"
            let stride = parameterValue "stride"
            $">=> siftKeypoints<{pixelType}> {sigma0} {scaleFactor} {scaleLevels} {contrastThreshold} {stride}"
        | "LogBlobKeypoints" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let sigma = parameterValue "sigma"
            let threshold = parameterValue "threshold"
            let stride = parameterValue "stride"
            $">=> logBlobKeypoints<{pixelType}> {sigma} {threshold} {stride}"
        | "HessianKeypoints" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let sigma = parameterValue "sigma"
            let responseKind = quotedParameter "responseKind"
            let threshold = parameterValue "threshold"
            let stride = parameterValue "stride"
            $">=> hessianKeypoints<{pixelType}> {sigma} {responseKind} {threshold} {stride}"
        | "Harris3DKeypoints" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let sigma = parameterValue "sigma"
            let rho = parameterValue "rho"
            let k = parameterValue "k"
            let threshold = parameterValue "threshold"
            let stride = parameterValue "stride"
            $">=> harris3DKeypoints<{pixelType}> {sigma} {rho} {k} {threshold} {stride}"
        | "Forstner3DKeypoints" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let sigma = parameterValue "sigma"
            let rho = parameterValue "rho"
            let threshold = parameterValue "threshold"
            let stride = parameterValue "stride"
            $">=> forstner3DKeypoints<{pixelType}> {sigma} {rho} {threshold} {stride}"
        | "PhaseCongruencyKeypoints" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let sigma = parameterValue "sigma"
            let threshold = parameterValue "threshold"
            let stride = parameterValue "stride"
            $">=> phaseCongruencyKeypoints<{pixelType}> {sigma} {threshold} {stride}"
        | "StreamConnectedObjects" ->
            let connectivity = parameterValue "connectivity"
            $">=> streamConnectedObjects<uint8> ObjectConnectivity.{connectivity}"
        | "MeasureObjects" ->
            ">=> measureObjects"
        | "ObjectSizeStats" ->
            ">=> objectSizeStats"
        | "ObjectSizes" ->
            ">=> objectSizes"
        | "PaintObjects" ->
            let width = parameterValue "width"
            let height = parameterValue "height"
            $">=> paintObjects {width} {height}"
        | "PaintObjectsCropped" ->
            $">=> paintObjectsCropped"
        | "SignedDistanceBand" ->
            let bandRadius = parameterValue "bandRadius"
            let stride = parameterValue "stride"
            $">=> signedDistanceBand {bandRadius} {stride} 1"
        | "PermuteAxes" ->
            let axes = parameterValue "axes" |> intArray3Literal
            $">=> permuteAxes {axes}"
        | "ResampleAffine" ->
            let lerp = parameterValue "lerp"
            let inputGeometry = parameterValue "inputGeometry"
            let outputGeometry = parameterValue "outputGeometry"
            let affine = parameterValue "affine"
            let background = parameterValue "background"
            $">=> resampleAffine {lerp} {inputGeometry} {outputGeometry} {affine} {background}"
        | "Cast" ->
            let sourceType = pixelTypeNameFromParameter "sourceType" "Float32" node
            let targetType = pixelTypeNameFromParameter "targetType" "Float32" node
            $">=> cast<{sourceType},{targetType}>"
        | _ ->
            $"// Unsupported element: {node.FunctionId}"

    let generateSavedGraph (graph: SavedGraph) =
        let graph = replaceFileDirectorySelectors graph
        let builder = StringBuilder()
        let nodesById = graph.Nodes |> Seq.map (fun node -> node.Id, node) |> Map.ofSeq
        let scalarNodes =
            graph.Nodes
            |> Array.filter (fun node ->
                node.FunctionId = "Scalar"
                || node.FunctionId = "ScalarOp"
                || node.FunctionId = "ScalarFunction"
                || node.FunctionId = "RandomRigidTransform"
                || node.FunctionId = "OtsuThresholdFromHistogram"
                || node.FunctionId = "MomentsThresholdFromHistogram")

        let scalarNamesByNodeId = scalarNames scalarNodes
        let expandNodesWithLinkedOutputs =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter isExpandNode
            |> Array.distinctBy _.Id

        let expandSourceNode (expandNode: SavedNode) =
            graph.Edges
            |> Array.tryFind (fun edge ->
                edge.ToNode = expandNode.Id
                && edge.ToKind <> "parameterInput"
                && edge.ToPort = 0)
            |> Option.bind (fun edge -> nodesById |> Map.tryFind edge.FromNode)

        let expandNodesFor predicate =
            expandNodesWithLinkedOutputs
            |> Array.filter (fun expandNode ->
                expandSourceNode expandNode
                |> Option.exists predicate)
            |> Array.distinctBy _.Id

        let statsExpandNodesWithLinkedOutputs =
            expandNodesFor (fun node -> node.FunctionId = "ComputeStats" || node.FunctionId = "ObjectSizeStats")

        let statsProducerNodesForExpand =
            statsExpandNodesWithLinkedOutputs
            |> Array.choose expandSourceNode
            |> Array.distinctBy _.Id

        let statsProducerNodesWithDirectLinkedOutputs =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter (fun node -> node.FunctionId = "ComputeStats" || node.FunctionId = "ObjectSizeStats")
            |> Array.distinctBy _.Id

        let statsNodesWithLinkedFields =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter isSingleValueReducerNode
            |> Array.distinctBy _.Id

        let statsNamesByNodeId =
            let producerNames =
                Array.concat [ statsProducerNodesForExpand; statsProducerNodesWithDirectLinkedOutputs ]
                |> Array.distinctBy _.Id
                |> Array.mapi (fun index node ->
                    let prefix = if node.FunctionId = "ObjectSizeStats" then "ObjectSizeStats" else "ImageStats"
                    node.Id, $"{prefix}{index}")

            let singleValueNames =
                statsNodesWithLinkedFields
                |> Array.mapi (fun index node -> node.Id, $"{node.FunctionId}{index}")

            let expandNames =
                statsExpandNodesWithLinkedOutputs
                |> Array.mapi (fun index node -> node.Id, $"Expand{index}")

            Array.concat [ producerNames; singleValueNames; expandNames ]
            |> Map.ofArray

        let translationTableNamesByNodeId =
            Map.empty<string, string>

        let histogramNodesWithLinkedOutputs =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter isHistogramDataNode
            |> Array.distinctBy _.Id

        let histogramNamesByNodeId =
            histogramNodesWithLinkedOutputs
            |> Array.mapi (fun index node -> node.Id, $"Histogram{index}")
            |> Map.ofArray

        let quantileNodesWithLinkedOutputs =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter isQuantilesNode
            |> Array.distinctBy _.Id

        let quantileNamesByNodeId =
            quantileNodesWithLinkedOutputs
            |> Array.mapi (fun index node -> node.Id, $"Quantiles{index}")
            |> Map.ofArray

        let imageInfoExpandNodesWithLinkedOutputs =
            expandNodesFor (fun node -> isImageInfoNode node || isWriteImageInfoNode node || isReadImageInfoNode node)

        let imageInfoProducerNodesForExpand =
            imageInfoExpandNodesWithLinkedOutputs
            |> Array.choose expandSourceNode
            |> Array.distinctBy _.Id

        let imageInfoProducerNodesWithDirectLinkedOutputs =
            graph.Edges
            |> Array.choose (fun edge ->
                if (edge.FromKind = "reducerOutput" || edge.FromKind = "output") && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter (fun node -> isImageInfoNode node || isWriteImageInfoNode node || isReadImageInfoNode node)
            |> Array.distinctBy _.Id

        let imageInfoProducerNodesWithLinkedOutputs =
            Array.concat [ imageInfoProducerNodesForExpand; imageInfoProducerNodesWithDirectLinkedOutputs ]
            |> Array.distinctBy _.Id

        let imageInfoNamesByNodeId =
            let producerNames =
                imageInfoProducerNodesWithLinkedOutputs
                |> Array.mapi (fun index node -> node.Id, $"ImageInfo{index}")

            let expandNames =
                imageInfoExpandNodesWithLinkedOutputs
                |> Array.mapi (fun index node -> node.Id, $"Expand{index + statsExpandNodesWithLinkedOutputs.Length}")

            Array.concat [ producerNames; expandNames ]
            |> Map.ofArray

        let chunkInfoNodesWithLinkedOutputs =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter isChunkInfoNode
            |> Array.distinctBy _.Id

        let chunkInfoExpandNodesWithLinkedOutputs =
            expandNodesFor (fun node -> isChunkInfoNode node || isReadChunkInfoNode node || isWriteChunkInfoNode node)

        let chunkInfoProducerNodesForExpand =
            chunkInfoExpandNodesWithLinkedOutputs
            |> Array.choose expandSourceNode
            |> Array.distinctBy _.Id

        let chunkInfoProducerNodesWithDirectLinkedOutputs =
            graph.Edges
            |> Array.choose (fun edge ->
                if (edge.FromKind = "reducerOutput" || edge.FromKind = "output") && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter (fun node -> isChunkInfoNode node || isReadChunkInfoNode node || isWriteChunkInfoNode node)
            |> Array.distinctBy _.Id

        let chunkInfoProducerNodesWithLinkedOutputs =
            Array.concat [ chunkInfoNodesWithLinkedOutputs; chunkInfoProducerNodesForExpand; chunkInfoProducerNodesWithDirectLinkedOutputs ]
            |> Array.distinctBy _.Id

        let chunkInfoNamesByNodeId =
            let producerNames =
                chunkInfoProducerNodesWithLinkedOutputs
                |> Array.mapi (fun index node -> node.Id, $"ChunkInfo{index}")

            let expandNames =
                chunkInfoExpandNodesWithLinkedOutputs
                |> Array.mapi (fun index node -> node.Id, $"Expand{index + statsExpandNodesWithLinkedOutputs.Length + imageInfoExpandNodesWithLinkedOutputs.Length}")

            Array.concat [ producerNames; expandNames ]
            |> Map.ofArray

        let serialGeometryNodesWithLinkedOutputs =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter isSerialVolumeGeometryNode
            |> Array.distinctBy _.Id

        let serialGeometryNamesByNodeId =
            serialGeometryNodesWithLinkedOutputs
            |> Array.mapi (fun index node -> node.Id, $"SerialVolumeGeometry{index}")
            |> Map.ofArray

        let newLine = Environment.NewLine

        let dataEdges =
            graph.Edges
            |> Array.filter (fun edge -> edge.FromKind <> "scalarOutput" && edge.ToKind <> "parameterInput")

        let printNodesUsedByTap =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "scalarOutput" && edge.ToKind = "parameterInput" then
                    match nodesById |> Map.tryFind edge.FromNode, nodesById |> Map.tryFind edge.ToNode with
                    | Some sourceNode, Some targetNode when sourceNode.FunctionId = "Tap" && targetNode.FunctionId = "Print" ->
                        Some targetNode.Id
                    | _ ->
                        None
                else
                    None)
            |> Set.ofArray

        let bindingNameForOutput edge =
            match edge.FromKind with
            | "scalarOutput" ->
                scalarNamesByNodeId |> Map.tryFind edge.FromNode
            | "reducerOutput" ->
                match statsNamesByNodeId |> Map.tryFind edge.FromNode with
                | Some name -> Some name
                | None ->
                    match translationTableNamesByNodeId |> Map.tryFind edge.FromNode with
                    | Some name -> Some name
                    | None ->
                        match histogramNamesByNodeId |> Map.tryFind edge.FromNode with
                        | Some name -> Some name
                        | None ->
                            match quantileNamesByNodeId |> Map.tryFind edge.FromNode with
                            | Some name -> Some name
                            | None ->
                                match imageInfoNamesByNodeId |> Map.tryFind edge.FromNode with
                                | Some name -> Some name
                                | None ->
                                    match chunkInfoNamesByNodeId |> Map.tryFind edge.FromNode with
                                    | Some name -> Some name
                                    | None -> serialGeometryNamesByNodeId |> Map.tryFind edge.FromNode
            | "output" ->
                imageInfoNamesByNodeId
                |> Map.tryFind edge.FromNode
                |> Option.orElseWith (fun () -> chunkInfoNamesByNodeId |> Map.tryFind edge.FromNode)
                |> Option.orElseWith (fun () -> serialGeometryNamesByNodeId |> Map.tryFind edge.FromNode)
            | _ ->
                None

        let parameterBindingDependencies (node: SavedNode) =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.ToNode = node.Id && edge.ToKind = "parameterInput" then
                    bindingNameForOutput edge
                else
                    None)
            |> Set.ofArray

        let rec pipelineBindingDependencies visited (node: SavedNode) =
            if visited |> Set.contains node.Id then
                Set.empty
            else
                let visited = visited |> Set.add node.Id
                let parameterDependencies = parameterBindingDependencies node

                let upstreamDependencies =
                    dataEdges
                    |> Array.filter (fun edge -> edge.ToNode = node.Id)
                    |> Array.choose (fun edge -> nodesById |> Map.tryFind edge.FromNode)
                    |> Array.map (pipelineBindingDependencies visited)
                    |> Set.unionMany

                Set.union parameterDependencies upstreamDependencies

        let indentBlock spaces (text: string) =
            let prefix = String.replicate spaces " "

            text.Split([| newLine |], StringSplitOptions.None)
            |> Array.map (fun line -> if String.IsNullOrWhiteSpace line then line else prefix + line)
            |> String.concat newLine

        let parenthesizeBlock (text: string) =
            $"({newLine}{indentBlock 4 text}{newLine})"

        let incomingDataEdge nodeId port =
            dataEdges
            |> Array.tryFind (fun edge -> edge.ToNode = nodeId && edge.ToPort = port)

        let sameOutputEdge (left: SavedEdge) (right: SavedEdge) =
            left.FromNode = right.FromNode
            && left.FromPort = right.FromPort
            && left.FromKind = right.FromKind

        let commonUpstreamInputEdge (node: SavedNode) =
            match incomingDataEdge node.Id 0 with
            | Some edge -> Some edge
            | None ->
                Some
                    { FromNode = node.Id
                      FromKind = "dataOutput"
                      FromPort = 0
                      ToNode = node.Id
                      ToKind = "dataInput"
                      ToPort = 0 }

        let validationError =
            let isReducerOutputNode (node: SavedNode) =
                isSingleValueReducerNode node
                || isHistogramDataNode node
                || isQuantilesNode node
                || isSerialVolumeGeometryNode node
                || node.FunctionId = "ComputeStats"
                || node.FunctionId = "ObjectSizeStats"

            let rec upstreamDataNodes visited nodeId =
                if visited |> Set.contains nodeId then
                    Set.empty
                else
                    let visited = visited |> Set.add nodeId

                    dataEdges
                    |> Array.filter (fun edge -> edge.ToNode = nodeId)
                    |> Array.map (fun edge ->
                        Set.add edge.FromNode (upstreamDataNodes visited edge.FromNode))
                    |> fun sets -> if sets.Length = 0 then Set.empty else Set.unionMany sets

            let rec forwardDataNodes visited nodeId =
                if visited |> Set.contains nodeId then
                    Set.empty
                else
                    let visited = visited |> Set.add nodeId

                    dataEdges
                    |> Array.filter (fun edge -> edge.FromNode = nodeId)
                    |> Array.map (fun edge ->
                        Set.add edge.ToNode (forwardDataNodes visited edge.ToNode))
                    |> fun sets -> if sets.Length = 0 then Set.empty else Set.unionMany sets

            graph.Edges
            |> Array.tryPick (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    match nodesById |> Map.tryFind edge.FromNode, nodesById |> Map.tryFind edge.ToNode with
                    | Some reducerNode, Some targetNode when isReducerOutputNode reducerNode ->
                        let reducerInputs =
                            upstreamDataNodes Set.empty reducerNode.Id
                            |> Set.filter (fun upstreamId ->
                                dataEdges
                                |> Array.exists (fun dataEdge -> dataEdge.ToNode = upstreamId))
                        let targetDependsOnReducerInput =
                            reducerInputs
                            |> Set.exists (fun upstreamId ->
                                forwardDataNodes Set.empty upstreamId
                                |> Set.contains targetNode.Id)

                        if targetDependsOnReducerInput then
                            Some $"reducer '{reducerNode.FunctionId}' feeds a parameter on '{targetNode.FunctionId}' while both depend on the same streaming data path. Use a separate source/proxy branch for the reducer."
                        else
                            None
                    | _ -> None
                else
                    None)

        let stageCall (node: SavedNode) =
            let line = savedElementLine graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node

            if line.StartsWith(">=> ", StringComparison.Ordinal) then
                Some(line.Substring(4))
            elif line.StartsWith(">>=> ", StringComparison.Ordinal) then
                Some(line.Substring(5))
            else
                None

        let rec inferredImagePixelType visited (targetNode: SavedNode) inputPort =
            if visited |> Set.contains targetNode.Id then
                None
            else
                let visited = visited |> Set.add targetNode.Id

                incomingDataEdge targetNode.Id inputPort
                |> Option.bind (fun edge ->
                    nodesById
                    |> Map.tryFind edge.FromNode
                    |> Option.bind (fun sourceNode ->
                        match sourceNode.FunctionId, edge.FromPort with
                        | "SerialEstTrans", 0 ->
                            inferredImagePixelType visited sourceNode 0
                        | "Cast", 0 ->
                            Some(pixelTypeNameFromParameter "targetType" "Float32" sourceNode)
                        | _ ->
                            let configured = savedParamValue "type" sourceNode
                            if String.IsNullOrWhiteSpace configured then None else Some(pixelTypeNameFromSuffix configured)))

        let composeStages (left: string) (right: string) =
            $"{left}{newLine}>=> {right}"

        let formatStageTuple (left: string) (right: string) =
            if left.Contains(newLine, StringComparison.Ordinal) || right.Contains(newLine, StringComparison.Ordinal) then
                $">=>> ({newLine}{indentBlock 4 left},{newLine}{indentBlock 4 right}{newLine})"
            else
                $">=>> ({left}, {right})"

        let rec pipelineExpression visited (node: SavedNode) =
            if visited |> Set.contains node.Id then
                $"// Cannot generate F#: cycle detected at {node.FunctionId}"
            else
                let visited = visited |> Set.add node.Id

                let inputExpression port =
                    match incomingDataEdge node.Id port with
                    | Some edge ->
                        match nodesById |> Map.tryFind edge.FromNode with
                        | Some upstream ->
                            let upstreamExpression = pipelineExpression visited upstream
                            if upstream.FunctionId = "PCA" && edge.FromPort >= 0 && edge.FromPort <= 8 then
                                let rawComponents = savedParamValue "components" upstream
                                let components =
                                    if String.IsNullOrWhiteSpace rawComponents then "3u"
                                    else numericLiteral UInt32 rawComponents
                                $"{upstreamExpression}{newLine}>=> selectGroupedVectorOutput ({components} + 1u) {edge.FromPort}u"
                            elif upstream.FunctionId = "AffineRegistration" && edge.FromPort >= 0 && edge.FromPort <= 1 then
                                $"{upstreamExpression}{newLine}>=> selectGroupedValueOutput 2u {edge.FromPort}u"
                            elif upstream.FunctionId = "EstimateHistogram" && edge.FromPort = 0 then
                                $"{upstreamExpression}{newLine}>=> histogramEstimateMap"
                            else
                                upstreamExpression
                        | None -> $"// Cannot generate F#: missing upstream node {edge.FromNode}"
                    | None -> $"// Cannot generate F#: missing input {port} for {node.FunctionId}"

                let rec branchStageExpression commonInputEdge visited (node: SavedNode) =
                    if visited |> Set.contains node.Id then
                        None
                    elif node.Id = commonInputEdge.FromNode then
                        Some "identity"
                    else
                        let visited = visited |> Set.add node.Id

                        match incomingDataEdge node.Id 0, stageCall node with
                        | Some incoming, Some call when sameOutputEdge incoming commonInputEdge ->
                            Some call
                        | Some incoming, Some call ->
                            nodesById
                            |> Map.tryFind incoming.FromNode
                            |> Option.bind (branchStageExpression commonInputEdge visited)
                            |> Option.map (fun upstream -> composeStages upstream call)
                        | _ ->
                            None

                let sharedFanOutExpression () =
                    match incomingDataEdge node.Id 0, incomingDataEdge node.Id 1 with
                    | Some leftEdge, Some rightEdge when sameOutputEdge leftEdge rightEdge ->
                        nodesById
                        |> Map.tryFind leftEdge.FromNode
                        |> Option.bind (fun sharedNode ->
                            let shared = pipelineExpression visited sharedNode
                            let fanOut =
                                formatStageTuple
                                    "identity"
                                    "identity"
                            pairStageFunctionName node
                            |> Option.map (fun pairFunction -> $"{shared}{newLine}{fanOut}{newLine}{pairCompositionOperator node} {pairFunction}"))
                    | Some leftEdge, Some rightEdge ->
                        match nodesById |> Map.tryFind leftEdge.FromNode, nodesById |> Map.tryFind rightEdge.FromNode with
                        | Some leftNode, Some rightNode ->
                            match commonUpstreamInputEdge leftNode, commonUpstreamInputEdge rightNode with
                            | Some leftInput, Some rightInput when sameOutputEdge leftInput rightInput ->
                                match nodesById |> Map.tryFind leftInput.FromNode with
                                | Some sharedNode ->
                                    match branchStageExpression leftInput Set.empty leftNode, branchStageExpression leftInput Set.empty rightNode with
                                    | Some leftStage, Some rightStage ->
                                        let shared = pipelineExpression visited sharedNode
                                        pairStageFunctionName node
                                        |> Option.map (fun pairFunction -> $"{shared}{newLine}{formatStageTuple leftStage rightStage}{newLine}{pairCompositionOperator node} {pairFunction}")
                                    | _ -> None
                                | None -> None
                            | _ -> None
                        | _ -> None
                    | _ -> None

                match node.FunctionId with
                | id when pairStageFunctionName node |> Option.isSome ->
                    match sharedFanOutExpression () with
                    | Some expression ->
                        expression
                    | None ->
                        let pairFunction = pairStageFunctionName node |> Option.get
                        let left = inputExpression 0
                        let right = inputExpression 1
                        let left = parenthesizeBlock left
                        let right = parenthesizeBlock right
                        $"({newLine}{indentBlock 4 left},{newLine}{indentBlock 4 right}{newLine}){newLine}||> zip{newLine}{pairCompositionOperator node} {pairFunction}"
                | _ when (incomingDataEdge node.Id 1 |> Option.isSome) && (stageCall node |> Option.isSome) ->
                    let stage = stageCall node |> Option.get
                    let left = inputExpression 0 |> parenthesizeBlock
                    let right = inputExpression 1 |> parenthesizeBlock
                    $"({newLine}{indentBlock 4 left},{newLine}{indentBlock 4 right}{newLine}){newLine}||> zip{newLine}>=> {stage}"
                | _ ->
                    let line = savedElementLine graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node

                    match incomingDataEdge node.Id 0 with
                    | Some _ ->
                        let input = inputExpression 0
                        $"{input}{newLine}{line}"
                    | None ->
                        line

        let generatedPipelines () =
            let isSinkTerminal (node: SavedNode) =
                match node.FunctionId with
                | "Write"
                | "WriteThrough"
                | "WriteChunks"
                | "WriteMesh"
                | "WritePointSet"
                | "WriteMatrix"
                | "WriteCSV"
                | "ImHistogram"
                | "ShowImage"
                | "Ignore"
                | "Empty" -> true
                | _ -> false

            let appendSinkIfTerminalWrite (node: SavedNode) expression =
                if isSinkTerminal node then
                    $"{expression}{newLine}|> sink"
                else
                    match node.FunctionId with
                    | "ComputeStats" ->
                        $"{expression}{newLine}|> drain"
                    | "SurfaceArea"
                    | "Volume"
                    | "PointPairDistances"
                    | "FitBiasModel"
                    | "FitBiasModelMasked"
                    | "AffineRegistration"
                    | "SerialEstBoundingBox" ->
                        $"{expression}{newLine}|> drain"
                    | "Histogram" ->
                        $"{expression}{newLine}|> drain"
                    | "ImHistogramData" ->
                        $"{expression}{newLine}|> drain"
                    | "EstimateHistogram" ->
                        $"{expression}{newLine}|> drain"
                    | "ObjectSizeStats" ->
                        $"{expression}{newLine}|> drain"
                    | _ ->
                        expression

            let expressionFromOutputEdge (edge: SavedEdge) =
                match nodesById |> Map.tryFind edge.FromNode with
                | Some upstream ->
                    let upstreamExpression = pipelineExpression Set.empty upstream
                    if upstream.FunctionId = "PCA" && edge.FromPort >= 0 && edge.FromPort <= 8 then
                        let rawComponents = savedParamValue "components" upstream
                        let components =
                            if String.IsNullOrWhiteSpace rawComponents then "3u"
                            else numericLiteral UInt32 rawComponents
                        $"{upstreamExpression}{newLine}>=> selectGroupedVectorOutput ({components} + 1u) {edge.FromPort}u"
                    elif upstream.FunctionId = "AffineRegistration" && edge.FromPort >= 0 && edge.FromPort <= 1 then
                        $"{upstreamExpression}{newLine}>=> selectGroupedValueOutput 2u {edge.FromPort}u"
                    elif upstream.FunctionId = "EstimateHistogram" && edge.FromPort = 0 then
                        $"{upstreamExpression}{newLine}>=> histogramEstimateMap"
                    else
                        upstreamExpression
                | None -> $"// Cannot generate F#: missing upstream node {edge.FromNode}"

            let terminalNodes =
                graph.Nodes
                |> Array.filter (fun node ->
                    node.FunctionId <> "Scalar"
                    && node.FunctionId <> "ScalarOp"
                    && node.FunctionId <> "ScalarFunction"
                    && node.FunctionId <> "RandomRigidTransform"
                    && node.FunctionId <> "OtsuThresholdFromHistogram"
                    && node.FunctionId <> "MomentsThresholdFromHistogram"
                    && node.FunctionId <> "GetChunkInfo"
                    && node.FunctionId <> "GetZarrInfo"
                    && node.FunctionId <> "GetNexusInfo"
                    && node.FunctionId <> "Expand"
                    && not (printNodesUsedByTap |> Set.contains node.Id)
                    && not (statsNamesByNodeId |> Map.containsKey node.Id)
                    && not (translationTableNamesByNodeId |> Map.containsKey node.Id)
                    && not (histogramNamesByNodeId |> Map.containsKey node.Id)
                    && not (quantileNamesByNodeId |> Map.containsKey node.Id)
                    && not (imageInfoNamesByNodeId |> Map.containsKey node.Id)
                    && not (chunkInfoNamesByNodeId |> Map.containsKey node.Id)
                    && not (serialGeometryNamesByNodeId |> Map.containsKey node.Id)
                    && not (dataEdges |> Array.exists (fun edge -> edge.FromNode = node.Id)))

            let sharedSinkGroups =
                terminalNodes
                |> Array.choose (fun node ->
                    match incomingDataEdge node.Id 0, stageCall node with
                    | Some edge, Some call when isSinkTerminal node ->
                        Some((edge.FromNode, edge.FromKind, edge.FromPort), (node, edge, call))
                    | _ ->
                        None)
                |> Array.groupBy fst
                |> Array.choose (fun (_, entries) ->
                    let branches = entries |> Array.map snd
                    if branches.Length >= 2 then Some branches else None)

            let groupedTerminalIds =
                sharedSinkGroups
                |> Array.collect (Array.map (fun (node, _, _) -> node.Id))
                |> Set.ofArray

            let sharedSinkExpressions =
                let formatForkStageTuple (left: string) (right: string) =
                    if left.Contains(newLine, StringComparison.Ordinal) || right.Contains(newLine, StringComparison.Ordinal) then
                        $"-->> ({newLine}{indentBlock 4 left},{newLine}{indentBlock 4 right}{newLine})"
                    else
                        $"-->> ({left}, {right})"

                let rec sinkTreeStageExpression (stageCalls: string list) =
                    match stageCalls with
                    | [] ->
                        "ignoreSingles ()"
                    | [ call ] ->
                        $"{call}{newLine}--> ignoreSingles ()"
                    | left :: right ->
                        let rightStage = sinkTreeStageExpression right
                        let rightStage =
                            if right.Length = 1 then
                                rightStage
                            else
                                parenthesizeBlock rightStage

                        $"identity{newLine}{formatForkStageTuple left rightStage}{newLine}--> ignorePairs ()"

                sharedSinkGroups
                |> Array.map (fun branches ->
                    let nodes = branches |> Array.map (fun (node, _, _) -> node)
                    let commonEdge = branches |> Array.head |> fun (_, edge, _) -> edge
                    let stageCalls = branches |> Array.map (fun (_, _, call) -> call) |> Array.toList
                    let shared = expressionFromOutputEdge commonEdge
                    let sinkStage = sinkTreeStageExpression stageCalls |> parenthesizeBlock

                    { Dependencies =
                          nodes
                          |> Array.map (pipelineBindingDependencies Set.empty)
                          |> Set.unionMany
                      Text =
                          $"{shared}{newLine}>=> {sinkStage}{newLine}|> sink" })

            let singleTerminalExpressions =
                terminalNodes
                |> Array.filter (fun node -> not (groupedTerminalIds |> Set.contains node.Id))
                |> Array.map (fun node ->
                    { Dependencies = pipelineBindingDependencies Set.empty node
                      Text = pipelineExpression Set.empty node |> appendSinkIfTerminalWrite node })

            Array.append sharedSinkExpressions singleTerminalExpressions

        let orderedBindings () =
            let statsBindings =
                let producerBindings =
                    Array.concat [ statsProducerNodesForExpand; statsProducerNodesWithDirectLinkedOutputs ]
                    |> Array.distinctBy _.Id
                    |> Array.map (fun node ->
                        let name = statsNamesByNodeId |> Map.find node.Id
                        let expression = pipelineExpression Set.empty node
                        let body = indentBlock 4 $"{expression}{newLine}|> drain"

                        { Name = name
                          Dependencies = pipelineBindingDependencies Set.empty node |> Set.remove name
                          Text = $"let {name} ={newLine}{body}" })

                let singleValueBindings =
                    statsNodesWithLinkedFields
                    |> Array.map (fun node ->
                        let name = statsNamesByNodeId |> Map.find node.Id
                        let expression = pipelineExpression Set.empty node
                        let body = indentBlock 4 $"{expression}{newLine}|> drain"

                        { Name = name
                          Dependencies = pipelineBindingDependencies Set.empty node |> Set.remove name
                          Text = $"let {name} ={newLine}{body}" })

                let expandBindings =
                    statsExpandNodesWithLinkedOutputs
                    |> Array.map (fun node ->
                        let name = statsNamesByNodeId |> Map.find node.Id

                        let sourceName =
                            graph.Edges
                            |> Array.tryFind (fun edge ->
                                edge.ToNode = node.Id
                                && edge.ToKind <> "parameterInput"
                                && edge.ToPort = 0)
                            |> Option.bind bindingNameForOutput
                            |> Option.defaultWith (fun () -> failwith $"Expand node '{node.Id}' is missing a record input.")

                        { Name = name
                          Dependencies = Set.singleton sourceName
                          Text = $"let {name} = {sourceName}" })

                Array.concat [ producerBindings; singleValueBindings; expandBindings ]

            let scalarBindings =
                scalarNodes
                |> Array.map (fun node ->
                    let name, text = scalarBinding graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node

                    { Name = name
                      Dependencies = parameterBindingDependencies node |> Set.remove name
                      Text = text })

            let translationTableBindings =
                Array.empty

            let histogramBindings =
                histogramNodesWithLinkedOutputs
                |> Array.map (fun node ->
                    let name = histogramNamesByNodeId |> Map.find node.Id
                    let expression = pipelineExpression Set.empty node
                    let body = indentBlock 4 $"{expression}{newLine}|> drain"

                    { Name = name
                      Dependencies = pipelineBindingDependencies Set.empty node |> Set.remove name
                      Text = $"let {name} ={newLine}{body}" })

            let quantileBindings =
                quantileNodesWithLinkedOutputs
                |> Array.map (fun node ->
                    let name = quantileNamesByNodeId |> Map.find node.Id

                    let parameter key =
                        node.Parameters
                        |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
                        |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node index key)
                        |> Option.defaultValue { Value = savedParamValue key node; IsLinked = false }

                    let numericParameter key =
                        let expression = parameter key
                        if expression.IsLinked then expression.Value else numericLiteral Float64 expression.Value

                    let boolParameter key =
                        savedParamValue key node
                        |> fun value -> value.Trim().Equals("true", StringComparison.OrdinalIgnoreCase)

                    let quantileValues =
                        [ "q1", true
                          "q2", boolParameter "useQ2"
                          "q3", boolParameter "useQ3"
                          "q4", boolParameter "useQ4"
                          "q5", boolParameter "useQ5" ]
                        |> List.choose (fun (key, enabled) -> if enabled then Some(numericParameter key) else None)
                        |> String.concat "; "

                    let histogram = (parameter "histogram").Value

                    { Name = name
                      Dependencies = parameterBindingDependencies node |> Set.remove name
                      Text = $"let {name} = quantiles [{quantileValues}] ({histogram} :> obj)" })

            let imageInfoBindings =
                let imageInfoProducerBindings =
                    imageInfoProducerNodesWithLinkedOutputs
                    |> Array.map (fun node ->
                        match node.FunctionId with
                        | "GetZarrInfo" ->
                            let name = imageInfoNamesByNodeId |> Map.find node.Id
                            let parameter key =
                                node.Parameters
                                |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
                                |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node index key)
                                |> Option.defaultValue { Value = savedParamValue key node; IsLinked = false }
                            let stringArgument key =
                                let expression = parameter key
                                if expression.IsLinked then expression.Value else quote expression.Value
                            let input = stringArgument "input"
                            let multiscaleIndex = (parameter "multiscaleIndex").Value
                            let datasetIndex = (parameter "datasetIndex").Value
                            { Name = name
                              Dependencies = parameterBindingDependencies node |> Set.remove name
                              Text = $"let {name} = getZarrInfo {input} {multiscaleIndex} {datasetIndex}" }
                        | "GetNexusInfo" ->
                            let name = imageInfoNamesByNodeId |> Map.find node.Id
                            let parameter key =
                                node.Parameters
                                |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
                                |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node index key)
                                |> Option.defaultValue { Value = savedParamValue key node; IsLinked = false }
                            let stringArgument key =
                                let expression = parameter key
                                if expression.IsLinked then expression.Value else quote expression.Value
                            let input = stringArgument "input"
                            let datasetPath = stringArgument "datasetPath"
                            let frameAxis = (parameter "frameAxis").Value
                            let yAxis = (parameter "yAxis").Value
                            let xAxis = (parameter "xAxis").Value
                            { Name = name
                              Dependencies = parameterBindingDependencies node |> Set.remove name
                              Text = $"let {name} = getNexusInfo {input} {datasetPath} {frameAxis} {yAxis} {xAxis}" }
                        | "Read" ->
                            let name = imageInfoNamesByNodeId |> Map.find node.Id
                            let parameter key =
                                node.Parameters
                                |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
                                |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node index key)
                                |> Option.defaultValue { Value = savedParamValue key node; IsLinked = false }
                            let stringArgument key =
                                let expression = parameter key
                                if expression.IsLinked then expression.Value else quote expression.Value
                            let stringArgumentOrDefault key fallback =
                                let expression = parameter key

                                if expression.IsLinked then
                                    expression.Value
                                else
                                    let value =
                                        if String.IsNullOrWhiteSpace expression.Value then fallback else expression.Value

                                    quote value
                            let input = stringArgument "input"
                            let suffix = stringArgumentOrDefault "suffix" ".tiff"
                            let format = savedParamValue "format" node
                            let text =
                                match format with
                                | "Volume file" -> $"let {name} = getImageFileInfo {input} {suffix}"
                                | "OME-Zarr" ->
                                    let multiscaleIndex = (parameter "multiscaleIndex").Value
                                    let datasetIndex = (parameter "datasetIndex").Value
                                    $"let {name} = getZarrInfo {input} {multiscaleIndex} {datasetIndex}"
                                | "NeXus/HDF5" ->
                                    let datasetPath = stringArgument "datasetPath"
                                    let frameAxis = (parameter "frameAxis").Value
                                    let yAxis = (parameter "yAxis").Value
                                    let xAxis = (parameter "xAxis").Value
                                    $"let {name} = getNexusInfo {input} {datasetPath} {frameAxis} {yAxis} {xAxis}"
                                | _ -> $"let {name} = getImageInfo {input} {suffix}"
                            { Name = name
                              Dependencies = parameterBindingDependencies node |> Set.remove name
                              Text = text }
                        | "ReadRandom" ->
                            let name = imageInfoNamesByNodeId |> Map.find node.Id
                            let parameter key =
                                node.Parameters
                                |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
                                |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node index key)
                                |> Option.defaultValue { Value = savedParamValue key node; IsLinked = false }
                            let stringArgument key =
                                let expression = parameter key
                                if expression.IsLinked then expression.Value else quote expression.Value
                            let stringArgumentOrDefault key fallback =
                                let expression = parameter key
                                if expression.IsLinked then expression.Value
                                elif String.IsNullOrWhiteSpace expression.Value then quote fallback
                                else quote expression.Value
                            let input = stringArgument "input"
                            let suffix = stringArgumentOrDefault "suffix" ".tiff"
                            let text =
                                match savedParamValue "format" node with
                                | "Volume file" -> $"let {name} = getImageFileInfo {input} {suffix}"
                                | "OME-Zarr" ->
                                    let multiscaleIndex = (parameter "multiscaleIndex").Value
                                    let datasetIndex = (parameter "datasetIndex").Value
                                    $"let {name} = getZarrInfo {input} {multiscaleIndex} {datasetIndex}"
                                | "NeXus/HDF5" ->
                                    let datasetPath = stringArgument "datasetPath"
                                    let frameAxis = (parameter "frameAxis").Value
                                    let yAxis = (parameter "yAxis").Value
                                    let xAxis = (parameter "xAxis").Value
                                    $"let {name} = getNexusInfo {input} {datasetPath} {frameAxis} {yAxis} {xAxis}"
                                | _ -> $"let {name} = getImageInfo {input} {suffix}"
                            { Name = name
                              Dependencies = parameterBindingDependencies node |> Set.remove name
                              Text = text }
                        | "ReadRange" ->
                            let name = imageInfoNamesByNodeId |> Map.find node.Id
                            let parameter key =
                                node.Parameters
                                |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
                                |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node index key)
                                |> Option.defaultValue { Value = savedParamValue key node; IsLinked = false }
                            let stringArgument key =
                                let expression = parameter key
                                if expression.IsLinked then expression.Value else quote expression.Value
                            let input = stringArgument "input"
                            let suffix = stringArgument "suffix"
                            let format = savedParamValue "format" node
                            let text =
                                match format with
                                | "Volume file" -> $"let {name} = getImageFileInfo {input} {suffix}"
                                | "OME-Zarr" ->
                                    let multiscaleIndex = (parameter "multiscaleIndex").Value
                                    let datasetIndex = (parameter "datasetIndex").Value
                                    $"let {name} = getZarrInfo {input} {multiscaleIndex} {datasetIndex}"
                                | "NeXus/HDF5" ->
                                    let datasetPath = stringArgument "datasetPath"
                                    let frameAxis = (parameter "frameAxis").Value
                                    let yAxis = (parameter "yAxis").Value
                                    let xAxis = (parameter "xAxis").Value
                                    $"let {name} = getNexusInfo {input} {datasetPath} {frameAxis} {yAxis} {xAxis}"
                                | _ -> $"let {name} = getImageInfo {input} {suffix}"
                            { Name = name
                              Dependencies = parameterBindingDependencies node |> Set.remove name
                              Text = text }
                        | "Write" ->
                            let parameter key =
                                node.Parameters
                                |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
                                |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node index key)
                                |> Option.defaultValue { Value = savedParamValue key node; IsLinked = false }

                            let stringArgument key =
                                let expression = parameter key
                                if expression.IsLinked then expression.Value else quote expression.Value

                            let name = imageInfoNamesByNodeId |> Map.find node.Id
                            let expression = pipelineExpression Set.empty node
                            let output = stringArgument "output"
                            let suffix = stringArgument "suffix"
                            let format = savedParamValue "format" node
                            let infoExpression =
                                match format with
                                | "Volume file" -> $"getImageFileInfo {output} {suffix}"
                                | "OME-Zarr" -> $"getZarrInfo {output} 0 0"
                                | "NeXus/HDF5" ->
                                    let datasetPath = stringArgument "datasetPath"
                                    let frameAxis = (parameter "frameAxis").Value
                                    let yAxis = (parameter "yAxis").Value
                                    let xAxis = (parameter "xAxis").Value
                                    $"getNexusInfo {output} {datasetPath} {frameAxis} {yAxis} {xAxis}"
                                | _ -> $"getImageInfo {output} {suffix}"
                            let body = indentBlock 4 $"{expression}{newLine}|> sink{newLine}{infoExpression}"

                            { Name = name
                              Dependencies =
                                  Set.union
                                      (pipelineBindingDependencies Set.empty node)
                                      (parameterBindingDependencies node)
                                  |> Set.remove name
                              Text = $"let {name} ={newLine}{body}" }
                        | _ ->
                            failwith $"Unsupported ImageInfo producer: {node.FunctionId}")

                let imageInfoExpandBindings =
                    imageInfoExpandNodesWithLinkedOutputs
                    |> Array.map (fun node ->
                        let name = imageInfoNamesByNodeId |> Map.find node.Id

                        let sourceName =
                            graph.Edges
                            |> Array.tryFind (fun edge ->
                                edge.ToNode = node.Id
                                && edge.ToKind <> "parameterInput"
                                && edge.ToPort = 0)
                            |> Option.bind bindingNameForOutput
                            |> Option.defaultWith (fun () -> failwith $"Expand node '{node.Id}' is missing a ImageInfo input.")

                        { Name = name
                          Dependencies = Set.singleton sourceName
                          Text = $"let {name} = {sourceName}" })

                Array.concat [ imageInfoProducerBindings; imageInfoExpandBindings ]

            let chunkInfoBindings =
                let chunkInfoProducerBindings =
                    chunkInfoProducerNodesWithLinkedOutputs
                    |> Array.map (fun node ->
                        let parameter key =
                            node.Parameters
                            |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
                            |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node index key)
                            |> Option.defaultValue { Value = savedParamValue key node; IsLinked = false }

                        let stringArgument key =
                            let expression = parameter key
                            if expression.IsLinked then expression.Value else quote expression.Value

                        match node.FunctionId with
                        | "GetChunkInfo" ->
                            let name, text = chunkInfoBinding graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId imageInfoNamesByNodeId chunkInfoNamesByNodeId serialGeometryNamesByNodeId node

                            { Name = name
                              Dependencies = parameterBindingDependencies node |> Set.remove name
                              Text = text }
                        | "WriteChunks" ->
                            let name = chunkInfoNamesByNodeId |> Map.find node.Id
                            let expression = pipelineExpression Set.empty node
                            let output = stringArgument "output"
                            let suffix = stringArgument "suffix"
                            let body = indentBlock 4 $"{expression}{newLine}|> sink{newLine}getChunkInfo {output} {suffix}"
                            { Name = name
                              Dependencies =
                                  Set.union
                                      (pipelineBindingDependencies Set.empty node)
                                      (parameterBindingDependencies node)
                                  |> Set.remove name
                              Text = $"let {name} ={newLine}{body}" }
                        | _ ->
                            failwith $"Unsupported ChunkInfo producer: {node.FunctionId}")

                let chunkInfoExpandBindings =
                    chunkInfoExpandNodesWithLinkedOutputs
                    |> Array.map (fun node ->
                        let name = chunkInfoNamesByNodeId |> Map.find node.Id
                        let sourceName =
                            graph.Edges
                            |> Array.tryFind (fun edge ->
                                edge.ToNode = node.Id
                                && edge.ToKind <> "parameterInput"
                                && edge.ToPort = 0)
                            |> Option.bind bindingNameForOutput
                            |> Option.defaultWith (fun () -> failwith $"Expand node '{node.Id}' is missing a ChunkInfo input.")

                        { Name = name
                          Dependencies = Set.singleton sourceName
                          Text = $"let {name} = {sourceName}" })

                Array.concat [ chunkInfoProducerBindings; chunkInfoExpandBindings ]

            let serialGeometryBindings =
                serialGeometryNodesWithLinkedOutputs
                |> Array.map (fun node ->
                    let name = serialGeometryNamesByNodeId |> Map.find node.Id
                    let expression = pipelineExpression Set.empty node
                    let body = indentBlock 4 $"{expression}{newLine}|> drain"

                    { Name = name
                      Dependencies = pipelineBindingDependencies Set.empty node |> Set.remove name
                      Text = $"let {name} ={newLine}{body}" })

            let bindings = Array.concat [ scalarBindings; statsBindings; translationTableBindings; histogramBindings; quantileBindings; imageInfoBindings; chunkInfoBindings; serialGeometryBindings ]
            let bindingsByName = bindings |> Array.map (fun binding -> binding.Name, binding) |> Map.ofArray
            let visited = HashSet<string>()
            let ordered = ResizeArray<NamedBinding>()

            let rec visit (binding: NamedBinding) =
                if visited.Add binding.Name then
                    binding.Dependencies
                    |> Seq.choose (fun dependency -> bindingsByName |> Map.tryFind dependency)
                    |> Seq.iter visit

                    ordered.Add binding

            bindings |> Array.iter visit
            ordered |> Seq.toArray

        builder.AppendLine("open StackProcessing") |> ignore

        builder.AppendLine() |> ignore

        let bindings = orderedBindings ()
        let bindingsByName = bindings |> Array.map (fun binding -> binding.Name, binding) |> Map.ofArray

        let postRootNames =
            Array.append
                (imageInfoProducerNodesWithLinkedOutputs
                 |> Array.choose (fun node ->
                     if node.FunctionId = "Write" then
                         imageInfoNamesByNodeId |> Map.tryFind node.Id
                     else
                         None))
                (chunkInfoProducerNodesWithLinkedOutputs
                 |> Array.choose (fun node ->
                     if node.FunctionId = "Write" then
                         chunkInfoNamesByNodeId |> Map.tryFind node.Id
                     else
                         None))
            |> Set.ofArray

        let rec bindingIsPost visited name =
            if postRootNames |> Set.contains name then
                true
            elif visited |> Set.contains name then
                false
            else
                match bindingsByName |> Map.tryFind name with
                | Some binding ->
                    binding.Dependencies
                    |> Seq.exists (bindingIsPost (visited |> Set.add name))
                | None ->
                    false

        let terminalExpressions = generatedPipelines ()

        let terminalIsPost (terminal: TerminalExpression) =
            terminal.Dependencies |> Seq.exists (bindingIsPost Set.empty)

        let preBindings =
            bindings |> Array.filter (fun binding -> not (bindingIsPost Set.empty binding.Name))

        let postBindings =
            bindings |> Array.filter (fun binding -> bindingIsPost Set.empty binding.Name)

        let preTerminals =
            terminalExpressions |> Array.filter (terminalIsPost >> not)

        let postTerminals =
            terminalExpressions |> Array.filter terminalIsPost

        let appendBlock (lines: string array) =
            if lines.Length > 0 then
                lines
                |> Array.iteri (fun index text ->
                    if index > 0 then
                        builder.AppendLine() |> ignore

                    builder.AppendLine(text) |> ignore)

                builder.AppendLine() |> ignore

        appendBlock (preBindings |> Array.map _.Text)
        appendBlock (preTerminals |> Array.map _.Text)
        appendBlock (postBindings |> Array.map _.Text)
        appendBlock (postTerminals |> Array.map _.Text)

        match validationError with
        | Some message -> $"// Cannot generate F#: {message}"
        | None -> builder.ToString().TrimEnd()
